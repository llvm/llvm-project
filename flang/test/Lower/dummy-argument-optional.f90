! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fdefault-integer-8 -emit-hlfir %s -o - | FileCheck %s

! Test OPTIONAL lowering on caller/callee and PRESENT intrinsic.
module opt
  implicit none
  type t
    real, allocatable :: p(:)
  end type
contains

! Test simple scalar optional
! CHECK-LABEL: func @_QMoptPintrinsic_scalar(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<f32> {fir.bindc_name = "x", fir.optional}) {
subroutine intrinsic_scalar(x)
  real, optional :: x
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]] {{.*}}uniq_name = "_QMoptFintrinsic_scalarEx"{{.*}}
  ! CHECK: fir.is_present %[[DECL]]#0 : (!fir.ref<f32>) -> i1
  print *, present(x)
end subroutine
! CHECK-LABEL: @_QMoptPcall_intrinsic_scalar()
subroutine call_intrinsic_scalar()
  ! CHECK: %[[X:.*]] = fir.alloca f32
  ! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]]
  real :: x
  ! CHECK: fir.call @_QMoptPintrinsic_scalar(%[[X_DECL]]#0) {{.*}}: (!fir.ref<f32>) -> ()
  call intrinsic_scalar(x)
  ! CHECK: %[[ABSENT:.*]] = fir.absent !fir.ref<f32>
  ! CHECK: fir.call @_QMoptPintrinsic_scalar(%[[ABSENT]]) {{.*}}: (!fir.ref<f32>) -> ()
  call intrinsic_scalar()
end subroutine

! Test explicit shape array optional
! CHECK-LABEL: func @_QMoptPintrinsic_f77_array(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "x", fir.optional}) {
subroutine intrinsic_f77_array(x)
  real, optional :: x(100)
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]]{{.*}}uniq_name = "_QMoptFintrinsic_f77_arrayEx"{{.*}}
  ! CHECK: fir.is_present %[[DECL]]#0 : (!fir.ref<!fir.array<100xf32>>) -> i1
  print *, present(x)
end subroutine
! CHECK-LABEL: func @_QMoptPcall_intrinsic_f77_array()
subroutine call_intrinsic_f77_array()
  ! CHECK: %[[X:.*]] = fir.alloca !fir.array<100xf32>
  real :: x(100)
  ! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]]
  ! CHECK: fir.call @_QMoptPintrinsic_f77_array(%[[X_DECL]]#0) {{.*}}: (!fir.ref<!fir.array<100xf32>>) -> ()
  call intrinsic_f77_array(x)
  ! CHECK: %[[ABSENT:.*]] = fir.absent !fir.ref<!fir.array<100xf32>>
  ! CHECK: fir.call @_QMoptPintrinsic_f77_array(%[[ABSENT]]) {{.*}}: (!fir.ref<!fir.array<100xf32>>) -> ()
  call intrinsic_f77_array()
end subroutine

! Test optional character scalar
! CHECK-LABEL: func @_QMoptPcharacter_scalar(
! CHECK-SAME: %[[ARG0:.*]]: !fir.boxchar<1> {fir.bindc_name = "x", fir.optional}) {
subroutine character_scalar(x)
  ! CHECK: %[[UNBOX:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[REF:.*]] = fir.convert %[[UNBOX]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[REF]] typeparams %c10 dummy_scope %{{.*}} arg 1 {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QMoptFcharacter_scalarEx"} : (!fir.ref<!fir.char<1,10>>, index, !fir.dscope) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>)
  character(10), optional :: x
  ! CHECK: fir.is_present %[[DECL]]#0 : (!fir.ref<!fir.char<1,10>>) -> i1
  print *, present(x)
end subroutine
! CHECK-LABEL: func @_QMoptPcall_character_scalar()
subroutine call_character_scalar()
  ! CHECK: %[[X:.*]] = fir.alloca !fir.char<1,10>
  character(10) :: x
  ! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]]
  ! CHECK: %[[EMBOX:.*]] = fir.emboxchar %[[X_DECL]]#0, %c10 : (!fir.ref<!fir.char<1,10>>, index) -> !fir.boxchar<1>
  ! CHECK: fir.call @_QMoptPcharacter_scalar(%[[EMBOX]]) {{.*}}: (!fir.boxchar<1>) -> ()
  call character_scalar(x)
  ! CHECK: %[[ABSENT:.*]] = fir.absent !fir.boxchar<1>
  ! CHECK: fir.call @_QMoptPcharacter_scalar(%[[ABSENT]]) {{.*}}: (!fir.boxchar<1>) -> ()
  call character_scalar()
end subroutine

! Test optional character function
! CHECK-LABEL: func @_QMoptPchar_proc(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.char<1,3>>,
character(len=3) function char_proc(i)
  integer :: i
  char_proc = "XYZ"
end function
! CHECK-LABEL: func @_QMoptPuse_char_proc(
! CHECK-SAME: %[[ARG0:.*]]: tuple<!fir.boxproc<() -> ()>, i64> {fir.char_proc},
subroutine use_char_proc(f, c)
  optional :: f
  interface
    character(len=3) function f(i)
      integer :: i
    end function
  end interface
  character(len=3) :: c
! CHECK: %[[BOXPROC:.*]] = fir.extract_value %[[ARG0]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK: %[[IS_PRESENT:.*]] = fir.is_present %{{.*}} : (!fir.boxproc<() -> ()>) -> i1
  if (present(f)) then
    c = f(0)
  else
    c = "ABC"
  end if
end subroutine
! CHECK-LABEL: func @_QMoptPcall_use_char_proc(
subroutine call_use_char_proc()
  character(len=3) :: c
! CHECK: %[[ABSENT:.*]] = fir.absent !fir.boxproc<() -> ()>
! CHECK: %[[UNDEF:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[TUPLE:.*]] = fir.insert_value %[[UNDEF]], %[[ABSENT]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[TUPLE2:.*]] = fir.insert_value %[[TUPLE]], %{{.*}}, [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: fir.call @_QMoptPuse_char_proc(%[[TUPLE2]], %{{.*}}){{.*}} : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxchar<1>) -> ()
  call use_char_proc(c=c)
! CHECK: %[[ADDR:.*]] = fir.address_of(@_QMoptPchar_proc) : (!fir.ref<!fir.char<1,3>>, index, {{.*}}) -> !fir.boxchar<1>
! CHECK: %[[BOXPROC:.*]] = fir.emboxproc %[[ADDR]] : ((!fir.ref<!fir.char<1,3>>, index, {{.*}}) -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
! CHECK: %[[UNDEF:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[TUPLE:.*]] = fir.insert_value %[[UNDEF]], %[[BOXPROC]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[TUPLE2:.*]] = fir.insert_value %[[TUPLE]], %{{.*}}, [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: fir.call @_QMoptPuse_char_proc(%[[TUPLE2]], {{.*}}){{.*}} : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxchar<1>) -> ()
  call use_char_proc(char_proc, c)
end subroutine

! Test optional assumed shape
! CHECK-LABEL: func @_QMoptPassumed_shape(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.optional}) {
subroutine assumed_shape(x)
  real, optional :: x(:)
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]] {{.*}}uniq_name = "_QMoptFassumed_shapeEx"{{.*}}
  ! CHECK: fir.is_present %[[DECL]]#1 : (!fir.box<!fir.array<?xf32>>) -> i1
  print *, present(x)
end subroutine
! CHECK: func @_QMoptPcall_assumed_shape()
subroutine call_assumed_shape()
  ! CHECK: %[[X:.*]] = fir.alloca !fir.array<100xf32>
  real :: x(100)
  ! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]]
  ! CHECK: %[[EMBOX:.*]] = fir.embox %[[X_DECL]]#0(%{{.*}}) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<100xf32>>
  ! CHECK: %[[BOX:.*]] = fir.convert %[[EMBOX]] : (!fir.box<!fir.array<100xf32>>) -> !fir.box<!fir.array<?xf32>>
  ! CHECK: fir.call @_QMoptPassumed_shape(%[[BOX]]) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
  call assumed_shape(x)
  ! CHECK: %[[ABSENT:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
  ! CHECK: fir.call @_QMoptPassumed_shape(%[[ABSENT]]) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
  call assumed_shape()
end subroutine

! Test optional allocatable
! CHECK: func @_QMoptPallocatable_array(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {fir.bindc_name = "x", fir.optional}) {
subroutine allocatable_array(x)
  real, allocatable, optional :: x(:)
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]] {{.*}}uniq_name = "_QMoptFallocatable_arrayEx"{{.*}}
  ! CHECK: fir.is_present %[[DECL]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> i1
  print *, present(x)
end subroutine
! CHECK: func @_QMoptPcall_allocatable_array()
subroutine call_allocatable_array()
  ! CHECK: %[[X:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
  real, allocatable :: x(:)
  ! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]]
  ! CHECK: fir.call @_QMoptPallocatable_array(%[[X_DECL]]#0) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ()
  call allocatable_array(x)
  ! CHECK: %[[ABSENT:.*]] = fir.absent !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK: fir.call @_QMoptPallocatable_array(%[[ABSENT]]) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ()
  call allocatable_array()
end subroutine

! CHECK: func @_QMoptPallocatable_to_assumed_optional_array(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>{{.*}}) {
subroutine allocatable_to_assumed_optional_array(x)
  real, allocatable :: x(:)
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]]
  ! CHECK: %[[LOAD:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  ! CHECK: %[[ADDR_I64:.*]] = fir.convert %[[ADDR]] : (!fir.heap<!fir.array<?xf32>>) -> i64
  ! CHECK: %[[IS_ALLOC:.*]] = arith.cmpi ne, %[[ADDR_I64]], %c0{{.*}} : i64
  ! CHECK: %[[ARG:.*]] = fir.if %[[IS_ALLOC]] -> (!fir.box<!fir.array<?xf32>>) {
  ! CHECK:   %[[LOAD2:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK:   %[[REBOX:.*]] = fir.rebox %[[LOAD2]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.box<!fir.array<?xf32>>
  ! CHECK:   fir.result %[[REBOX]] : !fir.box<!fir.array<?xf32>>
  ! CHECK: } else {
  ! CHECK:   %[[ABSENT:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
  ! CHECK:   fir.result %[[ABSENT]] : !fir.box<!fir.array<?xf32>>
  ! CHECK: }
  ! CHECK: fir.call @_QMoptPassumed_shape(%[[ARG]]) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
  call assumed_shape(x)
end subroutine

! CHECK-LABEL: func @_QMoptPalloc_component_to_optional_assumed_shape(
subroutine alloc_component_to_optional_assumed_shape(x)
  type(t) :: x(100)
  ! CHECK: %[[IS_ALLOC:.*]] = arith.cmpi ne
  ! CHECK: %[[SELECT:.*]] = fir.if %[[IS_ALLOC]] -> (!fir.box<!fir.array<?xf32>>) {
  ! CHECK:   fir.result %{{.*}} : !fir.box<!fir.array<?xf32>>
  ! CHECK: } else {
  ! CHECK:   %[[ABSENT:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
  ! CHECK:   fir.result %[[ABSENT]] : !fir.box<!fir.array<?xf32>>
  ! CHECK: }
  ! CHECK: fir.call @_QMoptPassumed_shape(%[[SELECT]])
  call assumed_shape(x(55)%p)
end subroutine

! CHECK-LABEL: func @_QMoptPalloc_component_eval_only_once(
subroutine alloc_component_eval_only_once(x)
  integer, external :: ifoo
  type(t) :: x(100)
  ! Verify that the index in the component reference are not evaluated twice
  ! because if the optional handling logic.
  ! CHECK: fir.call @_QPifoo()
  ! CHECK-NOT: fir.call @_QPifoo()
  call assumed_shape(x(ifoo())%p)
end subroutine

! CHECK-LABEL: func @_QMoptPnull_as_optional() {
subroutine null_as_optional
  ! CHECK: %[[ABSENT:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
  ! CHECK: fir.call @_QMoptPassumed_shape(%[[ABSENT]]) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
 call assumed_shape(null())
end subroutine null_as_optional

end module
