! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: flang-new -fc1 -fdefault-integer-8 -emit-fir %s -o - | FileCheck %s

! Test OPTIONAL lowering on caller/callee and PRESENT intrinsic.
module opt
  implicit none
  type t
    real, allocatable :: p(:)
  end type
contains

! Test simple scalar optional
! CHECK-LABEL: func @_QMoptPintrinsic_scalar(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<f32> {fir.bindc_name = "x", fir.optional}) {
subroutine intrinsic_scalar(x)
  real, optional :: x
  ! CHECK: fir.is_present %[[arg0]] : (!fir.ref<f32>) -> i1
  print *, present(x)
end subroutine
! CHECK-LABEL: @_QMoptPcall_intrinsic_scalar()
subroutine call_intrinsic_scalar()
  ! CHECK: %[[x:.*]] = fir.alloca f32
  real :: x
  ! CHECK: fir.call @_QMoptPintrinsic_scalar(%[[x]]) {{.*}}: (!fir.ref<f32>) -> ()
  call intrinsic_scalar(x)
  ! CHECK: %[[absent:.*]] = fir.absent !fir.ref<f32>
  ! CHECK: fir.call @_QMoptPintrinsic_scalar(%[[absent]]) {{.*}}: (!fir.ref<f32>) -> ()
  call intrinsic_scalar()
end subroutine

! Test explicit shape array optional
! CHECK-LABEL: func @_QMoptPintrinsic_f77_array(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "x", fir.optional}) {
subroutine intrinsic_f77_array(x)
  real, optional :: x(100)
  ! CHECK: fir.is_present %[[arg0]] : (!fir.ref<!fir.array<100xf32>>) -> i1
  print *, present(x)
end subroutine
! CHECK-LABEL: func @_QMoptPcall_intrinsic_f77_array()
subroutine call_intrinsic_f77_array()
  ! CHECK: %[[x:.*]] = fir.alloca !fir.array<100xf32>
  real :: x(100)
  ! CHECK: fir.call @_QMoptPintrinsic_f77_array(%[[x]]) {{.*}}: (!fir.ref<!fir.array<100xf32>>) -> ()
  call intrinsic_f77_array(x)
  ! CHECK: %[[absent:.*]] = fir.absent !fir.ref<!fir.array<100xf32>>
  ! CHECK: fir.call @_QMoptPintrinsic_f77_array(%[[absent]]) {{.*}}: (!fir.ref<!fir.array<100xf32>>) -> ()
  call intrinsic_f77_array()
end subroutine

! Test optional character scalar
! CHECK-LABEL: func @_QMoptPcharacter_scalar(
! CHECK-SAME: %[[arg0:.*]]: !fir.boxchar<1> {fir.bindc_name = "x", fir.optional}) {
subroutine character_scalar(x)
  ! CHECK: %[[unboxed:.*]]:2 = fir.unboxchar %[[arg0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  character(10), optional :: x
  ! CHECK: fir.is_present %[[unboxed]]#0 : (!fir.ref<!fir.char<1,?>>) -> i1
  print *, present(x)
end subroutine
! CHECK-LABEL: func @_QMoptPcall_character_scalar()
subroutine call_character_scalar()
  ! CHECK: %[[addr:.*]] = fir.alloca !fir.char<1,10>
  character(10) :: x
  ! CHECK: %[[addrCast:.*]] = fir.convert %[[addr]]
  ! CHECK: %[[x:.*]] = fir.emboxchar %[[addrCast]], {{.*}}
  ! CHECK: fir.call @_QMoptPcharacter_scalar(%[[x]]) {{.*}}: (!fir.boxchar<1>) -> ()
  call character_scalar(x)
  ! CHECK: %[[absent:.*]] = fir.absent !fir.boxchar<1>
  ! CHECK: fir.call @_QMoptPcharacter_scalar(%[[absent]]) {{.*}}: (!fir.boxchar<1>) -> ()
  call character_scalar()
end subroutine

! Test optional character function
! CHECK-LABEL: func @_QMoptPchar_proc(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.char<1,3>>,
character(len=3) function char_proc(i)
  integer :: i
  char_proc = "XYZ"
end function
! CHECK-LABEL: func @_QMoptPuse_char_proc(
! CHECK-SAME: %[[arg0:.*]]: tuple<!fir.boxproc<() -> ()>, i64> {fir.char_proc},
subroutine use_char_proc(f, c)
  optional :: f
  interface
    character(len=3) function f(i)
      integer :: i
    end function
  end interface
  character(len=3) :: c
! CHECK: %[[boxProc:.*]] = fir.extract_value %[[arg0]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK: %[[procAddr:.*]] = fir.box_addr %[[boxProc]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK: %{{.*}} = fir.is_present %[[procAddr]] : (() -> ()) -> i1
  if (present(f)) then
    c = f(0)
  else
    c = "ABC"
  end if
end subroutine
! CHECK-LABEL: func @_QMoptPcall_use_char_proc(
subroutine call_use_char_proc()
  character(len=3) :: c
! CHECK: %[[boxProc:.*]] = fir.absent !fir.boxproc<() -> ()>
! CHECK: %[[undef:.*]] = fir.undefined index
! CHECK: %[[charLen:.*]] = fir.convert %[[undef]] : (index) -> i64
! CHECK: %[[tuple:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple2:.*]] = fir.insert_value %[[tuple]], %[[boxProc]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple3:.*]] = fir.insert_value %[[tuple2]], %[[charLen]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: fir.call @_QMoptPuse_char_proc(%[[tuple3]], %{{.*}}){{.*}} : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxchar<1>) -> ()
  call use_char_proc(c=c)
! CHECK: %[[funcAddr:.*]] = fir.address_of(@_QMoptPchar_proc) : (!fir.ref<!fir.char<1,3>>, index, {{.*}}) -> !fir.boxchar<1>
! CHECK: %[[c3:.*]] = arith.constant 3 : i64
! CHECK: %[[boxProc2:.*]] = fir.emboxproc %[[funcAddr]] : ((!fir.ref<!fir.char<1,3>>, index, {{.*}}) -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
! CHECK: %[[tuple4:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple5:.*]] = fir.insert_value %[[tuple4]], %[[boxProc2]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple6:.*]] = fir.insert_value %[[tuple5]], %[[c3]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: fir.call @_QMoptPuse_char_proc(%[[tuple6]], {{.*}}){{.*}} : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxchar<1>) -> ()
  call use_char_proc(char_proc, c)
end subroutine

! Test optional assumed shape
! CHECK-LABEL: func @_QMoptPassumed_shape(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.optional}) {
subroutine assumed_shape(x)
  real, optional :: x(:)
  ! CHECK: fir.is_present %[[arg0]] : (!fir.box<!fir.array<?xf32>>) -> i1
  print *, present(x)
end subroutine
! CHECK: func @_QMoptPcall_assumed_shape()
subroutine call_assumed_shape()
  ! CHECK: %[[addr:.*]] = fir.alloca !fir.array<100xf32>
  real :: x(100)
  ! CHECK: %[[embox:.*]] = fir.embox %[[addr]]
  ! CHECK: %[[x:.*]] = fir.convert %[[embox]] : (!fir.box<!fir.array<100xf32>>) -> !fir.box<!fir.array<?xf32>>
  ! CHECK: fir.call @_QMoptPassumed_shape(%[[x]]) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
  call assumed_shape(x)
  ! CHECK: %[[absent:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
  ! CHECK: fir.call @_QMoptPassumed_shape(%[[absent]]) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
  call assumed_shape()
end subroutine

! Test optional allocatable
! CHECK: func @_QMoptPallocatable_array(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {fir.bindc_name = "x", fir.optional}) {
subroutine allocatable_array(x)
  real, allocatable, optional :: x(:)
  ! CHECK: fir.is_present %[[arg0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> i1
  print *, present(x)
end subroutine
! CHECK: func @_QMoptPcall_allocatable_array()
subroutine call_allocatable_array()
  ! CHECK: %[[x:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
  real, allocatable :: x(:)
  ! CHECK: fir.call @_QMoptPallocatable_array(%[[x]]) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ()
  call allocatable_array(x)
  ! CHECK: %[[absent:.*]] = fir.absent !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK: fir.call @_QMoptPallocatable_array(%[[absent]]) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ()
  call allocatable_array()
end subroutine

! CHECK: func @_QMoptPallocatable_to_assumed_optional_array(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>{{.*}}) {
subroutine allocatable_to_assumed_optional_array(x)
  real, allocatable :: x(:)

  ! CHECK: %[[xboxload:.*]] = fir.load %[[arg0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK: %[[xptr:.*]] = fir.box_addr %[[xboxload]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  ! CHECK: %[[xaddr:.*]] = fir.convert %[[xptr]] : (!fir.heap<!fir.array<?xf32>>) -> i64
  ! CHECK: %[[isAlloc:.*]] = arith.cmpi ne, %[[xaddr]], %c0{{.*}} : i64
  ! CHECK: %[[absent:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
  ! CHECK: %[[embox:.*]] = fir.embox %{{.*}}
  ! CHECK: %[[actual:.*]] = arith.select %[[isAlloc]], %[[embox]], %[[absent]] : !fir.box<!fir.array<?xf32>>
  ! CHECK: fir.call @_QMoptPassumed_shape(%[[actual]]) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
  call assumed_shape(x)
end subroutine

! CHECK-LABEL: func @_QMoptPalloc_component_to_optional_assumed_shape(
subroutine alloc_component_to_optional_assumed_shape(x)
  type(t) :: x(100)
  ! CHECK-DAG: %[[isAlloc:.*]] = arith.cmpi ne
  ! CHECK-DAG: %[[absent:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
  ! CHECK: %[[select:.*]] = arith.select %[[isAlloc]], %{{.*}}, %[[absent]] : !fir.box<!fir.array<?xf32>>
  ! CHECK: fir.call @_QMoptPassumed_shape(%[[select]])
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
  ! CHECK: %[[null_ptr:.*]] = fir.alloca !fir.box<!fir.ptr<none>>
  ! CHECK: %[[null:.*]] = fir.zero_bits !fir.ptr<none>
  ! CHECK: %[[null_box:.*]] = fir.embox %[[null]] : (!fir.ptr<none>) -> !fir.box<!fir.ptr<none>>
  ! CHECK: fir.store %[[null_box]] to %[[null_ptr]] : !fir.ref<!fir.box<!fir.ptr<none>>>
  ! CHECK: fir.call @_QMoptPassumed_shape(%{{.*}}) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
 call assumed_shape(null())
end subroutine null_as_optional

end module
