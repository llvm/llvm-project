! Test passing pointer, allocatables, and optional assumed shapes to optional
! explicit shapes (see F2018 15.5.2.12).
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s
module optional_tests
implicit none
interface
subroutine takes_opt_scalar(i)
  integer, optional :: i
end subroutine
subroutine takes_opt_scalar_char(c)
  character(*), optional :: c
end subroutine
subroutine takes_opt_explicit_shape(x)
  real, optional :: x(100)
end subroutine
subroutine takes_opt_explicit_shape_intentout(x)
  real, optional, intent(out) :: x(100)
end subroutine
subroutine takes_opt_explicit_shape_intentin(x)
  real, optional, intent(in) :: x(100)
end subroutine
subroutine takes_opt_explicit_shape_char(c)
  character(*), optional :: c(100)
end subroutine
function returns_pointer()
  real, pointer :: returns_pointer(:)
end function
end interface
contains

! -----------------------------------------------------------------------------
!     Test passing scalar pointers and allocatables to an optional
! -----------------------------------------------------------------------------
! Here, nothing optional specific is expected, the address is passed, and its
! allocation/association status match the dummy presence status.

! CHECK-LABEL: func @_QMoptional_testsPpass_pointer_scalar(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>>{{.*}}) {
subroutine pass_pointer_scalar(i)
  integer, pointer :: i
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}uniq_name = "_QMoptional_testsFpass_pointer_scalarEi"{{.*}}
  call takes_opt_scalar(i)
  ! CHECK: %[[LOAD:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
  ! CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
  ! CHECK: %[[ADDR_I64:.*]] = fir.convert %[[ADDR]] : (!fir.ptr<i32>) -> i64
  ! CHECK: %[[IS_PRESENT:.*]] = arith.cmpi ne, %[[ADDR_I64]], %c0_i64 : i64
  ! CHECK: %[[ARG:.*]] = fir.if %[[IS_PRESENT]] -> (!fir.ref<i32>) {
  ! CHECK:   %[[LOAD2:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
  ! CHECK:   %[[ADDR2:.*]] = fir.box_addr %[[LOAD2]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
  ! CHECK:   %[[REF:.*]] = fir.convert %[[ADDR2]] : (!fir.ptr<i32>) -> !fir.ref<i32>
  ! CHECK:   fir.result %[[REF]] : !fir.ref<i32>
  ! CHECK: } else {
  ! CHECK:   %[[ABSENT:.*]] = fir.absent !fir.ref<i32>
  ! CHECK:   fir.result %[[ABSENT]] : !fir.ref<i32>
  ! CHECK: }
  ! CHECK: fir.call @_QPtakes_opt_scalar(%[[ARG]]) {{.*}} : (!fir.ref<i32>) -> ()
end subroutine

! CHECK-LABEL: func @_QMoptional_testsPpass_allocatable_scalar(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>>{{.*}}) {
subroutine pass_allocatable_scalar(i)
  integer, allocatable :: i
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}uniq_name = "_QMoptional_testsFpass_allocatable_scalarEi"{{.*}}
  call takes_opt_scalar(i)
  ! CHECK: %[[LOAD:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
  ! CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  ! CHECK: %[[ADDR_I64:.*]] = fir.convert %[[ADDR]] : (!fir.heap<i32>) -> i64
  ! CHECK: %[[IS_PRESENT:.*]] = arith.cmpi ne, %[[ADDR_I64]], %c0_i64 : i64
  ! CHECK: %[[ARG:.*]] = fir.if %[[IS_PRESENT]] -> (!fir.ref<i32>) {
  ! CHECK:   %[[LOAD2:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
  ! CHECK:   %[[ADDR2:.*]] = fir.box_addr %[[LOAD2]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  ! CHECK:   %[[REF:.*]] = fir.convert %[[ADDR2]] : (!fir.heap<i32>) -> !fir.ref<i32>
  ! CHECK:   fir.result %[[REF]] : !fir.ref<i32>
  ! CHECK: } else {
  ! CHECK:   %[[ABSENT:.*]] = fir.absent !fir.ref<i32>
  ! CHECK:   fir.result %[[ABSENT]] : !fir.ref<i32>
  ! CHECK: }
  ! CHECK: fir.call @_QPtakes_opt_scalar(%[[ARG]]) {{.*}} : (!fir.ref<i32>) -> ()
end subroutine

! CHECK-LABEL: func @_QMoptional_testsPpass_pointer_scalar_char(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>{{.*}}) {
subroutine pass_pointer_scalar_char(c)
  character(:), pointer :: c
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}uniq_name = "_QMoptional_testsFpass_pointer_scalar_charEc"{{.*}}
  call takes_opt_scalar_char(c)
  ! CHECK: %[[LOAD:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
  ! CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> !fir.ptr<!fir.char<1,?>>
  ! CHECK: %[[ADDR_I64:.*]] = fir.convert %[[ADDR]] : (!fir.ptr<!fir.char<1,?>>) -> i64
  ! CHECK: %[[IS_PRESENT:.*]] = arith.cmpi ne, %[[ADDR_I64]], %c0_i64 : i64
  ! CHECK: %[[ARG:.*]] = fir.if %[[IS_PRESENT]] -> (!fir.boxchar<1>) {
  ! CHECK:   %[[LOAD2:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
  ! CHECK:   %[[ADDR2:.*]] = fir.box_addr %[[LOAD2]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> !fir.ptr<!fir.char<1,?>>
  ! CHECK:   %[[LOAD3:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
  ! CHECK:   %[[LEN:.*]] = fir.box_elesize %[[LOAD3]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> index
  ! CHECK:   %[[EMBOX:.*]] = fir.emboxchar %[[ADDR2]], %[[LEN]] : (!fir.ptr<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK:   fir.result %[[EMBOX]] : !fir.boxchar<1>
  ! CHECK: } else {
  ! CHECK:   %[[ABSENT:.*]] = fir.absent !fir.boxchar<1>
  ! CHECK:   fir.result %[[ABSENT]] : !fir.boxchar<1>
  ! CHECK: }
  ! CHECK: fir.call @_QPtakes_opt_scalar_char(%[[ARG]]) {{.*}} : (!fir.boxchar<1>) -> ()
end subroutine

! CHECK-LABEL: func @_QMoptional_testsPpass_allocatable_scalar_char(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>{{.*}}) {
subroutine pass_allocatable_scalar_char(c)
  character(:), allocatable :: c
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}uniq_name = "_QMoptional_testsFpass_allocatable_scalar_charEc"{{.*}}
  call takes_opt_scalar_char(c)
  ! CHECK: %[[LOAD:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  ! CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
  ! CHECK: %[[ADDR_I64:.*]] = fir.convert %[[ADDR]] : (!fir.heap<!fir.char<1,?>>) -> i64
  ! CHECK: %[[IS_PRESENT:.*]] = arith.cmpi ne, %[[ADDR_I64]], %c0_i64 : i64
  ! CHECK: %[[ARG:.*]] = fir.if %[[IS_PRESENT]] -> (!fir.boxchar<1>) {
  ! CHECK:   %[[LOAD2:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  ! CHECK:   %[[ADDR2:.*]] = fir.box_addr %[[LOAD2]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
  ! CHECK:   %[[LOAD3:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  ! CHECK:   %[[LEN:.*]] = fir.box_elesize %[[LOAD3]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
  ! CHECK:   %[[EMBOX:.*]] = fir.emboxchar %[[ADDR2]], %[[LEN]] : (!fir.heap<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK:   fir.result %[[EMBOX]] : !fir.boxchar<1>
  ! CHECK: } else {
  ! CHECK:   %[[ABSENT:.*]] = fir.absent !fir.boxchar<1>
  ! CHECK:   fir.result %[[ABSENT]] : !fir.boxchar<1>
  ! CHECK: }
  ! CHECK: fir.call @_QPtakes_opt_scalar_char(%[[ARG]]) {{.*}} : (!fir.boxchar<1>) -> ()
end subroutine

! -----------------------------------------------------------------------------
!     Test passing non contiguous pointers to explicit shape optional
! -----------------------------------------------------------------------------
! The pointer descriptor can be unconditionally read, but the copy-in/copy-out
! must be conditional on the pointer association status in order to get the
! correct present/absent aspect.

! CHECK-LABEL: func @_QMoptional_testsPpass_pointer_array(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>{{.*}}) {
subroutine pass_pointer_array(i)
  real, pointer :: i(:)
  ! CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}uniq_name = "_QMoptional_testsFpass_pointer_arrayEi"{{.*}}
  call takes_opt_explicit_shape(i)
  ! CHECK: %[[LOAD:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
  ! CHECK: %[[ADDR_I64:.*]] = fir.convert %[[ADDR]] : (!fir.ptr<!fir.array<?xf32>>) -> i64
  ! CHECK: %[[IS_PRESENT:.*]] = arith.cmpi ne, %[[ADDR_I64]], %c0_i64 : i64
  ! CHECK: %[[ARG:.*]]:3 = fir.if %[[IS_PRESENT]] -> (!fir.ref<!fir.array<100xf32>>, i1, !fir.box<!fir.ptr<!fir.array<?xf32>>>) {
  ! CHECK:   %[[LOAD2:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK:   %[[COPY_IN:.*]]:2 = hlfir.copy_in %[[LOAD2]] to %[[ALLOCA]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.box<!fir.ptr<!fir.array<?xf32>>>, i1)
  ! CHECK:   %[[BOX_ADDR:.*]] = fir.box_addr %[[COPY_IN]]#0 : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
  ! CHECK:   %[[REF:.*]] = fir.convert %[[BOX_ADDR]] : (!fir.ptr<!fir.array<?xf32>>) -> !fir.ref<!fir.array<100xf32>>
  ! CHECK:   fir.result %[[REF]], %[[COPY_IN]]#1, %[[LOAD2]] : !fir.ref<!fir.array<100xf32>>, i1, !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: } else {
  ! CHECK:   %[[ABSENT:.*]] = fir.absent !fir.ref<!fir.array<100xf32>>
  ! CHECK:   %[[FALSE:.*]] = arith.constant false
  ! CHECK:   %[[ABSENT_BOX:.*]] = fir.absent !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK:   fir.result %[[ABSENT]], %[[FALSE]], %[[ABSENT_BOX]] : !fir.ref<!fir.array<100xf32>>, i1, !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: }
  ! CHECK: fir.call @_QPtakes_opt_explicit_shape(%[[ARG]]#0) {{.*}} : (!fir.ref<!fir.array<100xf32>>) -> ()
  ! CHECK: hlfir.copy_out %[[ALLOCA]], %[[ARG]]#1 to %[[ARG]]#2 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, i1, !fir.box<!fir.ptr<!fir.array<?xf32>>>) -> ()
end subroutine

! CHECK-LABEL: func @_QMoptional_testsPpass_pointer_array_char(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>{{.*}}) {
subroutine pass_pointer_array_char(c)
  character(:), pointer :: c(:)
  ! CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}uniq_name = "_QMoptional_testsFpass_pointer_array_charEc"{{.*}}
  call takes_opt_explicit_shape_char(c)
  ! CHECK: %[[LOAD:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
  ! CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>) -> !fir.ptr<!fir.array<?x!fir.char<1,?>>>
  ! CHECK: %[[ADDR_I64:.*]] = fir.convert %[[ADDR]] : (!fir.ptr<!fir.array<?x!fir.char<1,?>>>) -> i64
  ! CHECK: %[[IS_PRESENT:.*]] = arith.cmpi ne, %[[ADDR_I64]], %c0_i64 : i64
  ! CHECK: %[[ARG:.*]]:3 = fir.if %[[IS_PRESENT]] -> (!fir.boxchar<1>, i1, !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>) {
  ! CHECK:   %[[LOAD2:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
  ! CHECK:   %[[COPY_IN:.*]]:2 = hlfir.copy_in %[[LOAD2]] to %[[ALLOCA]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>) -> (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>, i1)
  ! CHECK:   %[[BOX_ADDR:.*]] = fir.box_addr %[[COPY_IN]]#0 : (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>) -> !fir.ptr<!fir.array<?x!fir.char<1,?>>>
  ! CHECK:   %[[ELE_SIZE:.*]] = fir.box_elesize %[[COPY_IN]]#0 : (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>) -> index
  ! CHECK:   %[[REF:.*]] = fir.convert %[[BOX_ADDR]] : (!fir.ptr<!fir.array<?x!fir.char<1,?>>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK:   %[[EMBOX:.*]] = fir.emboxchar %[[REF]], %[[ELE_SIZE]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK:   fir.result %[[EMBOX]], %[[COPY_IN]]#1, %[[LOAD2]] : !fir.boxchar<1>, i1, !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK: } else {
  ! CHECK:   %[[ABSENT:.*]] = fir.absent !fir.boxchar<1>
  ! CHECK:   %[[FALSE:.*]] = arith.constant false
  ! CHECK:   %[[ABSENT_BOX:.*]] = fir.absent !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK:   fir.result %[[ABSENT]], %[[FALSE]], %[[ABSENT_BOX]] : !fir.boxchar<1>, i1, !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK: }
  ! CHECK: fir.call @_QPtakes_opt_explicit_shape_char(%[[ARG]]#0) {{.*}} : (!fir.boxchar<1>) -> ()
  ! CHECK: hlfir.copy_out %[[ALLOCA]], %[[ARG]]#1 to %[[ARG]]#2 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>, i1, !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>) -> ()
end subroutine

! This case is bit special because the pointer is not a symbol but a function
! result. Test that the copy-in/copy-out is the same as with normal pointers.

! CHECK-LABEL: func @_QMoptional_testsPforward_pointer_array() {
subroutine forward_pointer_array()
  ! CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
  ! CHECK: %[[RES:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>> {bindc_name = ".result"}
  call takes_opt_explicit_shape(returns_pointer())
  ! CHECK: %[[RET:.*]] = fir.call @_QPreturns_pointer() {{.*}} : () -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.save_result %[[RET]] to %[[RES]] : !fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[RES]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
  ! CHECK: %[[LOAD:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
  ! CHECK: %[[ADDR_I64:.*]] = fir.convert %[[ADDR]] : (!fir.ptr<!fir.array<?xf32>>) -> i64
  ! CHECK: %[[IS_PRESENT:.*]] = arith.cmpi ne, %[[ADDR_I64]], %c0_i64 : i64
  ! CHECK: %[[ARG:.*]]:3 = fir.if %[[IS_PRESENT]] -> (!fir.ref<!fir.array<100xf32>>, i1, !fir.box<!fir.ptr<!fir.array<?xf32>>>) {
  ! CHECK:   %[[LOAD2:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK:   %[[COPY_IN:.*]]:2 = hlfir.copy_in %[[LOAD2]] to %[[ALLOCA]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.box<!fir.ptr<!fir.array<?xf32>>>, i1)
  ! CHECK:   %[[BOX_ADDR:.*]] = fir.box_addr %[[COPY_IN]]#0 : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
  ! CHECK:   %[[REF:.*]] = fir.convert %[[BOX_ADDR]] : (!fir.ptr<!fir.array<?xf32>>) -> !fir.ref<!fir.array<100xf32>>
  ! CHECK:   fir.result %[[REF]], %[[COPY_IN]]#1, %[[LOAD2]] : !fir.ref<!fir.array<100xf32>>, i1, !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: } else {
  ! CHECK:   %[[ABSENT:.*]] = fir.absent !fir.ref<!fir.array<100xf32>>
  ! CHECK:   %[[FALSE:.*]] = arith.constant false
  ! CHECK:   %[[ABSENT_BOX:.*]] = fir.absent !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK:   fir.result %[[ABSENT]], %[[FALSE]], %[[ABSENT_BOX]] : !fir.ref<!fir.array<100xf32>>, i1, !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: }
  ! CHECK: fir.call @_QPtakes_opt_explicit_shape(%[[ARG]]#0) {{.*}} : (!fir.ref<!fir.array<100xf32>>) -> ()
  ! CHECK: hlfir.copy_out %[[ALLOCA]], %[[ARG]]#1 to %[[ARG]]#2 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, i1, !fir.box<!fir.ptr<!fir.array<?xf32>>>) -> ()
end subroutine

! -----------------------------------------------------------------------------
!    Test passing assumed shape optional to explicit shape optional
! -----------------------------------------------------------------------------
! The fix.box can only be read if the assumed shape is present,
! and the copy-in/copy-out must also be conditional on the assumed
! shape presence.

! CHECK-LABEL: func @_QMoptional_testsPpass_opt_assumed_shape(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.optional}) {
subroutine pass_opt_assumed_shape(x)
  real, optional :: x(:)
  ! CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}uniq_name = "_QMoptional_testsFpass_opt_assumed_shapeEx"{{.*}}
  call takes_opt_explicit_shape(x)
  ! CHECK: %[[IS_PRESENT:.*]] = fir.is_present %[[DECL]]#0 : (!fir.box<!fir.array<?xf32>>) -> i1
  ! CHECK: %[[ARG:.*]]:3 = fir.if %[[IS_PRESENT]] -> (!fir.ref<!fir.array<100xf32>>, i1, !fir.box<!fir.array<?xf32>>) {
  ! CHECK:   %[[COPY_IN:.*]]:2 = hlfir.copy_in %[[DECL]]#0 to %[[ALLOCA]] : (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.box<!fir.array<?xf32>>, i1)
  ! CHECK:   %[[BOX_ADDR:.*]] = fir.box_addr %[[COPY_IN]]#0 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  ! CHECK:   %[[REF:.*]] = fir.convert %[[BOX_ADDR]] : (!fir.ref<!fir.array<?xf32>>) -> !fir.ref<!fir.array<100xf32>>
  ! CHECK:   fir.result %[[REF]], %[[COPY_IN]]#1, %[[DECL]]#0 : !fir.ref<!fir.array<100xf32>>, i1, !fir.box<!fir.array<?xf32>>
  ! CHECK: } else {
  ! CHECK:   %[[ABSENT:.*]] = fir.absent !fir.ref<!fir.array<100xf32>>
  ! CHECK:   %[[FALSE:.*]] = arith.constant false
  ! CHECK:   %[[ABSENT_BOX:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
  ! CHECK:   fir.result %[[ABSENT]], %[[FALSE]], %[[ABSENT_BOX]] : !fir.ref<!fir.array<100xf32>>, i1, !fir.box<!fir.array<?xf32>>
  ! CHECK: }
  ! CHECK: fir.call @_QPtakes_opt_explicit_shape(%[[ARG]]#0) {{.*}} : (!fir.ref<!fir.array<100xf32>>) -> ()
  ! CHECK: hlfir.copy_out %[[ALLOCA]], %[[ARG]]#1 to %[[ARG]]#2 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, i1, !fir.box<!fir.array<?xf32>>) -> ()
end subroutine

! CHECK-LABEL: func @_QMoptional_testsPpass_opt_assumed_shape_char(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "c", fir.optional}) {
subroutine pass_opt_assumed_shape_char(c)
  character(*), optional :: c(:)
  ! CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}uniq_name = "_QMoptional_testsFpass_opt_assumed_shape_charEc"{{.*}}
  call takes_opt_explicit_shape_char(c)
  ! CHECK: %[[IS_PRESENT:.*]] = fir.is_present %[[DECL]]#0 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> i1
  ! CHECK: %[[ARG:.*]]:3 = fir.if %[[IS_PRESENT]] -> (!fir.boxchar<1>, i1, !fir.box<!fir.array<?x!fir.char<1,?>>>) {
  ! CHECK:   %[[COPY_IN:.*]]:2 = hlfir.copy_in %[[DECL]]#0 to %[[ALLOCA]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, i1)
  ! CHECK:   %[[BOX_ADDR:.*]] = fir.box_addr %[[COPY_IN]]#0 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
  ! CHECK:   %[[ELE_SIZE:.*]] = fir.box_elesize %[[COPY_IN]]#0 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
  ! CHECK:   %[[REF:.*]] = fir.convert %[[BOX_ADDR]] : (!fir.ref<!fir.array<?x!fir.char<1,?>>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK:   %[[EMBOX:.*]] = fir.emboxchar %[[REF]], %[[ELE_SIZE]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK:   fir.result %[[EMBOX]], %[[COPY_IN]]#1, %[[DECL]]#0 : !fir.boxchar<1>, i1, !fir.box<!fir.array<?x!fir.char<1,?>>>
  ! CHECK: } else {
  ! CHECK:   %[[ABSENT:.*]] = fir.absent !fir.boxchar<1>
  ! CHECK:   %[[FALSE:.*]] = arith.constant false
  ! CHECK:   %[[ABSENT_BOX:.*]] = fir.absent !fir.box<!fir.array<?x!fir.char<1,?>>>
  ! CHECK:   fir.result %[[ABSENT]], %[[FALSE]], %[[ABSENT_BOX]] : !fir.boxchar<1>, i1, !fir.box<!fir.array<?x!fir.char<1,?>>>
  ! CHECK: }
  ! CHECK: fir.call @_QPtakes_opt_explicit_shape_char(%[[ARG]]#0) {{.*}} : (!fir.boxchar<1>) -> ()
  ! CHECK: hlfir.copy_out %[[ALLOCA]], %[[ARG]]#1 to %[[ARG]]#2 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>, i1, !fir.box<!fir.array<?x!fir.char<1,?>>>) -> ()
end subroutine

! -----------------------------------------------------------------------------
!    Test passing contiguous optional assumed shape to explicit shape optional
! -----------------------------------------------------------------------------
! The fix.box can only be read if the assumed shape is present.
! There should be no copy-in/copy-out

! CHECK-LABEL: func @_QMoptional_testsPpass_opt_contiguous_assumed_shape(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.contiguous, fir.optional}) {
subroutine pass_opt_contiguous_assumed_shape(x)
  real, optional, contiguous :: x(:)
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}uniq_name = "_QMoptional_testsFpass_opt_contiguous_assumed_shapeEx"{{.*}}
  call takes_opt_explicit_shape(x)
  ! CHECK: %[[IS_PRESENT:.*]] = fir.is_present %[[DECL]]#0 : (!fir.box<!fir.array<?xf32>>) -> i1
  ! CHECK: %[[ARG:.*]] = fir.if %[[IS_PRESENT]] -> (!fir.ref<!fir.array<100xf32>>) {
  ! CHECK:   %[[BOX_ADDR:.*]] = fir.box_addr %[[DECL]]#1 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  ! CHECK:   %[[REF:.*]] = fir.convert %[[BOX_ADDR]] : (!fir.ref<!fir.array<?xf32>>) -> !fir.ref<!fir.array<100xf32>>
  ! CHECK:   fir.result %[[REF]] : !fir.ref<!fir.array<100xf32>>
  ! CHECK: } else {
  ! CHECK:   %[[ABSENT:.*]] = fir.absent !fir.ref<!fir.array<100xf32>>
  ! CHECK:   fir.result %[[ABSENT]] : !fir.ref<!fir.array<100xf32>>
  ! CHECK: }
  ! CHECK: fir.call @_QPtakes_opt_explicit_shape(%[[ARG]]) {{.*}} : (!fir.ref<!fir.array<100xf32>>) -> ()
end subroutine

! CHECK-LABEL: func @_QMoptional_testsPpass_opt_contiguous_assumed_shape_char(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "c", fir.contiguous, fir.optional}) {
subroutine pass_opt_contiguous_assumed_shape_char(c)
  character(*), optional, contiguous :: c(:)
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}uniq_name = "_QMoptional_testsFpass_opt_contiguous_assumed_shape_charEc"{{.*}}
  call takes_opt_explicit_shape_char(c)
  ! CHECK: %[[IS_PRESENT:.*]] = fir.is_present %[[DECL]]#0 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> i1
  ! CHECK: %[[ARG:.*]] = fir.if %[[IS_PRESENT]] -> (!fir.boxchar<1>) {
  ! CHECK:   %[[BOX_ADDR:.*]] = fir.box_addr %[[DECL]]#1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
  ! CHECK:   %[[ELE_SIZE:.*]] = fir.box_elesize %[[DECL]]#1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
  ! CHECK:   %[[REF:.*]] = fir.convert %[[BOX_ADDR]] : (!fir.ref<!fir.array<?x!fir.char<1,?>>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK:   %[[EMBOX:.*]] = fir.emboxchar %[[REF]], %[[ELE_SIZE]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK:   fir.result %[[EMBOX]] : !fir.boxchar<1>
  ! CHECK: } else {
  ! CHECK:   %[[ABSENT:.*]] = fir.absent !fir.boxchar<1>
  ! CHECK:   fir.result %[[ABSENT]] : !fir.boxchar<1>
  ! CHECK: }
  ! CHECK: fir.call @_QPtakes_opt_explicit_shape_char(%[[ARG]]) {{.*}} : (!fir.boxchar<1>) -> ()
end subroutine

! -----------------------------------------------------------------------------
!    Test passing allocatables and contiguous pointers to explicit shape optional
! -----------------------------------------------------------------------------
! The fix.box can be read and its address directly passed. There should be no
! copy-in/copy-out.

! CHECK-LABEL: func @_QMoptional_testsPpass_allocatable_array(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>{{.*}}) {
subroutine pass_allocatable_array(i)
  real, allocatable :: i(:)
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}uniq_name = "_QMoptional_testsFpass_allocatable_arrayEi"{{.*}}
  call takes_opt_explicit_shape(i)
  ! CHECK: %[[LOAD:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  ! CHECK: %[[ADDR_I64:.*]] = fir.convert %[[BOX_ADDR]] : (!fir.heap<!fir.array<?xf32>>) -> i64
  ! CHECK: %[[IS_PRESENT:.*]] = arith.cmpi ne, %[[ADDR_I64]], %c0_i64 : i64
  ! CHECK: %[[ARG:.*]] = fir.if %[[IS_PRESENT]] -> (!fir.ref<!fir.array<100xf32>>) {
  ! CHECK:   %[[LOAD2:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK:   %[[BOX_ADDR2:.*]] = fir.box_addr %[[LOAD2]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  ! CHECK:   %[[REF:.*]] = fir.convert %[[BOX_ADDR2]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<100xf32>>
  ! CHECK:   fir.result %[[REF]] : !fir.ref<!fir.array<100xf32>>
  ! CHECK: } else {
  ! CHECK:   %[[ABSENT:.*]] = fir.absent !fir.ref<!fir.array<100xf32>>
  ! CHECK:   fir.result %[[ABSENT]] : !fir.ref<!fir.array<100xf32>>
  ! CHECK: }
  ! CHECK: fir.call @_QPtakes_opt_explicit_shape(%[[ARG]]) {{.*}} : (!fir.ref<!fir.array<100xf32>>) -> ()
end subroutine

! CHECK-LABEL: func @_QMoptional_testsPpass_allocatable_array_char(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>{{.*}}) {
subroutine pass_allocatable_array_char(c)
  character(:), allocatable :: c(:)
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}uniq_name = "_QMoptional_testsFpass_allocatable_array_charEc"{{.*}}
  call takes_opt_explicit_shape_char(c)
  ! CHECK: %[[LOAD:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
  ! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>) -> !fir.heap<!fir.array<?x!fir.char<1,?>>>
  ! CHECK: %[[ADDR_I64:.*]] = fir.convert %[[BOX_ADDR]] : (!fir.heap<!fir.array<?x!fir.char<1,?>>>) -> i64
  ! CHECK: %[[IS_PRESENT:.*]] = arith.cmpi ne, %[[ADDR_I64]], %c0_i64 : i64
  ! CHECK: %[[ARG:.*]] = fir.if %[[IS_PRESENT]] -> (!fir.boxchar<1>) {
  ! CHECK:   %[[LOAD2:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
  ! CHECK:   %[[BOX_ADDR2:.*]] = fir.box_addr %[[LOAD2]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>) -> !fir.heap<!fir.array<?x!fir.char<1,?>>>
  ! CHECK:   %[[ELE_SIZE:.*]] = fir.box_elesize %[[LOAD2]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>) -> index
  ! CHECK:   %[[REF:.*]] = fir.convert %[[BOX_ADDR2]] : (!fir.heap<!fir.array<?x!fir.char<1,?>>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK:   %[[EMBOX:.*]] = fir.emboxchar %[[REF]], %[[ELE_SIZE]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK:   fir.result %[[EMBOX]] : !fir.boxchar<1>
  ! CHECK: } else {
  ! CHECK:   %[[ABSENT:.*]] = fir.absent !fir.boxchar<1>
  ! CHECK:   fir.result %[[ABSENT]] : !fir.boxchar<1>
  ! CHECK: }
  ! CHECK: fir.call @_QPtakes_opt_explicit_shape_char(%[[ARG]]) {{.*}} : (!fir.boxchar<1>) -> ()
end subroutine

! CHECK-LABEL: func @_QMoptional_testsPpass_contiguous_pointer_array(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "i", fir.contiguous}) {
subroutine pass_contiguous_pointer_array(i)
  real, pointer, contiguous :: i(:)
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}uniq_name = "_QMoptional_testsFpass_contiguous_pointer_arrayEi"{{.*}}
  call takes_opt_explicit_shape(i)
  ! CHECK: %[[LOAD:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
  ! CHECK: %[[ADDR_I64:.*]] = fir.convert %[[BOX_ADDR]] : (!fir.ptr<!fir.array<?xf32>>) -> i64
  ! CHECK: %[[IS_PRESENT:.*]] = arith.cmpi ne, %[[ADDR_I64]], %c0_i64 : i64
  ! CHECK: %[[ARG:.*]] = fir.if %[[IS_PRESENT]] -> (!fir.ref<!fir.array<100xf32>>) {
  ! CHECK:   %[[LOAD2:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK:   %[[BOX_ADDR2:.*]] = fir.box_addr %[[LOAD2]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
  ! CHECK:   %[[REF:.*]] = fir.convert %[[BOX_ADDR2]] : (!fir.ptr<!fir.array<?xf32>>) -> !fir.ref<!fir.array<100xf32>>
  ! CHECK:   fir.result %[[REF]] : !fir.ref<!fir.array<100xf32>>
  ! CHECK: } else {
  ! CHECK:   %[[ABSENT:.*]] = fir.absent !fir.ref<!fir.array<100xf32>>
  ! CHECK:   fir.result %[[ABSENT]] : !fir.ref<!fir.array<100xf32>>
  ! CHECK: }
  ! CHECK: fir.call @_QPtakes_opt_explicit_shape(%[[ARG]]) {{.*}} : (!fir.ref<!fir.array<100xf32>>) -> ()
end subroutine

! CHECK-LABEL: func @_QMoptional_testsPpass_contiguous_pointer_array_char(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>> {fir.bindc_name = "c", fir.contiguous}) {
subroutine pass_contiguous_pointer_array_char(c)
  character(:), pointer, contiguous :: c(:)
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}uniq_name = "_QMoptional_testsFpass_contiguous_pointer_array_charEc"{{.*}}
  call takes_opt_explicit_shape_char(c)
  ! CHECK: %[[LOAD:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
  ! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>) -> !fir.ptr<!fir.array<?x!fir.char<1,?>>>
  ! CHECK: %[[ADDR_I64:.*]] = fir.convert %[[BOX_ADDR]] : (!fir.ptr<!fir.array<?x!fir.char<1,?>>>) -> i64
  ! CHECK: %[[IS_PRESENT:.*]] = arith.cmpi ne, %[[ADDR_I64]], %c0_i64 : i64
  ! CHECK: %[[ARG:.*]] = fir.if %[[IS_PRESENT]] -> (!fir.boxchar<1>) {
  ! CHECK:   %[[LOAD2:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
  ! CHECK:   %[[BOX_ADDR2:.*]] = fir.box_addr %[[LOAD2]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>) -> !fir.ptr<!fir.array<?x!fir.char<1,?>>>
  ! CHECK:   %[[ELE_SIZE:.*]] = fir.box_elesize %[[LOAD2]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>) -> index
  ! CHECK:   %[[REF:.*]] = fir.convert %[[BOX_ADDR2]] : (!fir.ptr<!fir.array<?x!fir.char<1,?>>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK:   %[[EMBOX:.*]] = fir.emboxchar %[[REF]], %[[ELE_SIZE]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK:   fir.result %[[EMBOX]] : !fir.boxchar<1>
  ! CHECK: } else {
  ! CHECK:   %[[ABSENT:.*]] = fir.absent !fir.boxchar<1>
  ! CHECK:   fir.result %[[ABSENT]] : !fir.boxchar<1>
  ! CHECK: }
  ! CHECK: fir.call @_QPtakes_opt_explicit_shape_char(%[[ARG]]) {{.*}} : (!fir.boxchar<1>) -> ()
end subroutine

! -----------------------------------------------------------------------------
!    Test passing assumed shape optional to explicit shape optional with intents
! -----------------------------------------------------------------------------
! The fix.box can only be read if the assumed shape is present,
! and the copy-in/copy-out must also be conditional on the assumed
! shape presence. For intent(in), there should be no copy-out while for
! intent(out), there should be no copy-in.

! CHECK-LABEL: func @_QMoptional_testsPpass_opt_assumed_shape_to_intentin(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.optional}) {
subroutine pass_opt_assumed_shape_to_intentin(x)
  real, optional :: x(:)
  ! CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}uniq_name = "_QMoptional_testsFpass_opt_assumed_shape_to_intentinEx"{{.*}}
  call takes_opt_explicit_shape_intentin(x)
  ! CHECK: %[[IS_PRESENT:.*]] = fir.is_present %[[DECL]]#0 : (!fir.box<!fir.array<?xf32>>) -> i1
  ! CHECK: %[[ARG:.*]]:2 = fir.if %[[IS_PRESENT]] -> (!fir.ref<!fir.array<100xf32>>, i1) {
  ! CHECK:   %[[COPY_IN:.*]]:2 = hlfir.copy_in %[[DECL]]#0 to %[[ALLOCA]] : (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.box<!fir.array<?xf32>>, i1)
  ! CHECK:   %[[BOX_ADDR:.*]] = fir.box_addr %[[COPY_IN]]#0 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  ! CHECK:   %[[REF:.*]] = fir.convert %[[BOX_ADDR]] : (!fir.ref<!fir.array<?xf32>>) -> !fir.ref<!fir.array<100xf32>>
  ! CHECK:   fir.result %[[REF]], %[[COPY_IN]]#1 : !fir.ref<!fir.array<100xf32>>, i1
  ! CHECK: } else {
  ! CHECK:   %[[ABSENT:.*]] = fir.absent !fir.ref<!fir.array<100xf32>>
  ! CHECK:   %[[FALSE:.*]] = arith.constant false
  ! CHECK:   fir.result %[[ABSENT]], %[[FALSE]] : !fir.ref<!fir.array<100xf32>>, i1
  ! CHECK: }
  ! CHECK: fir.call @_QPtakes_opt_explicit_shape_intentin(%[[ARG]]#0) {{.*}} : (!fir.ref<!fir.array<100xf32>>) -> ()
  ! CHECK: hlfir.copy_out %[[ALLOCA]], %[[ARG]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, i1) -> ()
end subroutine

! CHECK-LABEL: func @_QMoptional_testsPpass_opt_assumed_shape_to_intentout(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.optional}) {
subroutine pass_opt_assumed_shape_to_intentout(x)
  real, optional :: x(:)
  ! CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}uniq_name = "_QMoptional_testsFpass_opt_assumed_shape_to_intentoutEx"{{.*}}
  call takes_opt_explicit_shape_intentout(x)
  ! CHECK: %[[IS_PRESENT:.*]] = fir.is_present %[[DECL]]#0 : (!fir.box<!fir.array<?xf32>>) -> i1
  ! CHECK: %[[ARG:.*]]:3 = fir.if %[[IS_PRESENT]] -> (!fir.ref<!fir.array<100xf32>>, i1, !fir.box<!fir.array<?xf32>>) {
  ! CHECK:   %[[COPY_IN:.*]]:2 = hlfir.copy_in %[[DECL]]#0 to %[[ALLOCA]] : (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.box<!fir.array<?xf32>>, i1)
  ! CHECK:   %[[BOX_ADDR:.*]] = fir.box_addr %[[COPY_IN]]#0 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  ! CHECK:   %[[REF:.*]] = fir.convert %[[BOX_ADDR]] : (!fir.ref<!fir.array<?xf32>>) -> !fir.ref<!fir.array<100xf32>>
  ! CHECK:   fir.result %[[REF]], %[[COPY_IN]]#1, %[[DECL]]#0 : !fir.ref<!fir.array<100xf32>>, i1, !fir.box<!fir.array<?xf32>>
  ! CHECK: } else {
  ! CHECK:   %[[ABSENT:.*]] = fir.absent !fir.ref<!fir.array<100xf32>>
  ! CHECK:   %[[FALSE:.*]] = arith.constant false
  ! CHECK:   %[[ABSENT_BOX:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
  ! CHECK:   fir.result %[[ABSENT]], %[[FALSE]], %[[ABSENT_BOX]] : !fir.ref<!fir.array<100xf32>>, i1, !fir.box<!fir.array<?xf32>>
  ! CHECK: }
  ! CHECK: fir.call @_QPtakes_opt_explicit_shape_intentout(%[[ARG]]#0) {{.*}} : (!fir.ref<!fir.array<100xf32>>) -> ()
  ! CHECK: hlfir.copy_out %[[ALLOCA]], %[[ARG]]#1 to %[[ARG]]#2 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, i1, !fir.box<!fir.array<?xf32>>) -> ()
end subroutine
end module
