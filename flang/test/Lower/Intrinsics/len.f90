! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPlen_test(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "i"},
! CHECK-SAME: %[[ARG1:.*]]: !fir.boxchar<1> {fir.bindc_name = "c"})
subroutine len_test(i, c)
  integer :: i
  character(*) :: c
! CHECK: %[[c:.*]]:2 = fir.unboxchar %[[ARG1]]
! CHECK: hlfir.declare %[[c]]#0 typeparams %[[c]]#1
! CHECK: %[[IVAL:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: %[[xx:.*]] = fir.convert %[[c]]#1 : (index) -> i32
! CHECK: hlfir.assign %[[xx]] to %[[IVAL]]#0 : i32, !fir.ref<i32>
  i = len(c)
end subroutine

! CHECK-LABEL: func.func @_QPlen_test_array(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "i"},
! CHECK-SAME: %[[ARG1:.*]]: !fir.boxchar<1> {fir.bindc_name = "c"})
subroutine len_test_array(i, c)
  integer :: i
  character(*) :: c(100)
! CHECK: %[[c:.*]]:2 = fir.unboxchar %[[ARG1]]
! CHECK: hlfir.declare {{.*}} typeparams %[[c]]#1
! CHECK: %[[IVAL:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: %[[xx:.*]] = fir.convert %[[c]]#1 : (index) -> i32
! CHECK: hlfir.assign %[[xx]] to %[[IVAL]]#0 : i32, !fir.ref<i32>
  i = len(c)
end subroutine

! CHECK-LABEL: func.func @_QPlen_test_assumed_shape_array(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "i"},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "c"}) {
subroutine len_test_assumed_shape_array(i, c)
  integer :: i
  character(*) :: c(:)
! CHECK-DAG:  %[[I:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK-DAG:  %[[C:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK:  %[[VAL_2:.*]] = fir.box_elesize %[[C]]#1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (index) -> i32
! CHECK:  hlfir.assign %[[VAL_3]] to %[[I]]#0 : i32, !fir.ref<i32>
  i = len(c)
end subroutine

! CHECK-LABEL: func.func @_QPlen_test_array_alloc(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "i"},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>> {fir.bindc_name = "c"}) {
subroutine len_test_array_alloc(i, c)
  integer :: i
  character(:), allocatable :: c(:)
! CHECK-DAG:  %[[I:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK-DAG:  %[[C:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK:  %[[VAL_2:.*]] = fir.load %[[C]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:  %[[VAL_3:.*]] = fir.box_elesize %[[VAL_2]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>) -> index
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (index) -> i32
! CHECK:  hlfir.assign %[[VAL_4]] to %[[I]]#0 : i32, !fir.ref<i32>
  i = len(c)
end subroutine

! CHECK-LABEL: func.func @_QPlen_test_array_local_alloc(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "i"})
subroutine len_test_array_local_alloc(i)
  integer :: i
  character(:), allocatable :: c(:)
! CHECK:  %[[I:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:  %[[C10:.*]] = arith.constant 10 : i32
  allocate(character(10):: c(100))
! CHECK:  %[[C_LOADED:.*]] = fir.load %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:  %[[ELESIZE:.*]] = fir.box_elesize %[[C_LOADED]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>) -> index
! CHECK:  %[[RESULT:.*]] = fir.convert %[[ELESIZE]] : (index) -> i32
! CHECK:  hlfir.assign %[[RESULT]] to %[[I]]#0 : i32, !fir.ref<i32>
  i = len(c)
end subroutine

! CHECK-LABEL: func.func @_QPlen_test_alloc_explicit_len(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "i"},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
! CHECK-SAME:  %[[VAL_2:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>> {fir.bindc_name = "c"}) {
subroutine len_test_alloc_explicit_len(i, n, c)
  integer :: i
  integer :: n
  character(n), allocatable :: c(:)
! CHECK-DAG:  %[[N:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK-DAG:  %[[I:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:  %[[VAL_3:.*]] = fir.load %[[N]]#0 : !fir.ref<i32>
! CHECK:  %[[c0_i32:.*]] = arith.constant 0 : i32
! CHECK:  %[[cmp:.*]] = arith.cmpi sgt, %[[VAL_3]], %[[c0_i32]] : i32
! CHECK:  %[[len:.*]] = arith.select %[[cmp]], %[[VAL_3]], %[[c0_i32]] : i32
! CHECK:  hlfir.assign %[[len]] to %[[I]]#0 : i32, !fir.ref<i32>
  i = len(c)
end subroutine

! CHECK-LABEL: func.func @_QPlen_test_pointer_explicit_len(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "i"},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
subroutine len_test_pointer_explicit_len(i, n, c)
  integer :: i
  integer :: n
  character(n), pointer :: c(:)
! CHECK-DAG:  %[[N:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK-DAG:  %[[I:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:  %[[VAL_3:.*]] = fir.load %[[N]]#0 : !fir.ref<i32>
! CHECK:  %[[c0_i32:.*]] = arith.constant 0 : i32
! CHECK:  %[[cmp:.*]] = arith.cmpi sgt, %[[VAL_3]], %[[c0_i32]] : i32
! CHECK:  %[[len:.*]] = arith.select %[[cmp]], %[[VAL_3]], %[[c0_i32]] : i32
! CHECK:  hlfir.assign %[[len]] to %[[I]]#0 : i32, !fir.ref<i32>
  i = len(c)
end subroutine

! CHECK-LABEL: func.func @_QPlen_test_assumed_shape_explicit_len(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "i"},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
subroutine len_test_assumed_shape_explicit_len(i, n, c)
  integer :: i
  integer :: n
  character(n) :: c(:)
! CHECK-DAG:  %[[N:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK-DAG:  %[[I:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:  %[[VAL_3:.*]] = fir.load %[[N]]#0 : !fir.ref<i32>
! CHECK:  %[[c0_i32:.*]] = arith.constant 0 : i32
! CHECK:  %[[cmp:.*]] = arith.cmpi sgt, %[[VAL_3]], %[[c0_i32]] : i32
! CHECK:  %[[len:.*]] = arith.select %[[cmp]], %[[VAL_3]], %[[c0_i32]] : i32
! CHECK:  hlfir.assign %[[len]] to %[[I]]#0 : i32, !fir.ref<i32>
  i = len(c)
end subroutine
