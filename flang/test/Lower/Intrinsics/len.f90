! RUN: bbc --use-desc-for-alloc=false -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: len_test
subroutine len_test(i, c)
  integer :: i
  character(*) :: c
  ! CHECK: %[[c:.*]]:2 = fir.unboxchar %arg1
  ! CHECK: %[[xx:.*]] = fir.convert %[[c]]#1 : (index) -> i32
  ! CHECK: fir.store %[[xx]] to %arg0
  i = len(c)
end subroutine

! CHECK-LABEL: len_test_array
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i32> {fir.bindc_name = "i"}, %[[arg1:.*]]: !fir.boxchar<1> {fir.bindc_name = "c"}
subroutine len_test_array(i, c)
  integer :: i
  character(*) :: c(100)
  ! CHECK: %[[c:.*]]:2 = fir.unboxchar %[[arg1]]
  ! CHECK: %[[xx:.*]] = fir.convert %[[c]]#1 : (index) -> i32
  ! CHECK: fir.store %[[xx]] to %[[arg0]]
  i = len(c)
end subroutine

! CHECK-LABEL: func @_QPlen_test_assumed_shape_array(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "i"},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "c"}) {
subroutine len_test_assumed_shape_array(i, c)
  integer :: i
  character(*) :: c(:)
! CHECK:  %[[VAL_2:.*]] = fir.box_elesize %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (index) -> i32
! CHECK:  fir.store %[[VAL_3]] to %[[VAL_0]] : !fir.ref<i32>
  i = len(c)
end subroutine

! CHECK-LABEL: func @_QPlen_test_array_alloc(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "i"},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>> {fir.bindc_name = "c"}) {
subroutine len_test_array_alloc(i, c)
  integer :: i
  character(:), allocatable :: c(:)
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:  %[[VAL_3:.*]] = fir.box_elesize %[[VAL_2]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>) -> index
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (index) -> i32
! CHECK:  fir.store %[[VAL_4]] to %[[VAL_0]] : !fir.ref<i32>
  i = len(c)
end subroutine

! CHECK-LABEL: func @_QPlen_test_array_local_alloc(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "i"})
subroutine len_test_array_local_alloc(i)
  integer :: i
  character(:), allocatable :: c(:)
! CHECK:  %[[VAL_5:.*]] = fir.alloca index {uniq_name = "_QFlen_test_array_local_allocEc.len"}
! CHECK:  %[[VAL_7:.*]] = arith.constant 10 : i32
! CHECK:  %[[VAL_10:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
! CHECK:  fir.store %[[VAL_10]] to %[[VAL_5]] : !fir.ref<index>
  allocate(character(10):: c(100))
! CHECK:  %[[VAL_13:.*]] = fir.load %[[VAL_5]] : !fir.ref<index>
! CHECK:  %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (index) -> i32
! CHECK:  fir.store %[[VAL_14]] to %[[VAL_0]] : !fir.ref<i32>
  i = len(c)
end subroutine

! CHECK-LABEL: func @_QPlen_test_alloc_explicit_len(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "i"},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
! CHECK-SAME:  %[[VAL_2:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>> {fir.bindc_name = "c"}) {
subroutine len_test_alloc_explicit_len(i, n, c)
  integer :: i
  integer :: n
  character(n), allocatable :: c(:)
! CHECK:  %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:  %[[c0_i32:.*]] = arith.constant 0 : i32
! CHECK:  %[[cmp:.*]] = arith.cmpi sgt, %[[VAL_3]], %[[c0_i32]] : i32
! CHECK:  %[[len:.*]] = arith.select %[[cmp]], %[[VAL_3]], %[[c0_i32]] : i32
! CHECK:  fir.store %[[len]] to %[[VAL_0]] : !fir.ref<i32>
  i = len(c)
end subroutine

! CHECK-LABEL: func @_QPlen_test_pointer_explicit_len(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "i"},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
subroutine len_test_pointer_explicit_len(i, n, c)
  integer :: i
  integer :: n
  character(n), pointer :: c(:)
! CHECK:  %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:  %[[c0_i32:.*]] = arith.constant 0 : i32
! CHECK:  %[[cmp:.*]] = arith.cmpi sgt, %[[VAL_3]], %[[c0_i32]] : i32
! CHECK:  %[[len:.*]] = arith.select %[[cmp]], %[[VAL_3]], %[[c0_i32]] : i32
! CHECK:  fir.store %[[len]] to %[[VAL_0]] : !fir.ref<i32>
  i = len(c)
end subroutine

! CHECK-LABEL: func @_QPlen_test_assumed_shape_explicit_len(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "i"},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
subroutine len_test_assumed_shape_explicit_len(i, n, c)
  integer :: i
  integer :: n
  character(n) :: c(:)
! CHECK:  %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:  %[[c0_i32:.*]] = arith.constant 0 : i32
! CHECK:  %[[cmp:.*]] = arith.cmpi sgt, %[[VAL_3]], %[[c0_i32]] : i32
! CHECK:  %[[len:.*]] = arith.select %[[cmp]], %[[VAL_3]], %[[c0_i32]] : i32
! CHECK:  fir.store %[[len]] to %[[VAL_0]] : !fir.ref<i32>
  i = len(c)
end subroutine
