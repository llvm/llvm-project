! Test lowering of SIZE/SIZEOF inquiry intrinsics with assumed-ranks
! arguments.
! RUN: bbc -emit-hlfir -o - %s -allow-assumed-rank | FileCheck %s


subroutine test_size_1(x)
  real :: x(..)
  call takes_integer(size(x))
end subroutine

subroutine test_size_2(x)
  real :: x(..)
  call takes_integer(size(x, 2))
end subroutine

subroutine test_size_3(x, d)
  real :: x(..)
  integer, optional :: d
  call takes_integer(size(x, d))
end subroutine

subroutine test_size_4(x)
  real, allocatable :: x(..)
  call takes_integer(size(x))
end subroutine


! CHECK-LABEL:   func.func @_QPtest_size_1(
! CHECK-SAME:                              %[[VAL_0:.*]]: !fir.box<!fir.array<*:f32>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {uniq_name = "_QFtest_size_1Ex"} : (!fir.box<!fir.array<*:f32>>, !fir.dscope) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> !fir.box<none>
! CHECK:           %[[VAL_7:.*]] = fir.call @_FortranASize(%[[VAL_5]]
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i64) -> i32
! CHECK:           %[[VAL_9:.*]]:3 = hlfir.associate %[[VAL_8]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:           fir.call @_QPtakes_integer(%[[VAL_9]]#1) fastmath<contract> : (!fir.ref<i32>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_9]]#1, %[[VAL_9]]#2 : !fir.ref<i32>, i1
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_size_2(
! CHECK-SAME:                              %[[VAL_0:.*]]: !fir.box<!fir.array<*:f32>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca i32
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] {uniq_name = "_QFtest_size_2Ex"} : (!fir.box<!fir.array<*:f32>>, !fir.dscope) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           %[[VAL_4:.*]] = arith.constant 2 : i32
! CHECK:           fir.store %[[VAL_4]] to %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<i32>) -> i64
! CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_7:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_6]] : i64
! CHECK:           %[[VAL_8:.*]] = fir.if %[[VAL_7]] -> (i32) {
! CHECK:             %[[VAL_11:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.box<!fir.array<*:f32>>) -> !fir.box<none>
! CHECK:             %[[VAL_13:.*]] = fir.call @_FortranASize(%[[VAL_11]]
! CHECK:             %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i64) -> i32
! CHECK:             fir.result %[[VAL_14]] : i32
! CHECK:           } else {
! CHECK:             %[[VAL_15:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:             %[[VAL_18:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.box<!fir.array<*:f32>>) -> !fir.box<none>
! CHECK:             %[[VAL_20:.*]] = fir.call @_FortranASizeDim(%[[VAL_18]]
! CHECK:             %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i64) -> i32
! CHECK:             fir.result %[[VAL_21]] : i32
! CHECK:           }
! CHECK:           %[[VAL_22:.*]]:3 = hlfir.associate %[[VAL_8]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:           fir.call @_QPtakes_integer(%[[VAL_22]]#1) fastmath<contract> : (!fir.ref<i32>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_22]]#1, %[[VAL_22]]#2 : !fir.ref<i32>, i1
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_size_3(
! CHECK-SAME:                              %[[VAL_0:.*]]: !fir.box<!fir.array<*:f32>> {fir.bindc_name = "x"},
! CHECK-SAME:                              %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "d", fir.optional}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_2]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFtest_size_3Ed"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] {uniq_name = "_QFtest_size_3Ex"} : (!fir.box<!fir.array<*:f32>>, !fir.dscope) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_3]]#1 : (!fir.ref<i32>) -> i64
! CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_7:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_6]] : i64
! CHECK:           %[[VAL_8:.*]] = fir.if %[[VAL_7]] -> (i32) {
! CHECK:             %[[VAL_11:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.box<!fir.array<*:f32>>) -> !fir.box<none>
! CHECK:             %[[VAL_13:.*]] = fir.call @_FortranASize(%[[VAL_11]],
! CHECK:             %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i64) -> i32
! CHECK:             fir.result %[[VAL_14]] : i32
! CHECK:           } else {
! CHECK:             %[[VAL_15:.*]] = fir.load %[[VAL_3]]#1 : !fir.ref<i32>
! CHECK:             %[[VAL_18:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.box<!fir.array<*:f32>>) -> !fir.box<none>
! CHECK:             %[[VAL_20:.*]] = fir.call @_FortranASizeDim(%[[VAL_18]]
! CHECK:             %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i64) -> i32
! CHECK:             fir.result %[[VAL_21]] : i32
! CHECK:           }
! CHECK:           %[[VAL_22:.*]]:3 = hlfir.associate %[[VAL_8]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:           fir.call @_QPtakes_integer(%[[VAL_22]]#1) fastmath<contract> : (!fir.ref<i32>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_22]]#1, %[[VAL_22]]#2 : !fir.ref<i32>, i1
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_size_4(
! CHECK-SAME:                              %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_size_4Ex"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>)
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>
! CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_3]] : (!fir.box<!fir.heap<!fir.array<*:f32>>>) -> !fir.box<none>
! CHECK:           %[[VAL_8:.*]] = fir.call @_FortranASize(%[[VAL_6]]
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i64) -> i32
! CHECK:           %[[VAL_10:.*]]:3 = hlfir.associate %[[VAL_9]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:           fir.call @_QPtakes_integer(%[[VAL_10]]#1) fastmath<contract> : (!fir.ref<i32>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_10]]#1, %[[VAL_10]]#2 : !fir.ref<i32>, i1
! CHECK:           return
! CHECK:         }
