! RUN: bbc -emit-hlfir -fopenmp -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s | FileCheck %s

! Test that we can reduce variables used in an equivalence statement.
! Unlike POINTER variables these are lowered to an unboxed !fir.ptr<>.

subroutine reduction_equivalence()
  integer::va
  equivalence (va,vva)
  va=1

!$omp parallel reduction(+:va)
  va=1
!$omp end parallel
end subroutine reduction_equivalence


! CHECK-LABEL:   omp.declare_reduction @add_reduction_i32 : i32 init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: i32):
! CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
! CHECK:           omp.yield(%[[VAL_1]] : i32)
! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32):
! CHECK:           %[[VAL_2:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : i32
! CHECK:           omp.yield(%[[VAL_2]] : i32)
! CHECK:         }

! CHECK-LABEL:   func.func @_QPreduction_equivalence() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.array<4xi8> {uniq_name = "_QFreduction_equivalenceEva"}
! CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_2:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_1]] : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] storage(%[[VAL_0]][0]) {uniq_name = "_QFreduction_equivalenceEva"} : (!fir.ptr<i32>, !fir.ref<!fir.array<4xi8>>) -> (!fir.ptr<i32>, !fir.ptr<i32>)
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_6:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_5]] : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<i8>) -> !fir.ptr<f32>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] storage(%[[VAL_0]][0]) {uniq_name = "_QFreduction_equivalenceEvva"} : (!fir.ptr<f32>, !fir.ref<!fir.array<4xi8>>) -> (!fir.ptr<f32>, !fir.ptr<f32>)
! CHECK:           %[[VAL_9:.*]] = arith.constant 1 : i32
! CHECK:           hlfir.assign %[[VAL_9]] to %[[VAL_4]]#0 : i32, !fir.ptr<i32>
! CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.ptr<i32>) -> !fir.ref<i32>
! CHECK:           omp.parallel reduction(@add_reduction_i32 %[[VAL_10]] -> %[[VAL_11:.*]] : !fir.ref<i32>) {
! CHECK:             %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_11]] {uniq_name = "_QFreduction_equivalenceEva"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:             %[[VAL_13:.*]] = arith.constant 1 : i32
! CHECK:             hlfir.assign %[[VAL_13]] to %[[VAL_12]]#0 : i32, !fir.ref<i32>
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }
