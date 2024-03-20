! RUN: bbc -emit-hlfir -fopenmp --force-byref-reduction %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --force-byref-reduction %s -o - | FileCheck %s

! CHECK-LABEL:   omp.declare_reduction @add_reduction_byref_i32 : !fir.ref<i32>
! CHECK-SAME:    init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<i32>):
! CHECK:            %[[C0_1:.*]] = arith.constant 0 : i32
! CHECK:            %[[REF:.*]] = fir.alloca i32
! CHECK:            fir.store %[[C0_1]] to %[[REF]] : !fir.ref<i32>
! CHECK:           omp.yield(%[[REF]] : !fir.ref<i32>)

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[ARG0:.*]]: !fir.ref<i32>, %[[ARG1:.*]]: !fir.ref<i32>):
! CHECK:           %[[LD0:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
! CHECK:           %[[LD1:.*]] = fir.load %[[ARG1]] : !fir.ref<i32>
! CHECK:           %[[RES:.*]] = arith.addi %[[LD0]], %[[LD1]] : i32
! CHECK:           fir.store %[[RES]] to %[[ARG0]] : !fir.ref<i32>
! CHECK:           omp.yield(%[[ARG0]] : !fir.ref<i32>)
! CHECK:         }

! CHECK-LABEL:   func.func @_QPsimple_int_reduction()
! CHECK:           %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsimple_int_reductionEi"}
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFsimple_int_reductionEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFsimple_int_reductionEx"}
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFsimple_int_reductionEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i32
! CHECK:           hlfir.assign %[[VAL_4]] to %[[VAL_3]]#0 : i32, !fir.ref<i32>
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_5:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
! CHECK:             %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] {uniq_name = "_QFsimple_int_reductionEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:             %[[VAL_7:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_8:.*]] = arith.constant 100 : i32
! CHECK:             %[[VAL_9:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop byref reduction(@add_reduction_byref_i32 %[[VAL_3]]#0 -> %[[VAL_10:.*]] : !fir.ref<i32>)  for  (%[[VAL_11:.*]]) : i32 = (%[[VAL_7]]) to (%[[VAL_8]]) inclusive step (%[[VAL_9]])
! CHECK:               fir.store %[[VAL_11]] to %[[VAL_6]]#1 : !fir.ref<i32>
! CHECK:               %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_10]] {uniq_name = "_QFsimple_int_reductionEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:               %[[VAL_13:.*]] = fir.load %[[VAL_12]]#0 : !fir.ref<i32>
! CHECK:               %[[VAL_14:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<i32>
! CHECK:               %[[VAL_15:.*]] = arith.addi %[[VAL_13]], %[[VAL_14]] : i32
! CHECK:               hlfir.assign %[[VAL_15]] to %[[VAL_12]]#0 : i32, !fir.ref<i32>
! CHECK:               omp.yield
! CHECK:             omp.terminator
! CHECK:           return


subroutine simple_int_reduction
  integer :: x
  x = 0
  !$omp parallel
  !$omp do reduction(+:x)
  do i=1, 100
    x = x + i
  end do
  !$omp end do
  !$omp end parallel
end subroutine
