! RUN: bbc -emit-hlfir -fopenmp -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s | FileCheck %s

module m1
  intrinsic max
end module m1
program main
  use m1, ren=>max
  n=0
  !$omp parallel reduction(ren:n)
  print *, "par"
  !$omp end parallel
end program main

! test that we understood that this should be a max reduction

! CHECK-LABEL:   omp.declare_reduction @max_i_32 : i32 init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: i32):
! CHECK:           %[[VAL_1:.*]] = arith.constant -2147483648 : i32
! CHECK:           omp.yield(%[[VAL_1]] : i32)

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32):
! CHECK:           %[[VAL_2:.*]] = arith.maxsi %[[VAL_0]], %[[VAL_1]] : i32
! CHECK:           omp.yield(%[[VAL_2]] : i32)
! CHECK:         }

! CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "main"} {
! CHECK:           %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFEn"}
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
! CHECK:           hlfir.assign %[[VAL_2]] to %[[VAL_1]]#0 : i32, !fir.ref<i32>
! CHECK:           omp.parallel reduction(@max_i_32 %[[VAL_1]]#0 -> %[[VAL_3:.*]] : !fir.ref<i32>) {
! ...
! CHECK:             omp.terminator

