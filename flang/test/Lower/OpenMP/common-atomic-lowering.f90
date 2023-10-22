!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

!CHECK: func.func @_QQmain() attributes {fir.bindc_name = "sample"} {
!CHECK: %[[val_0:.*]] = fir.alloca i32 {bindc_name = "a", uniq_name = "_QFEa"}
!CHECK: %[[val_1:.*]]:2 = hlfir.declare %[[val_0]] {uniq_name = "_QFEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[val_2:.*]] = fir.alloca i32 {bindc_name = "b", uniq_name = "_QFEb"}
!CHECK: %[[val_3:.*]]:2 = hlfir.declare %[[val_2]] {uniq_name = "_QFEb"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[val_4:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
!CHECK: %[[val_5:.*]]:2 = hlfir.declare %[[val_4]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[val_c5:.*]] = arith.constant 5 : index
!CHECK: %[[val_6:.*]] = fir.alloca !fir.array<5xi32> {bindc_name = "y", uniq_name = "_QFEy"}
!CHECK: %[[val_7:.*]] = fir.shape %[[val_c5]] : (index) -> !fir.shape<1>
!CHECK: %[[val_8:.*]]:2 = hlfir.declare %[[val_6]](%[[val_7]]) {uniq_name = "_QFEy"} : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<5xi32>>, !fir.ref<!fir.array<5xi32>>)
!CHECK: %[[val_c2:.*]] = arith.constant 2 : index
!CHECK: %[[val_9:.*]] = hlfir.designate %[[val_8]]#0 (%[[val_c2]])  : (!fir.ref<!fir.array<5xi32>>, index) -> !fir.ref<i32>
!CHECK: %[[val_c8:.*]] = arith.constant 8 : i32
!CHECK: %[[val_10:.*]] = fir.load %[[val_5]]#0 : !fir.ref<i32>
!CHECK: %[[val_11:.*]] = arith.addi %[[val_c8]], %[[val_10]] : i32
!CHECK: %[[val_12:.*]] = hlfir.no_reassoc %[[val_11]] : i32
!CHECK: omp.atomic.update %[[val_9]] : !fir.ref<i32> {
!CHECK:   ^bb0(%[[ARG:.*]]: i32):
!CHECK:     %[[val_18:.*]] = arith.muli %[[ARG]], %[[val_12]] : i32
!CHECK:     omp.yield(%[[val_18]] : i32)
!CHECK: }
!CHECK: %[[val_c2_0:.*]] = arith.constant 2 : index
!CHECK: %[[val_13:.*]] = hlfir.designate %[[val_8]]#0 (%[[val_c2_0]])  : (!fir.ref<!fir.array<5xi32>>, index) -> !fir.ref<i32>
!CHECK: %[[val_c8_1:.*]] = arith.constant 8 : i32
!CHECK: omp.atomic.update %[[val_13:.*]] : !fir.ref<i32> {
!CHECK:   ^bb0(%[[ARG:.*]]: i32):
!CHECK:     %[[val_18:.*]] = arith.divui %[[ARG]], %[[val_c8_1]] : i32
!CHECK:     omp.yield(%[[val_18]] : i32)
!CHECK: }
!CHECK: %[[val_c8_2:.*]] = arith.constant 8 : i32
!CHECK: %[[val_c4:.*]] = arith.constant 4 : index
!CHECK: %[[val_14:.*]] = hlfir.designate %[[val_8]]#0 (%[[val_c4]])  : (!fir.ref<!fir.array<5xi32>>, index) -> !fir.ref<i32>
!CHECK: %[[val_15:.*]] = fir.load %[[val_14]] : !fir.ref<i32>
!CHECK: %[[val_16:.*]] = arith.addi %[[val_c8_2]], %[[val_15]] : i32
!CHECK: %[[val_17:.*]] = hlfir.no_reassoc %[[val_16]] : i32
!CHECK: omp.atomic.update %[[val_5]]#1 : !fir.ref<i32> {
!CHECK:   ^bb0(%[[ARG:.*]]: i32):
!CHECK:      %[[val_18:.*]] = arith.addi %[[ARG]], %[[val_17]] : i32
!CHECK:      omp.yield(%[[val_18]] : i32)
!CHECK: }
!CHECK: %[[val_c8_3:.*]] = arith.constant 8 : i32
!CHECK: omp.atomic.update %[[val_5]]#1 : !fir.ref<i32> {
!CHECK:   ^bb0(%[[ARG]]: i32):
!CHECK:     %[[val_18:.*]] = arith.subi %[[ARG]], %[[val_c8_3]] : i32
!CHECK:     omp.yield(%[[val_18]] : i32)
!CHECK:   }
!CHECK: return
!CHECK: }
program sample

  integer :: x
  integer, dimension(5) :: y
  integer :: a, b

  !$omp atomic update
    y(2) =  (8 + x) * y(2)
  !$omp end atomic

  !$omp atomic update
    y(2) =  y(2) / 8
  !$omp end atomic

  !$omp atomic update
    x =  (8 + y(4)) + x
  !$omp end atomic

  !$omp atomic update
    x =  8 - x
  !$omp end atomic

end program sample
