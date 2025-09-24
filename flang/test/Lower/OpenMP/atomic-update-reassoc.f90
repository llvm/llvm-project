!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s

subroutine f00(x, y)
  implicit none
  integer :: x, y

  !$omp atomic update
  x = ((x + 1) + y) + 2
end

!CHECK-LABEL: func.func @_QPf00
!CHECK: %[[X:[0-9]+]]:2 = hlfir.declare %arg0
!CHECK: %[[Y:[0-9]+]]:2 = hlfir.declare %arg1
!CHECK: %c1_i32 = arith.constant 1 : i32
!CHECK: %[[LOAD_Y:[0-9]+]] = fir.load %[[Y]]#0 : !fir.ref<i32>
!CHECK: %[[Y_1:[0-9]+]] = arith.addi %c1_i32, %[[LOAD_Y]] : i32
!CHECK: %c2_i32 = arith.constant 2 : i32
!CHECK: %[[Y_1_2:[0-9]+]] = arith.addi %[[Y_1]], %c2_i32 : i32
!CHECK: omp.atomic.update memory_order(relaxed) %[[X]]#0 : !fir.ref<i32> {
!CHECK: ^bb0(%[[ARG:arg[0-9]+]]: i32):
!CHECK:   %[[ARG_P:[0-9]+]] = arith.addi %[[ARG]], %[[Y_1_2]] : i32
!CHECK:   omp.yield(%[[ARG_P]] : i32)
!CHECK: }


subroutine f01(x, y)
  implicit none
  real :: x
  integer :: y

  !$omp atomic update
  x = (int(x) + y) + 1
end

!CHECK-LABEL: func.func @_QPf01
!CHECK: %[[X:[0-9]+]]:2 = hlfir.declare %arg0
!CHECK: %[[Y:[0-9]+]]:2 = hlfir.declare %arg1
!CHECK: %[[LOAD_Y:[0-9]+]] = fir.load %[[Y]]#0 : !fir.ref<i32>
!CHECK: %c1_i32 = arith.constant 1 : i32
!CHECK: %[[Y_1:[0-9]+]] = arith.addi %[[LOAD_Y]], %c1_i32 : i32
!CHECK: omp.atomic.update memory_order(relaxed) %[[X]]#0 : !fir.ref<f32> {
!CHECK: ^bb0(%[[ARG:arg[0-9]+]]: f32):
!CHECK:   %[[ARG_I:[0-9]+]] = fir.convert %[[ARG]] : (f32) -> i32
!CHECK:   %[[ARG_P:[0-9]+]] = arith.addi %[[ARG_I]], %[[Y_1]] : i32
!CHECK:   %[[ARG_F:[0-9]+]] = fir.convert %[[ARG_P]] : (i32) -> f32
!CHECK:   omp.yield(%[[ARG_F]] : f32)
!CHECK: }


subroutine f02(x, a, b, c)
  implicit none
  integer(kind=4) :: x
  integer(kind=8) :: a, b, c

  !$omp atomic update
  x = ((b + a) + x) + c
end

!CHECK-LABEL: func.func @_QPf02
!CHECK: %[[A:[0-9]+]]:2 = hlfir.declare %arg1
!CHECK: %[[B:[0-9]+]]:2 = hlfir.declare %arg2
!CHECK: %[[C:[0-9]+]]:2 = hlfir.declare %arg3
!CHECK: %[[X:[0-9]+]]:2 = hlfir.declare %arg0
!CHECK: %[[LOAD_B:[0-9]+]] = fir.load %[[B]]#0 : !fir.ref<i64>
!CHECK: %[[LOAD_A:[0-9]+]] = fir.load %[[A]]#0 : !fir.ref<i64>
!CHECK: %[[A_B:[0-9]+]] = arith.addi %[[LOAD_B]], %[[LOAD_A]] : i64
!CHECK: %[[LOAD_C:[0-9]+]] = fir.load %[[C]]#0 : !fir.ref<i64>
!CHECK: %[[A_B_C:[0-9]+]] = arith.addi %[[A_B]], %[[LOAD_C]] : i64
!CHECK: omp.atomic.update memory_order(relaxed) %[[X]]#0 : !fir.ref<i32> {
!CHECK: ^bb0(%[[ARG:arg[0-9]+]]: i32):
!CHECK:   %[[ARG_8:[0-9]+]] = fir.convert %[[ARG]] : (i32) -> i64
!CHECK:   %[[ARG_P:[0-9]+]] = arith.addi %[[ARG_8]], %[[A_B_C]] : i64
!CHECK:   %[[ARG_4:[0-9]+]] = fir.convert %[[ARG_P]] : (i64) -> i32
!CHECK:   omp.yield(%[[ARG_4]] : i32)
!CHECK: }
