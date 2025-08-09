! This test checks lowering of OpenMP DO Directive(Worksharing) with collapse.

! RUN: bbc -fopenmp -fopenmp-version=51 -emit-hlfir %s -o - | FileCheck %s

!CHECK-LABEL: func.func @_QQmain() attributes {fir.bindc_name = "WSLOOP_TILE"} {
program wsloop_tile
  integer :: i, j, k
  integer :: a, b, c
  integer :: x

  a=30
  b=20
  c=50
  x=0

  !CHECK: omp.loop_nest (%[[IV_0:.*]], %[[IV_1:.*]], %[[IV_2:.*]]) : i32
  !CHECK-SAME: tiles(2, 5, 10)

  !$omp do
  !$omp tile sizes(2,5,10)
  do i = 1, a
     do j= 1, b
        do k = 1, c
  !CHECK: hlfir.assign %[[IV_0]] to %[[IV_0A:.*]] : i32
  !CHECK: hlfir.assign %[[IV_1]] to %[[IV_1A:.*]] : i32
  !CHECK: hlfir.assign %[[IV_2]] to %[[IV_2A:.*]] : i32
  !CHECK: %[[IVV_0:.*]] = fir.load %[[IV_0A]]
  !CHECK: %[[SUM0:.*]] = arith.addi %{{.*}}, %[[IVV_0]] : i32
  !CHECK: %[[IVV_1:.*]] = fir.load %[[IV_1A]]
  !CHECK: %[[SUM1:.*]] = arith.addi %[[SUM0]], %[[IVV_1]] : i32
  !CHECK: %[[IVV_2:.*]] = fir.load %[[IV_2A]]
  !CHECK: %[[SUM2:.*]] = arith.addi %[[SUM1]], %[[IVV_2]] : i32
           x = x + i + j + k
        end do
     end do
  end do
  !$omp end tile
  !$omp end do
end program wsloop_tile
