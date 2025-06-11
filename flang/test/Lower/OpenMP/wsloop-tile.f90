! This test checks lowering of OpenMP DO Directive(Worksharing) with collapse.

! RUN: bbc -fopenmp -fopenmp-version=51 -emit-hlfir %s -o - | FileCheck %s

!CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "wsloop_tile"} {
program wsloop_tile
  integer :: i, j, k
  integer :: a, b, c
  integer :: x

  a=30
  b=20
  c=50
  x=0

  !CHECK: omp.loop_nest
  !CHECK-SAME: tiles(2, 5, 10)

  !$omp do
  !$omp tile sizes(2,5,10)
  do i = 1, a
     do j= 1, b
        do k = 1, c
           x = x + i + j + k
        end do
     end do
  end do
  !$omp end tile
  !$omp end do
end program wsloop_tile
