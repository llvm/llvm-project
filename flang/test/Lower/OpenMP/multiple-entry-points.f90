!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! Check the first entry point
!CHECK: func.func @_QPprocess_a
!CHECK: omp.parallel
!CHECK: omp.wsloop
!CHECK: %[[V0:[0-9]+]] = fir.load %{{[0-9]+}} : !fir.ref<f32>
!CHECK: %[[V1:[a-z_0-9]+]] = arith.constant 2.000000e+00 : f32
!CHECK:   = arith.mulf %[[V0]], %[[V1]] fastmath<contract> : f32
!CHECK: omp.terminator
!CHECK-NOT: omp
!CHECK: return

! Check the second entry point
!CHECK: func.func @_QPprocess_b
!CHECK: omp.parallel
!CHECK: fir.do_loop
!CHECK: %[[V3:[0-9]+]] = fir.load %[[V2:[0-9]+]]#0 : !fir.ref<i32>
!CHECK: %[[V4:[0-9]+]] = fir.load %[[V2]]#0 : !fir.ref<i32>
!CHECK:   = arith.muli %[[V3]], %[[V4]] : i32
!CHECK: omp.terminator
!CHECK-NOT: omp
!CHECK: return

subroutine process_a(n, a)
  integer, intent(in) :: n
  real, intent(inout) :: a(n)
  integer :: i

  !$omp parallel do
  do i = 1, n
    a(i) = a(i) * 2.0
  end do
  !$omp end parallel do

  return

  entry process_b(n, b)
    
  !$omp parallel
  do i = 1, n
    a(i) = i * i
  end do
  !$omp end parallel

end subroutine process_a
