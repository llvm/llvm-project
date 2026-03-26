! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=host %s -o - \
! RUN:   | FileCheck %s

subroutine test1(x,s,N)
  real :: x(N), s
  integer :: N
  do concurrent(i=1:N) reduce(+:s)
     s=s+x(i)
  end do
end subroutine test1
subroutine test2(x,s,N)
  real :: x(N), s
  integer :: N
  do concurrent(i=1:N) reduce(+:s)
     s=s+x(i)
  end do
end subroutine test2

! CHECK:       omp.declare_reduction @[[RED_SYM:.*]] : f32 init
! CHECK-NOT:   omp.declare_reduction

! CHECK-LABEL: func.func @_QPtest1
! CHECK:         omp.parallel {
! CHECK:           omp.wsloop reduction(@[[RED_SYM]] {{.*}} : !fir.ref<f32>) {
! CHECK:           }
! CHECK:         }

! CHECK-LABEL: func.func @_QPtest2
! CHECK:         omp.parallel {
! CHECK:           omp.wsloop reduction(@[[RED_SYM]] {{.*}} : !fir.ref<f32>) {
! CHECK:           }
! CHECK:         }
