! Tests that the omp.integer_wrap_around module attribute is set when -fno-wrapv (default) is active with OpenMP enabled and absent when -fwrapv is specified.

! RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s --check-prefix=NOWRAPV
! RUN: %flang_fc1 -emit-fir -fopenmp -fwrapv %s -o - | FileCheck %s --check-prefix=WRAPV

! NOWRAPV: module attributes {{{.*}}omp.integer_wrap_around = #omp.integer_wrap_around<integer_wrap_around = false>{{.*}}}
! WRAPV-NOT: omp.integer_wrap_around

subroutine omp_loop(a, b, n)
  integer :: n, i
  real(8) :: a(n), b(n)
  !$omp parallel do
  do i = 1, n
    a(i) = b(i)
  end do
end subroutine
