! Check the flang-omp-report plugin for omp-declarative-directive.f90

! REQUIRES: plugins, examples, shell

! RUN: %flang_fc1 -load %llvmshlibdir/flangOmpReport.so -plugin flang-omp-report -fopenmp %s -o - | FileCheck %s

! Check OpenMP declarative directives

! 2.8.2 declare-simd

subroutine declare_simd_1(a, b)
  real(8), intent(inout) :: a, b
  !$omp declare simd(declare_simd_1) aligned(a)
  a = 3.14 + b
end subroutine declare_simd_1

! 2.10.6 declare-target
! 2.15.2 threadprivate

module m2
contains
  subroutine foo
    !$omp declare target
    integer, parameter :: N=10000, M=1024
    integer :: i
    real :: Q(N, N), R(N,M), S(M,M)
  end subroutine foo
end module m2

end

! CHECK:---
! CHECK-NEXT:- file:            '{{[^"]*}}omp-declarative-directive.f90'
! CHECK-NEXT:  line:            13
! CHECK-NEXT:  construct:       declare simd
! CHECK-NEXT:  clauses:
! CHECK-NEXT:    - clause:      aligned
! CHECK-NEXT:      details:     a
! CHECK-NEXT:- file:            '{{[^"]*}}omp-declarative-directive.f90'
! CHECK-NEXT:  line:            23
! CHECK-NEXT:  construct:       declare target
! CHECK-NEXT:  clauses:         []
! CHECK-NEXT:...
