! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! Test declare simd with linear clause does not cause an implicit declaration of i

module mod
contains
subroutine test(i)
!$omp declare simd linear(i:1)
  implicit none
  integer*8 i
  i=i+2
end subroutine
end module
