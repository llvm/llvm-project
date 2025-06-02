! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! Test declare simd with linear clause

module mod
contains
subroutine sub(m,i)
!$omp declare simd linear(i:1)
  implicit none
  integer*8 i,n
  value i
  parameter(n=10000)
  real*4 a,b,m
  common/com1/a(n)
  common/com2/b(n)
  a(i) = b(i) + m
  i=i+2
end subroutine
end module
