! RUN: %python %S/test_errors.py %s %flang_fc1
subroutine subr(a,b,n,m)
  real n,m
!ERROR: Must have INTEGER type, but is REAL(4)
!ERROR: Must have INTEGER type, but is REAL(4)
  integer a(n,m)
!ERROR: Rank of left-hand side is 2, but right-hand side has rank 1
!ERROR: Subscript expression has rank 2 greater than 1
  a = a(a,j)
end
