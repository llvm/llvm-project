! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! C701 The type-param-value for a kind type parameter shall be a constant
! expression.  This constraint looks like a mistake in the standard.
integer, parameter :: k = 8
real, parameter :: l = 8.0
integer :: n = 2
!ERROR: Must be a constant value
parameter(m=n)
integer(k) :: x
! C713 A scalar-int-constant-name shall be a named constant of type integer.
!ERROR: Must have INTEGER type, but is REAL(4)
integer(l) :: y
!ERROR: Must be a constant value
integer(n) :: z
type t(k)
  integer, kind :: k
end type
!ERROR: Type parameter 'k' lacks a value and has no default
type(t( &
!ERROR: Must have INTEGER type, but is LOGICAL(4)
  .true.)) :: w
!ERROR: Must have INTEGER type, but is REAL(4)
real :: u(l*2)
!ERROR: Must have INTEGER type, but is REAL(4)
character(len=l) :: v
!ERROR: Value of named constant 'o' (o) cannot be computed as a constant value
real, parameter ::  o = o
!ERROR: Must be a constant value
integer, parameter ::  p = 0/0
!ERROR: Must be a constant value
integer, parameter ::  q = 1+2*(1/0)
integer not_constant
!ERROR: Must be a constant value
integer, parameter :: s1 = not_constant/2
!ERROR: Must be a constant value
integer, parameter :: s2 = 3/not_constant
!ERROR: Must be a constant value
integer(kind=2/0) r
integer, parameter :: sok(*)=[1,2]/[1,2]
!ERROR: Must be a constant value
integer, parameter :: snok(*)=[1,2]/[1,0]
end
