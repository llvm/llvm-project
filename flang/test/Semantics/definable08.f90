!RUN: %python %S/test_errors.py %s %flang_fc1

subroutine s1(x)
  integer, intent(in) :: x
  real, save :: c[*]
  real :: r
  !ERROR: STAT variable 'x' is not definable
  !BECAUSE: 'x' is an INTENT(IN) dummy argument
  r = c[1, STAT=x]
end subroutine

module m
  integer, protected :: pv = 0
end module
subroutine s2
  use m
  real, save :: rCoarray[1,2,*]
  real :: rVar1
  !ERROR: STAT variable 'pv' is not definable
  !BECAUSE: 'pv' is protected in this scope
  rVar1 = rCoarray[1,2,3,STAT=pv]
end

subroutine s3()
  real, save :: c[10,20,*]
  real :: r
  integer :: i
  associate (z => i + 1)
    !ERROR: STAT variable 'z' is not definable
    !BECAUSE: 'z' is construct associated with an expression
    r = c[1,2,3, STAT=z]
  end associate
end subroutine

subroutine s4(a)
  integer, intent(in) :: a(2)
  real, save :: c[*]
  real :: r
  !ERROR: STAT variable 'a' is not definable
  !BECAUSE: 'a' is an INTENT(IN) dummy argument
  r = c[1, stat=a(1)]
end subroutine

subroutine s5
  real, save :: c[*]
  real :: r
  integer :: i
  do i = 1, 2
    !ERROR: Cannot redefine DO variable 'i'
    r = c[1, stat=i]
  end do
end subroutine
