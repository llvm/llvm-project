!RUN: not %flang_fc1 -fsyntax-only -pedantic %s 2>&1 | FileCheck %s

subroutine s1(k)
  integer, intent(in) :: x 
  real, save :: c[*]
  real :: r
! CHECK: error: STAT variable 'x' is not definable
  r = c[1, STAT=x]
end subroutine

module m
  integer, protected :: pv = 0
end module
subroutine s2
  use m
  real, save :: rCoarray[1,2,*]
  real :: rVar1
! CHECK: error: STAT variable 'pv' is not definable
  rVar1 = rCoarray[1,2,3,STAT=pv]
end

subroutine s3()
  real, save :: c[10,20,*]
  real :: r
  integer :: i
  associate (z => i + 1)
! CHECK: error: STAT variable 'z' is not definable
    r = c[1,2,3, STAT=z]
  end associate
end subroutine
