! RUN: %python %S/test_modfile.py %s %flang_fc1
! Ensure that m2.mod explicitly USEs a generic interface from m1
! so it can utilize its shadowed derived type.
module m1
  implicit none
  type :: xyz
    integer :: n
  end type
  interface xyz
    module procedure xzy
  end interface
 contains
  function xzy(j) result(res)
    integer, intent(in) :: j
    type(xyz) :: res
    res%n = j
  end function
end module

!Expect: m1.mod
!module m1
!type::xyz
!integer(4)::n
!end type
!interface xyz
!procedure::xzy
!end interface
!contains
!function xzy(j) result(res)
!integer(4),intent(in)::j
!type(xyz)::res
!end
!end

module m2
  implicit none
 contains
  function foo(j) result(res)
    use :: m1, only: xyz
    integer, intent(in) :: j
    type(xyz) :: res
    res = xyz(j)
  end function
end module

!Expect: m2.mod
!module m2
!contains
!function foo(j) result(res)
!use m1,only:xyz
!integer(4),intent(in)::j
!type(xyz)::res
!end
!end
