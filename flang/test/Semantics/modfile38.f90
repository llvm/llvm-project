! RUN: %python %S/test_modfile.py %s %flang_fc1

! Ensure that an interface with the same name as a derived type
! does not cause that shadowed name to be emitted later than its
! uses in the module file.

module m
  type :: t
  end type
  type :: t2
    type(t) :: c
  end type
  interface t
    module procedure f
  end interface
 contains
  type(t) function f
  end function
end module

!Expect: m.mod
!module m
!interface t
!procedure::f
!end interface
!type::t
!end type
!type::t2
!type(t)::c
!end type
!contains
!function f()
!type(t)::f
!end
!end
