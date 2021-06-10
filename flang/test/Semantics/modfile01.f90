! RUN: %S/test_modfile.sh %s %t %flang_fc1
! REQUIRES: shell
! Check correct modfile generation for type with private component.
module m
  integer :: i
  integer, private :: j
  type :: t
    integer :: i
    integer, private :: j
  end type
  type, private :: u
  end type
  type(t) :: x
end

!Expect: m.mod
!module m
!integer(4)::i
!integer(4),private::j
!type::t
!integer(4)::i
!integer(4),private::j
!end type
!type,private::u
!end type
!type(t)::x
!end

! Check correct modfile generation for type with private module procedure.

module m2
  private :: s1
contains
  subroutine s1()
  end
  subroutine s2()
  end
end

!Expect: m2.mod
!module m2
! private::s1
!contains
! subroutine s1()
! end
! subroutine s2()
! end
!end

module m3
  private
  public :: f1
contains
  real function f1()
  end
  real function f2()
  end
end

!Expect: m3.mod
!module m3
! private::f2
!contains
! function f1()
!  real(4)::f1
! end
! function f2()
!  real(4)::f2
! end
!end

! Test optional dummy procedure
module m4
contains
  subroutine s(f)
    interface
      logical recursive function f()
        implicit none
      end function
    end interface
    optional f
  end
end

!Expect: m4.mod
!module m4
!contains
! subroutine s(f)
!  optional::f
!  interface
!   recursive function f()
!    logical(4)::f
!   end
!  end interface
! end
!end
