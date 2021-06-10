! RUN: %S/test_modfile.sh %s %t %flang_fc1
! REQUIRES: shell
! Check modfile generation for external declarations
module m
  real, external :: a
  logical b
  external c
  complex c
  external b, d
  procedure() :: e
  procedure(real) :: f
  procedure(s) :: g
  type t
    procedure(), pointer, nopass :: e
    procedure(real), nopass, pointer :: f
    procedure(s), private, pointer :: g
  end type
contains
  subroutine s(x)
    class(t) :: x
  end
end

!Expect: m.mod
!module m
!  procedure(real(4))::a
!  procedure(logical(4))::b
!  procedure(complex(4))::c
!  procedure()::d
!  procedure()::e
!  procedure(real(4))::f
!  procedure(s)::g
!  type::t
!    procedure(),nopass,pointer::e
!    procedure(real(4)),nopass,pointer::f
!    procedure(s),pointer,private::g
!  end type
!contains
!  subroutine s(x)
!    class(t)::x
!  end
!end
