! RUN: %S/test_modfile.sh %s %t %f18
module m
  type :: t
    procedure(a), pointer, pass :: c
    procedure(a), pointer, pass(x) :: d
  contains
    procedure, pass(y) :: a, b
  end type
contains
  subroutine a(x, y)
    class(t) :: x, y
  end
  subroutine b(y)
    class(t) :: y
  end
end module

!Expect: m.mod
!module m
!  type::t
!    procedure(a),pass,pointer::c
!    procedure(a),pass(x),pointer::d
!  contains
!    procedure,pass(y)::a
!    procedure,pass(y)::b
!  end type
!contains
!  subroutine a(x,y)
!    class(t)::x
!    class(t)::y
!  end
!  subroutine b(y)
!    class(t)::y
!  end
!end
