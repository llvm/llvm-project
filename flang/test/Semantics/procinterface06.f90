! RUN: %python %S/test_errors.py %s %flang_fc1
! C1515: a name declared by a PROCEDURE statement may be used as an
! interface name only after that declaration.
module m
  implicit none
  abstract interface
    subroutine s_iface()
    end subroutine
  end interface
  !ERROR: 'q' must be declared before it is used as a procedure interface
  procedure(q), pointer :: r
  !ERROR: 'q' must be declared before it is used as a procedure interface
  procedure(q) :: r2
  procedure(s_iface), pointer :: q
  procedure(q), pointer :: t ! ok, q is declared above
  procedure(later), pointer :: u ! ok, later is declared by an interface body
  interface
    subroutine later()
    end subroutine
  end interface
  type dt
    !ERROR: 'q2' must be declared before it is used as a procedure interface
    procedure(q2), pointer, nopass :: comp
  end type
  procedure(s_iface), pointer :: q2
end module

module m2
  implicit none
  type, abstract :: t
  contains
    !ERROR: 'q' must be declared before it is used as a procedure interface
    procedure(q), deferred, nopass :: b
  end type
  abstract interface
    subroutine s_iface()
    end subroutine
  end interface
  procedure(s_iface), pointer :: q
end module

subroutine sub(a, b)
  abstract interface
    subroutine s_iface()
    end subroutine
  end interface
  !ERROR: 'b' must be declared before it is used as a procedure interface
  procedure(b), pointer :: a
  procedure(s_iface) :: b
end subroutine
