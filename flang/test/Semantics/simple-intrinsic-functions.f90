! RUN: %python %S/test_errors.py %s %flang_fc1
! Intrinsic functions are SIMPLE (F2023 16.1 p2). A simple target or actual
! may be associated with a pointer or dummy that is merely PURE, but a pointer
! or dummy whose interface is an intrinsic function is SIMPLE and so cannot be
! associated with a procedure that is only PURE (F2023 10.2.2.4 p3,
! F2023 15.5.2.10 p1).
module simple_intrinsic_function
  implicit none
  abstract interface
    simple real function simple_ifc(x)
      real, intent(in) :: x
    end function
    pure real function pure_ifc(x)
      real, intent(in) :: x
    end function
  end interface
contains
  pure real function pure_fn(x)
    real, intent(in) :: x
    pure_fn = x
  end function
  simple real function simple_fn(x)
    real, intent(in) :: x
    simple_fn = x
  end function
  subroutine take_pure(f)
    procedure(pure_ifc) :: f
  end subroutine
  subroutine take_intrinsic(f)
    procedure(sqrt) :: f
  end subroutine
  subroutine test()
    procedure(simple_ifc), pointer :: sp
    procedure(pure_ifc), pointer :: pp
    procedure(sqrt), pointer :: ip
    intrinsic :: sin
    sp => sin
    pp => sin
    call take_pure(sin)
    ip => simple_fn
    call take_intrinsic(simple_fn)
    !ERROR: Procedure pointer 'ip' associated with incompatible procedure designator 'pure_fn': incompatible procedure attributes: Simple
    ip => pure_fn
    !ERROR: Actual procedure argument has interface incompatible with dummy argument 'f=': incompatible procedure attributes: Simple
    call take_intrinsic(pure_fn)
  end subroutine
end module
