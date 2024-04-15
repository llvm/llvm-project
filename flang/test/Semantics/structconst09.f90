! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! Structure constructors with bad pointer targets
module m
  real, target, save :: x
  type t
    real, pointer :: rp => x
    procedure(f), pointer, nopass :: pp => f
  end type
 contains
  real function f()
    f = 0.
  end
  subroutine test(da, dp)
    real, target :: y, da
    procedure(f) dp
    procedure(f), pointer :: lpp
    external ext
    type(t) :: a1 = t() ! ok
    type(t) :: a2 = t(rp=x) ! ok
    type(t) :: a3 = t(pp=f) ! ok
    type(t) :: a4 = t(pp=ext) ! ok
    !ERROR: Must be a constant value
    type(t) :: a5 = t(rp=y)
    !ERROR: Must be a constant value
    type(t) :: a6 = t(rp=da)
    !ERROR: Must be a constant value
    type(t) :: a7 = t(pp=lpp)
    !ERROR: Must be a constant value
    type(t) :: a8 = t(pp=internal)
    !ERROR: Must be a constant value
    type(t) :: a9 = t(pp=dp)
   contains
    real function internal()
      internal = 666.
    end
  end
end
