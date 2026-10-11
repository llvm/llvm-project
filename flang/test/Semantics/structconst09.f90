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

module protected_target_constructors
  type ordinary
    integer, pointer :: p(:)
  end type
  type readonly
    integer, pointer, protected_target :: p(:)
  end type
  type data
    integer :: value
  end type
  type node
    integer :: array(3)
    integer, pointer :: link(:)
  end type
  integer, target :: values(3)
contains
  function get() result(r)
    integer, pointer, protected_target :: r(:)
    r => values
  end function
  subroutine constructors(p, n)
    integer, pointer, protected_target :: p(:)
    type(node), pointer, protected_target :: n
    type(ordinary) :: writable
    type(readonly) :: protected
    type(data) :: copied
    protected = readonly(p)
    protected = readonly(p(1:2))
    protected = readonly(n%array)
    protected = readonly(get())
    protected = readonly(null())
    writable = ordinary(n%link)
    copied = data(p(1))
    !ERROR: A PROTECTED_TARGET target may not be associated with pointer 'p' that is not PROTECTED_TARGET
    writable = ordinary(p)
    !ERROR: A PROTECTED_TARGET target may not be associated with pointer 'p' that is not PROTECTED_TARGET
    writable = ordinary(p(1:2))
    !ERROR: A PROTECTED_TARGET target may not be associated with pointer 'p' that is not PROTECTED_TARGET
    writable = ordinary(n%array)
    !ERROR: A PROTECTED_TARGET target may not be associated with pointer 'p' that is not PROTECTED_TARGET
    writable = ordinary(get())
    associate(a => n%array)
      !ERROR: A PROTECTED_TARGET target may not be associated with pointer 'p' that is not PROTECTED_TARGET
      writable = ordinary(a)
    end associate
  end subroutine
end module
