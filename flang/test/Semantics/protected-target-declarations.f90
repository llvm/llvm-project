! RUN: %python %S/test_errors.py %s %flang_fc1
! Declarations with the PROTECTED_TARGET attribute (F'2028 8.5.16, 8.6.14)

module m1
  real, target :: x
  real, pointer, protected_target :: p1 => x
  real, pointer :: p2, p3
  protected_target :: p2
  protected_target p3
  real, protected_target :: p4
  pointer :: p4
  real, pointer, protected, protected_target :: p5
  type :: t
    real, pointer, protected_target :: c1
    real, pointer, protected_target :: c2(:)
    !ERROR: A PROTECTED_TARGET entity must be a data pointer
    real, protected_target :: c3
    !ERROR: A PROTECTED_TARGET entity must be a data pointer
    real, allocatable, protected_target :: c4
  end type
  !ERROR: A PROTECTED_TARGET entity must be a data pointer
  real, protected_target :: n1
  !ERROR: A PROTECTED_TARGET entity must be a data pointer
  real, target, protected_target :: n2
  !ERROR: A PROTECTED_TARGET entity must be a data pointer
  real :: n3
  protected_target :: n3
  !ERROR: A PROTECTED_TARGET entity must be a data pointer
  real, allocatable, protected_target :: n4
  !ERROR: A PROTECTED_TARGET entity must be a data pointer
  procedure(f), pointer :: pp
  protected_target :: pp
 contains
  function f()
    real, pointer, protected_target :: f
    f => x
  end
  function g() result(r)
    real, pointer :: r
    protected_target :: r
    r => x
  end
  subroutine s1(d1, d2)
    real, pointer, protected_target, intent(in) :: d1
    real, pointer, protected_target :: d2
  end
end

subroutine s2
  use m1
  !ERROR: Cannot change PROTECTED_TARGET attribute on use-associated 'x'
  protected_target :: x
end

subroutine s3
  !ERROR: A PROTECTED_TARGET pointer may not be in a common block
  real, pointer, protected_target :: p
  !ERROR: A PROTECTED_TARGET pointer may not be in a common block
  integer, pointer, protected_target :: q
  real, pointer :: r
  common /blk/ p, r
  common q
end
