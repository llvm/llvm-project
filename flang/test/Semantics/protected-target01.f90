! RUN: %python %S/test_errors.py %s %flang_fc1
! F2028 PROTECTED_TARGET attribute and statement declarations
! (J3/26-007r1 C781, C867, C8109).  The attribute applies only to data
! pointers (J3/25-182r1 section 3), also in a type declaration statement.
module m
  type t
    real, pointer, protected_target :: c1
    class(*), pointer, protected_target :: c2(:)
    !ERROR: A component with the PROTECTED_TARGET attribute must also have the POINTER attribute
    real, protected_target :: c3
    !ERROR: A component with the PROTECTED_TARGET attribute must also have the POINTER attribute
    real, allocatable, protected_target :: c4
  end type
  real, pointer, protected_target :: p1
  real, pointer :: p2
  protected_target p2
  ! The POINTER attribute may be specified after the statement.
  protected_target :: p3
  real, pointer :: p3(:)
  real, pointer, protected, protected_target :: p4
  !WARNING: Attribute 'PROTECTED_TARGET' cannot be used more than once [-Wredundant-attribute]
  real, pointer, protected_target, protected_target :: p5
  real, pointer :: q
  !ERROR: An entity with the PROTECTED_TARGET attribute must be a data pointer
  real, protected_target :: x1
  !ERROR: An entity with the PROTECTED_TARGET attribute must be a data pointer
  real, allocatable, protected_target :: x2
  !ERROR: An entity with the PROTECTED_TARGET attribute must be a data pointer
  real, target :: x3
  protected_target x3
  !ERROR: An entity with the PROTECTED_TARGET attribute must be a data pointer
  real, parameter, protected_target :: x4 = 1.
  !ERROR: An entity with the PROTECTED_TARGET attribute must be a data pointer
  procedure(), pointer :: pp
  protected_target pp
  !ERROR: A PROTECTED_TARGET pointer may not be in a common block
  real, pointer, protected_target :: pc
  common /blk/ pc
 contains
  subroutine s1(d1, d2)
    real, pointer, protected_target, intent(in) :: d1
    real, pointer, intent(in out) :: d2
    protected_target d2
    block
      real, pointer :: b1
      protected_target :: b1
    end block
  end
  function f1() result(r)
    real, pointer, protected_target :: r
    nullify(r)
  end
  function f2()
    real, pointer, protected_target :: f2(:)
    nullify(f2)
  end
end

module m2
  use m, only: p1, q
  !WARNING: Use-associated 'p1' already has 'PROTECTED_TARGET' attribute [-Wredundant-attribute]
  protected_target p1
  !ERROR: Cannot change PROTECTED_TARGET attribute on use-associated 'q'
  protected_target q
end
