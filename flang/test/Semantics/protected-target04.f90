! RUN: %python %S/test_errors.py %s %flang_fc1
! Protection follows construct associations and subobjects, but an ordinary
! pointer component's target is not a subobject (F2028 9.4.2p5).

module associations
  type node
    integer :: value
    integer, allocatable :: allocation
    type(node), pointer :: link
    type(node), pointer, protected_target :: readonly_link
    procedure(sin), pointer, nopass :: callback
  end type
contains
  subroutine nested(p, x)
    type(node), pointer, protected_target :: p
    type(node), target :: x
    integer, pointer :: ordinary
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    p%value = 1
    !ERROR: pointer 'ordinary' must have the PROTECTED_TARGET attribute when associated with the target of PROTECTED_TARGET pointer 'p'
    ordinary => p%value
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    p%link => x
    !ERROR: 'link' may not appear in NULLIFY
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    nullify(p%link)
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    p%callback => sin
    !ERROR: 'callback' may not appear in NULLIFY
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    nullify(p%callback)
    !ERROR: Actual argument associated with INTENT(OUT) procedure pointer dummy argument 'callback=' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    call reset_callback(p%callback)
    !ERROR: Name in ALLOCATE statement is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    allocate(p%allocation, source=1)
    !ERROR: Name in ALLOCATE statement is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    allocate(p%link)
    !ERROR: Name in DEALLOCATE statement is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    deallocate(p%link)
    p%link%value = 1
    p%link%link => x
    nullify(p%link%link)
    p%link%callback => sin
    nullify(p%link%callback)
    ordinary => p%link%value
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'readonly_link' is not definable
    p%link%readonly_link%value = 1
    !ERROR: A PROTECTED_TARGET pointer may not be deallocated
    deallocate(p%link%readonly_link)
  end subroutine
  subroutine associate_array(p)
    integer, pointer, protected_target :: p(:)
    integer, pointer :: ordinary(:)
    external implicit
    associate(a => p(1:2))
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
      a = 1
      !ERROR: pointer 'ordinary' must have the PROTECTED_TARGET attribute when associated with the target of PROTECTED_TARGET pointer 'p'
      ordinary => a
      !ERROR: PROTECTED_TARGET pointer or subobject 'a' requires an explicit interface
      call implicit(a)
      associate(b => a)
        !ERROR: Left-hand side of assignment is not definable
        !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
        b(1) = 1
      end associate
    end associate
  end subroutine
  subroutine select_type(p)
    class(node), pointer, protected_target :: p
    select type(a => p)
    type is(node)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
      a%value = 1
      a%link%value = 1
    end select
  end subroutine
  subroutine select_rank(p)
    integer, pointer, protected_target :: p(..)
    integer, pointer, protected_target :: p2(:)
    integer, pointer :: w(:)
    select rank(a => p)
    rank(1)
      a => p2
      !ERROR: pointer 'w' must have the PROTECTED_TARGET attribute when associated with the target of PROTECTED_TARGET pointer 'p'
      w => a
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
      a(1) = 1
      !ERROR: Allocation of a PROTECTED_TARGET pointer requires SOURCE=
      allocate(a(2))
      allocate(a(2), source=[1, 2])
      nullify(a)
    rank default
      select rank(b => a)
      rank(1)
        b => p2
      end select
      !ERROR: PROTECTED_TARGET pointer or subobject 'a' may be associated only with an INTENT(IN) nonpointer dummy argument 'x='
      call assumed_rank(a)
    end select
  end subroutine
  subroutine assumed_rank(x)
    integer :: x(..)
  end subroutine
  subroutine reset_callback(callback)
    procedure(sin), pointer, intent(out) :: callback
    nullify(callback)
  end subroutine
end module
