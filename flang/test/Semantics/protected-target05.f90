! RUN: %python %S/test_errors.py %s %flang_fc1
! F2028 draft PROTECTED_TARGET across construct associations and procedure actuals.

module protected_target_constructs
  abstract interface
    subroutine iface()
    end subroutine
  end interface
  type node
    integer :: value
    integer, pointer :: link
    type(node), pointer :: next
    procedure(iface), pointer, nopass :: proc
  end type
contains
  subroutine associate_pointer(p)
    type(node), pointer, protected_target :: p
    integer, target :: target
    type(node), target :: object
    p%link = 1
    p%next%value = 1
    associate(alias => p)
      ! Independent pointer targets remain writable through an associate name.
      alias%link = 1
      alias%next%value = 1
      alias%next%link => target
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target is not definable
      alias%value = 1
      !ERROR: The left-hand side of a pointer assignment is not definable
      !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target is not definable
      alias%link => target
      !ERROR: The left-hand side of a pointer assignment is not definable
      !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target is not definable
      alias%next => object
      !ERROR: 'link' may not appear in NULLIFY
      !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target is not definable
      nullify(alias%link)
      associate(nested => alias)
        nested%link = 1
        nested%next%value = 1
        !ERROR: The left-hand side of a pointer assignment is not definable
        !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target is not definable
        nested%link => target
      end associate
      ! ASSOCIATE does not give its entity the selector's POINTER attribute.
      !ERROR: The left-hand side of a pointer assignment is not definable
      !BECAUSE: 'alias' is not a pointer
      alias => object
    end associate
  end subroutine

  function get_node() result(p)
    type(node), pointer, protected_target :: p
    type(node), target, save :: object
    p => object
  end function

  subroutine associate_function
    integer, target :: target
    type(node), target :: object
    associate(alias => get_node())
      alias%link = 1
      alias%next%value = 1
      alias%next%link => target
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target is not definable
      alias%value = 1
      !ERROR: The left-hand side of a pointer assignment is not definable
      !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target is not definable
      alias%link => target
      !ERROR: The left-hand side of a pointer assignment is not definable
      !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target is not definable
      alias%next => object
    end associate
  end subroutine

  subroutine select_type_pointer(p)
    class(node), pointer, protected_target :: p
    type(node), target :: object
    select type(alias => p)
    type is(node)
      alias%link = 1
      alias%next%value = 1
      !ERROR: The left-hand side of a pointer assignment is not definable
      !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target is not definable
      alias%link => null()
      ! SELECT TYPE does not give its entity the POINTER attribute either.
      !ERROR: The left-hand side of a pointer assignment is not definable
      !BECAUSE: 'alias' is not a pointer
      alias => object
    end select
  end subroutine

  subroutine rank_pointer(p, scalar, array)
    integer, pointer, protected_target :: p(..), scalar, array(:)
    select rank(p)
    rank(0)
      allocate(p, source=1)
      p => scalar
      !ERROR: ALLOCATE of a PROTECTED_TARGET pointer requires SOURCE=
      allocate(p)
      !ERROR: A PROTECTED_TARGET pointer may not appear in DEALLOCATE
      deallocate(p)
    rank(1)
      allocate(p(2), source=1)
      p => array
      !ERROR: ALLOCATE of a PROTECTED_TARGET pointer requires SOURCE=
      allocate(p(2))
      !ERROR: A PROTECTED_TARGET pointer may not appear in DEALLOCATE
      deallocate(p)
    rank default
      call read_rank(p)
      !ERROR: A PROTECTED_TARGET pointer may not appear in DEALLOCATE
      deallocate(p)
    end select
    select rank(alias => p)
    rank(0)
      allocate(alias, source=1)
      alias => scalar
      !ERROR: ALLOCATE of a PROTECTED_TARGET pointer requires SOURCE=
      allocate(alias)
    rank(1)
      allocate(alias(2), source=1)
      alias => array
      !ERROR: ALLOCATE of a PROTECTED_TARGET pointer requires SOURCE=
      allocate(alias(2))
    rank default
      call read_rank(alias)
      !ERROR: A PROTECTED_TARGET pointer may not appear in DEALLOCATE
      deallocate(alias)
      select rank(nested => alias)
      rank(0)
        allocate(nested, source=1)
        nested => scalar
        !ERROR: ALLOCATE of a PROTECTED_TARGET pointer requires SOURCE=
        allocate(nested)
      end select
    end select
  end subroutine

  subroutine read_rank(p)
    integer, pointer, protected_target, intent(in) :: p(..)
  end subroutine

  subroutine ordinary_rank(p, value)
    integer, pointer :: p(..)
    integer :: value(..)
    select rank(p)
    rank(0)
      allocate(p)
      p = 1
      deallocate(p)
    rank(1)
      allocate(p(2))
      p = 1
      deallocate(p)
    end select
    select rank(alias => value)
    rank(*)
      alias(1) = 1
    rank(1)
      alias = 1
    end select
  end subroutine

  subroutine procedure_arguments(p, ordinary)
    type(node), pointer, protected_target :: p
    type(node), pointer :: ordinary
    ! A pointer dummy receives the descriptor, even with INTENT(IN).
    !ERROR: PROTECTED_TARGET pointer or subobject 'p%proc' may not be associated with dummy argument 'q=' without PROTECTED_TARGET
    call proc_in(p%proc)
    !ERROR: PROTECTED_TARGET pointer or subobject 'p%proc' may not be associated with dummy argument 'q=' without PROTECTED_TARGET
    call proc_default(p%proc)
    !ERROR: PROTECTED_TARGET pointer or subobject 'p%proc' may not be associated with dummy argument 'q=' without PROTECTED_TARGET
    call proc_inout(p%proc)
    !ERROR: PROTECTED_TARGET pointer or subobject 'p%proc' may not be associated with dummy argument 'q=' without PROTECTED_TARGET
    call proc_out(p%proc)
    ! A nonpointer procedure dummy receives only the independent procedure target.
    call proc_nonpointer(p%proc)
    call proc_in(proc_target)
    call proc_in(ordinary%proc)
    call proc_default(ordinary%proc)
    call proc_inout(ordinary%proc)
    call proc_out(ordinary%proc)
    call proc_in(p%next%proc)
    call proc_default(p%next%proc)
    call proc_inout(p%next%proc)
    call proc_out(p%next%proc)
    associate(alias => p)
      !ERROR: PROTECTED_TARGET pointer or subobject 'alias%proc' may not be associated with dummy argument 'q=' without PROTECTED_TARGET
      call proc_in(alias%proc)
      call proc_nonpointer(alias%proc)
      call proc_inout(alias%next%proc)
    end associate
  end subroutine

  subroutine proc_in(q)
    procedure(iface), pointer, intent(in) :: q
  end subroutine
  subroutine proc_default(q)
    procedure(iface), pointer :: q
    nullify(q)
  end subroutine
  subroutine proc_inout(q)
    procedure(iface), pointer, intent(inout) :: q
    nullify(q)
  end subroutine
  subroutine proc_out(q)
    procedure(iface), pointer, intent(out) :: q
    nullify(q)
  end subroutine
  subroutine proc_nonpointer(q)
    procedure(iface) :: q
  end subroutine
  subroutine proc_target()
  end subroutine
end module
