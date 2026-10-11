! RUN: %python %S/test_errors.py %s %flang_fc1
! J3/26-007r1 C870-C871 preserve protection in assignments and constructors.

module associations
  type t
    integer :: value
    integer, pointer :: separate
  end type
  type writable
    integer, pointer :: component
  end type
  type readable
    integer, pointer, protected_target :: component
  end type
  integer, target :: storage
  integer, parameter :: storage_constant = 1
contains
  subroutine assignment(p, a, c, object)
    integer, pointer, protected_target :: p, a(:)
    character(:), pointer, protected_target :: c
    type(t), pointer, protected_target :: object
    integer, pointer :: ordinary, ordinary_array(:)
    integer, pointer, protected_target :: copy, copy_array(:)
    character(:), pointer :: ordinary_character
    type(writable) :: w
    type(readable) :: r
    copy => p
    copy_array => a(1:2)
    p => ordinary
    !ERROR: Target of a PROTECTED_TARGET pointer may not be associated with pointer 'ordinary' without PROTECTED_TARGET
    ordinary => p
    !ERROR: Target of a PROTECTED_TARGET pointer may not be associated with pointer 'ordinary_array' without PROTECTED_TARGET
    ordinary_array => a(1:2)
    !ERROR: Target of a PROTECTED_TARGET pointer may not be associated with pointer 'ordinary' without PROTECTED_TARGET
    ordinary => a(1)
    !ERROR: Target of a PROTECTED_TARGET pointer may not be associated with pointer 'ordinary_character' without PROTECTED_TARGET
    ordinary_character => c(1:2)
    !ERROR: Target of a PROTECTED_TARGET pointer may not be associated with pointer 'ordinary' without PROTECTED_TARGET
    ordinary => object%value
    ordinary => object%separate
    !ERROR: Target of a PROTECTED_TARGET pointer may not be associated with pointer 'component' without PROTECTED_TARGET
    w = writable(p)
    !ERROR: Target of a PROTECTED_TARGET pointer may not be associated with pointer 'component' without PROTECTED_TARGET
    w = writable(a(1))
    r = readable(p)
    r = readable(a(1))
    w = writable(object%separate)
    !ERROR: Target associated with pointer 'component' must be a designator or a call to a pointer-valued function
    w = writable((p))
    associate(alias => a(1:2))
      !ERROR: Target of a PROTECTED_TARGET pointer may not be associated with pointer 'ordinary_array' without PROTECTED_TARGET
      ordinary_array => alias
      copy_array => alias
    end associate
    !ERROR: Target associated with pointer 'p' must be a designator or a call to a pointer-valued function
    p => storage_constant
  end subroutine

  function read_only_result() result(p)
    integer, pointer, protected_target :: p
    p => storage
  end function

  function writable_result() result(p)
    integer, pointer :: p
    p => storage
  end function

  subroutine function_results
    integer, pointer :: ordinary
    integer, pointer, protected_target :: p
    type(writable) :: w
    type(readable) :: r
    p => read_only_result()
    !ERROR: Target of a PROTECTED_TARGET pointer may not be associated with pointer 'ordinary' without PROTECTED_TARGET
    ordinary => read_only_result()
    !ERROR: Target of a PROTECTED_TARGET pointer may not be associated with pointer 'component' without PROTECTED_TARGET
    w = writable(read_only_result())
    r = readable(read_only_result())
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    read_only_result() = 1
    !ERROR: Input variable 'read_only_result()' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    read *, read_only_result()
    writable_result() = 1
    read *, writable_result()
    associate(a => read_only_result())
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
      a = 1
      !ERROR: Target of a PROTECTED_TARGET pointer may not be associated with pointer 'ordinary' without PROTECTED_TARGET
      ordinary => a
    end associate
  end subroutine
end module
