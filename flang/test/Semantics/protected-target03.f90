! RUN: %python %S/test_errors.py %s %flang_fc1
! F'2028 C870 and C871: do not lose protection in pointer assignment or
! structure construction, including components, sections and association.

module protected_target_propagation
  type leaf
    integer :: value
    integer, pointer :: writable
  end type
  type wrapper
    type(leaf), pointer, protected_target :: p
  end type
  type writable_holder
    integer, pointer :: q
  end type
  type readonly_holder
    integer, pointer, protected_target :: q
  end type
contains
  subroutine test(p, a, obj, container, c, z)
    integer, pointer, protected_target :: p, a(:)
    type(leaf), pointer, protected_target :: obj
    type(wrapper) :: container
    character(8), pointer, protected_target :: c
    complex, pointer, protected_target :: z
    integer, pointer :: q, v(:)
    integer, pointer, protected_target :: r, b(:)
    real, pointer :: re
    character(3), pointer :: s
    type(writable_holder) :: w
    type(readonly_holder) :: ro
    procedure(result_pointer), pointer :: pf
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with pointer 'q' without PROTECTED_TARGET
    q => p
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with pointer 'v' without PROTECTED_TARGET
    v => a(1:3)
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with pointer 'q' without PROTECTED_TARGET
    q => a(1)
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with pointer 'q' without PROTECTED_TARGET
    q => obj%value
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with pointer 'q' without PROTECTED_TARGET
    q => container%p%value
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with pointer 's' without PROTECTED_TARGET
    s => c(1:3)
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with pointer 're' without PROTECTED_TARGET
    re => z%im
    r => p
    r => obj%value
    r => q
    b => a
    b(1:3) => a(2:4)
    q => obj%writable ! This target belongs to the ordinary pointer component.
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with pointer 'q' without PROTECTED_TARGET
    w = writable_holder(p)
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with pointer 'q' without PROTECTED_TARGET
    w = writable_holder(a(1))
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with pointer 'q' without PROTECTED_TARGET
    w = writable_holder(container%p%value)
    ro = readonly_holder(p)
    ro = readonly_holder(q)
    ro = readonly_holder(container%p%value)
    associate (alias => container%p%value)
      q => alias
      r => alias
      w = writable_holder(alias)
    end associate
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with pointer 'q' without PROTECTED_TARGET
    q => result_pointer()
    r => result_pointer()
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with pointer 'q' without PROTECTED_TARGET
    w = writable_holder(result_pointer())
    ro = readonly_holder(result_pointer())
    pf => result_pointer
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'result_pointer' is not definable
    result_pointer() = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'pf' is not definable
    pf() = 1
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with pointer 'q' without PROTECTED_TARGET
    q => pf()
    r => pf()
    associate (alias => result_pointer())
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'result_pointer' is not definable
      alias = 1
    end associate
  end subroutine

  function result_pointer() result(res)
    integer, pointer, protected_target :: res
    integer, target, save :: x = 1
    res => x
  end function
end module
