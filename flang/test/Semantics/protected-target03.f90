! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic

module protected_target_uses
  integer, target, protected :: protected_value
  integer, pointer, protected, protected_target :: protected_pointer
  type :: readonly_holder
    integer, pointer, protected_target :: p
  end type
  type :: ordinary_holder
    integer, pointer :: p
  end type
  type :: result_provider
  contains
    procedure, nopass :: get => pointer_result
    generic :: generic_get => get
  end type
  interface generic_result
    module procedure pointer_result
  end interface
  abstract interface
    subroutine ordinary_interface(p)
      integer, pointer, intent(in) :: p
    end subroutine
    subroutine protected_interface(p)
      integer, pointer, protected_target, intent(in) :: p
    end subroutine
  end interface
contains
  subroutine associate_in_module()
    protected_pointer => protected_value
  end subroutine
  subroutine positive(target, ordinary_pointer)
    integer, target, intent(in) :: target
    integer, pointer :: ordinary_pointer
    integer, pointer, protected_target :: p, q
    integer, pointer :: array(:)
    integer, pointer, protected_target :: protected_array(:)
    type(readonly_holder) :: readonly_value
    type(ordinary_holder) :: ordinary_value
    integer :: value
    logical :: is_associated
    protected_target :: ordinary_statement_pointer
    integer, pointer :: ordinary_statement_pointer
    p => target
    readonly_value = readonly_holder(target)
    call readonly_dummy(target)
    !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
    !BECAUSE: 'target' is an INTENT(IN) dummy argument
    ordinary_value = ordinary_holder(target)
    !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
    !BECAUSE: 'target' is an INTENT(IN) dummy argument
    call ordinary_dummy(target)
    p => ordinary_pointer
    q => p
    allocate(array(2), source=[1, 2])
    protected_array => array
    protected_array(1:) => array
    protected_array(1:2) => array
    nullify(p, q, protected_array, ordinary_statement_pointer)
    allocate(p, source=1)
    q => p
    allocate(protected_array(2), source=[1, 2])
    value = p + q
    is_associated = associated(p)
    value = size(protected_array) + sum(protected_array)
    p => null(mold=q)
    print *, value, is_associated
    deallocate(array)
  end subroutine

  subroutine readonly_dummy(p)
    integer, pointer, protected_target, intent(in) :: p
  end subroutine

  subroutine ordinary_dummy(p)
    integer, pointer, intent(in) :: p
  end subroutine

  subroutine characteristics(ordinary)
    procedure(ordinary_interface) :: ordinary
    procedure(protected_interface), pointer :: p
    ! PROTECTED_TARGET is not a dummy characteristic in F2028 15.3.2.2.
    p => ordinary
  end subroutine

  function pointer_result() result(p)
    integer, pointer, protected_target :: p
    nullify(p)
  end function

  function implicit_pointer_result()
    integer, pointer :: implicit_pointer_result
    protected_target :: implicit_pointer_result
    nullify(implicit_pointer_result)
  end function

  subroutine host_association(p)
    integer, pointer, protected_target :: p
  contains
    subroutine read_target()
      print *, p
    end subroutine
  end subroutine

  subroutine function_results()
    integer, pointer :: writable
    integer, pointer, protected_target :: readonly
    type(result_provider) :: provider
    !ERROR: Pointer 'writable' must have PROTECTED_TARGET to associate with target of PROTECTED_TARGET pointer 'p'
    writable => pointer_result()
    !ERROR: Pointer 'writable' must have PROTECTED_TARGET to associate with target of PROTECTED_TARGET pointer 'p'
    writable => generic_result()
    !ERROR: Pointer 'writable' must have PROTECTED_TARGET to associate with target of PROTECTED_TARGET pointer 'p'
    writable => provider%get()
    !ERROR: Pointer 'writable' must have PROTECTED_TARGET to associate with target of PROTECTED_TARGET pointer 'p'
    writable => provider%generic_get()
    readonly => pointer_result()
    readonly => provider%get()
    associate (a => pointer_result())
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable
      a = 1
    end associate
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable
    pointer_result() = 1
  end subroutine
end module

subroutine preserve_protected()
  use protected_target_uses
  integer, pointer, protected_target :: p
  ! Flang warns for a use-associated PROTECTED nonpointer target (F2028 C864).
  !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
  !BECAUSE: 'protected_value' is protected in this scope
  p => protected_value
  !ERROR: The left-hand side of a pointer assignment is not definable
  !BECAUSE: 'protected_pointer' is protected in this scope
  protected_pointer => p
end subroutine
