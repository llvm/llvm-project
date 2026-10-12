! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic

module protected_target_declarations
  ! F2028 8.5.16p1: the attribute imposes limitations on the usage of
  ! a data pointer.
  !ERROR: The PROTECTED_TARGET attribute may apply only to a data pointer
  integer, protected_target :: not_pointer
  !ERROR: The PROTECTED_TARGET attribute may apply only to a data pointer
  integer, pointer, external, protected_target :: procedure_pointer

  type :: t
    ! F2028 C781: POINTER must be in the same component attribute list.
    !ERROR: The PROTECTED_TARGET attribute may apply only to a data pointer
    integer, protected_target :: not_pointer_component
    integer, protected_target, pointer :: p
  end type
contains
  subroutine statement()
    ! F2028 C8109
    !ERROR: The PROTECTED_TARGET attribute may apply only to a data pointer
    integer :: not_pointer
    protected_target :: not_pointer
  end subroutine

  subroutine procedure_statement()
    !ERROR: The PROTECTED_TARGET attribute may apply only to a data pointer
    procedure(), pointer :: p
    protected_target p
  end subroutine

  ! F2028 R863: a subroutine name is not a variable-name.
  !ERROR: The PROTECTED_TARGET attribute may apply only to a data pointer
  subroutine subprogram_name()
    protected_target :: subprogram_name
  end subroutine

  !ERROR: The PROTECTED_TARGET attribute may apply only to a data pointer
  function separate_result() result(r)
    integer :: r
    protected_target :: separate_result
    r = 1
  end function

  subroutine common_pointer()
    ! F2028 C867
    !ERROR: A PROTECTED_TARGET pointer may not be in a common block
    integer, pointer, protected_target :: p
    common /block/ p
  end subroutine

  subroutine common_statement_pointer()
    ! F2028 C867 also applies to the statement form.
    !ERROR: A PROTECTED_TARGET pointer may not be in a common block
    integer, pointer :: p
    protected_target :: p
    common /statement_block/ p
  end subroutine

  subroutine common_nonpointer()
    !ERROR: The PROTECTED_TARGET attribute may apply only to a data pointer
    integer, protected_target :: p
    common /nonpointer_block/ p
  end subroutine

  subroutine repeated()
    !WARNING: Attribute 'PROTECTED_TARGET' cannot be used more than once [-Wredundant-attribute]
    integer, pointer, protected_target, protected_target :: p
  end subroutine
end module

module protected_target_declaration_association
  ! Diagnose the declaring entity, not each associated reference.
  !ERROR: The PROTECTED_TARGET attribute may apply only to a data pointer
  integer, protected_target :: invalid
  integer, pointer, protected_target :: valid
contains
  subroutine host_reference()
    print *, invalid, valid
  end subroutine
end module

subroutine renamed_reference()
  use protected_target_declaration_association, only: local => invalid, valid
  print *, local, valid
contains
  subroutine host_reference()
    print *, local, valid
  end subroutine
end subroutine
