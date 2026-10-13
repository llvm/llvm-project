! RUN: %python %S/test_errors.py %s %flang_fc1 -Wpointer-to-undefinable
! PROTECTED_TARGET attribute (F2028 draft J3/26-007r1): C864 forbids a
! USE-associated nonpointer PROTECTED object as a data-target, while 8.5.16
! NOTE 1 associates a PROTECTED_TARGET pointer with exactly such an object.
! Flang follows NOTE 1: the target cannot be defined via a PROTECTED_TARGET
! pointer, so that pointer assignment has no diagnostic.  An
! initial-data-target, which NOTE 1 does not show, keeps the existing optional
! warning, as does an ordinary pointer.
module m
  real, protected, target :: x = 1.
  real, pointer, protected_target :: p
 contains
  subroutine inside
    real, pointer :: w
    w => x
    p => x
  end
end

module initialization
  use m
  !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
  !BECAUSE: 'x' is protected in this scope
  real, pointer, protected_target :: ip => x
  type t
    !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
    !BECAUSE: 'x' is protected in this scope
    real, pointer, protected_target :: c => x
  end type
end

subroutine outside
  use m
  real, pointer :: w
  real, pointer, protected_target :: q
  p => x
  q => x
  q => p
  !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
  !BECAUSE: 'x' is protected in this scope
  w => x
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
  p = 999.
end

subroutine intentIn(y)
  real, intent(in), target :: y
  real, pointer :: w
  real, pointer, protected_target :: q
  q => y
  !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
  !BECAUSE: 'y' is an INTENT(IN) dummy argument
  w => y
  print *, associated(q), associated(q, y)
end

subroutine intrinsicPointerArgs(c)
  use iso_c_binding, only: c_ptr, c_f_pointer
  type(c_ptr), intent(in) :: c
  real, pointer, protected_target :: q
  call c_f_pointer(c, q)
  print *, associated(q)
end

! C870-C874 name the pointer and its subobjects, not an associate name, so
! an associate name used as a data-target or actual argument is not
! diagnosed by them; the data-target here gets only the optional warning.
! Defining the associate name is still an error (protected-target01.f90).
subroutine associateName(p)
  real, pointer, protected_target :: p
  real, pointer :: w
  external :: ext
  associate (a => p)
    !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    w => a
    call ext(a)
  end associate
end
