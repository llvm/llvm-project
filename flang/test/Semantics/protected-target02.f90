! RUN: %python %S/test_errors.py %s %flang_fc1
! F2028 C866, C870-C871 and 8.5.16p2.

subroutine definitions
  integer, target :: x(2)
  integer, pointer, protected_target :: p(:)
  integer, pointer :: w(:), scalar
  character(4), pointer, protected_target :: text
  character(:), pointer :: mutable_text
  complex, pointer, protected_target :: z
  real, pointer :: mutable_real
  type mutable
    integer, pointer :: component(:)
  end type
  type readonly
    integer, pointer, protected_target :: component(:)
  end type
  type(mutable) :: m
  type(readonly) :: r
  p => x
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
  p = 1
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
  p(1) = 1
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
  if (.false.) p(1:2) = 1
  !ERROR: Input variable 'p' is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
  read *, p
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'text' is not definable
  text(2:3) = 'ab'
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'z' is not definable
  z%re = 1.
  !ERROR: pointer 'mutable_text' must have the PROTECTED_TARGET attribute when associated with the target of PROTECTED_TARGET pointer 'text'
  mutable_text => text(2:3)
  !ERROR: pointer 'mutable_real' must have the PROTECTED_TARGET attribute when associated with the target of PROTECTED_TARGET pointer 'z'
  mutable_real => z%re
  !ERROR: pointer 'w' must have the PROTECTED_TARGET attribute when associated with the target of PROTECTED_TARGET pointer 'p'
  w => p
  !ERROR: pointer 'w' must have the PROTECTED_TARGET attribute when associated with the target of PROTECTED_TARGET pointer 'p'
  w => p(1:2)
  !ERROR: pointer 'scalar' must have the PROTECTED_TARGET attribute when associated with the target of PROTECTED_TARGET pointer 'p'
  scalar => p(1)
  !ERROR: pointer 'component' must have the PROTECTED_TARGET attribute when associated with the target of PROTECTED_TARGET pointer 'p'
  m = mutable(p)
  !ERROR: pointer 'component' must have the PROTECTED_TARGET attribute when associated with the target of PROTECTED_TARGET pointer 'p'
  m = mutable(p(1:2))
  r = readonly(p)
  r = readonly(p(1:2))
  r%component => p
  p => r%component
  x = 2
  nullify(p, r%component)
end subroutine

subroutine intent_in(p)
  integer, pointer, intent(in), protected_target :: p
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
  p = 1
  !ERROR: 'p' may not appear in NULLIFY
  !BECAUSE: 'p' is an INTENT(IN) dummy argument
  nullify(p)
end subroutine
