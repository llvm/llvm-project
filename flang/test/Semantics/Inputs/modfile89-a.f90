subroutine reader
  use protected_target_module
  integer, pointer :: ordinary
  integer, pointer, protected_target :: q
  type(t) :: object
  q => p
  q => get()
  object = t(p)
  call take(q)
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
  p = 1
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'component' is not definable via that pointer
  object%component = 1
  !ERROR: Target of a PROTECTED_TARGET pointer may not be associated with pointer 'ordinary' without PROTECTED_TARGET
  ordinary => get()
  !ERROR: Actual argument associated with POINTER dummy argument 'a=' must also be POINTER unless INTENT(IN)
  call take(storage)
end subroutine
