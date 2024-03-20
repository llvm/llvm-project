! RUN: %python %S/test_errors.py %s %flang_fc1
! 15.4.2.2. Test that errors are reported when an explicit interface
! is not provided for an external procedure that requires an explicit
! interface (the definition needs to be visible so that the compiler
! can detect the violation).

subroutine foo(a_pointer)
  real, pointer :: a_pointer(:)
end subroutine

subroutine bar(a_pointer)
  procedure(real), pointer :: a_pointer
end subroutine

subroutine baz(proc)
  external :: proc
  real, optional :: proc
end subroutine

subroutine test()
  real, pointer :: a_pointer(:)
  real, pointer :: an_array(:)
  intrinsic :: sin

  ! This call would be allowed if the interface was explicit here,
  ! but its handling with an implicit interface is different (no
  ! descriptor involved, copy-in/copy-out...)

  !ERROR: References to the procedure 'foo' require an explicit interface
  !BECAUSE: a dummy argument has the allocatable, asynchronous, optional, pointer, target, value, or volatile attribute
  call foo(a_pointer)

  ! This call would be error if the interface was explicit here.

  !ERROR: References to the procedure 'foo' require an explicit interface
  !BECAUSE: a dummy argument has the allocatable, asynchronous, optional, pointer, target, value, or volatile attribute
  call foo(an_array)

  !ERROR: References to the procedure 'bar' require an explicit interface
  !BECAUSE: a dummy procedure is optional or a pointer
  !WARNING: If the procedure's interface were explicit, this reference would be in error
  !BECAUSE: Actual argument associated with procedure pointer dummy argument 'a_pointer=' must be a pointer unless INTENT(IN)
  call bar(sin)

  !ERROR: References to the procedure 'baz' require an explicit interface
  !BECAUSE: a dummy procedure is optional or a pointer
  call baz(sin)
end subroutine
