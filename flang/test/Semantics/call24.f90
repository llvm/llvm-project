! RUN: %python %S/test_errors.py %s %flang_fc1
! 15.4.2.2. Test that errors are reported when an explicit interface
! is not provided for an external procedure that requires an explicit
! interface (the definition needs to be visible so that the compiler
! can detect the violation).

subroutine foo(a_pointer)
  real, pointer :: a_pointer(:)
end subroutine

subroutine test()
  real, pointer :: a_pointer(:)
  real, pointer :: an_array(:)

  ! This call would be allowed if the interface was explicit here,
  ! but its handling with an implicit interface is different (no
  ! descriptor involved, copy-in/copy-out...)

  !ERROR: References to the procedure 'foo' require an explicit interface
  call foo(a_pointer)

  ! This call would be error if the interface was explicit here.

  !ERROR: References to the procedure 'foo' require an explicit interface
  call foo(an_array)
end subroutine
