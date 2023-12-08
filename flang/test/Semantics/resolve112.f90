! RUN: %python %S/test_errors.py %s %flang_fc1

! If there are 2 or more use-associated symbols
! from different modules with the same name,
! the error should be generated only if
! the name is actually used.
module a
  contains
    function foo()
      foo = 42
    end function foo
end module a

module b
  contains
    function foo()
      foo = 42
    end function foo
end module b

subroutine without_error
  use a
  use b
end subroutine without_error

subroutine with_error
  use a
  use b
  integer :: res
  ! ERROR: Reference to 'foo' is ambiguous
  res = foo()
end subroutine with_error
