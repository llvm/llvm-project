! RUN: %python %S/test_errors.py %s %flang_fc1

module m1
  type :: t
  end type
 contains
  pure subroutine s1(x)
    class(t), intent(in out) :: x
    call s2(x)
    call s3(x)
  end subroutine
  pure subroutine s2(x)
    class(t), intent(in out) :: x
    !ERROR: Left-hand side of assignment may not be polymorphic unless assignment is to an entire allocatable
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'x' is polymorphic in a pure subprogram
    x = t()
  end subroutine
  pure subroutine s3(x)
    !ERROR: An INTENT(OUT) dummy argument of a pure subroutine may not be polymorphic
    class(t), intent(out) :: x
  end subroutine
end module
