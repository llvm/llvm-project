! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
subroutine foo(A, B, P)
  interface
    real elemental function foo_elemental(x)
      real, intent(in) :: x
    end function
    pure real function foo_pure(x)
      real, intent(in) :: x
    end function
    real function foo_nonelemental(x)
      real, intent(in) :: x
    end function
  end interface
  real :: A(:), B(:)
  !PORTABILITY: A dummy procedure should not have an ELEMENTAL intrinsic as its interface
  procedure(sqrt), pointer :: P
  !ERROR: Rank of dummy argument is 0, but actual argument has rank 1
  A = P(B)
  !ERROR: Procedure pointer 'p' associated with incompatible procedure designator 'foo_elemental': incompatible procedure attributes: Elemental
  P => foo_elemental
  P => foo_pure ! ok
  !ERROR: PURE procedure pointer 'p' may not be associated with non-PURE procedure designator 'foo_nonelemental'
  P => foo_nonelemental
end subroutine
