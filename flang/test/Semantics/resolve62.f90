! RUN: %python %S/test_errors.py %s %flang_fc1
! Resolve generic based on number of arguments
subroutine subr1
  interface f
    real function f1(x)
      optional :: x
    end
    real function f2(x, y)
    end
  end interface
  z = f(1.0)
  z = f(1.0, 2.0)
  !ERROR: No specific function of generic 'f' matches the actual arguments
  z = f(1.0, 2.0, 3.0)
end

! Elemental and non-element function both match: non-elemental one should be used
subroutine subr2
  interface f
    logical elemental function f1(x)
      intent(in) :: x
    end
    real function f2(x)
      real :: x(10)
    end
  end interface
  real :: x, y(10), z
  logical :: a
  a = f(1.0)
  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types LOGICAL(4) and REAL(4)
  a = f(y)
end

! Resolve named operator
subroutine s3
  interface operator(.foo.)
    pure integer(8) function f_real(x, y)
      real, intent(in) :: x, y
    end
    pure integer(8) function f_integer(x, y)
      integer, intent(in) :: x, y
    end
  end interface
  logical :: a, b, c
  x = y .foo. z  ! OK: f_real
  i = j .foo. k  ! OK: f_integer
  !ERROR: No intrinsic or user-defined .FOO. matches operand types LOGICAL(4) and LOGICAL(4)
  a = b .foo. c
end

! Generic resolves successfully but error analyzing call
module m4
  real, protected :: x
  real :: y
  interface s
    pure subroutine s101(x)
      real, intent(out) :: x
    end
    subroutine s102(x, y)
      real :: x, y
    end
  end interface
end
subroutine s4a
  use m4
  real :: z
  !OK
  call s(z)
end
subroutine s4b
  use m4
  !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'x=' is not definable
  !BECAUSE: 'x' is protected in this scope
  call s(x)
end
pure subroutine s4c
  use m4
  !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'x=' is not definable
  !BECAUSE: 'y' may not be defined in pure subprogram 's4c' because it is USE-associated
  call s(y)
end
