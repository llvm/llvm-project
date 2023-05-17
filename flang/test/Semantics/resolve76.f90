! RUN: %python %S/test_errors.py %s %flang_fc1

! 15.6.2.5(3)

module m1
  implicit logical(a-b)
  interface
    module subroutine sub1(a, b)
      real, intent(in) :: a
      real, intent(out) :: b
    end
    logical module function f()
    end
  end interface
end
submodule(m1) sm1
contains
  module procedure sub1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'a' is an INTENT(IN) dummy argument
    a = 1.0
    b = 2.0
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types REAL(4) and LOGICAL(4)
    b = .false.
  end
  module procedure f
    f = .true.
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types LOGICAL(4) and REAL(4)
    f = 1.0
  end
end
