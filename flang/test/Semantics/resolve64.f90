! RUN: %S/test_errors.sh %s %t %f18 -flogical-abbreviations -fxor-operator

! Like m4 in resolve63 but compiled with different options.
! Alternate operators are enabled so treat these as intrinsic.
module m4
contains
  subroutine s1(x, y, z)
    logical :: x
    real :: y, z
    !ERROR: Operands of .AND. must be LOGICAL; have REAL(4) and REAL(4)
    x = y .a. z
    !ERROR: Operands of .OR. must be LOGICAL; have REAL(4) and REAL(4)
    x = y .o. z
    !ERROR: Operand of .NOT. must be LOGICAL; have REAL(4)
    x = .n. y
    !ERROR: Operands of .NEQV. must be LOGICAL; have REAL(4) and REAL(4)
    x = y .xor. z
    !ERROR: Operands of .NEQV. must be LOGICAL; have REAL(4) and REAL(4)
    x = y .x. y
  end
end

! Like m4 in resolve63 but compiled with different options.
! Alternate operators are enabled so treat .A. as .AND.
module m5
  interface operator(.A.)
    logical function f1(x, y)
      integer, intent(in) :: x, y
    end
  end interface
  interface operator(.and.)
    logical function f2(x, y)
      real, intent(in) :: x, y
    end
  end interface
contains
  subroutine s1(x, y, z)
    logical :: x
    complex :: y, z
    !ERROR: No intrinsic or user-defined OPERATOR(.A.) matches operand types COMPLEX(4) and COMPLEX(4)
    x = y .and. z
    !ERROR: No intrinsic or user-defined OPERATOR(.A.) matches operand types COMPLEX(4) and COMPLEX(4)
    x = y .a. z
  end
end
