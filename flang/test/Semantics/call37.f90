! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror
! Test warnings on mismatching interfaces involvingCHARACTER arguments
subroutine constLen(s)
  character(len = 1) s
end
subroutine assumedLen(s)
  character(len = *) s
end
subroutine exprLen(s)
  common n
  character(len = n) s
end

module m0
  interface ! these are all OK
    subroutine constLen(s)
      character(len=1) s
    end
    subroutine assumedLen(s)
      character(len=*) s
    end
    subroutine exprLen(s)
      common n
      character(len=n) s
    end
  end interface
end

module m1
  interface
    !WARNING: The global subprogram 'constlen' is not compatible with its local procedure declaration (incompatible dummy argument #1: incompatible dummy data object types: CHARACTER(KIND=1,LEN=1_8) vs CHARACTER(KIND=1,LEN=2_8))
    subroutine constLen(s)
      character(len=2) s
    end
    !WARNING: The global subprogram 'assumedlen' is not compatible with its local procedure declaration (incompatible dummy argument #1: assumed-length character vs explicit-length character)
    subroutine assumedLen(s)
      character(len=2) s
    end
    !WARNING: The global subprogram 'exprlen' is not compatible with its local procedure declaration (incompatible dummy argument #1: constant-length vs non-constant-length character dummy arguments)
    subroutine exprLen(s)
      character(len=2) s
    end
  end interface
end

module m2
  interface
    !WARNING: The global subprogram 'constlen' is not compatible with its local procedure declaration (incompatible dummy argument #1: assumed-length character vs explicit-length character)
    subroutine constLen(s)
      character(len=*) s
    end
    !WARNING: The global subprogram 'exprlen' is not compatible with its local procedure declaration (incompatible dummy argument #1: assumed-length character vs explicit-length character)
    subroutine exprLen(s)
      character(len=*) s
    end
  end interface
end

module m3
  interface
    !WARNING: The global subprogram 'constlen' is not compatible with its local procedure declaration (incompatible dummy argument #1: constant-length vs non-constant-length character dummy arguments)
    subroutine constLen(s)
      common n
      character(len=n) s
    end
    !WARNING: The global subprogram 'assumedlen' is not compatible with its local procedure declaration (incompatible dummy argument #1: assumed-length character vs explicit-length character)
    subroutine assumedLen(s)
      common n
      character(len=n) s
    end
  end interface
end
