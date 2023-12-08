! RUN: %python %S/test_errors.py %s %flang_fc1
program test
  !ERROR: Generic interface 'generic' must not use abstract interface 'abstract' as a specific procedure
  interface generic
    subroutine explicit(n)
      integer, intent(in) :: n
    end subroutine
    procedure implicit
    procedure abstract
  end interface
  abstract interface
    subroutine abstract
    end subroutine
  end interface
!ERROR: Specific procedure 'implicit' of generic interface 'generic' must have an explicit interface
  external implicit
  call generic(1)
end
