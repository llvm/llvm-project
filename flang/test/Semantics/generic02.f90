! RUN: %python %S/test_errors.py %s %flang_fc1
program test
  interface generic
    subroutine explicit(n)
      integer, intent(in) :: n
    end subroutine
    procedure implicit
  end interface
!ERROR: Specific procedure 'implicit' of generic interface 'generic' must have an explicit interface
  external implicit
  call generic(1)
end
