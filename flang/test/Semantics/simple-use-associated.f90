! RUN: %flang_fc1 -fsyntax-only %s

module m
contains
  simple subroutine s()
  end subroutine
end module

program p
  use m
  implicit none

  call needs_simple(s)

contains

  subroutine needs_simple(p)
    abstract interface
      simple subroutine simple_proc()
      end subroutine
    end interface

    procedure(simple_proc) :: p
  end subroutine

end program
