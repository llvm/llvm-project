! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

module my_module
  interface foo
    subroutine foo_int(a)
    integer :: a
    end subroutine
    subroutine foo_real(a)
    real :: a
    end subroutine
  end interface
contains
  subroutine bar(N)
    integer :: N
    entry entry1(N)
  end subroutine
  subroutine foobar(N)
    integer::N
    !ERROR: The procedure 'entry1' in DECLARE TARGET construct cannot be an entry name.
    !$omp declare target(bar, entry1)
    call bar(N)
  end subroutine
end module

module other_mod
  abstract interface
    integer function foo(a)
      integer, intent(in) :: a
    end function
  end interface
  procedure(foo), pointer :: procptr
  !ERROR: The procedure 'procptr' in DECLARE TARGET construct cannot be a procedure pointer.
  !$omp declare target(procptr)
end module

subroutine baz(x)
    real, intent(inout) :: x
    real :: res 
    stmtfunc(x) = 4.0 * (x**3)
    !ERROR: The procedure 'stmtfunc' in DECLARE TARGET construct cannot be a statement function.
    !$omp declare target (stmtfunc)
    res = stmtfunc(x)
end subroutine

program main
  use my_module
  !ERROR: The procedure 'foo' in DECLARE TARGET construct cannot be a generic name.
  !$omp declare target(foo)
end
