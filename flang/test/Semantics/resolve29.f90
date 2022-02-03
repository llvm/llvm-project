! RUN: %python %S/test_errors.py %s %flang_fc1
module m1
  type t1
  end type
  type t3
  end type
  interface
    subroutine s1(x)
      !ERROR: 't1' from host is not accessible
      import :: t1
      type(t1) :: x
      integer :: t1
    end subroutine
    subroutine s2()
      !ERROR: 't2' not found in host scope
      import :: t2
    end subroutine
    subroutine s3(x, y)
      !ERROR: Derived type 't1' not found
      type(t1) :: x, y
    end subroutine
    subroutine s4(x, y)
      !ERROR: 't3' from host is not accessible
      import, all
      type(t1) :: x
      type(t3) :: y
      integer :: t3
    end subroutine
  end interface
contains
  subroutine s5()
  end
  subroutine s6()
    import, only: s5
    implicit none(external)
    call s5()
  end
  subroutine s7()
    import, only: t1
    implicit none(external)
    !ERROR: 's5' is an external procedure without the EXTERNAL attribute in a scope with IMPLICIT NONE(EXTERNAL)
    call s5()
  end
end module
module m2
  integer, parameter :: ck = kind('a')
end module
program main
  use m2
  interface
    subroutine s0(x)
      import :: ck
      character(kind=ck) :: x ! no error
    end subroutine
  end interface
end program
