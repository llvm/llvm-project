! RUN: %python %S/test_errors.py %s %flang_fc1
module m1
  type t1
  end type
  type t3
    integer t3c
  end type
  interface
    subroutine s1(x)
      !ERROR: 't1' from host is not accessible
      import :: t1
      type(t1) :: x
      !BECAUSE: 't1' is hidden by this entity
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
      !BECAUSE: 't3' is hidden by this entity
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
  subroutine s8()
    !This case is a dangerous ambiguity allowed by the standard.
    !ERROR: 't1' from host is not accessible
    type(t1), pointer :: p
    !BECAUSE: 't1' is hidden by this entity
    type t1
      integer n(2)
    end type
  end
  subroutine s9()
    !This case is a dangerous ambiguity allowed by the standard.
    type t2
      !ERROR: 't1' from host is not accessible
      type(t1), pointer :: p
    end type
    !BECAUSE: 't1' is hidden by this entity
    type t1
      integer n(2)
    end type
    type(t2) x
  end
  subroutine s10()
    !Forward shadowing derived type in IMPLICIT
    !(supported by all other compilers)
    implicit type(t1) (c) ! forward shadow
    implicit type(t3) (d) ! host associated
    type t1
      integer a
    end type
    c%a = 1
    d%t3c = 2
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
