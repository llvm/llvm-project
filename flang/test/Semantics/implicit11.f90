! RUN: %S/test_errors.sh %s %t %f18

! Test use of implicitly declared variable in specification expression

subroutine s1()
  m = 1
contains
  subroutine s1a()
    implicit none
    !ERROR: No explicit type declared for 'n'
    real :: a(m, n)
  end
  subroutine s1b()
    !ERROR: Implicitly typed local entity 'n' not allowed in specification expression
    real :: a(m, n)
  end
end

subroutine s2()
  type :: t(m, n)
    integer, len :: m
    integer, len :: n
  end type
  n = 1
contains
  subroutine s2a()
    !ERROR: Implicitly typed local entity 'm' not allowed in specification expression
    type(t(m, n)) :: a
  end
  subroutine s2b()
    implicit none
    !ERROR: No explicit type declared for 'm'
    character(m) :: a
  end
end

subroutine s3()
  m = 1
contains
  subroutine s3a()
    implicit none
    real :: a(m, n)
    !ERROR: No explicit type declared for 'n'
    common n
  end
  subroutine s3b()
    ! n is okay here because it is in a common block
    real :: a(m, n)
    common n
  end
end

subroutine s4()
  implicit none
contains
  subroutine s4a()
    !ERROR: No explicit type declared for 'n'
    real :: a(n)
  end
end

