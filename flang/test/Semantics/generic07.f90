! RUN: %python %S/test_errors.py %s %flang_fc1
module m
  type :: t1
    sequence
    real :: x
  end type
  type :: t2
    sequence
    real :: x
  end type
  type :: t3
    real :: x
  end type
  type :: t4
    real, private :: x
  end type
 contains
  subroutine s1a(x)
    type(t1), intent(in) :: x
  end
  subroutine s2a(x)
    type(t2), intent(in) :: x
  end
  subroutine s3a(x)
    type(t3), intent(in) :: x
  end
  subroutine s4a(x)
    type(t4), intent(in) :: x
  end
end

program test
  use m, only: s1a, s2a, s3a, s4a
  type :: t1
    sequence
    integer :: x ! distinct type
  end type
  type :: t2
    sequence
    real :: x
  end type
  type :: t3 ! no SEQUENCE
    real :: x
  end type
  type :: t4
    real :: x ! not PRIVATE
  end type
  interface distinguishable1
    procedure :: s1a, s1b
  end interface
  interface distinguishable2
    procedure :: s1a, s1b
  end interface
  interface distinguishable3
    procedure :: s1a, s1b
  end interface
  !ERROR: Generic 'indistinguishable' may not have specific procedures 's2a' and 's2b' as their interfaces are not distinguishable
  interface indistinguishable
    procedure :: s2a, s2b
  end interface
 contains
  subroutine s1b(x)
    type(t1), intent(in) :: x
  end
  subroutine s2b(x)
    type(t2), intent(in) :: x
  end
  subroutine s3b(x)
    type(t3), intent(in) :: x
  end
  subroutine s4b(x)
    type(t4), intent(in) :: x
  end
end
