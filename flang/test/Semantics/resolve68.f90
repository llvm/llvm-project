! RUN: %S/test_errors.sh %s %t %f18
! Test resolution of type-bound generics.

module m1
  type :: t
  contains
    procedure, pass(x) :: add1 => add
    procedure, nopass :: add2 => add
    procedure :: add_real
    generic :: g => add1, add2, add_real
  end type
contains
  integer function add(x, y)
    class(t), intent(in) :: x, y
  end
  integer function add_real(x, y)
    class(t), intent(in) :: x
    real, intent(in) :: y
  end
  subroutine test1(x, y, z)
    type(t) :: x
    integer :: y
    integer :: z
    !ERROR: No specific procedure of generic 'g' matches the actual arguments
    z = x%g(y)
  end
  subroutine test2(x, y, z)
    type(t) :: x
    real :: y
    integer :: z
    !ERROR: No specific procedure of generic 'g' matches the actual arguments
    z = x%g(x, y)
  end
end
