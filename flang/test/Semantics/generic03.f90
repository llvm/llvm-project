! RUN: %python %S/test_errors.py %s %flang_fc1
! Exercise function vs subroutine distinction in generics
module m1
  type t1
    integer n
  end type
  interface g1
    integer function f1(x, j)
      import t1
      class(t1), intent(in out) :: x
      integer, intent(in) :: j
    end
  end interface
end module

program test
  use m1
  !WARNING: Generic interface 'g1' has both a function and a subroutine
  interface g1
    subroutine s1(x, a)
      import t1
      class(t1), intent(in out) :: x
      real, intent(in) :: a
    end subroutine
  end interface
  type(t1) :: x
  print *, g1(x,1) ! ok
  !ERROR: No specific function of generic 'g1' matches the actual arguments
  print *, g1(x,1.)
  !ERROR: No specific subroutine of generic 'g1' matches the actual arguments
  call g1(x,1)
  call g1(x, 1.) ! ok
 contains
end
