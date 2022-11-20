! RUN: %python %S/test_errors.py %s %flang_fc1
! C911 - abstract derived type can be used only when polymorphic
program test
  type, abstract :: abstract
    integer :: j
  end type
  type, extends(abstract) :: concrete
    integer :: k
    class(concrete), allocatable :: a(:)
  end type
  type(concrete) :: x(2)
  call sub1(x(1)) ! ok
  call sub2(x) ! ok
  call sub1(x(1)%a(1)) ! ok
  call sub2(x(1)%a) ! ok
  !ERROR: Reference to object with abstract derived type 'abstract' must be polymorphic
  call sub1(x(1)%abstract) ! bad
  !ERROR: Reference to object with abstract derived type 'abstract' must be polymorphic
  call sub2(x%abstract) ! bad
  !ERROR: Reference to object with abstract derived type 'abstract' must be polymorphic
  call sub1(x(1)%a(1)%abstract) ! bad
  !ERROR: Reference to object with abstract derived type 'abstract' must be polymorphic
  call sub2(x(1)%a%abstract) ! bad
 contains
  subroutine sub1(d)
    class(abstract) d
  end subroutine
  subroutine sub2(d)
    class(abstract) d(:)
  end subroutine
end
