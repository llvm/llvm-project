! RUN: %python %S/test_errors.py %s %flang_fc1
! Test rename to the same name.
module m1
  integer, allocatable :: a(:)

  interface operator(.add.)
    module procedure add
  end interface

contains
  integer function add(a, b)
    integer, intent(in) :: a, b

    add = a + b
  end function
end

program p1
  use m1, a => a, operator(.add.) => operator(.add.)

  allocate(a(10))
  deallocate(a)
  print *, 2 .add. 2
end
