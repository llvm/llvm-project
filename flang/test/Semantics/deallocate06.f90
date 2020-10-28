! RUN: %S/test_errors.sh %s %t %f18

! Test deallocate of use- and host-associated variables
module m1
  real, pointer :: a(:)
  real, allocatable :: b(:)
end

subroutine s1()
  use m1
  complex, pointer :: c(:)
  complex, allocatable :: d(:)
  complex :: e(10)
  deallocate(a)
  deallocate(b)
contains
  subroutine s2()
    deallocate(a)
    deallocate(b)
    deallocate(c)
    deallocate(d)
    !ERROR: name in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute
    deallocate(e)
  end subroutine
end
