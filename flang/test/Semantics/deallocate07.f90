! RUN: %python %S/test_errors.py %s %flang_fc1

module m
  type t1
  end type
  type t2
    class(t2), allocatable :: pc
  end type
 contains
  pure subroutine subr(pp1, pp2, mp2)
    class(t1), intent(in out), pointer :: pp1
    class(t2), intent(in out) :: pp2
    type(t2), pointer :: mp2
    !ERROR: 'pp1' may not be deallocated in a pure procedure because it is polymorphic
    deallocate(pp1)
    !ERROR: 'pc' may not be deallocated in a pure procedure because it is polymorphic
    deallocate(pp2%pc)
    !ERROR: 'mp2' may not be deallocated in a pure procedure because its type has a polymorphic allocatable ultimate component 'pc'
    deallocate(mp2)
  end subroutine
end module
