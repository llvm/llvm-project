! RUN: %python %S/test_errors.py %s %flang_fc1

module m
  type t1
  end type
  type t2
    class(t2), allocatable :: pc
  end type
  class(t1), pointer :: mp1
  type(t2) :: mv1
 contains
  pure subroutine subr(pp1, pp2, mp2)
    class(t1), intent(in out), pointer :: pp1
    class(t2), intent(in out) :: pp2
    type(t2), pointer :: mp2
    !ERROR: Name in DEALLOCATE statement is not definable
    !BECAUSE: 'mp1' may not be defined in pure subprogram 'subr' because it is host-associated
    deallocate(mp1)
    !ERROR: Name in DEALLOCATE statement is not definable
    !BECAUSE: 'mv1' may not be defined in pure subprogram 'subr' because it is host-associated
    deallocate(mv1%pc)
    !ERROR: Object in DEALLOCATE statement is not deallocatable
    !BECAUSE: 'pp1' is polymorphic in a pure subprogram
    deallocate(pp1)
    !ERROR: Object in DEALLOCATE statement is not deallocatable
    !BECAUSE: 'pc' is polymorphic in a pure subprogram
    deallocate(pp2%pc)
    !ERROR: Object in DEALLOCATE statement is not deallocatable
    !BECAUSE: 'mp2' has polymorphic component '%pc' in a pure subprogram
    deallocate(mp2)
  end subroutine
end module
