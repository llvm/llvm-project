! RUN: not %flang_fc1 -pedantic %s 2>&1 | FileCheck %s
! Test extension: allow forward references to dummy arguments
! from specification expressions in scopes with IMPLICIT NONE(TYPE),
! as long as those symbols are eventually typed later with the
! same integer type they would have had without IMPLICIT NONE.

!CHECK: Dummy argument 'n1' was used without being explicitly typed
!CHECK: error: No explicit type declared for dummy argument 'n1'
subroutine foo1(a, n1)
  implicit none
  real a(n1)
end

!CHECK: Dummy argument 'n2' was used without being explicitly typed
subroutine foo2(a, n2)
  implicit none
  real a(n2)
!CHECK: error: The type of 'n2' has already been implicitly declared
  double precision n2
end

!CHECK: Dummy argument 'n3' was used without being explicitly typed
!CHECK-NOT: error: Dummy argument 'n3'
subroutine foo3(a, n3)
  implicit none
  real a(n3)
  integer n3
end
