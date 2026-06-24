! RUN: not %flang_fc1 -pedantic %s 2>&1 | FileCheck %s
! Test extension: allow forward references to dummy arguments or COMMON
! from specification expressions in scopes with IMPLICIT NONE(TYPE),
! as long as those symbols are eventually typed later.

!CHECK: warning: 'n1' was used without (or before) being explicitly typed
!CHECK: error: No explicit type declared for dummy argument 'n1'
subroutine foo1(a, n1)
  implicit none
  real a(n1)
end

!CHECK: warning: 'n2' was used without (or before) being explicitly typed
subroutine foo2(a, n2)
  implicit none
  real a(n2)
!CHECK: error: The type of 'n2' has already been implicitly declared
  double precision n2
end

!CHECK: warning: 'n3a' was used under IMPLICIT NONE(TYPE) before being explicitly typed
!CHECK: warning: 'n3b' was used under IMPLICIT NONE(TYPE) before being explicitly typed
!CHECK-NOT: error: Dummy argument 'n3a'
!CHECK-NOT: error: Dummy argument 'n3b'
subroutine foo3(a, n3a, n3b)
  implicit none
  integer a(n3a, n3b)
  integer n3a
  integer(8) n3b
end

!CHECK: warning: 'n4' was used without (or before) being explicitly typed
!CHECK: error: No explicit type declared for 'n4'
subroutine foo4(a)
  implicit none
  real a(n4)
  common /b4/ n4
end

!CHECK: warning: 'n5' was used without (or before) being explicitly typed
subroutine foo5(a)
  implicit none
  real a(n5)
  common /b5/ n5
!CHECK: error: The type of 'n5' has already been implicitly declared
  double precision n5
end

!CHECK: warning: 'n6' was used without (or before) being explicitly typed
subroutine foo6(a)
  implicit none
  real a(n6)
  common /b6/ n6
  integer n6
end
