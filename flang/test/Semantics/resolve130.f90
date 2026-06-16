! RUN: not %flang_fc1 -pedantic %s 2>&1 | FileCheck %s
! Test extension: a named constant defined by a PARAMETER statement may appear
! before its explicit type declaration in a scope with IMPLICIT NONE(TYPE).
! It acquires the type it would have had under implicit typing rules, which a
! later explicit declaration must match.

! Forward reference accepted, matching INTEGER declaration appears later.
!CHECK: warning: 'n' was used without (or before) being explicitly typed
subroutine s1
  implicit none
  parameter(n=4096)
  integer n
end

! The would-be-implicit type need not be INTEGER; a matching REAL declaration
! is accepted.
!CHECK: warning: 'x' was used without (or before) being explicitly typed
subroutine s2
  implicit none
  parameter(x=1.5)
  real x
end

! A later declaration whose type differs from the would-be-implicit type is
! rejected.  'm' begins with a letter in I-N, so under implicit typing rules
! its type would be default INTEGER (reported below as INTEGER(4), the default
! integer kind on this target), which the later REAL declaration contradicts.
!CHECK: warning: 'm' was used without (or before) being explicitly typed
subroutine s3
  implicit none
  parameter(m=4096)
!CHECK: error: The type of 'm' has already been implicitly declared as INTEGER(4)
  real m
end

! If no explicit type declaration ever appears, IMPLICIT NONE(TYPE) is still
! enforced.
!CHECK: warning: 'k' was used without (or before) being explicitly typed
!CHECK: error: No explicit type declared for 'k'
subroutine s4
  implicit none
  parameter(k=4096)
end

! Without IMPLICIT NONE(TYPE) this is not an extension: implicit typing is in
! effect, so no forward-reference portability warning is emitted.  A later
! declaration that contradicts the would-be-implicit type is still rejected.
!CHECK: error: The type of 'mm' has already been implicitly declared as INTEGER(4)
subroutine s5
  parameter(mm=4096)
  real mm
end
