! RUN: split-file %s %t
! RUN: cat %t/bound.f90 %t/unbound.f90 %t/other.f90 > %t/test.f90
! RUN: not %flang_fc1 -fsyntax-only %t/test.f90 2>&1 | FileCheck %s --check-prefix=OTHER
! RUN: cat %t/bound.f90 %t/other.f90 %t/unbound.f90 > %t/test.f90
! RUN: not %flang_fc1 -fsyntax-only %t/test.f90 2>&1 | FileCheck %s --check-prefix=UNBOUND
! RUN: cat %t/unbound.f90 %t/bound.f90 %t/other.f90 > %t/test.f90
! RUN: not %flang_fc1 -fsyntax-only %t/test.f90 2>&1 | FileCheck %s --check-prefix=OTHER
! RUN: cat %t/unbound.f90 %t/other.f90 %t/bound.f90 > %t/test.f90
! RUN: not %flang_fc1 -fsyntax-only %t/test.f90 2>&1 | FileCheck %s --check-prefix=OTHER
! RUN: cat %t/other.f90 %t/bound.f90 %t/unbound.f90 > %t/test.f90
! RUN: not %flang_fc1 -fsyntax-only %t/test.f90 2>&1 | FileCheck %s --check-prefix=UNBOUND
! RUN: cat %t/other.f90 %t/unbound.f90 %t/bound.f90 > %t/test.f90
! RUN: not %flang_fc1 -fsyntax-only %t/test.f90 2>&1 | FileCheck %s --check-prefix=UNBOUND
! RUN: cat %t/bound.f90 %t/empty.f90 %t/other.f90 > %t/test.f90
! RUN: not %flang_fc1 -fsyntax-only %t/test.f90 2>&1 | FileCheck %s --check-prefix=OTHER
! RUN: cat %t/bound.f90 %t/other.f90 %t/blank.f90 > %t/test.f90
! RUN: not %flang_fc1 -fsyntax-only %t/test.f90 2>&1 | FileCheck %s --check-prefix=UNBOUND
! RUN: cat %t/bound.f90 %t/same-scope.f90 > %t/test.f90
! RUN: not %flang_fc1 -fsyntax-only %t/test.f90 2>&1 | FileCheck %s --check-prefix=SAME
! RUN: cat %t/same-scope.f90 %t/bound.f90 > %t/test.f90
! RUN: not %flang_fc1 -fsyntax-only %t/test.f90 2>&1 | FileCheck %s --check-prefix=SAME

! Check every ordering of a three-way collision.  A bound /com/ is
! compatible with either of the other blocks, but the unbound /com/ and
! bound /other/ do not share storage.  The attachment must name the actual
! conflicting member, even when it is not the first member of the group.
! OTHER: error: Two entities have the same global name 'com'
! OTHER-NEXT: common /other/ k
! OTHER: Conflicting declaration
! OTHER-NEXT: common /com/ j
! OTHER-NOT: error:
! UNBOUND: error: Two entities have the same global name 'com'
! UNBOUND-NEXT: common /com/ j
! UNBOUND: Conflicting declaration
! UNBOUND-NEXT: common /other/ k
! UNBOUND-NOT: error:

! An earlier scope must not mask a collision in a later scope.
! SAME: error: Two entities have the same global name 'com'
! SAME-NEXT: common /b/ k
! SAME: Conflicting declaration
! SAME-NEXT: common /a/ j
! SAME-NOT: error:

!--- bound.f90
subroutine bound
  integer :: i
  common /com/ i
  bind(c, name="com") :: /com/
end subroutine

!--- unbound.f90
subroutine unbound
  integer :: j
  common /com/ j
end subroutine

!--- other.f90
subroutine other
  integer :: k
  common /other/ k
  bind(c, name="com") :: /other/
end subroutine

!--- empty.f90
subroutine empty_label
  integer :: j
  common /com/ j
  bind(c, name="") :: /com/
end subroutine

!--- blank.f90
subroutine blank_label
  integer :: j
  common /com/ j
  bind(c, name="   ") :: /com/
end subroutine

!--- same-scope.f90
subroutine same_scope
  integer :: j, k
  common /a/ j
  common /b/ k
  bind(c, name="com") :: /a/
  bind(c, name="com") :: /b/
end subroutine
