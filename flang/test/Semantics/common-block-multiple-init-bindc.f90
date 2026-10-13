! RUN: not %flang -fsyntax-only 2>&1 %s | FileCheck %s

! Test that when two differently-named Fortran COMMON blocks are merged by
! a shared BIND(C) name, the "Previous initialization" attachment names the
! block where the previous initialization actually appeared, not the
! current appearance's own (different) Fortran name.
!
! This file also has an unrelated BIND(C) name-clash error (the two
! merged blocks having a shared BIND(C) name is itself illegal), which
! prevents DATA statement compilation from running, so the
! multiple-initialization check below conservatively reports a hard error
! here rather than the usual portability warning for a duplicate -- see
! common-block-multiple-init.f90 for that.

subroutine s1
  integer :: xa
  common /a/ xa
  bind(c, name="cblk") :: /a/
  data xa /1/
end subroutine

subroutine s2
  integer :: xb
  common /b/ xb
  bind(c, name="cblk") :: /b/
  ! CHECK: error: Multiple initialization of COMMON block /b/
  ! CHECK: Previous initialization of COMMON block /a/
  data xb /1/
end subroutine
