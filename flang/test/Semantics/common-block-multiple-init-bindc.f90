! RUN: %flang -fsyntax-only 2>&1 %s | FileCheck %s

! Test that when two differently-named Fortran COMMON blocks are merged by
! a shared BIND(C) name, the "Previous initialization" attachment names the
! block where the previous initialization actually appeared, not the
! current appearance's own (different) Fortran name.
!
! Merging /a/ and /b/ through their shared BIND(C) label is accepted as an
! extension.  Their identical initializations are also accepted as the
! duplicate initialization extension tested in common-block-multiple-init.f90,
! with a portability warning rather than an error.

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
  ! CHECK: portability: Multiple initialization of COMMON block /b/ is not standard; this appearance duplicates the previous initialization [-Wmultiple-common-block-init]
  ! CHECK: Previous initialization of COMMON block /a/
  data xb /1/
end subroutine
