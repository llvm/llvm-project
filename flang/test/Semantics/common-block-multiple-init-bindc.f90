! RUN: %flang -fsyntax-only 2>&1 %s | FileCheck %s

! Test that when two differently-named Fortran COMMON blocks are merged by
! a shared BIND(C) name, the "Previous initialization" attachment names the
! block where the previous initialization actually appeared, not the
! current appearance's own (different) Fortran name.
!
! /a/ and /b/ both carry the same explicit BIND(C) binding label ("cblk"),
! so they denote one object and the merge itself is not an error (19.2 p2
! does not apply: every common block sharing the key genuinely carries
! that label). Since both appearances initialize that one object to the
! identical value, this is accepted as the nonstandard "duplicate
! initialization" extension -- the same one common-block-multiple-init.f90
! exercises for same-named commons -- and reported as a portability
! warning rather than an error.

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
