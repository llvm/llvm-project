! Regression test: array named constants declared directly in the intrinsic
! iso_fortran_env module (e.g. character_kinds) must be emitted as a DEFINITION
! with an initializer, not as a bodyless external declaration. Otherwise the
! reference is undefined at link time.
! See createIntrinsicModuleDefinitions in lib/Lower/Bridge.cpp.

! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine test_character_kinds(i, r)
  use iso_fortran_env, only: character_kinds
  integer :: i, r
  r = character_kinds(i)
end subroutine

! The global must be DEFINED (linkonce_odr linkage + a parenthesized
! initializer), not a bodyless "fir.global @_QM...character_kinds ... constant"
! external declaration.
! CHECK:     fir.global linkonce_odr @_QMiso_fortran_envECcharacter_kinds({{.*}}) {{.*}}constant : !fir.array<{{[0-9]+}}xi32>
! CHECK-NOT: fir.global @_QMiso_fortran_envECcharacter_kinds {{.*}}constant
