! Tests -fget-symbols-sources with modules.

PROGRAM helloworld
    use mm2b
    implicit none
    integer::i
    i = callget5()
ENDPROGRAM

! RUN: %f18 -fparse-only %S/Inputs/getsymbols02-a.f90
! RUN: %f18 -fparse-only %S/Inputs/getsymbols02-b.f90
! RUN: %f18 -fget-symbols-sources -fparse-only %s 2>&1 | FileCheck %s
! CHECK: get5: mm2a
! CHECK: callget5: mm2b
