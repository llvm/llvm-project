! Tests -fget-symbols-sources with modules.

PROGRAM helloworld
    use mm2b
    implicit none
    integer::i
    i = callget5()
ENDPROGRAM

! RUN: %flang_fc1 -fsyntax-only %S/Inputs/getsymbols02-a.f90
! RUN: %flang_fc1 -fsyntax-only %S/Inputs/getsymbols02-b.f90
! RUN: %flang_fc1 -fget-symbols-sources %s 2>&1 | FileCheck %s
! CHECK: callget5: .{{[/\\]}}mm2b.mod,
! CHECK: get5: .{{[/\\]}}.{{[/\\]}}mm2a.mod,
