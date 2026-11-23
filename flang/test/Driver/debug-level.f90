! RUN: %flang %s -g -c -### 2>&1 | FileCheck %s --check-prefix=FULL
! RUN: %flang %s -g1 -c -### 2>&1 | FileCheck %s --check-prefix=LINE
! RUN: %flang %s -gline-tables-only -c -### 2>&1 | FileCheck %s --check-prefix=LINE
! RUN: %flang %s -gline-directives-only -### 2>&1 | FileCheck %s --check-prefix=DIRECTIVES

! LINE: -debug-info-kind=line-tables-only
! DIRECTIVES: -debug-info-kind=line-directives-only
! FULL: -debug-info-kind=standalone

