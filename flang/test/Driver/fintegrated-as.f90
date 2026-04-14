! Tests for the `-f(no-)integrated-as` flag.

! UNSUPPORTED: system-windows

!--------------------------
! With `-fno-integrated-as`
!--------------------------
! Verify that there _is_ a separate line with an assembler invocation, and
! that -no-integrated-as is specified to generate GNU Assembler compatible assembly.
! RUN: %flang -c -fno-integrated-as %s -### 2>&1 | FileCheck %s
! CHECK-LABEL: "-fc1"
! CHECK-SAME: "-no-integrated-as"
! CHECK-SAME: "-o" "[[assembly_file:.*]].s"
! CHECK-NEXT: "-o" "{{.*}}.o" "[[assembly_file:.*]].s"

!----------------------------------------------------------------
! Without `-fno-integrated-as` / With `-fintegrated-as` (default)
!----------------------------------------------------------------
! Verify that there _is no_ separate line with an assembler invocation
! RUN: %flang -c -fintegrated-as %s -### 2>&1 | FileCheck %s -check-prefix=DEFAULT
! RUN: %flang -c %s -### 2>&1 | FileCheck %s -check-prefix=DEFAULT
! DEFAULT-LABEL: "-fc1"
! DEFAULT-NOT: "-no-integrated-as"
! DEFAULT-SAME: "-o" "{{.*}}.o" "{{.*}}fintegrated-as.f90"
