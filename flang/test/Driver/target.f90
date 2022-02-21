!  Test that --target indeed sets the target

!-----------------------------------------
! RUN LINES
!-----------------------------------------
! RUN: %flang --target=unknown-unknown-unknown -emit-llvm -c %s \
! RUN:   -o %t.o -### 2>&1 | FileCheck %s

!-----------------
! EXPECTED OUTPUT
!-----------------
! CHECK: Target: unknown-unknown-unknown
! CHECK: "-triple" "unknown-unknown-unknown"
