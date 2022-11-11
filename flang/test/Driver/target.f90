!  Test that --target indeed sets the target

! RUN: %flang --target=unknown-unknown-unknown -emit-llvm -c %s \
! RUN:   -o %t.o -### 2>&1 | FileCheck %s

! CHECK: Target: unknown-unknown-unknown
! CHECK: "-triple" "unknown-unknown-unknown"
