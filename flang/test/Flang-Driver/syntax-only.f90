! RUN: not %flang-new -fc1 -fsyntax-only %s 2>&1 | FileCheck %s
! RUN: not %f18 -fsyntax-only %s 2>&1 | FileCheck %s

! REQUIRES: new-flang-driver

! CHECK: IF statement is not allowed in IF statement
! CHECK: Semantic errors in {{.*}}syntax-only.f90
IF (A > 0.0) IF (B < 0.0) A = LOG (A)
END
