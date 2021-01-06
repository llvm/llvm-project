! RUN: not %flang-new -fc1 -E %s 2>&1 | FileCheck %s --check-prefix=ERROR
! RUN: not %f18 -E %s 2>&1 | FileCheck %s --check-prefix=ERROR

! REQUIRES: new-flang-driver

! ERROR: Could not scan {{.*}}scanning-error.f95

#NOT-PP-DIRECTIVE
