! RUN: not %flang-new -fc1 -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=ERROR
! RUN: not %f18 -parse-only %s 2>&1 | FileCheck %s --check-prefix=ERROR

! REQUIRES: new-flang-driver

! ERROR: Could not parse {{.*}}parse-error.f95

"This file will not parse"
