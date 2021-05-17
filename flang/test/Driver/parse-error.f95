! Verify that parsing errors are correctly reported by the driver

! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=ERROR

! ERROR: Could not parse {{.*}}parse-error.f95

"This file will not parse"
