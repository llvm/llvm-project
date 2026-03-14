! RUN: not %flang_fc1 -E %s 2>&1 | FileCheck %s --check-prefix=ERROR

! ERROR: Could not scan {{.*}}scanning-error.f95

#NOT-PP-DIRECTIVE
