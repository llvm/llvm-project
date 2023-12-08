! Verify that the frontend driver error-out if multiple actions are specified

! RUN: not %flang_fc1 -E -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=ERROR
! RUN: not %flang_fc1 -fsyntax-only -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=ERROR

! ERROR: error: Only one action option is allowed

end progream
