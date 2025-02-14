! Ensure that only argument -Werror is supported.

! RUN: not %flang_fc1 -fsyntax-only -Wall %s  2>&1 | FileCheck %s --check-prefix=WRONG
! RUN: not %flang_fc1 -fsyntax-only -WX %s  2>&1 | FileCheck %s --check-prefix=WRONG

! WRONG: Only `-Werror`, `-W[no]unused-dummy-argument` and `-W[no]unused-variable` are supported currently.
