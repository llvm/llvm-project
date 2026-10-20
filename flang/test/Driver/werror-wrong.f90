! Ensure that only argument -Werror is supported.

! RUN: not %flang_fc1 -fsyntax-only -Wall %s  2>&1 | FileCheck %s --check-prefix=WRONG1
! RUN: not %flang_fc1 -fsyntax-only -WX %s  2>&1 | FileCheck %s --check-prefix=WRONG2

! WRONG1: error: Unknown diagnostic option: -Wall
! WRONG2: error: Unknown diagnostic option: -WX
