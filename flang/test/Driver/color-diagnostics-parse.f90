! Test the behaviors of -f{no-}color-diagnostics when emitting parsing
! diagnostics.
! Windows command prompt doesn't support ANSI escape sequences.
! REQUIRES: shell

! RUN: not %flang %s -fcolor-diagnostics 2>&1 \
! RUN:     | FileCheck %s --check-prefix=CHECK_CD
! RUN: not %flang %s -fno-color-diagnostics 2>&1 \
! RUN:     | FileCheck %s --check-prefix=CHECK_NCD
! RUN: not %flang_fc1 %s -fcolor-diagnostics 2>&1 \
! RUN:     | FileCheck %s --check-prefix=CHECK_CD
! RUN: not %flang_fc1 %s 2>&1 | FileCheck %s --check-prefix=CHECK_NCD

! CHECK_CD: {{.*}}[0;1;31merror: {{.*}}[0mexpected '('

! CHECK_NCD: error: expected '('

program m
  integer :: i =
end
