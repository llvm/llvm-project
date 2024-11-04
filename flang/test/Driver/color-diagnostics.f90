! Test the behaviors of -f{no-}color-diagnostics and -f{no}-diagnostics-color.
! Windows command prompt doesn't support ANSI escape sequences.
! REQUIRES: shell

! RUN: not %flang %s -fcolor-diagnostics 2>&1 \
! RUN:     | FileCheck %s --check-prefix=CHECK_CD
! RUN: not %flang %s -fno-color-diagnostics 2>&1 \
! RUN:     | FileCheck %s --check-prefix=CHECK_NCD
! RUN: not %flang_fc1 %s -fcolor-diagnostics 2>&1 \
! RUN:     | FileCheck %s --check-prefix=CHECK_CD
! RUN: not %flang_fc1 %s -fno-color-diagnostics 2>&1 \
! RUN:     | FileCheck %s --check-prefix=UNSUPPORTED_COLOR_DIAGS

! RUN: not %flang %s -fdiagnostics-color 2>&1 \
! RUN:     | FileCheck %s --check-prefix=CHECK_CD
! RUN: not %flang %s -fno-diagnostics-color 2>&1 \
! RUN:     | FileCheck %s --check-prefix=CHECK_NCD
! RUN: not %flang_fc1 %s -fdiagnostics-color 2>&1 \
! RUN:     | FileCheck %s --check-prefix=UNSUPPORTED_DIAGS_COLOR
! RUN: not %flang_fc1 %s -fno-diagnostics-color 2>&1 \
! RUN:     | FileCheck %s --check-prefix=UNSUPPORTED_NO_DIAGS_COLOR

! RUN: not %flang %s -fdiagnostics-color=always 2>&1 \
! RUN:     | FileCheck %s --check-prefix=CHECK_CD
! RUN: not %flang %s -fdiagnostics-color=never 2>&1 \
! RUN:     | FileCheck %s --check-prefix=CHECK_NCD

! RUN: not %flang_fc1 %s 2>&1 | FileCheck %s --check-prefix=CHECK_NCD

! CHECK_CD: {{.*}}[0;1;31merror: {{.*}}[0m{{.*}}[1mSemantic errors in {{.*}}color-diagnostics.f90{{.*}}[0m

! CHECK_NCD: Semantic errors in {{.*}}color-diagnostics.f90

! UNSUPPORTED_COLOR_DIAGS: error: unknown argument: '-fno-color-diagnostics'
! UNSUPPORTED_DIAGS_COLOR: error: unknown argument: '-fdiagnostics-color'
! UNSUPPORTED_NO_DIAGS_COLOR: error: unknown argument: '-fno-diagnostics-color'

! Check that invalid values of -fdiagnostics-color= are disallowed.
! RUN: not %flang %s -fdiagnostics-color=sometimes 2>&1 \
! RUN:     | FileCheck %s --check-prefix=DCEQ_BAD
! DCEQ_BAD: error: invalid argument 'sometimes' to -fdiagnostics-color=

program m
  integer :: i = k
end
