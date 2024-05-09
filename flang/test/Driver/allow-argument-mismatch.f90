! Test that -fallow-argument-mismatch flag is accepted with warning, while
! -fno-allow-argument-mismatch is rejected with error

! RUN: %flang -S %s -fallow-argument-mismatch -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-WARNING
! RUN: %flang_fc1 -S %s -fallow-argument-mismatch -o /dev/null
! RUN: not %flang -S %s -fno-allow-argument-mismatch -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
! RUN: not %flang_fc1 -S %s -fno-allow-argument-mismatch -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

! CHECK-WARNING: warning:
! CHECK_WARNING-SAME: argument unused
! CHECK_WARNING-SAME: -fallow-argument-mismatch

! CHECK-ERROR: error:
! CHECK-ERROR-SAME: unknown argument
! CHECK-ERROR-SAME: -fno-allow-argument-mismatch
