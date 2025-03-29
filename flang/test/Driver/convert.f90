! Ensure argument -fconvert=<value> accepts all relevant options and produces an
! error if an invalid value is specified. 

!--------------------------
! FLANG DRIVER (flang)
!--------------------------
! RUN: %flang -### -fconvert=unknown %s  2>&1 | FileCheck %s --check-prefix=VALID
! RUN: %flang -### -fconvert=native %s  2>&1 | FileCheck %s --check-prefix=VALID
! RUN: %flang -### -fconvert=little-endian %s  2>&1 | FileCheck %s --check-prefix=VALID
! RUN: %flang -### -fconvert=big-endian %s  2>&1 | FileCheck %s --check-prefix=VALID
! RUN: %flang -### -fconvert=swap %s  2>&1 | FileCheck %s --check-prefix=VALID
! RUN: not %flang -fconvert=foobar %s  2>&1 | FileCheck %s --check-prefix=INVALID

!-----------------------------------------
! FRONTEND FLANG DRIVER (flang -fc1)
!-----------------------------------------
! RUN: %flang_fc1 -emit-mlir -fconvert=unknown %s -o - | FileCheck %s --check-prefix=VALID_FC1
! RUN: %flang_fc1 -emit-mlir -fconvert=native %s -o - | FileCheck %s --check-prefix=VALID_FC1
! RUN: %flang_fc1 -emit-mlir -fconvert=little-endian %s -o - | FileCheck %s --check-prefix=VALID_FC1
! RUN: %flang_fc1 -emit-mlir -fconvert=big-endian %s -o - | FileCheck %s --check-prefix=VALID_FC1
! RUN: %flang_fc1 -emit-mlir -fconvert=swap %s -o - | FileCheck %s --check-prefix=VALID_FC1
! RUN: not %flang_fc1 -fconvert=foobar %s  2>&1 | FileCheck %s --check-prefix=INVALID

! Only test that the command executes without error. Correct handling of each
! option is handled in Lowering tests.
! VALID: -fconvert
! VALID_FC1: module

! INVALID: error: invalid value 'foobar' in '-fconvert=foobar'
