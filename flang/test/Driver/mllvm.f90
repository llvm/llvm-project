! Test the `-mllvm` option

!------------
! RUN COMMAND
!------------
! 1. Test typical usage.
! RUN: %flang -S -mllvm -print-before-all %s -o - 2>&1 | FileCheck %s --check-prefix=OUTPUT
! RUN: %flang_fc1 -S -mllvm -print-before-all %s -o - 2>&1 | FileCheck %s --check-prefix=OUTPUT

! 2. Does the option forwarding from `flang-new` to `flang-new -fc1` work?
! RUN: %flang -### -S -mllvm -print-before-all %s -o - 2>&1 | FileCheck %s --check-prefix=OPTION_FORWARDING

! 3. Test invalid usage (`-print-before` requires an argument)
! RUN: not %flang -S -mllvm -print-before %s -o - 2>&1 | FileCheck %s --check-prefix=INVALID_USAGE

!----------------
! EXPECTED OUTPUT
!----------------
! OUTPUT: *** IR Dump Before Pre-ISel Intrinsic Lowering (pre-isel-intrinsic-lowering) ***
! OUTPUT-NEXT: ; ModuleID = 'FIRModule'
! OUTPUT-NEXT: source_filename = "FIRModule"

! Verify that `-mllvm <option>` is forwarded to flang -fc1
! OPTION_FORWARDING: flang-new" "-fc1"
! OPTION_FORWARDING-SAME: "-mllvm" "-print-before-all"

! INVALID_USAGE: flang (LLVM option parsing): for the --print-before option: requires a value!

!------
! INPUT
!------
end program
