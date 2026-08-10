! Test the `-mllvm` option

! 1. Test typical usage.
! RUN: %flang -S -mllvm -print-before-all %s -o - 2>&1 | FileCheck %s --check-prefix=OUTPUT
! RUN: %flang_fc1 -S -mllvm -print-before-all %s -o - 2>&1 | FileCheck %s --check-prefix=OUTPUT

! 2. Test invalid usage (`-print-before` requires an argument)
! RUN: not %flang -S -mllvm -print-before %s -o - 2>&1 | FileCheck %s --check-prefix=INVALID_USAGE

! OUTPUT: *** IR Dump Before Pre-ISel Intrinsic Lowering (pre-isel-intrinsic-lowering) ***
! OUTPUT-NEXT: ; ModuleID = 'FIRModule'
! OUTPUT-NEXT: source_filename = "FIRModule"

! INVALID_USAGE: flang (LLVM option parsing): for the --print-before option: requires a value!

end program
