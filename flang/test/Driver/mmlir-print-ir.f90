! Verify that --mlir-print-ir-before=<pass> and --mlir-print-ir-after=<pass>
! work for registered passes.

! RUN: %flang_fc1 -emit-llvm -mmlir --mlir-print-ir-before=cse -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=BEFORE
! RUN: %flang_fc1 -emit-llvm -mmlir --mlir-print-ir-after=cse -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=AFTER

! BEFORE: IR Dump Before CSE
! BEFORE: func.func @_QQmain

! AFTER: IR Dump After CSE
! AFTER: func.func @_QQmain

end program
