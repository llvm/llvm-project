! Verify that --mlir-print-ir-before=<pass> works for registered passes.

! RUN: %flang_fc1 -emit-llvm -mmlir --mlir-print-ir-before=cse -o /dev/null %s 2>&1 | FileCheck %s

! CHECK: IR Dump Before CSE
! CHECK: func.func @_QQmain

end program
