! RUN: %flang_fc1 -mllvm -print-pipeline-passes -emit-llvm-bc -o /dev/null -O0 %s 2>&1 | FileCheck %s

! Just check a few passes to ensure that something reasonable is being printed.
! CHECK: always-inline
! CHECK-SAME: annotation-remarks

end program
