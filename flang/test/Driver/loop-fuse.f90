! RUN: %flang -### -S -fexperimental-loop-fusion %s 2>&1 | FileCheck -check-prefix=CHECK-LOOP-FUSE %s
! RUN: %flang -### -S -fno-experimental-loop-fusion %s 2>&1 | FileCheck -check-prefix=CHECK-NO-LOOP-FUSE %s
! RUN: %flang -### -S -O0 %s 2>&1 | FileCheck -check-prefix=CHECK-NO-LOOP-FUSE %s
! RUN: %flang -### -S -O1 %s 2>&1 | FileCheck -check-prefix=CHECK-NO-LOOP-FUSE %s
! RUN: %flang -### -S -O2 %s 2>&1 | FileCheck -check-prefix=CHECK-NO-LOOP-FUSE %s
! RUN: %flang -### -S -O3 %s 2>&1 | FileCheck -check-prefix=CHECK-NO-LOOP-FUSE %s
! RUN: %flang -### -S -Os %s 2>&1 | FileCheck -check-prefix=CHECK-NO-LOOP-FUSE %s
! RUN: %flang -### -S -Oz %s 2>&1 | FileCheck -check-prefix=CHECK-NO-LOOP-FUSE %s
! CHECK-LOOP-FUSE: "-fexperimental-loop-fusion"
! CHECK-NO-LOOP-FUSE-NOT: "-fexperimental-loop-fusion"
! RUN: %flang_fc1 -emit-llvm -O2 -fexperimental-loop-fusion -mllvm -print-pipeline-passes -o /dev/null %s 2>&1 | FileCheck -check-prefix=CHECK-LOOP-FUSE-PASS %s
! RUN: %flang_fc1 -emit-llvm -O2 -fno-experimental-loop-fusion -mllvm -print-pipeline-passes -o /dev/null %s 2>&1 | FileCheck -check-prefix=CHECK-NO-LOOP-FUSE-PASS %s
! CHECK-LOOP-FUSE-PASS: loop-fusion
! CHECK-NO-LOOP-FUSE-PASS-NOT: loop-fusion

program test
end program
