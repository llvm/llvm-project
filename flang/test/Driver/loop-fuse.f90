! RUN: %flang -### -S -floop-fuse %s 2>&1 | FileCheck -check-prefix=CHECK-LOOP-FUSE %s
! RUN: %flang -### -S -fno-loop-fuse %s 2>&1 | FileCheck -check-prefix=CHECK-NO-LOOP-FUSE %s
! RUN: %flang -### -S -O0 %s 2>&1 | FileCheck -check-prefix=CHECK-NO-LOOP-FUSE %s
! RUN: %flang -### -S -O1 %s 2>&1 | FileCheck -check-prefix=CHECK-NO-LOOP-FUSE %s
! RUN: %flang -### -S -O2 %s 2>&1 | FileCheck -check-prefix=CHECK-NO-LOOP-FUSE %s
! RUN: %flang -### -S -O3 %s 2>&1 | FileCheck -check-prefix=CHECK-NO-LOOP-FUSE %s
! RUN: %flang -### -S -Os %s 2>&1 | FileCheck -check-prefix=CHECK-NO-LOOP-FUSE %s
! RUN: %flang -### -S -Oz %s 2>&1 | FileCheck -check-prefix=CHECK-NO-LOOP-FUSE %s
! CHECK-LOOP-FUSE: "-floop-fuse"
! CHECK-NO-LOOP-FUSE-NOT: "-floop-fuse"
! RUN: %flang_fc1 -emit-llvm -O2 -floop-fuse -mllvm -print-pipeline-passes -o /dev/null %s 2>&1 | FileCheck -check-prefix=CHECK-LOOP-FUSE-PASS %s
! RUN: %flang_fc1 -emit-llvm -O2 -fno-loop-fuse -mllvm -print-pipeline-passes -o /dev/null %s 2>&1 | FileCheck -check-prefix=CHECK-NO-LOOP-FUSE-PASS %s
! CHECK-LOOP-FUSE-PASS: loop-fusion
! CHECK-NO-LOOP-FUSE-PASS-NOT: loop-fusion

program test
end program
