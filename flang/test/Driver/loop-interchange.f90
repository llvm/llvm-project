! The driver only forwards an explicit -f[no-]loop-interchange; the -O3
! default is applied in the frontend, so it is not visible in -###.
! RUN: %flang -### -S -floop-interchange %s 2>&1 | FileCheck -check-prefix=CHECK-LOOP-INTERCHANGE %s
! RUN: %flang -### -S -fno-loop-interchange %s 2>&1 | FileCheck -check-prefix=CHECK-NO-LOOP-INTERCHANGE %s
! RUN: %flang -### -S -O3 %s 2>&1 | FileCheck -check-prefix=CHECK-NO-LOOP-INTERCHANGE %s
! CHECK-LOOP-INTERCHANGE: "-floop-interchange"
! CHECK-NO-LOOP-INTERCHANGE-NOT: "-floop-interchange"
!
! Loop interchange matches the LLVM pipeline default: on whenever the
! optimization pipeline runs (-O1 and above), off with an explicit
! -fno-loop-interchange.
! RUN: %flang_fc1 -emit-llvm -O1 -mllvm -print-pipeline-passes -o /dev/null %s 2>&1 | FileCheck -check-prefix=CHECK-LOOP-INTERCHANGE-PASS %s
! RUN: %flang_fc1 -emit-llvm -O2 -mllvm -print-pipeline-passes -o /dev/null %s 2>&1 | FileCheck -check-prefix=CHECK-LOOP-INTERCHANGE-PASS %s
! RUN: %flang_fc1 -emit-llvm -O3 -mllvm -print-pipeline-passes -o /dev/null %s 2>&1 | FileCheck -check-prefix=CHECK-LOOP-INTERCHANGE-PASS %s
! RUN: %flang_fc1 -emit-llvm -O3 -fno-loop-interchange -mllvm -print-pipeline-passes -o /dev/null %s 2>&1 | FileCheck -check-prefix=CHECK-NO-LOOP-INTERCHANGE-PASS %s
! CHECK-LOOP-INTERCHANGE-PASS: loop-interchange
! CHECK-NO-LOOP-INTERCHANGE-PASS-NOT: loop-interchange

program test
end program
