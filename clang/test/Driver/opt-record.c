// RUN: %clang -### -S -o FOO -fsave-optimization-record %s 2>&1 | FileCheck %s
// RUN: %clang -### -c -o FOO -fsave-optimization-record %s 2>&1 | FileCheck %s
// RUN: %clang -### -c -fsave-optimization-record %s 2>&1 | FileCheck %s -check-prefix=CHECK-NO-O
// RUN: %clang -### -fsave-optimization-record %s 2>&1 | FileCheck %s -check-prefix=CHECK-NO-O
// RUN: %clang -### -S -fsave-optimization-record -x cuda -nocudainc -nocudalib %s 2>&1 | FileCheck %s -check-prefix=CHECK-NO-O -check-prefix=CHECK-CUDA-DEV
// RUN: %clang -### -fsave-optimization-record -x cuda -nocudainc -nocudalib %s 2>&1 | FileCheck %s -check-prefix=CHECK-NO-O -check-prefix=CHECK-CUDA-DEV
// RUN: %clang -### -S -o FOO -fsave-optimization-record -foptimization-record-file=BAR.txt %s 2>&1 | FileCheck %s -check-prefix=CHECK-EQ

// CHECK: "-cc1"
// CHECK: "-opt-record-file" "FOO.opt.yaml"

// CHECK-NO-O: "-cc1"
// CHECK-NO-O-DAG: "-opt-record-file" "opt-record.opt.yaml"
// CHECK-CUDA-DEV-DAG: "-opt-record-file" "opt-record-cuda-{{nvptx64|nvptx}}-nvidia-cuda-sm_20.opt.yaml"

// CHECK-EQ: "-cc1"
// CHECK-EQ: "-opt-record-file" "BAR.txt"

