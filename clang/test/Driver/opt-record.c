// RUN: %clang -### -S -o FOO -fsave-optimization-record %s 2>&1 | FileCheck %s
// RUN: %clang -### -c -o FOO -fsave-optimization-record %s 2>&1 | FileCheck %s
// RUN: %clang -### -c -o FOO.o -fsave-optimization-record %s 2>&1 | FileCheck %s
// RUN: %clang -### -save-temps -S -o FOO -fsave-optimization-record %s 2>&1 | FileCheck %s
// RUN: %clang -### -save-temps -c -o FOO.o -fsave-optimization-record %s 2>&1 | FileCheck %s
// RUN: %clang -### -c -fsave-optimization-record %s 2>&1 | FileCheck %s -check-prefix=CHECK-NO-O
// RUN: %clang -### -save-temps -c -fsave-optimization-record %s 2>&1 | FileCheck %s -check-prefix=CHECK-NO-O
// RUN: %clang -### -fsave-optimization-record %s 2>&1 | FileCheck %s -check-prefix=CHECK-NO-O
// RUN: %clang -### -S -fsave-optimization-record -x cuda -nocudainc -nocudalib %s 2>&1 | FileCheck %s -check-prefix=CHECK-NO-O -check-prefix=CHECK-CUDA-DEV
// RUN: %clang -### -fsave-optimization-record -x cuda -nocudainc -nocudalib %s 2>&1 | FileCheck %s -check-prefix=CHECK-NO-O -check-prefix=CHECK-CUDA-DEV
// RUN: %clang -### -S -o FOO -fsave-optimization-record -foptimization-record-file=BAR.txt %s 2>&1 | FileCheck %s -check-prefix=CHECK-EQ
// RUN: %clang -### -S -o FOO -foptimization-record-file=BAR.txt %s 2>&1 | FileCheck %s -check-prefix=CHECK-EQ
// RUN: %clang -### -S -o FOO -foptimization-record-file=BAR.txt -fno-save-optimization-record %s 2>&1 | FileCheck %s --check-prefix=CHECK-FOPT-DISABLE

// RUN: %clang -### -S -o FOO -fsave-optimization-record -foptimization-record-passes=inline %s 2>&1 | FileCheck %s -check-prefix=CHECK-EQ-PASSES
// RUN: %clang -### -S -o FOO -foptimization-record-passes=inline %s 2>&1 | FileCheck %s -check-prefix=CHECK-EQ-PASSES
// RUN: %clang -### -S -o FOO -foptimization-record-passes=inline -fno-save-optimization-record %s 2>&1 | FileCheck %s --check-prefix=CHECK-FOPT-DISABLE-PASSES
// RUN: %clang -### -S -o FOO -fsave-optimization-record -fsave-optimization-record=some-format %s 2>&1 | FileCheck %s -check-prefix=CHECK-EQ-FORMAT
// RUN: %clang -### -S -o FOO -fsave-optimization-record=some-format %s 2>&1 | FileCheck %s -check-prefix=CHECK-EQ-FORMAT
// RUN: %clang -### -S -o FOO -fsave-optimization-record=some-format -fno-save-optimization-record %s 2>&1 | FileCheck %s --check-prefix=CHECK-FOPT-DISABLE-FORMAT
// RUN: %clang -### -S -o FOO -fsave-optimization-record -arch x86_64 -arch x86_64h %s 2>&1 | FileCheck %s --check-prefix=CHECK-MULTIPLE-ARCH
//
// CHECK: "-cc1"
// CHECK: "-opt-record-file" "FOO.opt.yaml"

// CHECK-NO-O: "-cc1"
// CHECK-NO-O-DAG: "-opt-record-file" "opt-record.opt.yaml"
// CHECK-CUDA-DEV-DAG: "-opt-record-file" "opt-record-cuda-{{nvptx64|nvptx}}-nvidia-cuda-sm_20.opt.yaml"

// CHECK-EQ: "-cc1"
// CHECK-EQ: "-opt-record-file" "BAR.txt"

// CHECK-FOPT-DISABLE-NOT: "-fno-save-optimization-record"

// CHECK-EQ-PASSES: "-cc1"
// CHECK-EQ-PASSES: "-opt-record-passes" "inline"

// CHECK-FOPT-DISABLE-PASSES-NOT: "-fno-save-optimization-record"

// CHECK-EQ-FORMAT: "-cc1"
// CHECK-EQ-FORMAT: "-opt-record-file" "FOO.opt.some-format"
// CHECK-EQ-FORMAT: "-opt-record-format" "some-format"

// CHECK-FOPT-DISABLE-FORMAT-NOT: "-fno-save-optimization-record"

// CHECK-MULTIPLE-ARCH: "-cc1"
// CHECK-MULTIPLE-ARCH: "-opt-record-file" "FOO-x86_64.opt.yaml"
// CHECK-MULTIPLE-ARCH: "-cc1"
// CHECK-MULTIPLE-ARCH: "-opt-record-file" "FOO-x86_64h.opt.yaml"
