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
//
// CHECK: "-cc1"
// CHECK: "-opt-record-file" "FOO.opt.yaml"

// CHECK-NO-O: "-cc1"
// CHECK-NO-O-DAG: "-opt-record-file" "opt-record.opt.yaml"
// CHECK-CUDA-DEV-DAG: "-opt-record-file" "opt-record-cuda-{{nvptx64|nvptx}}-nvidia-cuda-sm_{{.*}}.opt.yaml"

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

// Test remarks options pass-through
// No pass-through: lto is disabled
// RUN: %clang -target x86_64 -### -o FOO -fdiagnostics-hotness-threshold=100 -fsave-optimization-record %s 2>&1 | FileCheck %s -check-prefix=CHECK-NOPASS

// Pass-through:
// RUN: %clang -target x86_64-linux -### -fuse-ld=lld -flto -fdiagnostics-hotness-threshold=100 -fsave-optimization-record -foptimization-record-passes=inline %s 2>&1 | FileCheck %s -check-prefix=CHECK-PASS-A
// RUN: %clang -target x86_64-linux -### -o FOO -fuse-ld=gold -flto -fdiagnostics-hotness-threshold=100 -fsave-optimization-record -foptimization-record-passes=inline %s 2>&1 | FileCheck %s -check-prefix=CHECK-PASS
// RUN: %clang -target x86_64-linux -### -o FOO -fuse-ld=lld -flto=thin -fdiagnostics-hotness-threshold=100 -fsave-optimization-record=some-format -foptimization-record-file=FOO.txt %s 2>&1 | FileCheck %s -check-prefix=CHECK-PASS-CUSTOM
// RUN: %clang -target x86_64-linux -### -o FOO -fuse-ld=lld -flto=thin -fdiagnostics-hotness-threshold=100 -Rpass=inline -Rpass-missed=inline -Rpass-analysis=inline %s 2>&1 | FileCheck %s -check-prefix=CHECK-PASS-RPASS
// RUN: %clang -target x86_64-linux -### -o FOO -fuse-ld=lld -flto=thin -fdiagnostics-hotness-threshold=auto -Rpass=inline -Rpass-missed=inline -Rpass-analysis=inline %s 2>&1 | FileCheck %s -check-prefix=CHECK-PASS-AUTO

// CHECK-NOPASS-NOT: "--plugin-opt=opt-remarks-filename="
// CHECK-NOPASS-NOT: "--plugin-opt=opt-remarks-passes=inline"
// CHECK-NOPASS-NOT: "--plugin-opt=opt-remarks-format=yaml"
// CHECK-NOPASS-NOT: "--plugin-opt=opt-remarks-hotness-threshold=100"

// CHECK-PASS-A:      "--plugin-opt=opt-remarks-filename=a.out.opt.ld.yaml"
// CHECK-PASS-A-SAME: "--plugin-opt=opt-remarks-passes=inline"
// CHECK-PASS-A-SAME: "--plugin-opt=opt-remarks-format=yaml"
// CHECK-PASS-A-SAME: "--plugin-opt=opt-remarks-hotness-threshold=100"

// CHECK-PASS:      "--plugin-opt=opt-remarks-filename=FOO.opt.ld.yaml"
// CHECK-PASS-SAME: "--plugin-opt=opt-remarks-passes=inline"
// CHECK-PASS-SAME: "--plugin-opt=opt-remarks-format=yaml"
// CHECK-PASS-SAME: "--plugin-opt=opt-remarks-hotness-threshold=100"

// CHECK-PASS-CUSTOM:      "--plugin-opt=opt-remarks-filename=FOO.txt.opt.ld.some-format"
// CHECK-PASS-CUSTOM-SAME: "--plugin-opt=opt-remarks-format=some-format"
// CHECK-PASS-CUSTOM-SAME: "--plugin-opt=opt-remarks-hotness-threshold=100"

// CHECK-PASS-RPASS:      "--plugin-opt=-pass-remarks=inline"
// CHECK-PASS-RPASS-SAME: "--plugin-opt=-pass-remarks-missed=inline"
// CHECK-PASS-RPASS-SAME: "--plugin-opt=-pass-remarks-analysis=inline"
// CHECK-PASS-RPASS-SAME: "--plugin-opt=opt-remarks-hotness-threshold=100"

// CHECK-PASS-AUTO:   "--plugin-opt=opt-remarks-hotness-threshold=auto"
