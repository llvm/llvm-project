// Check that -march works for all supported targets.

// RUN: not %clang -target s390x -S -emit-llvm -march=z9 %s -o - 2>&1 | FileCheck --check-prefix=CHECK-Z9 %s
// RUN: %clang -target s390x -### -S -emit-llvm -march=z10 %s 2>&1 | FileCheck --check-prefix=CHECK-Z10 %s
// RUN: %clang -target s390x -### -S -emit-llvm -march=arch8 %s 2>&1 | FileCheck --check-prefix=CHECK-ARCH8 %s
// RUN: %clang -target s390x -### -S -emit-llvm -march=z196 %s 2>&1 | FileCheck --check-prefix=CHECK-Z196 %s
// RUN: %clang -target s390x -### -S -emit-llvm -march=arch9 %s 2>&1 | FileCheck --check-prefix=CHECK-ARCH9 %s
// RUN: %clang -target s390x -### -S -emit-llvm -march=zEC12 %s 2>&1 | FileCheck --check-prefix=CHECK-ZEC12 %s
// RUN: %clang -target s390x -### -S -emit-llvm -march=arch10 %s 2>&1 | FileCheck --check-prefix=CHECK-ARCH10 %s
// RUN: %clang -target s390x -### -S -emit-llvm -march=z13 %s 2>&1 | FileCheck --check-prefix=CHECK-Z13 %s
// RUN: %clang -target s390x -### -S -emit-llvm -march=arch11 %s 2>&1 | FileCheck --check-prefix=CHECK-ARCH11 %s
// RUN: %clang -target s390x -### -S -emit-llvm -march=z14 %s 2>&1 | FileCheck --check-prefix=CHECK-Z14 %s
// RUN: %clang -target s390x -### -S -emit-llvm -march=arch12 %s 2>&1 | FileCheck --check-prefix=CHECK-ARCH12 %s
// RUN: %clang -target s390x -### -S -emit-llvm -march=z15 %s 2>&1 | FileCheck --check-prefix=CHECK-Z15 %s
// RUN: %clang -target s390x -### -S -emit-llvm -march=arch13 %s 2>&1 | FileCheck --check-prefix=CHECK-ARCH13 %s
// RUN: %clang -target s390x -### -S -emit-llvm -march=z16 %s 2>&1 | FileCheck --check-prefix=CHECK-Z16 %s
// RUN: %clang -target s390x -### -S -emit-llvm -march=arch14 %s 2>&1 | FileCheck --check-prefix=CHECK-ARCH14 %s

// CHECK-Z9: error: unknown target CPU 'z9'
// CHECK-Z10: "-target-cpu" "z10"
// CHECK-ARCH8: "-target-cpu" "arch8"
// CHECK-Z196: "-target-cpu" "z196"
// CHECK-ARCH9: "-target-cpu" "arch9"
// CHECK-ZEC12: "-target-cpu" "zEC12"
// CHECK-ARCH10: "-target-cpu" "arch10"
// CHECK-Z13: "-target-cpu" "z13"
// CHECK-ARCH11: "-target-cpu" "arch11"
// CHECK-Z14: "-target-cpu" "z14"
// CHECK-ARCH12: "-target-cpu" "arch12"
// CHECK-Z15: "-target-cpu" "z15"
// CHECK-ARCH13: "-target-cpu" "arch13"
// CHECK-Z16: "-target-cpu" "z16"
// CHECK-ARCH14: "-target-cpu" "arch14"

int x;
