//// Enabled by default for assembly
// RUN: %clang --target=riscv32 -### %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-ENABLED
// RUN: %clang --target=riscv64 -### %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-ENABLED

/// Can be forced on or off for assembly.
// RUN: %clang --target=riscv32 -### %s 2>&1 -mno-default-build-attributes \
// RUN:   | FileCheck %s --check-prefix=CHECK-DISABLED
// RUN: %clang --target=riscv64 -### %s 2>&1 -mno-default-build-attributes \
// RUN:   | FileCheck %s --check-prefix=CHECK-DISABLED
// RUN: %clang --target=riscv32 -### %s 2>&1 -mdefault-build-attributes \
// RUN:   | FileCheck %s --check-prefix=CHECK-ENABLED
// RUN: %clang --target=riscv64 -### %s 2>&1 -mdefault-build-attributes \
// RUN:   | FileCheck %s --check-prefix=CHECK-ENABLED

/// Option ignored for C/C++ (since we always emit hardware and ABI build
/// attributes during codegen).
// RUN: %clang --target=riscv32 -### -x c %s -mdefault-build-attributes 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DISABLED
// RUN: %clang --target=riscv64 -### -x c %s -mdefault-build-attributes 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DISABLED
// RUN: %clang --target=riscv32 -### -x c++ %s -mdefault-build-attributes 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DISABLED
// RUN: %clang --target=riscv64 -### -x c++ %s -mdefault-build-attributes 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DISABLED

// CHECK-DISABLED-NOT: "-riscv-add-build-attributes"
// CHECK-ENABLED: "-riscv-add-build-attributes"
