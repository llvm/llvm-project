/// Enabled by default for assembly
// RUN: %clang --target=hexagon-unknown-elf -### %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-ENABLED

/// Can be forced on or off for assembly.
// RUN: %clang --target=hexagon-unknown-elf -### %s 2>&1 -mno-default-build-attributes \
// RUN:    | FileCheck %s -check-prefix CHECK-DISABLED
// RUN: %clang --target=hexagon-unknown-elf -### %s 2>&1 -mdefault-build-attributes \
// RUN:    | FileCheck %s -check-prefix CHECK-ENABLED

/// Option ignored C/C++ (since we always emit hardware and ABI build attributes
/// during codegen).
// RUN: %clang --target=hexagon-unknown-elf -### -x c %s -mdefault-build-attributes 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-DISABLED-C
// RUN: %clang --target=hexagon-unknown-elf -### -x c++ %s -mdefault-build-attributes 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-DISABLED-C

// CHECK-DISABLED-NOT: "-hexagon-add-build-attributes"
// CHECK-DISABLED-C-NOT: "-hexagon-add-build-attributes"
// CHECK-ENABLED: "-hexagon-add-build-attributes"
// CHECK-DISABLED-C: argument unused during compilation: '-mdefault-build-attributes'
