// Enabled by default for assembly
// RUN: %clang -target armv7--none-eabi -### %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-ENABLED

// Can be forced on or off for assembly.
// RUN: %clang -target armv7--none-eabi -### %s 2>&1 -mno-default-build-attributes \
// RUN:    | FileCheck %s -check-prefix CHECK-DISABLED
// RUN: %clang -target armv7--none-eabi -### %s 2>&1 -mdefault-build-attributes \
// RUN:    | FileCheck %s -check-prefix CHECK-ENABLED

// Option ignored C/C++ (since we always emit hardware and ABI build attributes
// during codegen).
// RUN: %clang -target armv7--none-eabi -### -x c %s -mdefault-build-attributes 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-DISABLED
// RUN: %clang -target armv7--none-eabi -### -x c++ %s -mdefault-build-attributes 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-DISABLED

// CHECK-DISABLED-NOT: "-arm-add-build-attributes"
// CHECK-ENABLED: "-arm-add-build-attributes"
// expected-warning {{argument unused during compilation: '-mno-default-build-attributes'}}
