// RUN: %clang -target armv7-apple-darwin -### %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-DEFAULT

// RUN: %clang -target armv7-apple-darwin -mkernel -### %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-KERNEL

// CHECK-DEFAULT-NOT: "-target-feature" "+no-movt"

// CHECK-KERNEL: "-target-feature" "+no-movt"
