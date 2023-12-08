// RUN: %clang -target arm-none-gnueabi -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT
// RUN: %clang -target arm-none-gnueabi -mno-neg-immediates -### %s 2>&1 | FileCheck %s

// RUN: %clang -target aarch64-none-gnueabi -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT
// RUN: %clang -target aarch64-none-gnueabi -mno-neg-immediates -### %s 2>&1 | FileCheck %s

// CHECK: "-target-feature" "+no-neg-immediates"
// CHECK-DEFAULT-NOT: "+no-neg-immediates"
