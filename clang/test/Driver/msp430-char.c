/// Check that char is unsigned by default.
// RUN: %clang -### %s --target=msp430 -c 2>&1 | FileCheck %s
// CHECK: "-cc1" "-triple" "msp430"
// CHECK-SAME: "-fno-signed-char"
