/// Check that char is unsigned by default.
// RUN: %clang -### %s --target=xtensa -c 2>&1 | FileCheck %s
// CHECK: "-cc1" "-triple" "xtensa"
// CHECK-SAME: "-fno-signed-char"
