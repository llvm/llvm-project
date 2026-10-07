// RUN: not %clang_cc1 -std=c++20 -fmodules -fsyntax-only %s 2> %t
// RUN: FileCheck %s < %t

#pragma clang module build N
module N {}
#pragma clang module contents
#pragma clang module begin N
int x;

// CHECK: error: no matching '#pragma clang module endbuild'
// CHECK: error: no matching '#pragma clang module end'
// CHECK-NOT: Assertion
