// RUN: not %clang_cc1 -std=c++20 -fmodules -fsyntax-only -I %S/Inputs %s 2> %t
// RUN: FileCheck %s < %t

#include "GH203167.h"

// CHECK: error: no matching '#pragma clang module endbuild'
// CHECK: error: no matching '#pragma clang module end'
// CHECK-NOT: Assertion
