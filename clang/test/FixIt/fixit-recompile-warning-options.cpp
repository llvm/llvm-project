// RUN: %clang_cc1 -std=c++14 %s -fixit-recompile -fixit-to-temporary -Werror \
// RUN:   -Wno-deprecated-register 2>&1 | FileCheck %s

//In c++14 this produces a fixit, which we fix with -fixit-recompile
static_assert(true);

// During the recompile we ensure that the -Wno-deprecated-register option
// is properly applied
void f() {
  register int data;
}

// CHECK: error: 'static_assert' with no message is a C++17 extension
// CHECK: note: FIX-IT applied suggested code changes
// CHECK-NOT: 'register' storage class specifier is deprecated and incompatible with C++17
