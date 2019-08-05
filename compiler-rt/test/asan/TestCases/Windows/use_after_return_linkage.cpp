// Make sure LIBCMT doesn't accidentally get added to the list of DEFAULTLIB
// directives.  REQUIRES: asan-dynamic-runtime
// RUN: %clang_cl_asan -LD %s | FileCheck %s
// CHECK: Creating library
// CHECK-NOT: LIBCMT

void foo(int *p) { *p = 42; }

__declspec(dllexport) void bar() {
  int x;
  foo(&x);
}
