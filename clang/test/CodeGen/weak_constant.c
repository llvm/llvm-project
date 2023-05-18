// RUN: %clang_cc1 -w -emit-llvm %s -O1 -o - | FileCheck %s
// This used to "check for bug compatibility with gcc".
// Now it checks that that the "weak" declaration makes the value
// fully interposable whereas a "selectany" one is handled as constant
// and propagated.

// CHECK: @x = weak {{.*}}constant i32 123
const int x __attribute((weak)) = 123;

// CHECK: @y = weak_odr {{.*}}constant i32 234
const int y __attribute((selectany)) = 234;

int* f(void) {
  return &x;
}

int g(void) {
  // CHECK: load i32, ptr @x
  // CHECK-NOT: ret i32 123
  return *f();
}

int *k(void) {
  return &y;
}

int l(void) {
  // CHECK-NOT: load i32, ptr @y
  // CHECK: ret i32 234
  return *k();
}
