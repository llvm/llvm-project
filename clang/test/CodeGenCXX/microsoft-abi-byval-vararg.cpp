// RUN: %clang_cc1 -Wno-non-pod-varargs -emit-llvm %s -o - -triple=i686-pc-win32 -mconstructor-aliases -fno-rtti | FileCheck %s

#include <stdarg.h>

struct A {
  A(int a) : a(a) {}
  A(const A &o) : a(o.a) {}
  ~A() {}
  int a;
};

int foo(A a, ...) {
  va_list ap;
  va_start(ap, a);
  int sum = 0;
  for (int i = 0; i < a.a; ++i)
    sum += va_arg(ap, int);
  va_end(ap);
  return sum;
}

// CHECK-LABEL: define dso_local noundef i32 @"?foo@@YAHUA@@ZZ"(ptr inalloca(<{ %struct.A }>) %0, ...)

int main() {
  return foo(A(3), 1, 2, 3);
}
// CHECK-LABEL: define dso_local noundef i32 @main()
// CHECK: %[[argmem:[^ ]*]] = alloca inalloca <{ %struct.A, i32, i32, i32 }>
// CHECK: call noundef i32 (ptr, ...) @"?foo@@YAHUA@@ZZ"{{.*}}(ptr inalloca(<{ %struct.A, i32, i32, i32 }>) %[[argmem]])

void varargs_zero(...);
void varargs_one(int, ...);
void varargs_two(int, int, ...);
void varargs_three(int, int, int, ...);
void call_var_args() {
  A x(3);
  varargs_zero(x);
  varargs_one(1, x);
  varargs_two(1, 2, x);
  varargs_three(1, 2, 3, x);
}

// CHECK-LABEL: define dso_local void @"?call_var_args@@YAXXZ"()
// Passing non-POD to varargs ellipsis (...) traps. Since the trap is noreturn,
// it is followed by unreachable, and subsequent dead code (like inalloca stack
// allocations and varargs calls) is pruned and not emitted.
// CHECK: call void @llvm.trap()
// CHECK-NEXT: unreachable

// CHECK-LABEL: declare dso_local void @"?varargs_zero@@YAXZZ"(...)
