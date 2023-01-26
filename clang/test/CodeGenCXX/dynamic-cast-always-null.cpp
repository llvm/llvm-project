// RUN: %clang_cc1 -I%S %s -triple x86_64-apple-darwin10 -emit-llvm -fcxx-exceptions -fexceptions -std=c++11 -o - | FileCheck %s
struct A { virtual ~A(); };
struct B final : A { };
struct C { virtual ~C(); int c; };

// CHECK: @_Z1fP1B
C *f(B* b) {
  // CHECK-NOT: call ptr @__dynamic_cast
  // CHECK: ret ptr null
  return dynamic_cast<C*>(b);
}

// CHECK: @_Z1fR1B
C &f(B& b) {
  // CHECK-NOT: call ptr @__dynamic_cast
  // CHECK: call void @__cxa_bad_cast() [[NR:#[0-9]+]]
  // CHECK: ret ptr undef
  return dynamic_cast<C&>(b);
}

void dont_crash() {
  (void) dynamic_cast<void*>((A*)0);
  (void) dynamic_cast<void*>((B*)0);
}

// CHECK: attributes [[NR]] = { noreturn }
