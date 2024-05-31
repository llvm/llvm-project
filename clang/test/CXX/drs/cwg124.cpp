// RUN: %clang_cc1 -std=c++98 %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++11 %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++14 %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++17 %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++20 %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++23 %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++2c %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK

#if __cplusplus == 199711L
#define NOTHROW throw()
#else
#define NOTHROW noexcept(true)
#endif

namespace cwg124 { // cwg124: 2.7

extern void full_expr_fence() NOTHROW;

struct A {
  A() NOTHROW {}
  ~A() NOTHROW {}
};

struct B {
  B(A = A()) NOTHROW {}
  ~B() NOTHROW {}
};

void f() {
  full_expr_fence();
  B b[2];
  full_expr_fence();
}

// CHECK-LABEL: define {{.*}} void @cwg124::f()()
// CHECK:         call void @cwg124::full_expr_fence()
// CHECK:         br label %arrayctor.loop
// CHECK-LABEL: arrayctor.loop:
// CHECK:         call void @cwg124::A::A()
// CHECK:         call void @cwg124::B::B(cwg124::A)
// CHECK:         call void @cwg124::A::~A()
// CHECK:         br {{.*}}, label %arrayctor.cont, label %arrayctor.loop
// CHECK-LABEL: arrayctor.cont:
// CHECK:         call void @cwg124::full_expr_fence()
// CHECK:         br label %arraydestroy.body
// CHECK-LABEL: arraydestroy.body:
// CHECK:         call void @cwg124::B::~B()
// CHECK-LABEL: }


} // namespace cwg124
