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

namespace dr201 { // dr201: 2.8

extern void full_expr_fence() NOTHROW;

struct A {
  ~A() NOTHROW {}
};

struct B {
  B(A) NOTHROW {}
  ~B() NOTHROW {}
};

void foo() {
  full_expr_fence();
  B b = A();
  full_expr_fence();
}

// CHECK-LABEL: define {{.*}} void @dr201::foo()
// CHECK:         call void @dr201::full_expr_fence()
// CHECK:         call void @dr201::B::B(dr201::A)
// CHECK:         call void @dr201::A::~A()
// CHECK:         call void @dr201::full_expr_fence()
// CHECK:         call void @dr201::B::~B()
// CHECK-LABEL: }

} // namespace dr201
