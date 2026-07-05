// RUN: %clang_cc1 -std=c++98 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++11 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++14 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++17 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++20 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++23 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++2c %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK

#if __cplusplus == 199711L
#define NOTHROW throw()
#else
#define NOTHROW noexcept(true)
#endif

namespace cwg193 { // cwg193: 2.7
struct A {
  ~A() NOTHROW {}
};

struct B {
  ~B() NOTHROW {}
};

struct C {
  ~C() NOTHROW {}
};

struct D : A {
  B b;
  ~D() NOTHROW { C c; }
};

void foo() {
  D d;
}

// skipping over D1 (complete object destructor)
// CHECK-LABEL: define {{.*}} void @cwg193::D::~D(){{.*}}
// CHECK-LABEL: define {{.*}} void @cwg193::D::~D(){{.*}}
// CHECK-NOT:     call void @cwg193::A::~A()
// CHECK-NOT:     call void @cwg193::B::~B()
// CHECK:         call void @cwg193::C::~C()
// CHECK:         call void @cwg193::B::~B()
// CHECK:         call void @cwg193::A::~A()
// CHECK-LABEL: }
} // namespace cwg193
