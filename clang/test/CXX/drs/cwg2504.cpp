// RUN: %clang_cc1 -std=c++98 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++11 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK,SINCE-CXX11
// RUN: %clang_cc1 -std=c++14 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK,SINCE-CXX11
// RUN: %clang_cc1 -std=c++17 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK,SINCE-CXX11
// RUN: %clang_cc1 -std=c++20 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK,SINCE-CXX11
// RUN: %clang_cc1 -std=c++23 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK,SINCE-CXX11
// RUN: %clang_cc1 -std=c++2c %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK,SINCE-CXX11

namespace cwg2504 { // cwg2504: no
#if __cplusplus >= 201103L
struct V { V() = default; V(int); };
struct Q { Q(); };
struct A : virtual V, Q {
  using V::V;
  A() = delete;
};
int bar() { return 42; }
struct B : A {
  B() : A(bar()) {}  // ok
};
struct C : B {};
void foo() { C c; } // bar is not invoked, because the V subobject is not initialized as part of B
#endif
}

// FIXME: As specified in the comment above (which comes from an example in the Standard),
//        we are not supposed to unconditionally call `bar()` and call a constructor
//        inherited from `V`.

// SINCE-CXX11-LABEL: define linkonce_odr void @cwg2504::B::B()
// SINCE-CXX11-NOT:     br
// SINCE-CXX11:         call noundef i32 @cwg2504::bar()
// SINCE-CXX11-NOT:     br
// SINCE-CXX11:         call void @cwg2504::A::A(int)
// SINCE-CXX11-LABEL: }

// CHECK: {{.*}}
