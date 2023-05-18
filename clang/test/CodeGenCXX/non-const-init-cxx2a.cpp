// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s -std=c++2a | FileCheck %s

// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-pch -o %t.pch %s -std=c++2a
// RUN: %clang_cc1 -triple x86_64-apple-darwin -include-pch %t.pch -x c++ /dev/null -emit-llvm -o - -std=c++2a | FileCheck %s

struct B {
  constexpr B() {}
  constexpr ~B() { n *= 5; }
  int n = 123;
};

// We emit a dynamic destructor here because b.n might have been modified
// before b is destroyed.
//
// CHECK: @b ={{.*}} global {{.*}} i32 123
B b = B();

// CHECK: define {{.*}}cxx_global_var_init
// CHECK: call {{.*}} @__cxa_atexit({{.*}} @_ZN1BD1Ev, {{.*}} @b
