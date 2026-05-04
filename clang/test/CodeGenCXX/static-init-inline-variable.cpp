// RUN: %clang_cc1 -std=c++17 -emit-llvm -disable-llvm-passes -o - %s -triple x86_64-linux-gnu | FileCheck %s

struct A {
  int x;
  A(int x) : x(x) {}
  ~A() {}
};

namespace DeferredSequence {
inline int a = 1;
inline int b = a + 1;
inline int c = b + 1;
inline int d = c + 1;
int e = d;
}

namespace MixedSequence {
inline A a(1);
inline int x = a.x + 1;
inline int y = x + 1;
inline A b(y);
inline int z = b.x + 1;
inline int w = z + 1;
inline A c(b.x);
inline A d(c.x);
int t = w;
}

namespace NonDeferredSequence {
inline A a(1);
inline A b(a.x);
inline A c(b.x);
inline A d(c.x);
}

// CHECK: @llvm.global_ctors = appending global [16 x { i32, ptr, ptr }] [
// CHECK-SAME: { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.12, ptr @_ZN16DeferredSequence1bE },
// CHECK-SAME: { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.11, ptr @_ZN16DeferredSequence1cE },
// CHECK-SAME: { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.10, ptr @_ZN16DeferredSequence1dE },
// CHECK-SAME: { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.1,  ptr @_ZN13MixedSequence1aE },
// CHECK-SAME: { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.14, ptr @_ZN13MixedSequence1xE },
// CHECK-SAME: { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.13, ptr @_ZN13MixedSequence1yE },
// CHECK-SAME: { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.2,  ptr @_ZN13MixedSequence1bE },
// CHECK-SAME: { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.16, ptr @_ZN13MixedSequence1zE },
// CHECK-SAME: { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.15, ptr @_ZN13MixedSequence1wE },
// CHECK-SAME: { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.3,  ptr @_ZN13MixedSequence1cE },
// CHECK-SAME: { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.4,  ptr @_ZN13MixedSequence1dE },
// CHECK-SAME: { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.6,  ptr @_ZN19NonDeferredSequence1aE },
// CHECK-SAME: { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.7,  ptr @_ZN19NonDeferredSequence1bE },
// CHECK-SAME: { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.8,  ptr @_ZN19NonDeferredSequence1cE },
// CHECK-SAME: { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.9,  ptr @_ZN19NonDeferredSequence1dE },
// CHECK-SAME: { i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_static_init_inline_variable.cpp, ptr null }
