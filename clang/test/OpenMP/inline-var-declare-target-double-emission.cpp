// Fixed triple: the reproducer needs LP64 layout so the folded initializer
// type differs from the declared type, and ELF for the comdat key.
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fopenmp \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

struct P {
  constexpr P() : x(0) {}
  ~P();

  union {
    int x;
    long y;
  };
};

inline P S;
#pragma omp declare target(S)

inline const int *PTR = &S.x;
int use() { return S.x + *PTR; }

// CHECK: @S = linkonce_odr{{.*}} global { { i32, [4 x i8] } }
// S must be emitted before its guard variable: a second emission erases and
// re-creates @S, moving it after the surviving @_ZGV1S.
// CHECK-NEXT: @_ZGV1S = linkonce_odr global i64 0
// CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }]
// CHECK-SAME: ptr @__cxx_global_var_init, ptr @S
// CHECK: define internal void @__cxx_global_var_init()
// CHECK-NOT: define {{.*}} @__cxx_global_var_init
