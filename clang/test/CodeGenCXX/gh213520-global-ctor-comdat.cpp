// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fopenmp \
// RUN:     -emit-llvm -o - %s | FileCheck %s

struct Base {
  int x;
  constexpr Base() : x(0) {}
};

struct T : Base {
  ~T();
};

struct S;

template <class Tag>
inline T g;

T *a() { return &g<S>; }
T *b() { return &g<S>; }

// The COMDAT key must be the surviving global, not null or a freed value.
// CHECK: @llvm.global_ctors = appending global {{.*}} { i32 65535, ptr @__cxx_global_var_init{{.*}}, ptr @_Z1gI1SE }
