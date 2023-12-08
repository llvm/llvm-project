// RUN: %clang_cc1 -std=c++98 -Wno-constant-evaluated -triple x86_64-linux -include %s -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++98 -Wno-constant-evaluated -triple x86_64-linux -emit-pch %s -o %t
// RUN: %clang_cc1 -std=c++98 -Wno-constant-evaluated -triple x86_64-linux -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -std=c++11 -Wno-constant-evaluated -triple x86_64-linux -include %s -verify %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CXX11
// RUN: %clang_cc1 -std=c++11 -Wno-constant-evaluated -triple x86_64-linux -emit-pch %s -o %t-cxx11
// RUN: %clang_cc1 -std=c++11 -Wno-constant-evaluated -triple x86_64-linux -include-pch %t-cxx11 -verify %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CXX11

// RUN: %clang_cc1 -std=c++20 -Wno-constant-evaluated -triple x86_64-linux -include %s -verify %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CXX11
// RUN: %clang_cc1 -std=c++20 -Wno-constant-evaluated -triple x86_64-linux -emit-pch %s -o %t-cxx11
// RUN: %clang_cc1 -std=c++20 -Wno-constant-evaluated -triple x86_64-linux -include-pch %t-cxx11 -verify %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CXX11

// expected-no-diagnostics

#ifndef HEADER_INCLUDED
#define HEADER_INCLUDED

// CHECK-DAG: @a ={{.*}} global i8 1,
// CHECK-DAG: @b ={{.*}} constant i8 1,
// CXX11-DAG: @c ={{.*}} constant i8 1,
// CHECK-DAG: @d ={{.*}} global float 1.000000e+00
// CHECK-DAG: @e ={{.*}} constant float 1.000000e+00

bool a = __builtin_is_constant_evaluated();
extern const bool b = __builtin_is_constant_evaluated();
#if __cplusplus >= 201103L
extern constexpr bool c = __builtin_is_constant_evaluated();
#endif
float d = __builtin_is_constant_evaluated();
extern const float e = __builtin_is_constant_evaluated();

void g(...);

// CHECK-LABEL: define {{.*}} @_Z1fv(
// CHECK:       store i8 0, ptr %[[A:.*]],
// CHECK:       store i8 1, ptr %[[B:.*]],
// CXX11:       store i8 1, ptr %[[C:.*]],
// CHECK:       store float 0.000000e+00, ptr %[[D:.*]],
// CHECK:       store float 0.000000e+00, ptr %[[E:.*]],
// CHECK:       load i8, ptr %[[A]],
// CHECK:       call {{.*}} @_Z1gz(i32 noundef %{{[^,]+}}, i32 noundef 1
// CXX11-SAME:  , i32 noundef 1
// CHECK-SAME:  , double noundef %{{[^,]+}}, double noundef 0.000000e+00)
void f() {
  bool a = __builtin_is_constant_evaluated();
  const bool b = __builtin_is_constant_evaluated();
#if __cplusplus >= 201103L
  constexpr bool c = __builtin_is_constant_evaluated();
#endif
  float d = __builtin_is_constant_evaluated();
  const float e = __builtin_is_constant_evaluated();
  g(a, b
#if __cplusplus >= 201103L
      , c
#endif
      , d, e);
}

#else

_Static_assert(b, "");
#if __cplusplus >= 201103L
static_assert(c, "");
#endif
_Static_assert(__builtin_constant_p(1) ? e == 1.0f : false, "");

#endif
