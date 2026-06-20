//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// REQUIRES: has-filecheck

// In case the function signature is taking an argument by value,
// when the type is small and trivial, we pass it internally by value,
// otherwise, we pass it by rvalue reference

#include <cassert>
#include <functional>
#include <utility>
#include <type_traits>

// RUN: %{cxx} -c %s %{flags} %{compile_flags} -S -emit-llvm --target=arm64-apple-darwin25.4.0 -O0 -o - | FileCheck %s

struct Small {
  char c[8];
};

struct Big {
  char c[256];
};

struct SmallButNonTrivial {
  Small s;
  SmallButNonTrivial() = default;
  SmallButNonTrivial(const SmallButNonTrivial&) {}
};

// CHECK: define{{.*}} @_Z10test_small
// CHECK: define{{.*}} @_ZNK{{.*}}function_ref{{.*}}Fv5Small{{.*}}ES1_(ptr noundef %0, i64 %1)
// CHECK: %3 = alloca %struct.Small, align 1
// CHECK: %6 = alloca %struct.Small, align 1
// CHECK: %7 = getelementptr inbounds nuw %struct.Small, ptr %3, i32 0, i32 0
// CHECK: store i64 %1, ptr %7, align 1
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %6, ptr align 1 %3, i64 8, i1 false)
// CHECK: %14 = getelementptr inbounds nuw %struct.Small, ptr %6, i32 0, i32 0
// CHECK: %15 = load i64, ptr %14, align 1
// CHECK: call void %10(ptr %13, i64 %15)
//
// The above has two local copies of Small object. It loads %1 to local copy %3, and then
// memcpy to the second copy %6 and eventually load into %15, which passed to the type erased function
void test_small(std::function_ref<void(Small)> f) { f(Small{}); }

// CHECK: define{{.*}} @_Z8test_big
// CHECK: define{{.*}} @_ZNKSt3__112function_ref{{.*}}Fv3Big{{.*}}ES1_(ptr noundef %0, ptr noundef %1)
// CHECK: call void {{.*}}(ptr {{.*}}, ptr noundef nonnull align 1 dereferenceable(256) %1)
//
// The above directly calls the function pointer with the parameter Big object pointer
// points to the operator()'s parameter %1, thus, no extra copies/moves
// So we internally pass the Big object by reference
void test_big(std::function_ref<void(Big)> f) { f(Big{}); }

// CHECK: define{{.*}} @_Z22test_small_non_trivial
// CHECK: define{{.*}} @_ZNKSt3__112function_ref{{.*}}Fv18SmallButNonTrivial{{.*}}ES1_(ptr noundef %0, ptr noundef %1)
// CHECK: call void {{.*}}(ptr {{.*}}, ptr noundef nonnull align 1 dereferenceable(8) %1)
//
// The above directly calls the function pointer with the parameter SmallButNonTrivial object pointer
// points to the operator()'s parameter %1, thus, no extra copies/moves
// So we internally pass the SmallButNonTrivial object by reference
void test_small_non_trivial(std::function_ref<void(SmallButNonTrivial)> f) { f(SmallButNonTrivial{}); }