// RUN: %clang_cc1 -std=c++20 -fincremental-extensions -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s
// Entities in a top-level statement mangle as <local-name>s of that statement,
// so same-named locals in different top-level statements do not collide.

extern "C" int printf(const char *, ...);
namespace ns { template <typename F> void call(F f) { f(); } }

ns::call([] { printf("ONE\n"); });
ns::call([] { printf("TWO\n"); });
// CHECK-DAG: define internal void @"_ZN2ns4callIZL9__stmt__0vE3$_0EEvT_"
// CHECK-DAG: define internal void @"_ZN2ns4callIZL9__stmt__1vE3$_1EEvT_"
// CHECK-DAG: define internal void @"_ZZL9__stmt__0vENK3$_0clEv"
// CHECK-DAG: define internal void @"_ZZL9__stmt__1vENK3$_1clEv"

{ struct S { int f() { return 1; } }; printf("%d\n", S().f()); }
{ struct S { int f() { return 2; } }; printf("%d\n", S().f()); }
// CHECK-DAG: define internal noundef i32 @_ZZL9__stmt__2vEN1S1fEv
// CHECK-DAG: define internal noundef i32 @_ZZL9__stmt__3vEN1S1fEv
