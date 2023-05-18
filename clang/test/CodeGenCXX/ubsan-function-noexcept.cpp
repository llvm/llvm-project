// RUN: %clang_cc1 -std=c++17 -fsanitize=function -emit-llvm -triple x86_64-linux-gnu %s -o - | FileCheck %s

/// Check the following two functions have the same func_sanitize metadata, i.e.
/// they have the same type hash despite the exception specifier.
// CHECK: define{{.*}} void @_Z1fv() #[[#]] !func_sanitize ![[FUNCSAN:.*]] {
// CHECK: define{{.*}} void @_Z10f_noexceptv() #[[#]] !func_sanitize 
// CHECK-SAME: ![[FUNCSAN]] {
void f() {}
void f_noexcept() noexcept {}

// CHECK: define{{.*}} void @_Z1gPDoFvvE
void g(void (*p)() noexcept) {
  // CHECK: icmp eq i32 %{{.*}}, -1056584962, !nosanitize
  // CHECK: icmp eq i32 %{{.*}}, [[Hash:[-0-9]+]], !nosanitize
  p();
}

// CHECK: ![[FUNCSAN]] = !{i32 -1056584962, i32 [[Hash]]}
