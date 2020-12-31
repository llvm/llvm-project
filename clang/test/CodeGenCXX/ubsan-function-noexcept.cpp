// RUN: %clang_cc1 -std=c++17 -fsanitize=function -emit-llvm -triple x86_64-linux-gnu %s -o - | FileCheck %s

// Check that typeinfo recorded in function prolog doesn't have "Do" noexcept
// qualifier in its mangled name.
// CHECK: @[[RTTI:[0-9]+]] = private constant i8* bitcast ({ i8*, i8* }* @_ZTIFvvE to i8*)
// CHECK: define{{.*}} void @_Z1fv() #{{.*}} prologue <{ i32, i32 }> <{ i32 {{.*}}, i32 trunc (i64 sub (i64 ptrtoint (i8** @[[RTTI]] to i64), i64 ptrtoint (void ()* @_Z1fv to i64)) to i32) }>
void f() noexcept {}

// CHECK: define{{.*}} void @_Z1gPDoFvvE
void g(void (*p)() noexcept) {
  // Check that reference typeinfo at call site doesn't have "Do" noexcept
  // qualifier in its mangled name, either.
  // CHECK: icmp eq i8* %{{.*}}, bitcast ({ i8*, i8* }* @_ZTIFvvE to i8*), !nosanitize
  p();
}
