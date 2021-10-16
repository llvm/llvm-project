// RUN: %clang_cc1 -fsanitize=nullability-return -emit-llvm %s -o - -triple x86_64-apple-macosx10.10.0 | FileCheck %s

// CHECK: [[ATTR_LOC:@[0-9]+]] = {{.*}} global { {{.*}} i32 15, i32 38

// CHECK-LABEL: define{{.*}} i8* @_Z3foov()
// CHECK: [[CALL:%.*]] = call noundef i8* @_Z6helperv()
// CHECK: icmp ne i8* [[CALL]]
// CHECK: call void @__ubsan_handle_nullability_return_v1_abort({{.*}}[[ATTR_LOC]]

struct S {
  using PtrTy = id;
};

#pragma clang assume_nonnull begin
__attribute__((ns_returns_retained)) S::PtrTy foo(void) {
  extern S::PtrTy helper(void);
  return helper();
}
#pragma clang assume_nonnull end
