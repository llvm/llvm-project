// RUN: %clang_cc1 -no-opaque-pointers -o - -emit-llvm -triple x86_64-linux-pc %s | FileCheck %s
int(&&intu_rvref)[] {1,2,3,4};
// CHECK: @_ZGR10intu_rvref_ = internal global [4 x i32] [i32 1, i32 2, i32 3, i32 4]
// CHECK: @intu_rvref ={{.*}} constant [4 x i32]* @_ZGR10intu_rvref_

void foo() {
  static const int(&&intu_rvref)[] {1,2,3,4};
  // CHECK: @_ZZ3foovE10intu_rvref = internal constant [4 x i32]* @_ZGRZ3foovE10intu_rvref_
  // CHECK: @_ZGRZ3foovE10intu_rvref_ = internal constant [4 x i32] [i32 1, i32 2, i32 3, i32 4]
}

// Example given on review, ensure this doesn't crash as well.
constexpr int f() {
  // CHECK: i32 @_Z1fv()
  int(&&intu_rvref)[]{1, 2, 3, 4};
  // CHECK: %{{.*}} = alloca [4 x i32]*
  return intu_rvref[2];
}

void use_f() {
  int i = f();
}
