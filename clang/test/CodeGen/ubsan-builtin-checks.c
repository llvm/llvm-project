// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -w -emit-llvm -o - %s -fsanitize=builtin | FileCheck %s --check-prefixes=CHECK,POISON
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -w -emit-llvm -o - %s -fsanitize=builtin | FileCheck %s --check-prefixes=CHECK,NOPOISON

// A zero input to __bultin_ctz/clz is considered UB even if the target does not
// want to optimize based on zero input being undefined.

// CHECK: define{{.*}} void @check_ctz
void check_ctz(int n) {
  // CHECK: [[NOT_ZERO:%.*]] = icmp ne i32 [[N:%.*]], 0, !nosanitize
  // CHECK-NEXT: br i1 [[NOT_ZERO]]
  //
  // Handler block:
  // CHECK: call void @__ubsan_handle_invalid_builtin
  // CHECK-NEXT: unreachable
  //
  // Continuation block:
  // POISON: call i32 @llvm.cttz.i32(i32 [[N]], i1 true)
  // NOPOISON: call i32 @llvm.cttz.i32(i32 [[N]], i1 false)
  __builtin_ctz(n);

  // CHECK: call void @__ubsan_handle_invalid_builtin
  __builtin_ctzl(n);

  // CHECK: call void @__ubsan_handle_invalid_builtin
  __builtin_ctzll(n);

  // CHECK: call void @__ubsan_handle_invalid_builtin
  __builtin_ctzg((unsigned int)n);
}

// CHECK: define{{.*}} void @check_clz
void check_clz(int n) {
  // CHECK: [[NOT_ZERO:%.*]] = icmp ne i32 [[N:%.*]], 0, !nosanitize
  // CHECK-NEXT: br i1 [[NOT_ZERO]]
  //
  // Handler block:
  // CHECK: call void @__ubsan_handle_invalid_builtin
  // CHECK-NEXT: unreachable
  //
  // Continuation block:
  // POISON: call i32 @llvm.ctlz.i32(i32 [[N]], i1 true)
  // NOPOISON: call i32 @llvm.ctlz.i32(i32 [[N]], i1 false)
  __builtin_clz(n);

  // CHECK: call void @__ubsan_handle_invalid_builtin
  __builtin_clzl(n);

  // CHECK: call void @__ubsan_handle_invalid_builtin
  __builtin_clzll(n);

  // CHECK: call void @__ubsan_handle_invalid_builtin
  __builtin_clzg((unsigned int)n);
}

// CHECK: define{{.*}} void @check_assume
void check_assume(int n) {
  // CHECK: [[TOBOOL:%.*]] = icmp ne i32 [[N:%.*]], 0
  // CHECK-NEXT: br i1 [[TOBOOL]]
  //
  // Handler block:
  // CHECK: call void @__ubsan_handle_invalid_builtin
  // CHECK-NEXT: unreachable
  //
  // Continuation block:
  // CHECK: call void @llvm.assume(i1 [[TOBOOL]])
  __builtin_assume(n);

  // CHECK: call void @__ubsan_handle_invalid_builtin
  __attribute__((assume(n)));
}
