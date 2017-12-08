// RUN: %clang_cc1 %s -emit-llvm -w -triple x86_64-apple-darwin10 -fsanitize=array-bounds -o - | FileCheck %s

// CHECK-LABEL: define i32 @foo(
int foo(int *const p __attribute__((pass_object_size(0))), int n) {
  int x = (p)[n];

  // CHECK: [[SIZE_ALLOCA:%.*]] = alloca i64, align 8
  // CHECK: store i64 %{{.*}}, i64* [[SIZE_ALLOCA]], align 8
  // CHECK: [[LOAD_SIZE:%.*]] = load i64, i64* [[SIZE_ALLOCA]], align 8, !nosanitize
  // CHECK-NEXT: [[SCALED_SIZE:%.*]] = udiv i64 [[LOAD_SIZE]], 4, !nosanitize
  // CHECK-NEXT: [[SEXT_N:%.*]] = sext i32 %{{.*}} to i64, !nosanitize
  // CHECK-NEXT: [[ICMP:%.*]] = icmp ult i64 [[SEXT_N]], [[SCALED_SIZE]], !nosanitize
  // CHECK-NEXT: br i1 [[ICMP]], {{.*}} !nosanitize
  // CHECK: __ubsan_handle_out_of_bounds

  {
    int **p = &p; // Shadow the parameter. The pass_object_size info is lost.
    // CHECK-NOT: __ubsan_handle_out_of_bounds
    x = *p[n];
  }

  // CHECK: ret i32
  return x;
}

typedef struct {} ZeroSizedType;

// CHECK-LABEL: define void @bar(
ZeroSizedType bar(ZeroSizedType *const p __attribute__((pass_object_size(0))), int n) {
  // CHECK-NOT: __ubsan_handle_out_of_bounds
  // CHECK: ret void
  return p[n];
}

// CHECK-LABEL: define i32 @baz(
int baz(int *const p __attribute__((pass_object_size(1))), int n) {
  // CHECK: __ubsan_handle_out_of_bounds
  // CHECK: ret i32
  return p[n];
}

// CHECK-LABEL: define i32 @mat(
int mat(int *const p __attribute__((pass_object_size(2))), int n) {
  // CHECK-NOT: __ubsan_handle_out_of_bounds
  // CHECK: ret i32
  return p[n];
}

// CHECK-LABEL: define i32 @pat(
int pat(int *const p __attribute__((pass_object_size(3))), int n) {
  // CHECK-NOT: __ubsan_handle_out_of_bounds
  // CHECK: ret i32
  return p[n];
}

// CHECK-LABEL: define i32 @cat(
int cat(int p[static 10], int n) {
  // CHECK: icmp ult i64 {{.*}}, 10, !nosanitize
  // CHECK: __ubsan_handle_out_of_bounds
  // CHECK: ret i32
  return p[n];
}

// CHECK-LABEL: define i32 @bat(
int bat(int n, int p[n]) {
  // CHECK-NOT: __ubsan_handle_out_of_bounds
  // CHECK: ret i32
  return p[n];
}
