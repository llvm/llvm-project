// REQUIRES: asserts
// RUN: %clang_cc1 -disable-noundef-analysis -x objective-c -emit-llvm -triple x86_64-apple-macosx10.10.0 -fblocks -fobjc-arc -fsanitize=nullability-arg,nullability-assign,nullability-return -w %s -o - | FileCheck %s
// RUN: %clang_cc1 -disable-noundef-analysis -x objective-c++ -emit-llvm -triple x86_64-apple-macosx10.10.0 -fblocks -fobjc-arc -fsanitize=nullability-arg,nullability-assign,nullability-return -w %s -o - | FileCheck %s

// CHECK: [[NONNULL_RV_LOC1:@.*]] = private unnamed_addr global {{.*}} i32 100, i32 6
// CHECK: [[NONNULL_ARG_LOC:@.*]] = private unnamed_addr global {{.*}} i32 204, i32 15 {{.*}} i32 190, i32 23
// CHECK: [[NONNULL_ASSIGN1_LOC:@.*]] = private unnamed_addr global {{.*}} i32 305, i32 9
// CHECK: [[NONNULL_ASSIGN2_LOC:@.*]] = private unnamed_addr global {{.*}} i32 405, i32 10
// CHECK: [[NONNULL_ASSIGN3_LOC:@.*]] = private unnamed_addr global {{.*}} i32 506, i32 10
// CHECK: [[NONNULL_INIT1_LOC:@.*]] = private unnamed_addr global {{.*}} i32 604, i32 25
// CHECK: [[NONNULL_INIT2_LOC1:@.*]] = private unnamed_addr global {{.*}} i32 707, i32 26
// CHECK: [[NONNULL_INIT2_LOC2:@.*]] = private unnamed_addr global {{.*}} i32 707, i32 29
// CHECK: [[NONNULL_RV_LOC2:@.*]] = private unnamed_addr global {{.*}} i32 800, i32 6

#define NULL ((void *)0)
#define INULL ((int *)NULL)
#define INNULL ((int *_Nonnull)NULL)

// CHECK-LABEL: define{{.*}} i32* @{{.*}}nonnull_retval1
#line 100
int *_Nonnull nonnull_retval1(int *p) {
  // CHECK: [[ICMP:%.*]] = icmp ne i32* {{.*}}, null, !nosanitize
  // CHECK: br i1 [[ICMP]], {{.*}}, !nosanitize
  // CHECK: call void @__ubsan_handle_nullability_return{{.*}}[[NONNULL_RV_LOC1]]
  return p;
  // CHECK: ret i32*
}

#line 190
void nonnull_arg(int *_Nonnull p) {}

// CHECK-LABEL: define{{.*}} void @{{.*}}call_func_with_nonnull_arg
#line 200
void call_func_with_nonnull_arg(int *_Nonnull p) {
  // CHECK: [[ICMP:%.*]] = icmp ne i32* {{.*}}, null, !nosanitize
  // CHECK-NEXT: br i1 [[ICMP]], {{.*}}, !nosanitize
  // CHECK: call void @__ubsan_handle_nullability_arg{{.*}}[[NONNULL_ARG_LOC]]
  nonnull_arg(p);
}

// CHECK-LABEL: define{{.*}} void @{{.*}}nonnull_assign1
#line 300
void nonnull_assign1(int *p) {
  // CHECK: [[ICMP:%.*]] = icmp ne i32* {{.*}}, null, !nosanitize
  // CHECK-NEXT: br i1 [[ICMP]], {{.*}}, !nosanitize
  // CHECK: call void @__ubsan_handle_type_mismatch{{.*}}[[NONNULL_ASSIGN1_LOC]]
  int *_Nonnull local;
  local = p;
}

// CHECK-LABEL: define{{.*}} void @{{.*}}nonnull_assign2
#line 400
void nonnull_assign2(int *p) {
  // CHECK: [[ICMP:%.*]] = icmp ne i32* %{{.*}}, null, !nosanitize
  // CHECK-NEXT: br i1 [[ICMP]], {{.*}}, !nosanitize
  // CHECK: call void @__ubsan_handle_type_mismatch{{.*}}[[NONNULL_ASSIGN2_LOC]]
  int *_Nonnull arr[1];
  arr[0] = p;
}

struct S1 {
  int *_Nonnull mptr;
};

// CHECK-LABEL: define{{.*}} void @{{.*}}nonnull_assign3
#line 500
void nonnull_assign3(int *p) {
  // CHECK: [[ICMP:%.*]] = icmp ne i32* %{{.*}}, null, !nosanitize
  // CHECK-NEXT: br i1 [[ICMP]], {{.*}}, !nosanitize
  // CHECK: call void @__ubsan_handle_type_mismatch{{.*}}[[NONNULL_ASSIGN3_LOC]]
  // CHECK-NOT: call void @__ubsan_handle_type_mismatch
  struct S1 s;
  s.mptr = p;
}

// CHECK-LABEL: define{{.*}} void @{{.*}}nonnull_init1
#line 600
void nonnull_init1(int *p) {
  // CHECK: [[ICMP:%.*]] = icmp ne i32* %{{.*}}, null, !nosanitize
  // CHECK-NEXT: br i1 [[ICMP]], {{.*}}, !nosanitize
  // CHECK: call void @__ubsan_handle_type_mismatch{{.*}}[[NONNULL_INIT1_LOC]]
  int *_Nonnull local = p;
}

// CHECK-LABEL: define{{.*}} void @{{.*}}nonnull_init2
#line 700
void nonnull_init2(int *p) {
  // CHECK: [[ICMP:%.*]] = icmp ne i32* %{{.*}}, null, !nosanitize
  // CHECK-NEXT: br i1 [[ICMP]], {{.*}}, !nosanitize
  // CHECK: call void @__ubsan_handle_type_mismatch{{.*}}[[NONNULL_INIT2_LOC1]]
  // CHECK: [[ICMP:%.*]] = icmp ne i32* %{{.*}}, null, !nosanitize
  // CHECK-NEXT: br i1 [[ICMP]], {{.*}}, !nosanitize
  // CHECK: call void @__ubsan_handle_type_mismatch{{.*}}[[NONNULL_INIT2_LOC2]]
  int *_Nonnull arr[] = {p, p};
}

// CHECK-LABEL: define{{.*}} i32* @{{.*}}nonnull_retval2
#line 800
int *_Nonnull nonnull_retval2(int *_Nonnull arg1,  //< Test this.
                              int *_Nonnull arg2,  //< Test this.
                              int *_Nullable arg3, //< Don't test the rest.
                              int *arg4,
                              int arg5, ...) {
  // CHECK: [[ARG1CMP:%.*]] = icmp ne i32* %arg1, null, !nosanitize
  // CHECK-NEXT: [[DO_RV_CHECK_1:%.*]] = and i1 true, [[ARG1CMP]], !nosanitize
  // CHECK: [[ARG2CMP:%.*]] = icmp ne i32* %arg2, null, !nosanitize
  // CHECK-NEXT: [[DO_RV_CHECK_2:%.*]] = and i1 [[DO_RV_CHECK_1]], [[ARG2CMP]]
  // CHECK: [[SLOC_PTR:%.*]] = load i8*, i8** %return.sloc.ptr
  // CHECK-NEXT: [[SLOC_NONNULL:%.*]] = icmp ne i8* [[SLOC_PTR]], null
  // CHECK-NEXT: [[DO_RV_CHECK_3:%.*]] = and i1 [[SLOC_NONNULL]], [[DO_RV_CHECK_2]]
  // CHECK: br i1 [[DO_RV_CHECK_3]], label %[[NULL:.*]], label %[[NONULL:.*]], !nosanitize
  // CHECK: [[NULL]]:
  // CHECK-NEXT: [[ICMP:%.*]] = icmp ne i32* {{.*}}, null, !nosanitize
  // CHECK: br i1 [[ICMP]], {{.*}}, !nosanitize
  // CHECK: call void @__ubsan_handle_nullability_return{{.*}}[[NONNULL_RV_LOC2]]
  return arg1;
  // CHECK: [[NONULL]]:
  // CHECK-NEXT: ret i32*
}

@interface A
+(int *_Nonnull) objc_clsmethod: (int *_Nonnull) arg1;
-(int *_Nonnull) objc_method: (int *_Nonnull) arg1;
@end

@implementation A

// CHECK-LABEL: define internal i32* @"\01+[A objc_clsmethod:]"
+(int *_Nonnull) objc_clsmethod: (int *_Nonnull) arg1 {
  // CHECK: [[ARG1CMP:%.*]] = icmp ne i32* %arg1, null, !nosanitize
  // CHECK-NEXT: [[DO_RV_CHECK:%.*]] = and i1 true, [[ARG1CMP]]
  // CHECK: [[SLOC_PTR:%.*]] = load i8*, i8** %return.sloc.ptr
  // CHECK-NEXT: [[SLOC_NONNULL:%.*]] = icmp ne i8* [[SLOC_PTR]], null
  // CHECK-NEXT: [[DO_RV_CHECK_2:%.*]] = and i1 [[SLOC_NONNULL]], [[DO_RV_CHECK]]
  // CHECK: br i1 [[DO_RV_CHECK_2]], label %[[NULL:.*]], label %[[NONULL:.*]], !nosanitize
  // CHECK: [[NULL]]:
  // CHECK-NEXT: [[ICMP:%.*]] = icmp ne i32* {{.*}}, null, !nosanitize
  // CHECK: br i1 [[ICMP]], {{.*}}, !nosanitize
  // CHECK: call void @__ubsan_handle_nullability_return{{.*}}
  return arg1;
  // CHECK: [[NONULL]]:
  // CHECK-NEXT: ret i32*
}

// CHECK-LABEL: define internal i32* @"\01-[A objc_method:]"
-(int *_Nonnull) objc_method: (int *_Nonnull) arg1 {
  // CHECK: [[ARG1CMP:%.*]] = icmp ne i32* %arg1, null, !nosanitize
  // CHECK-NEXT: [[DO_RV_CHECK:%.*]] = and i1 true, [[ARG1CMP]]
  // CHECK: [[SLOC_PTR:%.*]] = load i8*, i8** %return.sloc.ptr
  // CHECK-NEXT: [[SLOC_NONNULL:%.*]] = icmp ne i8* [[SLOC_PTR]], null
  // CHECK-NEXT: [[DO_RV_CHECK_2:%.*]] = and i1 [[SLOC_NONNULL]], [[DO_RV_CHECK]]
  // CHECK: br i1 [[DO_RV_CHECK_2]], label %[[NULL:.*]], label %[[NONULL:.*]], !nosanitize
  // CHECK: [[NULL]]:
  // CHECK-NEXT: [[ICMP:%.*]] = icmp ne i32* {{.*}}, null, !nosanitize
  // CHECK: br i1 [[ICMP]], {{.*}}, !nosanitize
  // CHECK: call void @__ubsan_handle_nullability_return{{.*}}
  return arg1;
  // CHECK: [[NONULL]]:
  // CHECK-NEXT: ret i32*
}
@end

// CHECK-LABEL: define{{.*}} void @{{.*}}call_A
void call_A(A *a, int *p) {
  // CHECK: [[ICMP:%.*]] = icmp ne i32* [[P1:%.*]], null, !nosanitize
  // CHECK-NEXT: br i1 [[ICMP]], {{.*}}, !nosanitize
  // CHECK: call void @__ubsan_handle_nullability_arg{{.*}} !nosanitize
  // CHECK: call i32* {{.*}} @objc_msgSend to i32* {{.*}}({{.*}}, i32* [[P1]])
  [a objc_method: p];

  // CHECK: [[ICMP:%.*]] = icmp ne i32* [[P2:%.*]], null, !nosanitize
  // CHECK-NEXT: br i1 [[ICMP]], {{.*}}, !nosanitize
  // CHECK: call void @__ubsan_handle_nullability_arg{{.*}} !nosanitize
  // CHECK: call i32* {{.*}} @objc_msgSend to i32* {{.*}}({{.*}}, i32* [[P2]])
  [A objc_clsmethod: p];
}

void dont_crash(int *_Nonnull p, ...) {}

@protocol NSObject
- (id)init;
@end
@interface NSObject <NSObject> {}
@end

#pragma clang assume_nonnull begin

/// Create a "NSObject * _Nonnull" instance.
NSObject *get_nonnull_error() {
  // Use nil for convenience. The actual object doesn't matter.
  return (NSObject *)NULL;
}

NSObject *_Nullable no_null_return_value_diagnostic(int flag) {
// CHECK-LABEL: define internal {{.*}}no_null_return_value_diagnostic{{i?}}_block_invoke
// CHECK-NOT: @__ubsan_handle_nullability_return
  NSObject *_Nullable (^foo)() = ^() {
    if (flag) {
      // Clang should not infer a nonnull return value for this block when this
      // call is present.
      return get_nonnull_error();
    } else {
      return (NSObject *)NULL;
    }
  };
  return foo();
}

#pragma clang assume_nonnull end

int main() {
  nonnull_retval1(INULL);
  nonnull_retval2(INNULL, INNULL, INULL, (int *_Nullable)NULL, 0, 0, 0, 0);
  call_func_with_nonnull_arg(INNULL);
  nonnull_assign1(INULL);
  nonnull_assign2(INULL);
  nonnull_assign3(INULL);
  nonnull_init1(INULL);
  nonnull_init2(INULL);
  call_A((A *)NULL, INULL);
  dont_crash(INNULL, NULL);
  no_null_return_value_diagnostic(0);
  no_null_return_value_diagnostic(1);
  return 0;
}
