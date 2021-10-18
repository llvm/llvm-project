// RUN: %clang_cc1 -x objective-c -emit-llvm -triple x86_64-apple-macosx10.10.0 -fsanitize=nonnull-attribute %s -o - -w | FileCheck %s

@interface A

-(void) one_arg: (__attribute__((nonnull)) int *) arg1;

-(void) varargs: (__attribute__((nonnull)) int *) arg1, ...;

+(void) clsmethod: (__attribute__((nonnull)) int *) arg1;

@end

@implementation A

// CHECK-LABEL: define internal void @"\01-[A one_arg:]"
// CHECK-SAME: i32* nonnull
-(void) one_arg: (__attribute__((nonnull)) int *) arg1 {}

// CHECK-LABEL: define internal void @"\01-[A varargs:]"
// CHECK-SAME: i32* nonnull
-(void) varargs: (__attribute__((nonnull)) int *) arg1, ... {}

// CHECK-LABEL: define internal void @"\01+[A clsmethod:]"
// CHECK-SAME: i32* nonnull
+(void) clsmethod: (__attribute__((nonnull)) int *) arg1 {}

@end

// CHECK-LABEL: define{{.*}} void @call_A
void call_A(A *a, int *p) {
  // CHECK: [[ICMP:%.*]] = icmp ne i32* [[P1:%.*]], null, !nosanitize
  // CHECK: br i1 [[ICMP]], {{.*}}, !nosanitize
  // CHECK: call void @__ubsan_handle_nonnull_arg{{.*}} !nosanitize
  // CHECK: call void {{.*}} @objc_msgSend {{.*}} ({{.*}}, i32* [[P1]])
  [a one_arg: p];

  // CHECK: [[ICMP:%.*]] = icmp ne i32* [[P2:%.*]], null, !nosanitize
  // CHECK: br i1 [[ICMP]], {{.*}}, !nosanitize
  // CHECK: call void @__ubsan_handle_nonnull_arg{{.*}} !nosanitize
  // CHECK: call void {{.*}} @objc_msgSend {{.*}} ({{.*}}, i32* [[P2]], {{.*}})
  [a varargs: p, p];

  // CHECK: [[ICMP:%.*]] = icmp ne i32* [[P3:%.*]], null, !nosanitize
  // CHECK: br i1 [[ICMP]], {{.*}}, !nosanitize
  // CHECK: call void @__ubsan_handle_nonnull_arg{{.*}} !nosanitize
  // CHECK: call void {{.*}} @objc_msgSend {{.*}} ({{.*}}, i32* [[P3]])
  [A clsmethod: p];
}
