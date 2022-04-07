// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin11 -fobjc-arc -fobjc-runtime-has-weak %s -emit-llvm -o - | FileCheck %s

// CHECK: bitcast {{.*}} %self_weak_s_w_s
// CHECK-NEXT: llvm.objc.destroyWeak
// CHECK-NEXT: bitcast {{.*}} %self_strong_w_s
// CHECK-NEXT: llvm.objc.storeStrong
// CHECK-NEXT: bitcast {{.*}} %self_weak_s
// CHECK-NEXT: llvm.objc.destroyWeak
// CHECK-NEXT: bitcast {{.*}} %self_weak_s3
// CHECK-NEXT: llvm.objc.destroyWeak
// CHECK-NEXT: bitcast {{.*}} %self_strong3
// CHECK-NEXT: llvm.objc.storeStrong
// CHECK-NEXT: bitcast {{.*}} %self_strong2
// CHECK-NEXT: llvm.objc.storeStrong
// CHECK-NEXT: bitcast {{.*}} %self_strong
// CHECK-NEXT: llvm.objc.storeStrong
@interface NSObject
@end
@interface A : NSObject
@end
@implementation A
- (void)test {
  __attribute__((objc_ownership(strong))) __typeof__(self) self_strong;
  __attribute__((objc_ownership(strong))) __typeof__(self_strong) self_strong2;
  __attribute__((objc_ownership(strong))) __typeof__(self_strong2) self_strong3;
  __attribute__((objc_ownership(weak))) __typeof__(self_strong3) self_weak_s3;
 
  __attribute__((objc_ownership(weak))) __typeof__(self_strong) self_weak_s;
  __attribute__((objc_ownership(strong))) __typeof__(self_weak_s) self_strong_w_s;
  __attribute__((objc_ownership(weak))) __typeof__(self_strong_w_s) self_weak_s_w_s;
}
@end
