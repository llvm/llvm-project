// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fobjc-arc -fobjc-runtime-has-weak %s -emit-llvm -o - | FileCheck %s

// CHECK-DAG: llvm.objc.destroyWeak(ptr %self_weak_s_w_s)
// CHECK-DAG: llvm.objc.storeStrong(ptr %self_strong_w_s, ptr null)
// CHECK-DAG: llvm.objc.destroyWeak(ptr %self_weak_s)
// CHECK-DAG: llvm.objc.destroyWeak(ptr %self_weak_s3)
// CHECK-DAG: llvm.objc.storeStrong(ptr %self_strong3, ptr null)
// CHECK-DAG: llvm.objc.storeStrong(ptr %self_strong2, ptr null)
// CHECK-DAG: llvm.objc.storeStrong(ptr %self_strong, ptr null)
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
