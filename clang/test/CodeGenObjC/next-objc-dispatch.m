// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - %s \
// RUN:   -fobjc-dispatch-method=legacy | \
// RUN:   FileCheck -check-prefix CHECK-FRAGILE_LEGACY %s
//
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -emit-llvm -o - %s    \
// RUN:   -fobjc-dispatch-method=legacy | \
// RUN:   FileCheck -check-prefix CHECK-NONFRAGILE_LEGACY %s
//
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -emit-llvm -o - %s    \
// RUN:   -fobjc-dispatch-method=non-legacy | \
// RUN:   FileCheck -check-prefix CHECK-NONFRAGILE_NONLEGACY %s
//
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -emit-llvm -o - %s    \
// RUN:   -fobjc-dispatch-method=mixed | \
// RUN:   FileCheck -check-prefix CHECK-NONFRAGILE_MIXED %s

// There are basically four ways that we end up doing message dispatch for the
// NeXT runtime. They are:
//  (1) fragile ABI, legacy dispatch
//  (2) non-fragile ABI, legacy dispatch
//  (2) non-fragile ABI, non-legacy dispatch
//  (2) non-fragile ABI, mixed dispatch
//
// Note that fragile ABI and non-fragile ABI legacy dispatch are not the same,
// they use some different API calls (objc_msgSendSuper vs objc_msgSendSuper2).

// CHECK-FRAGILE_LEGACY: ModuleID
// CHECK-FRAGILE_LEGACY-NOT: declare ptr @objc_msgSendSuper2_fixup(
// CHECK-FRAGILE_LEGACY-NOT: declare ptr @objc_msgSend_fixup(
// CHECK-FRAGILE_LEGACY: declare ptr @objc_msgSendSuper(
// CHECK-FRAGILE_LEGACY: declare ptr @objc_msgSend(

// CHECK-NONFRAGILE_LEGACY: ModuleID
// CHECK-NONFRAGILE_LEGACY-NOT: declare ptr @objc_msgSendSuper2_fixup(
// CHECK-NONFRAGILE_LEGACY-NOT: declare ptr @objc_msgSend_fixup(
// CHECK-NONFRAGILE_LEGACY: declare ptr @objc_msgSendSuper2(
// CHECK-NONFRAGILE_LEGACY: declare ptr @objc_msgSend(

// CHECK-NONFRAGILE_NONLEGACY: ModuleID
// CHECK-NONFRAGILE_NONLEGACY: declare ptr @objc_msgSendSuper2_fixup(
// CHECK-NONFRAGILE_NONLEGACY: declare ptr @objc_msgSend_fixup(

// CHECK-NONFRAGILE_MIXED: declare ptr @objc_msgSendSuper2_fixup(
// CHECK-NONFRAGILE_MIXED: declare ptr @objc_msgSendSuper2(
// CHECK-NONFRAGILE_MIXED: declare ptr @objc_msgSend_fixup(
// CHECK-NONFRAGILE_MIXED: declare ptr @objc_msgSend(

@interface NSObject
+ (id)alloc;
- (id)init;
@end

@interface I0 : NSObject
-(void) im0;
@end

@implementation I0
+(id) alloc {
  return [super alloc];
}
-(id) init {
 [super init];
 return self;
}
-(void) im0 {}
@end

void f0(I0 *a) {
  [I0 alloc];
  [a im0];
}
