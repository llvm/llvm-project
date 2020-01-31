// Objective-C recovery
// RUN: not %clang_cc1 -triple x86_64-apple-darwin10 -fdiagnostics-parseable-fixits -x objective-c %s 2>&1 | FileCheck -check-prefix=CHECK-MRR %s
// RUN: not %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-arc -fdiagnostics-parseable-fixits -x objective-c %s 2>&1 | FileCheck -check-prefix=CHECK-ARC %s

__attribute__((objc_root_class))
@interface Root
+ (void)classDirectMethod __attribute__((objc_direct));
+ (void)classDirectMethod2 __attribute__((objc_direct));
- (void)instanceDirectMethod __attribute__((objc_direct));
@end

@interface A : Root
@end

@implementation A
+ (void)classMethod {
  // CHECK-MRR: {18:4-18:8}:"Root"
  [self classDirectMethod];
}
+ (void)classMethod2 {
  // CHECK-MRR: {23:4-23:9}:"Root"
  // CHECK-ARC: {23:4-23:9}:"self"
  [super classDirectMethod2];
}
- (void)instanceMethod {
  // CHECK-MRR: {28:4-28:9}:"self"
  // CHECK-ARC: {28:4-28:9}:"self"
  [super instanceDirectMethod];
}
@end
