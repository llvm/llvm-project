// This file checks that certain nil checks can be removed from direct method calls.

// RUN: %clang_cc1 -fobjc-emit-nil-check-thunk -O0 -emit-llvm -fobjc-arc -triple arm64-apple-darwin %s -o - | FileCheck %s
// RUN: %clang_cc1 -fobjc-emit-nil-check-thunk -O2 -emit-llvm -fobjc-arc -triple arm64-apple-darwin %s -o - | FileCheck %s --check-prefixes=OPT

__attribute__((objc_root_class))
@interface Root
@property(direct) int idx;
- (int)privateLoadWithOffset:(int *)ptr __attribute__((objc_direct));
@end

// Optimization is enabled because the source code is available, and the compiler can reason that some methods don't require nil checks.
@interface Fib : Root
- (int)fibWithN:(int)n __attribute__((objc_direct));
@end

@implementation Fib
// With optimization, the inner function call should be a tail call.
// OPT-LABEL: define hidden i32 @"\01-[Fib fibWithN:]_nonnull"
// OPT: {{.*}} tail call i32 @"\01-[Fib fibWithN:]_nonnull"

// The inner function knows that self is non null so it can call the method without the nil check.
// CHECK-LABEL: define hidden i32 @"\01-[Fib fibWithN:]_nonnull"
// CHECK: {{.*}} call i32 @"\01-[Fib fibWithN:]_nonnull"
// CHECK: {{.*}} call i32 @"\01-[Fib fibWithN:]_nonnull"

// Thunk function calls the inner function as usual.
// CHECK-LABEL: define hidden i32 @"\01-[Fib fibWithN:]"
// CHECK: {{.*}} call i32 @"\01-[Fib fibWithN:]_nonnull"
- (int)fibWithN:(int)n {
  if (n <= 0) return 0;
  if (n == 1) return 1;
  return [self fibWithN:n-1] + [self fibWithN:n-2];
}
@end

@interface SubRoot : Root
@property(direct) int val;

- (int)calculateWithPtr:(int*)ptr __attribute__((objc_direct));
- (int)privateMethod:(int)n __attribute__((objc_direct));
@end
@implementation SubRoot
- (int)calculateWithPtr:(int*)ptr {
  // For inner functions, it is trivial to reason that the receiver `self` can't be null
  // CHECK-LABEL: define hidden i32 @"\01-[SubRoot calculateWithPtr:]_nonnull"
  // CHECK: {{.*}} = call i32 @"\01-[SubRoot val]_nonnull"
  // CHECK: call void @"\01-[SubRoot setVal:]_nonnull"
  // CHECK: {{.*}} = call i32 @"\01-[Root privateLoadWithOffset:]_nonnull"
  // CHECK: {{.*}} = call i32 @"\01-[SubRoot privateMethod:]_nonnull"
  // CHECK: {{.*}} = call i32 @"\01-[Root idx]_nonnull"
  // CHECK: call void @"\01-[Root setIdx:]_nonnull"
  int ret = [self val];
  [self setVal:*ptr];
  ret += [self privateLoadWithOffset:ptr];
  ret += [self privateMethod:ret];
  ret += [self idx];
  [self setIdx:ret];
  return ret;
}
@end

// The thunk declarations don't exist since all calls to them are non null.
// We trust that these symbols will be generated when the definition is available.
// CHECK-LABEL: declare i32 @"\01-[Root privateLoadWithOffset:]_nonnull"(ptr, ptr)
// CHECK-LABEL: declare i32 @"\01-[SubRoot privateMethod:]_nonnull"(ptr, i32)
// CHECK-LABEL: declare i32 @"\01-[Root idx]_nonnull"(ptr)
// CHECK-LABEL: declare void @"\01-[Root setIdx:]_nonnull"(ptr, i32)
