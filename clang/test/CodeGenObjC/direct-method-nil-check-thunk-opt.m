// This file checks that certain nil checks can be removed from direct method calls.

// RUN: %clang_cc1 -fobjc-export-direct-methods -O0 -emit-llvm -fobjc-arc -triple arm64-apple-darwin %s -o - | FileCheck %s
// RUN: %clang_cc1 -fobjc-export-direct-methods -O2 -emit-llvm -fobjc-arc -triple arm64-apple-darwin %s -o - | FileCheck %s --check-prefixes=OPT

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
// OPT-LABEL: define dso_local i32 @"-<Fib fibWithN:>_inner"
// OPT: {{.*}} tail call i32 @"-<Fib fibWithN:>_inner"

// The inner function knows that self is non null so it can call the method without the nil check.
// CHECK-LABEL: define dso_local i32 @"-<Fib fibWithN:>_inner"
// CHECK: {{.*}} call i32 @"-<Fib fibWithN:>_inner"
// CHECK: {{.*}} call i32 @"-<Fib fibWithN:>_inner"

// Thunk function calls the inner function as usual.
// CHECK-LABEL: define dso_local i32 @"-<Fib fibWithN:>"
// CHECK: {{.*}} call i32 @"-<Fib fibWithN:>_inner"
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
  // CHECK-LABEL: define dso_local i32 @"-<SubRoot calculateWithPtr:>_inner"
  // CHECK: {{.*}} = call i32 @"-<SubRoot val>_inner"
  // CHECK: call void @"-<SubRoot setVal:>_inner"
  // CHECK: {{.*}} = call i32 @"-<Root privateLoadWithOffset:>_inner"
  // CHECK: {{.*}} = call i32 @"-<SubRoot privateMethod:>_inner"
  // CHECK: {{.*}} = call i32 @"-<Root idx>_inner"
  // CHECK: call void @"-<Root setIdx:>_inner"
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
// CHECK-LABEL: declare i32 @"-<Root privateLoadWithOffset:>_inner"(ptr, ptr)
// CHECK-LABEL: declare i32 @"-<SubRoot privateMethod:>_inner"(ptr, i32)
// CHECK-LABEL: declare i32 @"-<Root idx>_inner"(ptr)
// CHECK-LABEL: declare void @"-<Root setIdx:>_inner"(ptr, i32)
