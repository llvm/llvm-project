// RUN: %clang_cc1 -no-opaque-pointers -fsanitize=nullability-return -emit-llvm %s -o - -triple x86_64-apple-macosx10.10.0 -Wno-objc-root-class | FileCheck %s

// CHECK-LABEL: define internal i8* @"\01-[I init]"
// CHECK: unreachable
// CHECK-NEXT: }

#pragma clang assume_nonnull begin
@interface I
- (instancetype)init __attribute__((unavailable));
@end
@implementation I
- (instancetype)init __attribute__((unavailable)) { __builtin_unreachable(); }
@end
#pragma clang assume_nonnull end
