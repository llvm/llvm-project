/// -fstrict-return is the default.
// RUN: %clang_cc1 -Wno-error=return-type -emit-llvm -fblocks -triple x86_64-apple-darwin -o - %s | FileCheck %s
// RUN: %clang_cc1 -Wno-error=return-type -emit-llvm -fblocks -triple x86_64-apple-darwin -O -o - %s | FileCheck %s

@interface I
@end

@implementation I

- (int)method {
}

@end

// Ensure that methods don't use the -fstrict-return undefined behaviour optimization.

// CHECK-NOT: call void @llvm.trap
// CHECK-NOT: unreachable
