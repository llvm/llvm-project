// RUN: %clang_cc1 -triple x86_64-apple-macosx10.10 %s -S -emit-llvm -o - | FileCheck %s

// CHECK: @llvm.used =
// CHECK-SAME: @"\01-[X m]"

// CHECK: define internal void @"\01-[X m]"(

@interface X @end
@implementation X
-(void) m __attribute__((used)) {}
@end
