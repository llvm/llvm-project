// RUN: %clang_cc1 -emit-llvm %s -debug-info-kind=limited -o - | FileCheck %s
// CHECK-NOT: DW_TAG_member
// Interface P should not be a member of interface I in debug info.
@interface P 
@end

@interface I : P 
@end

void fn(I *iptr) {}
