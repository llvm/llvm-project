// RUN: %clang_cc1 -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s
// Do not emit AT_MIPS_linkage_name for static variable i

// CHECK: !DIGlobalVariable(name: "i"
// CHECK-NOT:               linkageName:
// CHECK-SAME:              ){{$}}

@interface A {
}
-(void) foo;
@end

@implementation A 
-(void)foo {
  static int i = 1;
}
@end

