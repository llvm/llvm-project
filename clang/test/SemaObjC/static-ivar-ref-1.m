// RUN: %clang_cc1 -triple i386-unknown-unknown -ast-print %s 2>&1 | FileCheck  %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10  -ast-print %s  2>&1  | FileCheck %s

@interface current 
{
@public
  int ivar;
  int ivar1;
  int ivar2;
}
@end

current *pc;

int foo(void)
{
  return pc->ivar2 + (*pc).ivar + pc->ivar1;
}

// CHECK: @interface current{
// CHECK:     int ivar;
// CHECK:     int ivar1;
// CHECK:     int ivar2;
// CHECK: }
// CHECK: @end
// CHECK: current *pc;
// CHECK: int foo(void) {
// CHECK:     return pc->ivar2 + (*pc).ivar + pc->ivar1;
// CHECK: }

