@implementation I
-(void)meth {
  int x;
  if (x) {
}
@end

// RUN: c-index-test -cursor-at=%s:3:7 %s > %t
// RUN: FileCheck %s -input-file %t

// CHECK: VarDecl=x:3:7
