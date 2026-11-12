@interface I
-(void)foo;
@end

struct S {
  int x,y;
};

@implementation I
-(void) foo {
  struct S s;
  if (1) {
    s.
}
@end

// RUN: c-index-test -code-completion-at=%s:13:7 -fobjc-nonfragile-abi %s | FileCheck %s
// CHECK: FieldDecl:{ResultType int}{TypedText x}
// CHECK: FieldDecl:{ResultType int}{TypedText y}
