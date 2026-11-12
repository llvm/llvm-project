// RUN: %clang_cc1 -E %s -o %t.m
// RUN: %clang_cc1 -fobjc-runtime=macosx-fragile-10.5 -rewrite-objc %t.m -o - | FileCheck %s

@interface I {
  id _delegate;
}
-(void)foo;
@end

@implementation I

static void KKKK(int w);

-(void) foo {
  KKKK(0);
}

static void KKKK(int w) {
  I *self = (I *)0;
  if ([self->_delegate respondsToSelector:@selector(handlePortMessage:)]) {
  }
}

-(void) foo2 {
  KKKK(0);
}

@end

// CHECK: if (((id (*)(id, SEL, ...))(void *)objc_msgSend)((id)((struct I_IMPL *)self)->_delegate, sel_registerName("respondsToSelector:"), sel_registerName("handlePortMessage:")))
