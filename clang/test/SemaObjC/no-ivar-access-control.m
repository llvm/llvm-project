// RUN: %clang_cc1 -fsyntax-only -fdebugger-support -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -fdebugger-support -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics

@interface I
{
@private
int ivar;
}
@end

@implementation I
- (int) meth {
  return self->ivar;
}
int foo1(I* p) {
  return p->ivar;
}
@end

int foo(I* p) {
  return p->ivar;
}

@interface B 
@end

@implementation B 
- (int) meth : (I*) arg {
  return arg->ivar;
}
@end


@interface I1 {
 int protected_ivar;
}
@property int PROP_INMAIN;
@end

@interface I1() {
 int private_ivar;
}
@property int PROP_INCLASSEXT;
@end

@implementation I1
@synthesize PROP_INMAIN, PROP_INCLASSEXT;

- (int) Meth {
   PROP_INMAIN = 1;
   PROP_INCLASSEXT = 2;
   protected_ivar = 1;  // OK
   return private_ivar; // OK
}
@end


@interface DER : I1
@end

@implementation DER
- (int) Meth {
   protected_ivar = 1;  // OK
   PROP_INMAIN = 1;
   PROP_INCLASSEXT = 2; 
   return private_ivar;
}
@end

