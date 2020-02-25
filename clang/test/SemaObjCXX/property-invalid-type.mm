// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface I
{
  A* response; // expected-error {{unknown type name 'A'}}
}
@end
@interface I ()
@property A* response;  // expected-error {{unknown type name 'A'}}
@property  int helper;
@end
@implementation I
@synthesize response;
- (void) foo :(A*) a   // expected-error {{expected a type}}
{
  self.response = a; // expected-error{{incompatible pointer types assigning to 'int *' from 'id'}}
}
@end

void foo(I *i)
{
  i.helper; // expected-warning{{property access result unused - getters should not be used for side effects}}
}
