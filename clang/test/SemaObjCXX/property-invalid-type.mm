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

@interface J
@property (nonnull) auto a; // expected-error {{'auto' not allowed in interface member}}
@property auto b; // expected-error {{'auto' not allowed in interface member}}
@property (nullable) auto c; // expected-error {{'auto' not allowed in interface member}}
@end

@interface J (Cat)
@property (nonnull) auto catprop; // expected-error {{'auto' not allowed in interface member}}
@end
