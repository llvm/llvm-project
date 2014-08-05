// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface I
{
  A* response; // expected-error {{unknown type name 'A'}}
}
@end
@interface I ()
@property A* response;  // expected-error {{unknown type name 'A'}}
@end
@implementation I
@synthesize response;
- (void) foo :(A*) a   // expected-error {{expected a type}}
{
  self.response = a;
}
@end


