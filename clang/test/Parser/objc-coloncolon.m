// RUN: %clang_cc1 -x objective-c -fsyntax-only -Wno-objc-root-class -verify %s

int GV = 42;

@interface A
+ (int) getGV;
- (instancetype)init:(::A *) foo; // expected-error {{expected a type}}
@end

@implementation A
- (void)performSelector:(SEL)selector {}
- (void)double:(int)firstArg :(int)secondArg colon:(int)thirdArg {}
- (void)test {
  // The `::` below should not trigger an error.
  [self performSelector:@selector(double::colon:)];
}
+ (int) getGV { return ::GV; } // expected-error {{expected a type}}
- (instancetype)init:(::A *) foo { return self; } // expected-error {{expected a type}}
@end
