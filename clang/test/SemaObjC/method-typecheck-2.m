// RUN: %clang_cc1 -fsyntax-only -verify %s

@protocol P
- (void) doSomethingInProtocol: (float) x; // expected-note {{previous definition is here}}
+ (void) doSomethingClassyInProtocol: (float) x; // expected-note {{previous definition is here}}
- (void) doNothingInProtocol : (float) x;
+ (void) doNothingClassyInProtocol : (float) x;
@end

@interface I <P>
- (void) doSomething: (float) x; // expected-note {{previous definition is here}}
+ (void) doSomethingClassy: (int) x; // expected-note {{previous definition is here}}
@end

@interface Bar : I
@end

@implementation Bar
- (void) doSomething: (int) x {} // expected-warning {{conflicting parameter types}}
+ (void) doSomethingClassy: (float) x{}  // expected-warning {{conflicting parameter types}}
- (void) doSomethingInProtocol: (id) x {}  // expected-warning {{conflicting parameter types}}
+ (void) doSomethingClassyInProtocol: (id) x {}  // expected-warning {{conflicting parameter types}}
@end


