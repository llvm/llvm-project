// RUN: %clang_cc1  -fsyntax-only -Wcustom-atomic-properties -verify -Wno-objc-root-class %s

@interface Foo
@property (assign) Foo *myProp; // expected-note {{property declared here}} expected-note {{property declared here}}
@end

@implementation Foo
 -(Foo*)myProp {return 0;} // expected-warning {{atomic by default property 'myProp' has a user defined getter (property should be marked 'atomic' if this is intended)}}
 -(void)setMyProp:(Foo*)e {} // expected-warning {{atomic by default property 'myProp' has a user defined setter (property should be marked 'atomic' if this is intended)}}
@end

@interface Foo2 {
  Foo *myProp;
}
@property (assign) Foo *myProp;
@end

@implementation Foo2
@synthesize myProp; // no warnings.
@end
