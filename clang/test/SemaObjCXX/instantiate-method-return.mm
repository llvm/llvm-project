// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// PR7386

@class NSObject;

class A; // expected-note {{forward declaration of 'A'}}
template<class T> class V { T x; }; // expected-error {{field has incomplete type 'A'}}

@protocol Protocol
- (V<A*>)protocolMethod;
- (V<A>)method2;
@end


@interface I<Protocol>
@end


@implementation I
- (void)randomMethod:(id)info {
  V<A*> vec([self protocolMethod]);
}

- (V<A*>)protocolMethod {
  V<A*> va; return va;
}
- (V<A>)method2 { // expected-note {{in instantiation of}}
}
@end
