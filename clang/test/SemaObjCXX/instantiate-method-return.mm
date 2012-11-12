// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics
// PR7386

@class NSObject;

class A;
template<class T> class V {};

@protocol Protocol
- (V<A*>)protocolMethod;
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
@end
