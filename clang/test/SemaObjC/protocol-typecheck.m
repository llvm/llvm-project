// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface NSObject @end
@protocol XCElementP @end
@protocol XCElementSpacerP <XCElementP>  @end

@protocol PWhatever @end

@interface XX

- (void)setFlexElement:(NSObject <PWhatever, XCElementP> *)flexer;
- (void)setFlexElement2:(NSObject <PWhatever, XCElementSpacerP> *)flexer; // expected-note{{passing argument to parameter 'flexer' here}}

@end

void func(void) {
  NSObject <PWhatever, XCElementSpacerP> * flexer;
  NSObject <PWhatever, XCElementP> * flexer2;
  XX *obj;
  [obj setFlexElement:flexer];
  // FIXME: GCC provides the following diagnostic (which is much better):
  // protocol-typecheck.m:21: warning: class 'NSObject <PWhatever, XCElementP>' does not implement the 'XCElementSpacerP' protocol
  [obj setFlexElement2:flexer2]; // expected-warning{{incompatible pointer types sending 'NSObject<PWhatever,XCElementP> *' to parameter of type 'NSObject<PWhatever,XCElementSpacerP> *'}}
}

