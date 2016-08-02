// RUN: %clang_cc1 -triple x86_64-apple-macosx10.8.0 -fsyntax-only -verify %s

// This test case shows that 'availability' and 'deprecated' do not inherit
// when a property is redeclared in a subclass.  This is intentional.

@interface NSObject @end
@protocol myProtocol
@property int myProtocolProperty __attribute__((availability(macosx,introduced=10.7,deprecated=10.8))); // expected-note {{'myProtocolProperty' has been explicitly marked deprecated here}} \
                                                                                                        // expected-note {{property 'myProtocolProperty' is declared deprecated here}}
@end

@interface Foo : NSObject
@property int myProperty __attribute__((availability(macosx,introduced=10.7,deprecated=10.8)));  // expected-note 2 {{'myProperty' has been explicitly marked deprecated here}} \
								// expected-note {{property 'myProperty' is declared deprecated here}}
@end

@interface Bar : Foo <myProtocol>
@property int myProperty;
@property int myProtocolProperty;
@end

void test(Foo *y, Bar *x, id<myProtocol> z) {
  y.myProperty = 0; // expected-warning {{'myProperty' is deprecated: first deprecated in OS X 10.8}}
  (void)[y myProperty];   // expected-warning {{'myProperty' is deprecated: first deprecated in OS X 10.8}} 

  x.myProperty = 1; // no-warning
  (void)[x myProperty]; // no-warning

  x.myProtocolProperty = 0; // no-warning

  (void)[x myProtocolProperty]; // no-warning
  (void)[z myProtocolProperty]; // expected-warning {{'myProtocolProperty' is deprecated: first deprecated in OS X 10.8}}
}
