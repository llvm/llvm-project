// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface MyClass {
  int prop;
};
@property unsigned char bufferedUTF8Bytes[4];  // expected-error {{property cannot have array or function type}}
@property unsigned char bufferedUTFBytes:1;    // expected-error {{property name cannot be a bit-field}}
@property(nonatomic, retain, setter=ab_setDefaultToolbarItems) MyClass *ab_defaultToolbarItems; // expected-error {{method name referenced in property setter attribute must end with ':'}}

@property int prop;
@end

@implementation MyClass
@dynamic ab_defaultToolbarItems // expected-error{{expected ';' after @dynamic}}
@synthesize prop // expected-error{{expected ';' after @synthesize}}
@end

