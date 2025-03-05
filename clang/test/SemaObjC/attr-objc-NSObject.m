// RUN: %clang_cc1 -verify -Wno-objc-root-class -fsyntax-only %s

@interface NSArray<__covariant ObjectType>
- (void)containsObject:(ObjectType)anObject; // expected-note {{passing argument to parameter 'anObject' here}}
- (void)description;
@end

typedef __attribute__((NSObject)) struct Foo *FooRef;
typedef struct Bar *BarRef;

void good() {
  FooRef object;
  NSArray<FooRef> *array;
  [array containsObject:object];
  [object description];
}

void bad() {
  BarRef object;
  NSArray<BarRef> *array; // expected-error {{type argument 'BarRef' (aka 'struct Bar *') is neither an Objective-C object nor a block type}}
  [array containsObject:object]; // expected-warning {{incompatible pointer types sending 'BarRef' (aka 'struct Bar *') to parameter of type 'id'}}
  [object description]; // expected-warning {{receiver type 'BarRef' (aka 'struct Bar *') is not 'id' or interface pointer, consider casting it to 'id'}}
}
