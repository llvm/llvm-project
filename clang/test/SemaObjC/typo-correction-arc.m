// RUN: %clang_cc1 -triple i386-apple-macosx10.10 -fobjc-arc -fsyntax-only -Wno-objc-root-class %s -verify

typedef unsigned long NSUInteger;

id nameless;                                  // expected-note{{'nameless' declared here}}

@interface NSArray
- (instancetype)initWithObjects:(const id[])objects count:(NSUInteger)count;
@end

@interface I
@property NSArray *array;
- (id)getArrayById:(id)name;
- (void)setArrayValue:(id)array;
@end

@interface J
- (void)setArray:(id)array;
- (void)setIvarArray;
@end

@implementation J {
  I *i;
}
- (void)setArray:(id)array {  // expected-note{{'array' declared here}}
  i.array = aray;             // expected-error{{use of undeclared identifier 'aray'; did you mean 'array'}}
}
- (void)setIvarArray {
  [i setArrayValue:[i getArrayById:nameles]]; // expected-error{{use of undeclared identifier 'nameles'; did you mean 'nameless'}}
}
@end

