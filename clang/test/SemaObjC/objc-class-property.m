// RUN: %clang_cc1 -fsyntax-only -verify %s

#if !__has_feature(objc_class_property)
#error does not support class property
#endif

@interface Root
-(id) alloc;
-(id) init;
@end

@interface A : Root {
  int x;
  int z;
}
@property int x;
@property int y;
@property int z;
@property(readonly) int ro, ro2;
@property (class) int c;
@property (class) int c2; // expected-note {{property declared here}} \
                          // expected-note {{property declared here}}
@property (class) int x;
@property (class, setter=customSet:) int customSetterProperty;
@property (class, getter=customGet) int customGetterProperty;
@end

@implementation A // expected-warning {{class property 'c2' requires method 'c2' to be defined}} \
                  // expected-warning {{class property 'c2' requires method 'setC2:' to be defined}}
@dynamic x; // refers to the instance property
@dynamic (class) x; // refers to the class property
@synthesize z, c2; // expected-error {{@synthesize not allowed on a class property 'c2'}}
@dynamic c; // refers to the class property
@dynamic customSetterProperty;
@dynamic customGetterProperty;
@end

int test(void) {
  A *a = [[A alloc] init];
  a.c; // expected-error {{property 'c' is a class property; did you mean to access it with class 'A'}}
  return a.x + A.c;
}

void customSelectors(void) {
  A.customSetterProperty = 1;
  (void)A.customGetterProperty;
}

void message_id(id me) {
  [me y];
}

void message_class(Class me) {
  [me c2];
}

@interface NSObject
@end

@interface MyClass : NSObject
@property(class, readonly) int classProp; // expected-note {{property declared here}}
@end

@implementation MyClass // expected-warning {{class property 'classProp' requires method 'classProp' to be defined}}
- (int)classProp { // Oops, mistakenly made this an instance method.
  return 8;
}
@end
