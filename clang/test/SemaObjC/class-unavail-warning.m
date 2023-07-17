// RUN: %clang_cc1  -fsyntax-only -fblocks -triple x86_64-apple-darwin10 -verify %s
// rdar://9092208

__attribute__((unavailable("not available")))
@interface MyClass { // expected-note 7 {{'MyClass' has been explicitly marked unavailable here}}
@public
    void *_test;
    MyClass *ivar; // no error.
}

- (id)self;
- new;
+ (void)addObject:(id)anObject;
- (MyClass *)meth; // no error.

@end

@interface Gorf {
  MyClass *ivar; // expected-error {{unavailable}}
}
- (MyClass *)meth; // expected-error {{unavailable}}
@end

@interface MyClass (Cat1)
- (MyClass *)meth; // no error.
@end

@interface MyClass (Cat2) // no error.
@end

@implementation MyClass (Cat2) // no error.
@end

int main(void) {
 [MyClass new]; // expected-error {{'MyClass' is unavailable: not available}}
 [MyClass self]; // expected-error {{'MyClass' is unavailable: not available}}
 [MyClass addObject:((void *)0)]; // expected-error {{'MyClass' is unavailable: not available}}

 MyClass *foo = [MyClass new]; // expected-error 2 {{'MyClass' is unavailable: not available}}

 return 0;
}

// rdar://16681279
@interface NSObject @end

__attribute__((visibility("default"))) __attribute__((availability(macosx,unavailable)))
@interface Foo : NSObject @end // expected-note 3 {{'Foo' has been explicitly marked unavailable here}}
@interface AppDelegate  : NSObject
@end

@class Foo;

@implementation AppDelegate
- (void) applicationDidFinishLaunching
{
  Foo *foo = 0; // expected-error {{'Foo' is unavailable}}
}
@end

@class Foo;
Foo *g_foo = 0; // expected-error {{'Foo' is unavailable}}

@class Foo;
@class Foo;
@class Foo;
Foo * f_func(void) { // expected-error {{'Foo' is unavailable}}
  return 0; 
}

#define UNAVAILABLE __attribute__((unavailable("not available")))

UNAVAILABLE
@interface Base // expected-note {{unavailable here}}
@end

UNAVAILABLE
@protocol SomeProto // expected-note 4 {{unavailable here}}
@end

@interface Sub : Base<SomeProto> // expected-error 2 {{unavailable}}
@end
@interface IP<SomeProto> // expected-error {{unavailable}}
@end
@protocol SubProt<SomeProto> // expected-error {{unavailable}}
@end
@interface Sub(cat)<SomeProto> // expected-error {{unavailable}}
@end

UNAVAILABLE
@interface UnavailSub : Base<SomeProto> // no error
@end
UNAVAILABLE
@interface UnavailIP<SomeProto> // no error
@end
UNAVAILABLE
@protocol UnavailProt<SomeProto> // no error
@end
@interface UnavailSub(cat)<SomeProto> // no error
@end

int unavail_global UNAVAILABLE;

UNAVAILABLE __attribute__((objc_root_class))
@interface TestAttrContext
-meth;
@end

@implementation TestAttrContext
-meth {
  unavail_global = 2; // no warn
  (void) ^{
    unavail_global = 4; // no warn
  };
}
@end

typedef int unavailable_int UNAVAILABLE; // expected-note {{'unavailable_int' has been explicitly marked unavailable here}}

UNAVAILABLE
@interface A
extern unavailable_int global_unavailable; // expected-error {{'unavailable_int' is unavailable: not available}}
@end
