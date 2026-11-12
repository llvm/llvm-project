// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR3234

@protocol NSCopying @end
@interface NSObject @end

void f1(NSObject *o)
{
  o.foo; // expected-error{{property 'foo' not found on object of type 'NSObject *'}}
}

void f2(id<NSCopying> o)
{
  o.foo; // expected-error{{property 'foo' not found on object of type 'id<NSCopying>'}}
}

void f3(id o)
{
  o.foo; // expected-error{{property 'foo' not found on object of type 'id'}}
}

@class SomeOtherClass; // expected-note {{forward declaration of class here}}

@interface MyClass {
    SomeOtherClass *someOtherObject;
}
@end

void foo(MyClass *myObject) {
	myObject.someOtherObject.someProperty = 0; // expected-error {{property 'someOtherObject' refers to an incomplete Objective-C class 'SomeOtherClass' (with no @interface available)}}
}

