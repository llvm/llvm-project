// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface ReadOnly 
{
  id _object;
  id _object1;
}
@property(readonly) id object;
@property(readwrite, assign) id object1; // expected-note {{property declared here}}
@property (readonly) int indentLevel;
@end

@interface ReadOnly ()
@property(readwrite, copy) id object; // Ok. declaring memory model in class extension - primary has none.
@property(readonly) id object1; // expected-error {{illegal redeclaration of property in class extension 'ReadOnly' (attribute must be 'readwrite', while its primary must be 'readonly')}}
@property (readwrite, assign) int indentLevel; // OK. assign the default in any case.
@end

@protocol Proto
  @property (copy) id fee; // expected-note {{property declared here}}
@end

@protocol Foo<Proto>
  @property (copy) id foo; // expected-note {{property declared here}}
@end

@interface Bar  <Foo> {
        id _foo;
        id _fee;
}
@end

@interface Bar ()
@property (copy) id foo; // expected-error {{illegal redeclaration of property in class extension 'Bar' (attribute must be 'readwrite', while its primary must be 'readonly')}}
@property (copy) id fee; // expected-error {{illegal redeclaration of property in class extension 'Bar' (attribute must be 'readwrite', while its primary must be 'readonly')}}
@end

@implementation Bar
@synthesize foo = _foo;
@synthesize fee = _fee;
@end

// rdar://10752081
@interface MyOtherClass() // expected-error {{cannot find interface declaration for 'MyOtherClass'}}
{
 id array;
}
@end

@implementation MyOtherClass // expected-warning {{cannot find interface declaration for 'MyOtherClass'}}
@end
