// RUN: %clang_cc1  -fsyntax-only -verify %s

@interface Foo 
@property (readonly) char foo; 
@property (readwrite) char bar; // expected-note {{property declared here}}
@end

@interface Foo ()
@property (readwrite) char foo; // expected-note 2 {{property declared here}} 
@property (readwrite) char NewProperty; // expected-note 2 {{property declared here}} 
@property (readwrite) char bar; // expected-error{{illegal redeclaration of 'readwrite' property in class extension 'Foo' (perhaps you intended this to be a 'readwrite' redeclaration of a 'readonly' public property?)}}
@end

@interface Foo ()
@property (readwrite) char foo;	 // expected-error {{property has a previous declaration}}
@property (readwrite) char NewProperty; // expected-error {{property has a previous declaration}}
@end

@interface Foo ()
@property (readonly) char foo; // expected-error {{property has a previous declaration}}
@property (readwrite) char NewProperty; // expected-error {{property has a previous declaration}}
@end

