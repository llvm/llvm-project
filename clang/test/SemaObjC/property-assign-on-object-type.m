// RUN: %clang_cc1 -fsyntax-only -verify -Wobjc-property-assign-on-object-type %s

@interface Foo @end
@protocol Prot @end

@interface Bar
@property(assign, readonly) Foo* o1; // expected-warning {{'assign' property of object type may become a dangling reference; consider using 'unsafe_unretained'}}
@property(unsafe_unretained, readonly) Foo* o2;

@property(assign) Class classProperty;
@property(assign) Class<Prot> classWithProtocolProperty;
@property(assign) int s1;
@property(assign) int* s2;
@end

@interface Bar ()
@property(readwrite) Foo* o1;
@property(readwrite) Foo* o2;
@end
