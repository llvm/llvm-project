// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

typedef const void * VoidStar;

typedef struct __CFDictionary * CFMDRef;

void RandomFunc(CFMDRef theDict, const void *key, const void *value);

@interface Foo
- (void)_apply:(void (*)(const void *, const void *, void *))func context:(void *)context;
- (void)a:(id *)objects b:(id *)keys;
@end

@implementation Foo
- (void)_apply:(void (*)(const void *, const void *, void *))func context:(void *)context {
	id item;
	id obj;
    func(item, obj, context);
}

- (void)a:(id *)objects b:(id *)keys {
    VoidStar dict;
	id key;
    RandomFunc((CFMDRef)dict, key, objects[3]);
}
@end

@interface I
- (void) Meth : (I*) Arg; // expected-note{{passing argument to parameter 'Arg' here}}
@end

void Func (I* arg);  // expected-note {{candidate function not viable: no known conversion from 'const I *' to 'I *' for 1st argument}}

void foo(const I *p, I* sel) {
  [sel Meth : p];	// expected-error {{cannot initialize a parameter of type 'I *' with an lvalue of type 'const I *'}}
  Func(p);		// expected-error {{no matching function for call to 'Func'}}
}

@interface DerivedFromI : I
@end

void accept_derived(DerivedFromI*);

void test_base_to_derived(I* i) {
  accept_derived(i); // expected-warning{{incompatible pointer types passing 'I *' to parameter of type 'DerivedFromI *'}}
  DerivedFromI *di = i; // expected-warning{{incompatible pointer types initializing 'DerivedFromI *' with an expression of type 'I *'}}
  DerivedFromI *di2 = (DerivedFromI *)i;
}
