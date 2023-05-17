// RUN: %clang_cc1 -verify -std=c2x %s

// Demonstrate that we get the correct type information. Do this by leaning
// heavily on redeclarations needing to use the same type for both decls.
extern int i;
extern typeof(i) i;
extern typeof_unqual(i) i;

extern const int j;
extern typeof(j) j;

extern const int n;         // expected-note 2 {{previous declaration is here}}
extern typeof(i) n;         // expected-error {{redeclaration of 'n' with a different type: 'typeof (i)' (aka 'int') vs 'const int'}}
extern typeof_unqual(n) n;  // expected-error {{redeclaration of 'n' with a different type: 'typeof_unqual (n)' (aka 'int') vs 'const int'}}

// Ensure we get a redeclaration error here for the types not matching.
extern typeof(j) k;        // expected-note {{previous declaration is here}}
extern typeof_unqual(j) k; // expected-error {{redeclaration of 'k' with a different type: 'typeof_unqual (j)' (aka 'int') vs 'typeof (j)' (aka 'const int')}}

// Make sure the type-form of the operator also works.
extern typeof(int) l;
extern typeof_unqual(const int) l;

extern typeof(const int) m;        // expected-note {{previous declaration is here}}
extern typeof_unqual(const int) m; // expected-error {{redeclaration of 'm' with a different type: 'typeof_unqual(const int)' (aka 'int') vs 'typeof(const int)' (aka 'const int')}}

// Show that we can use an incomplete type which is then completed later.
extern typeof(struct T) *o;
struct T { int a; } t;
extern typeof(struct T) *o;
extern typeof(t) *o;
extern typeof(&t) o;
extern typeof_unqual(volatile struct T) *o;
extern typeof_unqual(t) *o;
extern typeof_unqual(&t) o;

// Show that we properly strip the _Atomic qualifier.
extern _Atomic int i2;
extern _Atomic(int) i2;
extern typeof(i2) i2;        // expected-note {{previous declaration is here}}
extern typeof_unqual(i2) i2; // expected-error {{redeclaration of 'i2' with a different type: 'typeof_unqual (i2)' (aka 'int') vs 'typeof (i2)' (aka '_Atomic(int)')}}

// We cannot take the type of a bit-field.
struct S {
  int bit : 4;
} s;

typeof(s.bit) nope1; // expected-error {{invalid application of 'typeof' to bit-field}}
typeof_unqual(s.bit) nope2; // expected-error {{invalid application of 'typeof_unqual' to bit-field}}

// Show that we properly resolve nested typeof specifiers.
extern typeof(typeof(0)) i3;
extern typeof(typeof(int)) i3;
extern typeof(typeof_unqual(0)) i3;
extern typeof(typeof_unqual(int)) i3;
extern typeof_unqual(typeof(0)) i3;
extern typeof_unqual(typeof(int)) i3;
extern typeof_unqual(typeof_unqual(0)) i3;
extern typeof_unqual(typeof_unqual(int)) i3;
extern typeof(typeof_unqual(j)) i3;
extern typeof(typeof_unqual(const int)) i3;
extern typeof_unqual(typeof(j)) i3;
extern typeof_unqual(typeof(const int)) i3;
extern typeof_unqual(typeof_unqual(j)) i3;
extern typeof_unqual(typeof_unqual(const int)) i3;

// Both of these result in a const int rather than an int.
extern typeof(typeof(j)) i4;
extern typeof(typeof(const int)) i4;

// Ensure that redundant qualifiers are allowed, same as with typedefs.
typedef const int CInt;
extern CInt i4;
extern const CInt i4;
extern const typeof(j) i4;
extern const typeof(const int) i4;
extern const typeof(CInt) i4;

// Qualifiers are not redundant here, but validating that the qualifiers are
// still honored.
extern const typeof_unqual(j) i4;
extern const typeof_unqual(const int) i4;
extern const typeof_unqual(CInt) i4;

// Show that type attributes are stripped from the unqualified version.
extern __attribute__((address_space(0))) int type_attr_test_2_obj;
extern int type_attr_test_2;
extern typeof_unqual(type_attr_test_2_obj) type_attr_test_2;            // expected-note {{previous declaration is here}}
extern __attribute__((address_space(0))) int type_attr_test_2;          // expected-error {{redeclaration of 'type_attr_test_2' with a different type: '__attribute__((address_space(0))) int' vs 'typeof_unqual (type_attr_test_2_obj)' (aka 'int')}}

// Ensure that an invalid type doesn't cause crashes.
void invalid_param_fn(__attribute__((address_space(1))) int i); // expected-error {{parameter may not be qualified with an address space}}
typeof(invalid_param_fn) invalid_param_1;
typeof_unqual(invalid_param_fn) invalid_param_2;
