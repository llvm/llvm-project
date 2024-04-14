// RUN: %clang_cc1 -std=c23 -verify -pedantic %s

/* WG14 N3030: Partial
 * Enhancements to Enumerations
 */

#include <limits.h>

enum a : unsigned long long {
	a0 = 0xFFFFFFFFFFFFFFFFULL
	// ^ not a constraint violation with a 64-bit unsigned long long
};


static_assert(_Generic(a0, unsigned long long: 0, default: 1) == 0);

enum e : unsigned short {
    x
};

static_assert(_Generic(x, enum e: 0, default: 1) == 0);

static_assert(_Generic(x, unsigned short: 0, default: 1) == 0);


static_assert(_Generic(x, enum e: 0, unsigned short: 2, default: 1) == 0); // expected-error {{type 'unsigned short' in generic association compatible with previously specified type 'enum e'}} 
// expected-note@-1 {{compatible type 'enum e' specified here}}



enum us : unsigned short {
	us_max = USHRT_MAX,
	us_violation, // expected-error {{enumerator value 65536 is not representable in the underlying type 'unsigned short'}} 
	us_violation_2 = us_max + 1, // expected-error {{enumerator value is not representable in the underlying type 'unsigned short'}}
	us_wrap_around_to_zero = (unsigned short)(USHRT_MAX + 1) /* Okay: conversion
	                          done in constant expression before conversion to
	                          underlying type: unsigned semantics okay. */
};


enum ui : unsigned int {
	ui_max = UINT_MAX,
	ui_violation, // expected-error {{enumerator value 4294967296 is not representable in the underlying type 'unsigned int'}} 
	ui_no_violation = ui_max + 1, /* Okay: Arithmetic performed as typical
	                                  unsigned integer arithmetic: conversion
	                                  from a value that is already 0 to 0. */
	ui_wrap_around_to_zero = (unsigned int)(UINT_MAX + 1) /* Okay: conversion
	                          done in constant expression before conversion to
	                          underlying type: unsigned semantics okay. */
};

static_assert(ui_wrap_around_to_zero + us_wrap_around_to_zero == 0);

enum E1: short;
enum E2: short; // expected-note {{previous declaration is here}}
enum E3; // expected-warning {{ISO C forbids forward references to 'enum' types}}
enum E4 : unsigned long long;

enum E1 : short { m11, m12 };
enum E1 x1 = m11;

enum E2 : long { m21, m22 }; // expected-error {{enumeration redeclared with different underlying type 'long' (was 'short')}} 

enum E3 { // expected-note {{definition of 'enum E3' is not complete until the closing '}'}}
// expected-note@-1 {{previous declaration is here}}
	m31,
	m32,
	m33 = sizeof(enum E3) // expected-error {{invalid application of 'sizeof' to an incomplete type 'enum E3'}} 
};
enum E3 : int; // expected-error {{enumeration previously declared with nonfixed underlying type}} 

enum E4 : unsigned long long {
	m40 = sizeof(enum E4),
	m41 = ULLONG_MAX,
	m42 // expected-error {{enumerator value 18446744073709551616 is not representable in the underlying type 'unsigned long long'}} 
};

enum E5 y; // expected-error {{tentative definition has type 'enum E5' that is never completed}}
// expected-warning@-1 {{ISO C forbids forward references to 'enum' types}}
// expected-note@-2 {{forward declaration of 'enum E5'}}
enum E6 : long int z; // expected-error {{non-defining declaration of enumeration with a fixed underlying type is only permitted as a standalone declaration; missing list of enumerators?}}
enum E7 : long int = 0; // expected-error {{non-defining declaration of enumeration with a fixed underlying type is only permitted as a standalone declaration; missing list of enumerators?}}
// expected-error@-1 {{expected identifier or '('}}


enum underlying : unsigned char {
	b0
};

static_assert(_Generic(b0, int: 2, unsigned char: 1, default: 0 ) == 1);

static_assert(_Generic((enum underlying)b0, int: 2, unsigned char: 1, default: 0 ) == 1);


void f1 (enum a2 : long b2); // expected-error {{non-defining declaration of enumeration with a fixed underlying type is only permitted as a standalone declaration; missing list of enumerators?}}
// expected-warning@-1 {{declaration of 'enum a2' will not be visible outside of this function}}
void f2 (enum c2 : long { x2 } d2);
// expected-warning@-1 {{declaration of 'enum c2' will not be visible outside of this function}}
enum e2 : int f3(); // expected-error {{non-defining declaration of enumeration with a fixed underlying type is only permitted as a standalone declaration; missing list of enumerators?}}

typedef enum t u; // expected-warning {{ISO C forbids forward references to 'enum' types}}
typedef enum v : short W; // expected-error {{non-defining declaration of enumeration with a fixed underlying type is only permitted as a standalone declaration; missing list of enumerators?}}
typedef enum q : short { s } R;

struct s1 {
	int x2;
	enum e2 : int : 1; // expected-error {{non-defining declaration of enumeration with a fixed underlying type is only permitted as a standalone declaration; missing list of enumerators?}}
	int y2;
};

enum forward; // expected-warning {{ISO C forbids forward references to 'enum' types}}
//TODO
extern enum forward fwd_val0; /* Constraint violation: incomplete type */
extern enum forward* fwd_ptr0; /* Constraint violation: enums cannot be
                                  used like other incomplete types */
// expected-note@-2 {{previous declaration is here}}
extern int* fwd_ptr0; // expected-error {{redeclaration of 'fwd_ptr0' with a different type: 'int *' vs 'enum forward *'}}

enum forward1 : int;
extern enum forward1 fwd_val1;
extern int fwd_val1;
extern enum forward1* fwd_ptr1;
extern int* fwd_ptr1;

int foo () {
	enum e : short;
	enum e : short f = 0; // expected-error {{non-defining declaration of enumeration with a fixed underlying type is only permitted as a standalone declaration; missing list of enumerators?}}
	enum g : short { y } h = y;
	return 0;
}
