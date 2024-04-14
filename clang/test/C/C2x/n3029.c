// RUN: %clang_cc1 -std=c23 -verify -pedantic %s

/* WG14 N3029: Yes
 * Improved Normal Enumerations
 */

 // expected-no-diagnostics


enum a {
	a0 = 0xFFFFFFFFFFFFFFFFULL
};


static_assert(_Generic(a0, unsigned long long: 0, int: 1, default: 2) == 0);


// 6.7.2.2.5

// During the processing of each enumeration constant in the enumerator list, the type of the enumeration constant shall be:

// int, if there are no previous enumeration constants in the enumerator list and no explicit = with a defining integer constant expression; or,


enum b {
	b0
};


static_assert(_Generic(b0, int: 1, default: 2) == 1);

// int, if given explicitly with = and the value of the integer constant expression is representable by an int; or,

enum c {
	c0 = 1
};

static_assert(_Generic(c0, int: 1, default: 2) == 1);

// the type of the integer constant expression, if given explicitly with = and if the value of the integer constant expression is not representable by int; or,


enum d {
	d0 = 1U
};

static_assert(_Generic(d0, int: 1, default: 2) == 1);

enum e {
	e0 = 0XFFFFFFFFU
};

static_assert(_Generic(e0, int: 1, unsigned int: 2, default: 3) == 2);

enum f {
	f0 = 0xFFFFFFFFFFFFFFFLL
};

static_assert(_Generic(f0, int: 1, long long: 2, default: 3) == 2);

// the type of the value from last enumeration constant with 1 added to it. If such an integer constant expression would overflow or wraparound the value of the previous enumeration constant from the addition of 1, the type takes on either:

// a suitably sized signed integer type (excluding the bit-precise signed integer types) capable of representing the value of the previous enumeration constant plus 1; or,

// a suitably sized unsigned integer type (excluding the bit-precise unsigned integer types) capable of representing the value of the previous enumeration constant plus 1.

// A signed integer type is chosen if the previous enumeration constant being added is of signed integer type. An unsigned integer type is chosen if the previous enumeration constant is of unsigned integer type. If there is no suitably sized integer type described previous which can represent the new value, then the enumeration has no type which is capable of representing all of its values).

enum g {
     g0 = 2147483647, g1 
};

static_assert(_Generic(g0, int: 1, long: 2, default: 3) == 2);
static_assert(_Generic(g1, int: 1, long: 2, default: 3) == 2);

enum h {
     h0 = 4294967295U, h1 
};

static_assert(_Generic(h0, int: 1, unsigned long: 2, default: 3) == 2);
static_assert(_Generic(h1, int: 1, unsigned long: 2, default: 3) == 2);
