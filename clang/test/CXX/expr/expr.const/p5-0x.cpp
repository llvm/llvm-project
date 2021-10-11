// RUN: %clang_cc1 -triple %itanium_abi_triple -fsyntax-only -std=c++11 -verify %s

// If an expression of literal class type is used in a context where an integral
// constant expression is required, then that class type shall have a single
// non-explicit conversion function to an integral or unscoped enumeration type
namespace std_example {

struct A {
  constexpr A(int i) : val(i) { }
  constexpr operator int() const { return val; } // expected-note {{here}}
  constexpr operator long() const { return 43; } // expected-note {{here}}
private:
  int val;
};
template<int> struct X { };
constexpr A a = 42;
X<a> x;     // ok, unique conversion to int
int ary[a]; // expected-error {{ambiguous conversion from type 'const A' to an integral or unscoped enumeration type}}

}

struct OK {
  constexpr OK() {}
  constexpr operator int() const { return 8; }
} constexpr ok;
extern struct Incomplete incomplete; // expected-note 5{{forward decl}}
struct Explicit {
  constexpr Explicit() {}
  constexpr explicit operator int() const { return 4; } // expected-note 5{{here}}
} constexpr expl;
struct Ambiguous {
  constexpr Ambiguous() {}
  constexpr operator int() const { return 2; } // expected-note 5{{here}}
  constexpr operator long() const { return 1; } // expected-note 5{{here}}
} constexpr ambig;

constexpr int test_ok = ok; // ok
constexpr int test_explicit(expl); // ok
constexpr int test_ambiguous = ambig; // ok

static_assert(test_ok == 8, "");
static_assert(test_explicit == 4, "");
static_assert(test_ambiguous == 2, "");

// [expr.new]p6: Every constant-expression in a noptr-new-declarator shall be
// an integral constant expression
auto new1 = new int[1][ok];
auto new2 = new int[1][incomplete]; // expected-error {{incomplete}}
auto new3 = new int[1][expl]; // expected-error {{explicit conversion}}
auto new4 = new int[1][ambig]; // expected-error {{ambiguous conversion}}

// [dcl.enum]p5: If the underlying type is not fixed [...] the initializing
// value [...] shall be an integral constant expression.
enum NotFixed {
  enum1 = ok,
  enum2 = incomplete, // expected-error {{incomplete}}
  enum3 = expl, // expected-error {{explicit conversion}}
  enum4 = ambig // expected-error {{ambiguous conversion}}
};

// [dcl.align]p2: When the alignment-specifier is of the form
// alignas(assignment-expression), the assignment-expression shall be an
// integral constant expression
alignas(ok) int alignas1;
alignas(incomplete) int alignas2; // expected-error {{incomplete}}
alignas(expl) int alignas3; // expected-error {{explicit conversion}}
alignas(ambig) int alignas4; // expected-error {{ambiguous conversion}}

// [dcl.array]p1: If the constant-expression is present, it shall be an integral
// constant expression
int array1[ok];
int array2[incomplete]; // expected-error {{incomplete}}
int array3[expl]; // expected-error {{explicit conversion}}
int array4[ambig]; // expected-error {{ambiguous conversion}}

// [class.bit]p1: The constasnt-expression shall be an integral constant
// expression
struct Bitfields {
  int bitfield1 : ok;
  int bitfield2 : incomplete; // expected-error {{incomplete}}
  int bitfield3 : expl; // expected-error {{explicit conversion}}
  int bitfield4 : ambig; // expected-error {{ambiguous conversion}}
};
