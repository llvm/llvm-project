// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 -pedantic %s
// C++ [expr.const]p1:
//   In several places, C++ requires expressions that evaluate to an integral
//   or enumeration constant: as array bounds, as case expressions, as
//   bit-field lengths, as enumerator initializers, as static member
//   initializers, and as integral or enumeration non-type template arguments.
//   An integral constant-expression can involve only literals, enumerators,
//   const variables or static data members of integral or enumeration types
//   initialized with constant expressions, and sizeof expressions. Floating
//   literals can appear only if they are cast to integral or enumeration types.

enum Enum { eval = 1 };
const int cval = 2;
const Enum ceval = eval;
struct Struct {
  static const int sval = 3;
  static const Enum seval = eval;
};

template <int itval, Enum etval> struct C {
  enum E {
    v1 = 1,
    v2 = eval,
    v3 = cval,
    v4 = ceval,
    v5 = Struct::sval,
    v6 = Struct::seval,
    v7 = itval,
    v8 = etval,
    v9 = (int)1.5,
    v10 = sizeof(Struct),
    v11 = true? 1 + cval * Struct::sval ^ itval / (int)1.5 - sizeof(Struct) : 0
  };
  unsigned
    b1 : 1,
    b2 : eval,
    b3 : cval,
    b4 : ceval,
    b5 : Struct::sval,
    b6 : Struct::seval,
    b7 : itval,
    b8 : etval,
    b9 : (int)1.5,
    b10 : sizeof(Struct),
    b11 : true? 1 + cval * Struct::sval ^ itval / (int)1.5 - sizeof(Struct) : 0
    ;
  static const int
    i1 = 1,
    i2 = eval,
    i3 = cval,
    i4 = ceval,
    i5 = Struct::sval,
    i6 = Struct::seval,
    i7 = itval,
    i8 = etval,
    i9 = (int)1.5,
    i10 = sizeof(Struct),
    i11 = true? 1 + cval * Struct::sval ^ itval / (int)1.5 - sizeof(Struct) : 0
    ;
  void f(int cond) {
    switch(cond) {
    case    0 + 1:
    case  100 + eval:
    case  200 + cval:
    case  300 + ceval:
    case  400 + Struct::sval:
    case  500 + Struct::seval:
    case  600 + itval:
    case  700 + etval:
    case  800 + (int)1.5:
    case  900 + sizeof(Struct):
    case 1000 + (true? 1 + cval * Struct::sval ^
                 itval / (int)1.5 - sizeof(Struct) : 0):
      ;
    }
  }
  typedef C<itval, etval> T0;
};

template struct C<1, eval>;
template struct C<cval, ceval>;
template struct C<Struct::sval, Struct::seval>;

enum {
  a = sizeof(int) == 8,
  b = a? 8 : 4
};

void diags(int n) {
  switch (n) {
    case (1/0, 1): // expected-error {{not an integral constant expression}} expected-note {{division by zero}}
    case (int)(1/0, 2.0): // expected-error {{not an integral constant expression}} expected-note {{division by zero}}
    case __imag(1/0): // expected-error {{not an integral constant expression}} expected-note {{division by zero}}
    case (int)__imag((double)(1/0)): // expected-error {{not an integral constant expression}} expected-note {{division by zero}}
      ;
  }
}

namespace IntOrEnum {
  const int k = 0;
  const int &p = k;
  template<int n> struct S {};
  S<p> s; // expected-error {{not an integral constant expression}}
}

extern const int recurse1;
// recurse2 cannot be used in a constant expression because it is not
// initialized by a constant expression. The same expression appearing later in
// the TU would be a constant expression, but here it is not.
const int recurse2 = recurse1;
const int recurse1 = 1;
int array1[recurse1]; // ok
int array2[recurse2]; // expected-warning {{variable length array}} expected-warning {{integer constant expression}}

namespace FloatConvert {
  typedef int a[(int)42.3];
  typedef int a[(int)42.997];
  typedef int b[(long long)4e20]; // expected-warning {{variable length}} expected-error {{variable length}} expected-warning {{'long long' is a C++11 extension}}
}

// PR12626
namespace test3 {
  struct X; // expected-note {{forward declaration of 'test3::X'}}
  struct Y { bool b; X x; }; // expected-error {{field has incomplete type 'test3::X'}}
  int f() { return Y().b; }
}

// PR18283
namespace test4 {
  template <int> struct A {};
  int const i = { 42 };
  // i can be used as non-type template-parameter as "const int x = { 42 };" is
  // equivalent to "const int x = 42;" as per C++03 8.5/p13.
  typedef A<i> Ai; // ok
}

// rdar://16064952
namespace rdar16064952 {
  template < typename T > void fn1() {
   T b;
   unsigned w = ({int a = b.val[sizeof(0)]; 0; }); // expected-warning {{use of GNU statement expression extension}}
  }
}

char PR17381_ice = 1000000 * 1000000; // expected-warning {{overflow}} expected-warning {{changes value}}
