// RUN: %clang_cc1 -fsyntax-only -pedantic -verify %s

void f() {
  typedef int T;
  int x, *px;
  
  // Type id.
  (T())x;    // expected-error {{cast from 'int' to 'T ()'}}
  (T())+x;   // expected-error {{cast from 'int' to 'T ()'}}
  (T())*px;  // expected-error {{cast from 'int' to 'T ()'}}
  
  // Expression.
  x = (T());
  x = (T())/x;

  typedef int *PT;
  // Make sure stuff inside the parens are parsed only once (only one warning).
  x = (PT()[(int){1}]); // expected-warning {{compound literals}}

  // Special case: empty parens is a call, not an expression
  struct S{int operator()();};
  (S())();

  // Special case: "++" is postfix here, not prefix
  (S())++; // expected-error {{cannot increment value of type 'S'}}

  struct X { int &operator++(int); X operator[](int); int &operator++(); };
  int &postfix_incr = (X()[3])++;
  (X())++ ++; // ok, not a C-style cast
  (X())++ ++X(); // expected-error {{C-style cast from 'int' to 'X ()'}}
  int q = (int)++(x);
}

// Make sure we do tentative parsing correctly in conditions.
typedef int type;
struct rec { rec(int); };

namespace ns {
  typedef int type;
  struct rec { rec(int); };
}

struct cls {
  typedef int type;
  struct rec { rec(int); };
};

struct result {
  template <class T> result(T);
  bool check();
};

void test(int i) {
  if (result((cls::type) i).check())
    return;

  if (result((ns::type) i).check())
    return;

  if (result((::type) i).check())
    return;

  if (result((cls::rec) i).check())
    return;

  if (result((ns::rec) i).check())
    return;

  if (result((::rec) i).check())
    return;
}

namespace GH221890 {
template <class T> struct S {}; // expected-note 2 {{'S' declared here}}
struct Plain {}; // expected-note {{'Plain' declared here}}
namespace foo {}
namespace ns1 { template <class T> struct Q {}; } // expected-note {{'ns1::Q' declared here}}
namespace ba { template <class T> struct T2 {}; struct P {}; } // expected-note 2 {{'ba' declared here}}
namespace bar {}

// Typo correction dropped or replaced the qualifier while the parser was
// tentatively deciding whether the parenthesized construct is a type-id, and
// the tokens of the original qualifier resurfaced after backtracking.
int a = (void(foo::S<int>)); // expected-error {{no template named 'S' in namespace 'GH221890::foo'; did you mean simply 'S'?}} \
                             // expected-error {{expected '(' for function-style cast or type construction}}
int b = (void(foo::Q<int>)); // expected-error {{no template named 'Q' in namespace 'GH221890::foo'; did you mean 'ns1::Q'?}} \
                             // expected-error {{expected '(' for function-style cast or type construction}}
int c = (void(bar::ba::T2<int>)); // expected-error {{no member named 'ba' in namespace 'GH221890::bar'; did you mean simply 'ba'?}} \
                                  // expected-error {{expected '(' for function-style cast or type construction}}
int d = (void(bar::ba::P)); // expected-error {{no member named 'ba' in namespace 'GH221890::bar'; did you mean simply 'ba'?}} \
                            // expected-error {{expected '(' for function-style cast or type construction}}
void f() {
  void(foo::S<int>); // expected-error {{no template named 'S' in namespace 'GH221890::foo'; did you mean simply 'S'?}} \
                     // expected-error {{expected '(' for function-style cast or type construction}}
}

// The same constructs without a typo, and with a non-template name.
int e = (void(S<int>)); // expected-error {{expected '(' for function-style cast or type construction}}
int g = (void(foo::Plain)); // expected-error {{no member named 'Plain' in namespace 'GH221890::foo'; did you mean simply 'Plain'?}}
}

