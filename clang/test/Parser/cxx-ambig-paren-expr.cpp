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

