// RUN: %clang_cc1 -std=c++1z -verify %s -Wno-vexing-parse

int g, h;
typedef int T;
int f() {
  // init-statement declarations
  if (T n = 0; n != 0) {} // expected-error {{not yet supported}}
  if (T f(); f()) {} // expected-error {{not yet supported}}
  if (T(f()); f()) {} // expected-error {{not yet supported}}
  if (T(f()), g, h; f()) {} // expected-error {{not yet supported}}
  if (T f(); f()) {} // expected-error {{not yet supported}}
  if (T f(), g, h; f()) {} // expected-error {{not yet supported}}

  // init-statement expressions
  if (T{f()}; f()) {} // expected-error {{not yet supported}}
  if (T{f()}, g, h; f()) {} // expected-error {{not yet supported}} expected-warning 2{{unused}}
  if (T(f()), g, h + 1; f()) {} // expected-error {{not yet supported}} expected-warning 2{{unused}}

  // condition declarations
  if (T(n){g}) {}
  if (T f()) {} // expected-error {{function type}}
  if (T f(), g, h) {} // expected-error {{function type}}

  // condition expressions
  if (T(f())) {}
  if (T{f()}) {}
  if (T(f()), g, h) {} // expected-warning 2{{unused}}
  if (T{f()}, g, h) {} // expected-warning 2{{unused}}

  // none of the above
  // FIXME: This causes a typo-correction crash, as does "void f() { +T(n)(g); }"
  //if (T(n)(g)) {} // expected-err-FIXME {{not a function}}

  // Likewise for 'switch'
  switch (int n; n) {} // expected-error {{not yet supported}}
  switch (g; int g = 5) {} // expected-error {{not yet supported}}

  if (int a, b; int c = a) { // expected-error {{not yet supported}} expected-note 6{{previous}}
    int a; // expected-error {{redefinition}}
    int b; // expected-error {{redefinition}}
    int c; // expected-error {{redefinition}}
  } else {
    int a; // expected-error {{redefinition}}
    int b; // expected-error {{redefinition}}
    int c; // expected-error {{redefinition}}
  }
}
