// RUN: %clang_cc1 -x c++ -verify -fsyntax-only %s

void f1();

struct S {
  static void f2();
};

extern void f3() __attribute__((weak_import));

struct S2 {
  static void f4() __attribute__((weak_import));
};

bool f5();
bool f6(int);

void bar() {
  bool b;

  b = f1; // expected-warning {{address of function 'f1' will always evaluate to 'true'}} \
             expected-note {{prefix with the address-of operator to silence this warning}}
  if (f1) {} // expected-warning {{address of function 'f1' will always evaluate to 'true'}} \
                expected-note {{prefix with the address-of operator to silence this warning}}
  b = S::f2; // expected-warning {{address of function 'S::f2' will always evaluate to 'true'}} \
                expected-note {{prefix with the address-of operator to silence this warning}}
  if (S::f2) {} // expected-warning {{address of function 'S::f2' will always evaluate to 'true'}} \
                   expected-note {{prefix with the address-of operator to silence this warning}}
  b = f5; // expected-warning {{address of function 'f5' will always evaluate to 'true'}} \
             expected-note {{prefix with the address-of operator to silence this warning}} \
             expected-note {{suffix with parentheses to turn this into a function call}}
  b = f6; // expected-warning {{address of function 'f6' will always evaluate to 'true'}} \
             expected-note {{prefix with the address-of operator to silence this warning}}

  // implicit casts of weakly imported symbols are ok:
  b = f3;
  if (f3) {}
  b = S2::f4;
  if (S2::f4) {}
}
