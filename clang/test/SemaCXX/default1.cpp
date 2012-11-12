// RUN: %clang_cc1 -fsyntax-only -verify %s
void f(int i);
void f(int i = 0); // expected-note {{previous definition is here}}
void f(int i = 17); // expected-error {{redefinition of default argument}}


void g(int i, int j, int k = 3);
void g(int i, int j = 2, int k);
void g(int i = 1, int j, int k);

void h(int i, int j = 2, int k = 3, 
       int l, // expected-error {{missing default argument on parameter 'l'}}
       int,   // expected-error {{missing default argument on parameter}}
       int n);// expected-error {{missing default argument on parameter 'n'}}

struct S { } s;
void i(int = s) { } // expected-error {{no viable conversion}} \
// expected-note{{passing argument to parameter here}}

struct X { 
  X(int);
};

void j(X x = 17);

struct Y { // expected-note 2{{candidate}}
  explicit Y(int);
};

void k(Y y = 17); // expected-error{{no viable conversion}} \
// expected-note{{passing argument to parameter 'y' here}}

void kk(Y = 17); // expected-error{{no viable conversion}} \
// expected-note{{passing argument to parameter here}}

int l () {
  int m(int i, int j, int k = 3);
  if (1)
  {
    int m(int i, int j = 2, int k = 4);
    m(8);
  }
  return 0;
}

int i () {
  void j (int f = 4);
  {
    void j (int f); // expected-note{{'j' declared here}}
    j(); // expected-error{{too few arguments to function call, single argument 'f' was not specified}}
  }
}

int i2() {
  void j(int f = 4); // expected-note{{'j' declared here}}
  {
    j(2, 3); // expected-error{{too many arguments to function call, expected at most single argument 'f', have 2}}
  }
}
