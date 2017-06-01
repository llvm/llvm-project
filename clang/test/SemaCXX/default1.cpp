// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

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

void j(X x = 17); // expected-note{{'::j' declared here}}

struct Y { // expected-note 2{{candidate constructor (the implicit copy constructor) not viable}}
#if __cplusplus >= 201103L // C++11 or later
// expected-note@-2 2 {{candidate constructor (the implicit move constructor) not viable}}
#endif

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
    void j (int f);
    j(); // expected-error{{too few arguments to function call, expected 1, have 0; did you mean '::j'?}}
  }
  void jj (int f = 4);
  {
    void jj (int f); // expected-note{{'jj' declared here}}
    jj(); // expected-error{{too few arguments to function call, single argument 'f' was not specified}}
  }
}

int i2() {
  void j(int f = 4); // expected-note{{'j' declared here}}
  {
    j(2, 3); // expected-error{{too many arguments to function call, expected at most single argument 'f', have 2}}
  }
}

int pr20055_f(int x = 0, int y = UNDEFINED); // expected-error{{use of undeclared identifier}}
int pr20055_v = pr20055_f(0);

void PR20769() { void PR20769(int = 1); }
void PR20769(int = 2);

void PR20769_b(int = 1);
void PR20769_b() { void PR20769_b(int = 2); }
