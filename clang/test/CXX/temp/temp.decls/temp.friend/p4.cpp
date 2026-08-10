// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T>
struct X1 {
  friend void f6(int) { } // expected-error{{redefinition of}} \
                          // expected-note{{previous definition}}
};

X1<int> x1a; 
X1<float> x1b; // expected-note {{in instantiation of}}

template<typename T>
struct X2 {
  operator int();

  friend void f(int x) { } // expected-error{{redefinition}} \
                           // expected-note{{previous definition}}
};

int array0[sizeof(X2<int>)]; 
int array1[sizeof(X2<float>)]; // expected-note{{instantiation of}}

void g() {
  X2<int> xi;
  f(xi);
  X2<float> xf; 
  f(xf);
}

template<typename T>
struct X3 {
  operator int();

  friend void h(int x);
};

int array2[sizeof(X3<int>)]; 
int array3[sizeof(X3<float>)];

void i() {
  X3<int> xi;
  h(xi);
  X3<float> xf; 
  h(xf);
}
