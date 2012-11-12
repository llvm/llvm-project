// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T>
struct X {
  template<typename U> struct Inner { };
  
  template<typename U> void f(T, U) { }
};

template<> template<typename U>
struct X<int>::Inner {
  U member;
};

template<> template<typename U>
void X<int>::f(int x, U y) { 
  x = y; // expected-error{{incompatible type}}
}

void test(X<int> xi, X<long> xl, float *fp) {
  X<int>::Inner<float*> xii;
  xii.member = fp;
  xi.f(17, 25);
  xi.f(17, 3.14159);
  xi.f(17, fp); // expected-note{{instantiation}}
  X<long>::Inner<float*> xli;
  
  xli.member = fp; // expected-error{{no member}}
  xl.f(17, fp); // okay
}
