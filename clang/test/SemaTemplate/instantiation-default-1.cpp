// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T, typename U = const T> struct Def1;

template<> struct Def1<int> { 
  void foo();
};

template<> struct Def1<const int> { // expected-note{{previous definition is here}}
  void bar();
};

template<> struct Def1<int&> {
  void wibble();
};

void test_Def1(Def1<int, const int> *d1, Def1<const int, const int> *d2,
               Def1<int&, int&> *d3) {
  d1->foo();
  d2->bar();
  d3->wibble();
}

template<typename T,  // FIXME: bad error message below, needs better location info
         typename T2 = const T*>  // expected-error{{'T2' declared as a pointer to a reference}}
  struct Def2;

template<> struct Def2<int> { 
  void foo();
};

void test_Def2(Def2<int, int const*> *d2) {
  d2->foo();
}

typedef int& int_ref_t;
Def2<int_ref_t> *d2; // expected-note{{in instantiation of default argument for 'Def2<int &>' required here}}


template<> struct Def1<const int> { }; // expected-error{{redefinition of 'Def1<const int>'}}

template<typename T, typename T2 = T&> struct Def3;

template<> struct Def3<int> { 
  void foo();
};

template<> struct Def3<int&> { 
  void bar();
};

void test_Def3(Def3<int, int&> *d3a, Def3<int&, int&> *d3b) {
  d3a->foo();
  d3b->bar();
}


template<typename T, typename T2 = T[]> struct Def4;

template<> struct Def4<int> {
  void foo();
};

void test_Def4(Def4<int, int[]> *d4a) {
  d4a->foo();
}

template<typename T, typename T2 = T const[12]> struct Def5;

template<> struct Def5<int> {
  void foo();
};

template<> struct Def5<int, int const[13]> {
  void bar();
};

void test_Def5(Def5<int, const int[12]> *d5a, Def5<int, const int[13]> *d5b) {
  d5a->foo();
  d5b->bar();
}

template<typename R, typename Arg1, typename Arg2 = Arg1,
         typename FuncType = R (*)(Arg1, Arg2)>
  struct Def6;

template<> struct Def6<int, float> { 
  void foo();
};

template<> struct Def6<bool, int[5], float(double, double)> {
  void bar();
};

bool test_Def6(Def6<int, float, float> *d6a, 
               Def6<int, float, float, int (*)(float, float)> *d6b,
               Def6<bool, int[5], float(double, double),
                    bool(*)(int*, float(*)(double, double))> *d6c) {
  d6a->foo();
  d6b->foo();
  d6c->bar();
  return d6a == d6b;
}
