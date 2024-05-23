// RUN: %clang_cc1 -fsyntax-only -std=c++98 %s -verify=cxx98,cxx98-cxx11,expected
// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify=cxx98-cxx11,cxx11,expected
// RUN: %clang_cc1 -fsyntax-only -std=c++14 %s -verify
// RUN: %clang_cc1 -fsyntax-only -std=c++20 %s -verify
// RUN: %clang_cc1 -fsyntax-only -std=c++23 %s -verify=cxx23,expected

// Introduced in C++14 by N3323
// Moved in C++20 to [conv.general]p5 by P1076R1

template<class T>
struct X0 {
  operator T();
};

void f0() {
  delete X0<int*>();
  delete X0<int*&>();
  delete X0<int*&&>();  // cxx98-warning {{C++11 extension}}
  switch (X0<int>()) {}
  switch (X0<int&>()) {}
  switch (X0<int&&>()) {}  // cxx98-warning {{C++11 extension}}
  delete X0<int(&)[1]>();
// expected-error@-1 {{cannot delete expression of type 'X0<int (&)[1]>'}}
}

template<class T>
struct zero_init {
  operator T&();  // #zero_init_mut
  operator T() const;  // #zero_init_const
};

void f1(zero_init<int*> p, const zero_init<int*> q) {
  delete p;
// cxx98-cxx11-error@-1 {{ambiguous conversion of delete expression of type 'zero_init<int *>' to a pointer}}
//   cxx98-cxx11-note@#zero_init_mut {{conversion to pointer type 'int *'}}
//   cxx98-cxx11-note@#zero_init_const {{conversion to pointer type 'int *'}}
  delete q;
// cxx98-cxx11-error@-1 {{ambiguous conversion of delete expression of type 'const zero_init<int *>' to a pointer}}
//   cxx98-cxx11-note@#zero_init_mut {{conversion to pointer type 'int *'}}
//   cxx98-cxx11-note@#zero_init_const {{conversion to pointer type 'int *'}}
}

void f2(zero_init<int> i, const zero_init<int> j) {
  switch (i) {}
// cxx98-cxx11-error@-1 {{multiple conversions from switch condition type 'zero_init<int>' to an integral or enumeration type}}
//   cxx98-cxx11-note@#zero_init_mut {{conversion to integral type 'int'}}
//   cxx98-cxx11-note@#zero_init_const {{conversion to integral type 'int'}}
  switch (j) {}
// cxx98-cxx11-error@-1 {{multiple conversions from switch condition type 'const zero_init<int>' to an integral or enumeration type}}
//   cxx98-cxx11-note@#zero_init_mut {{conversion to integral type 'int'}}
//   cxx98-cxx11-note@#zero_init_const {{conversion to integral type 'int'}}
}

template<class T>
struct X1 {
  template<class Result = T>  // cxx98-warning {{C++11 extension}}
  operator Result();
};

void f2(X1<int> i, X1<int*> p) {
  delete p;
// expected-error@-1 {{cannot delete expression of type 'X1<int *>'}}
  switch (i) {}
// expected-error@-1 {{statement requires expression of integer type ('X1<int>' invalid)}}
}

#if __cplusplus >= 201102L
template<class T>
struct X2 {
  T v;
  constexpr operator T() const { return v; }
};

enum E { E_0, E_4 = 4, E_5 = 5 };
enum class EC : unsigned { _0 };

static constexpr int _1 = 1;
static constexpr unsigned _3 = 3;
static constexpr E _5 = E::E_5;
static constexpr EC EC_0 = EC::_0;

void f3() {
  switch (0) {
  case X2<int>{0}:;
  case X2<const int&>{_1}:;
  case X2<unsigned>{2}:;
  case X2<const unsigned&>{_3}:;
  case X2<E>{E_4}:;
  case X2<const E&>{_5}:;
  }
  switch (EC::_0) { case X2<EC>{}:; }
  switch (EC::_0) { case X2<const EC&>{EC_0}:; }

  int a1[X2<__SIZE_TYPE__>{1}];
  new int[1][X2<__SIZE_TYPE__>{1}];
  // FIXME: Should only allow conversion operators to std::size_t
  int a2[X2<int>{1}];
  new int[1][X2<int>{1}];
}
#endif

#if __cplusplus >= 202302L
template<typename Derived, typename T, typename U>
struct X3 {
  T val;
  // There is only one type to convert to, so multiple overloads does not affect search result for contextual conversion (T)
  constexpr operator T(this Derived&& self) { return self.val; }
  constexpr operator U(this X3&& self) { return self.val; }
};

template<typename T>
struct X4 : X3<X4<T>, T, T> {
  constexpr X3<X4<T>, T, T>&& base(this X3<X4<T>, T, T>&& self) { return self; }
};

void f4() {
  delete X4<int*>{};
  delete X4<int*>{}.base();
  switch (X4<int>{}) { case 0:; }
  switch (X4<int>{}.base()) { case 0:; }
  switch (1) {
  case X4<int>{}:
  case X4<int>{{1}}.base():
  }
}

struct Unrelated {};

template<typename T1, typename T2>
struct X5 {
  operator T1(this X5&&);  // #X5-1
  operator T2(this Unrelated&&);  // #X5-2
};

void f5() {
  delete X5<int*, int*>{};
  delete X5<int*, int*&>{};
  delete X5<int* volatile&&, int* const&>{};
  delete X5<int*, const int*>{};
// cxx23-error@-1 {{ambiguous conversion of delete expression of type 'X5<int *, const int *>' to a pointer}}
//   cxx23-note@#X5-1 {{conversion to pointer type 'int *'}}
//   cxx23-note@#X5-2 {{conversion to pointer type 'const int *'}}
  switch (X5<int, int>{}) {}
  switch (X5<int, const int>{}) {}
  switch (X5<int, int&>{}) {}
  switch (X5<volatile int&&, const int&>{}) {}
  switch (X5<int, unsigned>{}) {}
// cxx23-error@-1 {{multiple conversions from switch condition type 'X5<int, unsigned int>' to an integral or enumeration type}}
//   cxx23-note@#X5-1 {{conversion to integral type 'int'}}
//   cxx23-note@#X5-2 {{conversion to integral type 'unsigned int'}}
}
#endif
