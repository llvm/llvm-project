// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++03 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// PR5336
template<typename FromCl>
struct isa_impl_cl {
 template<class ToCl>
 static void isa(const FromCl &Val) { }
};

template<class X, class Y>
void isa(const Y &Val) {   return isa_impl_cl<Y>::template isa<X>(Val); }

class Value;
void f0(const Value &Val) { isa<Value>(Val); }

// Implicit template-ids.
template<typename T>
struct X0 {
  template<typename U>
  void f1();
  
  template<typename U>
  void f2(U) {
    f1<U>();
  }
};

void test_X0_int(X0<int> xi, float f) {
  xi.f2(f);
}

// Not template-id expressions, but they almost look like it.
template<typename F>
struct Y {
  Y(const F&);
};

template<int I>
struct X {
  X(int, int);
  void f() { 
    Y<X<I> >(X<I>(0, 0)); 
    Y<X<I> >(::X<I>(0, 0)); 
  }
};

template struct X<3>;

// 'template' as a disambiguator.
// PR7030
struct Y0 {
  template<typename U>
  void f1(U);

  template<typename U>
  static void f2(U);

  void f3(int);

  static int f4(int);
  template<typename U>
  static void f4(U);

  template<typename U>
  void f() {
    Y0::template f1<U>(0);
    Y0::template f1(0);
    this->template f1(0);

    Y0::template f2<U>(0);
    Y0::template f2(0);

    Y0::template f3(0); // expected-error {{'f3' following the 'template' keyword does not refer to a template}}
    Y0::template f3(); // expected-error {{'f3' following the 'template' keyword does not refer to a template}}

    int x;
    x = Y0::f4(0);
    x = Y0::f4<int>(0); // expected-error {{assigning to 'int' from incompatible type 'void'}}
    x = Y0::template f4(0); // expected-error {{assigning to 'int' from incompatible type 'void'}}

    x = this->f4(0);
    x = this->f4<int>(0); // expected-error {{assigning to 'int' from incompatible type 'void'}}
    x = this->template f4(0); // expected-error {{assigning to 'int' from incompatible type 'void'}}
  }
};

template<typename U> void Y0
  ::template // expected-error {{expected unqualified-id}}
    f1(U) {}

// FIXME: error recovery is awful without this.
    ;

template<typename T>
struct Y1 {
  template<typename U>
  void f1(U);

  template<typename U>
  static void f2(U);

  void f3(int);

  static int f4(int);
  template<typename U>
  static void f4(U);

  template<typename U>
  void f() {
    Y1::template f1<U>(0);
    Y1::template f1(0);
    this->template f1(0);

    Y1::template f2<U>(0);
    Y1::template f2(0);

    Y1::template f3(0); // expected-error {{'f3' following the 'template' keyword does not refer to a template}}
    Y1::template f3(); // expected-error {{'f3' following the 'template' keyword does not refer to a template}}

    int x;
    x = Y1::f4(0);
    x = Y1::f4<int>(0); // expected-error {{assigning to 'int' from incompatible type 'void'}}
    x = Y1::template f4(0); // expected-error {{assigning to 'int' from incompatible type 'void'}}

    x = this->f4(0);
    x = this->f4<int>(0); // expected-error {{assigning to 'int' from incompatible type 'void'}}
    x = this->template f4(0); // expected-error {{assigning to 'int' from incompatible type 'void'}}
  }
};

void use_Y1(Y1<int> y1) { y1.f<int>(); } // expected-note {{in instantiation of}}

struct A {
  template<int I>
  struct B {
    static void b1();
  };
};

template<int I>
void f5() {
  A::template B<I>::template b1(); // expected-error {{'b1' following the 'template' keyword does not refer to a template}}
}

template void f5<0>(); // expected-note {{in instantiation of function template specialization 'f5<0>' requested here}}

class C {};
template <template <typename> class D>
class E {
  template class D<C>;  // expected-error {{expected '<' after 'template'}}
  template<> class D<C>;  // expected-error {{cannot specialize a template template parameter}}
  friend class D<C>; // expected-error {{type alias template 'D' cannot be referenced with a class specifier}}
};
#if __cplusplus <= 199711L
// expected-warning@+2 {{extension}}
#endif
template<typename T> using D = int; // expected-note {{declared here}} 
E<D> ed; // expected-note {{instantiation of}}
