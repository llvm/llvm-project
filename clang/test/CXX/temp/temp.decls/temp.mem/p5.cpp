// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify=expected,cxx23    %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected,cxx98_20 %s
// RUN: %clang_cc1 -std=c++98 -fsyntax-only -verify=expected,cxx98_20 %s
// RUN: %clang_cc1            -fsyntax-only -verify=expected,cxx98_20 %s

struct A {
  template <class T> operator T*();
};

template <class T> A::operator T*() { return 0; }
template <> A::operator char*(){ return 0; } // specialization
template A::operator void*(); // explicit instantiation

int main() {
  A a;
  int *ip;
  ip = a.operator int*();
}

// PR5742
namespace PR5742 {
  template <class T> struct A { };
  template <class T> struct B { };

  struct S {
    template <class T> operator T();
  } s;

  void f() {
    s.operator A<A<int> >();
    s.operator A<B<int> >();
    s.operator A<B<A<int> > >();
  }
}

// PR5762
class Foo {
 public:
  template <typename T> operator T();

  template <typename T>
  T As() {
    return this->operator T();
  }

  template <typename T>
  T As2() {
    return operator T();
  }

  int AsInt() {
    return this->operator int();
  }
};

template float Foo::As();
template double Foo::As2();

template<int B1> struct B {};
template<class C1> struct C {};
template<class D1, int D2> struct D {};

// Partial ordering with conversion function templates.
struct X0 {
  template<typename T> operator T*() {
    T x = 1; // expected-note{{variable 'x' declared const here}}
    x = 17; // expected-error{{cannot assign to variable 'x' with const-qualified type 'const int'}}
  }

  template<typename T> operator T*() const; // expected-note{{explicit instantiation refers here}}
  template<int V> operator B<V>() const; // expected-note{{explicit instantiation refers here}}
  template<class T, int V> operator C<T[V]>() const; // expected-note{{explicit instantiation refers here}}
#if __cplusplus >= 201103L
  template<int V> operator D<decltype(V), V>() const; // expected-note{{explicit instantiation refers here}}
#endif

  template<typename T> operator const T*() const {
    T x = T();
    return x; // cxx98_20-error{{cannot initialize return object of type 'const char *' with an lvalue of type 'char'}} \
    // cxx98_20-error{{cannot initialize return object of type 'const int *' with an lvalue of type 'int'}} \
    // cxx23-error{{cannot initialize return object of type 'const char *' with an rvalue of type 'char'}} \
    // cxx23-error{{cannot initialize return object of type 'const int *' with an rvalue of type 'int'}}
  }
};

template X0::operator const char*() const; // expected-note{{'X0::operator const char *<char>' requested here}}
template X0::operator const int*(); // expected-note{{'X0::operator const int *<const int>' requested here}}
// FIXME: These diagnostics are printing canonical types.
template X0::operator float*() const; // expected-error{{explicit instantiation of undefined function template 'operator type-parameter-0-0 *'}}
template X0::operator B<0>() const; // expected-error {{undefined function template 'operator B<value-parameter-0-0>'}}
// FIXME: Within the above issue were we print canonical types here, printing the array
// index expression as non-canonical is extra bad.
template X0::operator C<int[1]>() const; // expected-error {{undefined function template 'operator C<type-parameter-0-0[V]>'}}
#if __cplusplus >= 201103L
template X0::operator D<int, 0>() const; // expected-error {{undefined function template 'operator D<decltype(value-parameter-0-0), value-parameter-0-0>'}}
#endif

void test_X0(X0 x0, const X0 &x0c) {
  x0.operator const int*(); // expected-note{{in instantiation of function template specialization}}
  x0.operator float *();
  x0c.operator const char*();
}

namespace PR14211 {
template <class U> struct X {
  void foo(U){}
  template <class T> void foo(T){}

  template <class T> void bar(T){}
  void bar(U){}
};

template void X<int>::foo(int);
template void X<int>::bar(int);
}
