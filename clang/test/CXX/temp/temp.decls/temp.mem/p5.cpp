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

// Partial ordering with conversion function templates.
struct X0 {
  template<typename T> operator T*() {
    T x = 1; // expected-note{{variable 'x' declared const here}}
    x = 17; // expected-error{{cannot assign to variable 'x' with const-qualified type 'const int'}}
  }

  template<typename T> operator T*() const; // expected-note{{explicit instantiation refers here}}

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
template X0::operator float*() const; // expected-error{{explicit instantiation of undefined function template 'operator type-parameter-0-0 *'}}

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
