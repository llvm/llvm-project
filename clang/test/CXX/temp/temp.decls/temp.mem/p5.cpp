// RUN: %clang_cc1 -fsyntax-only -verify %s
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
    T x = 1;
    x = 17; // expected-error{{read-only variable is not assignable}}
  }
  
  template<typename T> operator T*() const; // expected-note{{explicit instantiation refers here}}
  
  template<typename T> operator const T*() const {
    T x = T();
    return x; // expected-error{{cannot initialize return object of type 'const char *' with an lvalue of type 'char'}} \
    // expected-error{{cannot initialize return object of type 'const int *' with an lvalue of type 'int'}}
  }
};

template X0::operator const char*() const; // expected-note{{'X0::operator const char *<char>' requested here}}
template X0::operator const int*(); // expected-note{{'X0::operator const int *<const int>' requested here}}
template X0::operator float*() const; // expected-error{{explicit instantiation of undefined function template}}

void test_X0(X0 x0, const X0 &x0c) {
  x0.operator const int*(); // expected-note{{in instantiation of function template specialization}}
  x0.operator float *();
  x0c.operator const char*();
}
