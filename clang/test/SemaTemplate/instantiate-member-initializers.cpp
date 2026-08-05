// RUN: %clang_cc1 -fsyntax-only -Wall -verify %s

template<typename T> struct A {
  A() : a(1) { } // expected-error{{cannot initialize a member subobject of type 'void *' with an rvalue of type 'int'}}

  T a;
};

A<int> a0;
A<void*> a1; // expected-note{{in instantiation of member function 'A<void *>::A' requested here}}

template<typename T> struct B {
  B() : b(1), // expected-warning {{field 'b' will be initialized after field 'a'}}
    a(2) { }
  
  int a;
  int b;
};

B<int> b0; // expected-note {{in instantiation of member function 'B<int>::B' requested here}}

template <class T> struct AA { AA(int); };
template <class T> class BB : public AA<T> {
public:
  BB() : AA<T>(1) {}
};
BB<int> x;

struct X {
  X();
};
template<typename T>
struct Y {
  Y() : x() {}
  X x;
};
Y<int> y;

template<typename T> struct Array {
  int a[3];
  Array() : a() {}
};
Array<int> s;
