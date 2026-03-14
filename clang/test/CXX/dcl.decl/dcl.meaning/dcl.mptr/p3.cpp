// RUN: %clang_cc1 -fsyntax-only -verify %s
class A { 
public:
  int& i; 

  A(int& i) : i(i) { }
  
  static int s;
}; 

template<typename T> void ft(T& t) {
  t.*&T::i = 10; // expected-error{{cannot form a pointer-to-member to member 'i' of reference type 'int &'}}
}

void f() {
  int b;
  A a(b); 
  
  int A::*ip = &A::s; // expected-error {{cannot initialize a variable of type 'int A::*' with an rvalue of type 'int *'}}
  a.*&A::s = 10; // expected-error{{right hand operand to .* has non-pointer-to-member type 'int *'}}
  
  a.*&A::i = 10; // expected-error{{cannot form a pointer-to-member to member 'i' of reference type 'int &'}}
  ft(a); // expected-note{{in instantiation of function template specialization 'ft<A>' requested here}}
  
  void A::*p = 0; // expected-error{{'p' declared as a member pointer to void}}
}
