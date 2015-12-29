// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

class Class_With_Destructor {
  ~Class_With_Destructor() { }
};

template <class T>
class Base { };

template<class T,  // Should be angle bracket instead of comma
class Derived : public Base<T> { // expected-error{{'Derived' cannot be defined in a type specifier}}
  Class_With_Destructor member;
}; // expected-error{{a non-type template parameter cannot have type 'class Derived'}}
   // expected-error@-1{{expected ',' or '>' in template-parameter-list}}
   // expected-warning@-2{{declaration does not declare anything}}

