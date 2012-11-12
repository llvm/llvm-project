// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

// Member function declarations with the same name and the same
// parameter-type-list as well as mem- ber function template
// declarations with the same name, the same parameter-type-list, and
// the same template parameter lists cannot be overloaded if any of
// them, but not all, have a ref-qualifier (8.3.5).

class Y { 
  void h() &; 
  void h() const &; 
  void h() &&; 
  void i() &; // expected-note{{previous declaration}}
  void i() const; // expected-error{{cannot overload a member function without a ref-qualifier with a member function with ref-qualifier '&'}}

  template<typename T> void f(T*) &;
  template<typename T> void f(T*) &&;

  template<typename T> void g(T*) &; // expected-note{{previous declaration}}
  template<typename T> void g(T*); // expected-error{{cannot overload a member function without a ref-qualifier with a member function with ref-qualifier '&'}}

  void k(); // expected-note{{previous declaration}}
  void k() &&; // expected-error{{cannot overload a member function with ref-qualifier '&&' with a member function without a ref-qualifier}}
};
