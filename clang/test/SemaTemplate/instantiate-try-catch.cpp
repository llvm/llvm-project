// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -std=c++11 -verify %s

template<typename T> struct TryCatch0 {
  void f() {
    try {
    } catch (T&&) { // expected-error 2{{cannot catch exceptions by rvalue reference}}
    }
  }
};

template struct TryCatch0<int&>; // okay
template struct TryCatch0<int&&>; // expected-note{{instantiation}}
template struct TryCatch0<int>; // expected-note{{instantiation}}


namespace PR10232 {
  template <typename T>
  class Templated {
    struct Exception {
    private:
      Exception(const Exception&); // expected-note{{declared private here}}
    };
    void exception() {
      try {
      } catch(Exception e) {  // expected-error{{calling a private constructor of class 'PR10232::Templated<int>::Exception'}}
      }
    }
  };

  template class Templated<int>; // expected-note{{in instantiation of member function 'PR10232::Templated<int>::exception' requested here}}
}
