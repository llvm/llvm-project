//RUN: %clang_cc1 %s -triple spir -cl-std=clc++ -verify -fsyntax-only

// Check that pointer to member functions are diagnosed
struct C {
  void f(int n);
};

typedef void (C::*p_t)(int);

template <class T> struct remove_reference { typedef T type; };
template <class T> struct remove_reference<T &> { typedef T type; };

template <typename T>
void templ_test() {
  typename remove_reference<T>::type *ptr; //expected-error{{pointers to functions are not allowed}}
}

void test() {
  void (C::*p)(int);       //expected-error{{pointers to functions are not allowed}}
  p_t p1;                  //expected-error{{pointers to functions are not allowed}}
  templ_test<int (&)()>(); //expected-note{{in instantiation of function template specialization}}
}
