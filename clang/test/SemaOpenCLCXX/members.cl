//RUN: %clang_cc1 %s -triple spir -cl-std=clc++ -verify -fsyntax-only
//RUN: %clang_cc1 %s -triple spir -cl-std=clc++ -verify -fsyntax-only -DFUNCPTREXT

#ifdef FUNCPTREXT
#pragma OPENCL EXTENSION __cl_clang_function_pointers : enable
//expected-no-diagnostics
#endif

// Check that pointer to member functions are diagnosed
// unless specific clang extension is enabled.
struct C {
  void f(int n);
};

typedef void (C::*p_t)(int);

template <class T> struct remove_reference { typedef T type; };
template <class T> struct remove_reference<T &> { typedef T type; };

template <typename T>
void templ_test() {
  typename remove_reference<T>::type *ptr;
#ifndef FUNCPTREXT
  //expected-error@-2{{pointers to functions are not allowed}}
#endif
}

void test() {
  void (C::*p)(int);
#ifndef FUNCPTREXT
//expected-error@-2{{pointers to functions are not allowed}}
#endif

  p_t p1;
#ifndef FUNCPTREXT
//expected-error@-2{{pointers to functions are not allowed}}
#endif

  templ_test<int (&)()>();
#ifndef FUNCPTREXT
//expected-note@-2{{in instantiation of function template specialization}}
#endif
}
