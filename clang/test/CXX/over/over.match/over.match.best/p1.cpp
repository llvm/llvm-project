// RUN: %clang_cc1 -fsyntax-only -verify -std=c++1z %s
// expected-no-diagnostics

template<typename T> int &f0(T*, int);
float &f0(void*, int);

void test_f0(int* ip, void *vp) {
  // One argument is better...
  int &ir = f0(ip, 0);
  
  // Prefer non-templates to templates
  float &fr = f0(vp, 0);
}

namespace deduction_guide_example {
  template<typename T> struct A {
    A(T, int*);
    A(A<T>&, int*);
    enum { value };
  };

  template<typename T> struct remove_ref_impl;
  template<typename T> struct remove_ref_impl<T&> { using type = T; };
  template<typename T> using remove_ref = typename remove_ref_impl<T>::type;

  // FIXME: The standard's example is wrong; we add a remove_ref<...> here to
  // fix it.
  template<typename T, int N = remove_ref<T>::value> A(T&&, int*) -> A<T>;
  A a{1, 0};
  extern A<int> a;
  A b{a, 0};

  A<int> *pa = &a;
  A<A<int>&> *pb = &b;
}

// Partial ordering of function template specializations will be tested 
// elsewhere
// FIXME: Initialization by user-defined conversion is tested elsewhere
