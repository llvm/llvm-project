// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T>
const T& min(const T&, const T&); // expected-note{{candidate template ignored: deduced conflicting types for parameter 'T' ('int' vs. 'long')}}

void test_min() {
  (void)min(1, 2l); // expected-error{{no matching function for call to 'min'}}
}

template<typename R, typename T>
R *dyn_cast(const T&); // expected-note{{candidate template ignored: couldn't infer template argument 'R'}}

void test_dyn_cast(int* ptr) {
  (void)dyn_cast(ptr); // expected-error{{no matching function for call to 'dyn_cast'}}
}

template<int I, typename T> 
  void get(const T&); // expected-note{{candidate template ignored: invalid explicitly-specified argument for template parameter 'I'}}
template<template<class T> class, typename T> 
  void get(const T&); // expected-note{{candidate template ignored: invalid explicitly-specified argument for 1st template parameter}}

void test_get(void *ptr) {
  get<int>(ptr); // expected-error{{no matching function for call to 'get'}}
}

template<typename T>
  typename T::type get_type(const T&); // expected-note{{candidate template ignored: substitution failure [with T = int *]: type 'int *' cannot be used prior to '::'}}
template<typename T>
  void get_type(T *, int[(int)sizeof(T) - 9] = 0); // expected-note{{candidate template ignored: substitution failure [with T = int]: array size is negative}}

void test_get_type(int *ptr) {
  (void)get_type(ptr); // expected-error{{no matching function for call to 'get_type'}}
}

struct X {
  template<typename T>
  const T& min(const T&, const T&); // expected-note{{candidate template ignored: deduced conflicting types for parameter 'T' ('int' vs. 'long')}}
};

void test_X_min(X x) {
  (void)x.min(1, 2l); // expected-error{{no matching member function for call to 'min'}}
}

namespace boost {
  template<bool, typename = void> struct enable_if {};
  template<typename T> struct enable_if<true, T> { typedef T type; };
}
template<typename T> typename boost::enable_if<sizeof(T) == 4, int>::type if_size_4(); // expected-note{{candidate template ignored: disabled by 'enable_if' [with T = char]}}
int k = if_size_4<char>(); // expected-error{{no matching function}}

namespace llvm {
  template<typename Cond, typename T = void> struct enable_if : boost::enable_if<Cond::value, T> {};
}
template<typename T> struct is_int { enum { value = false }; };
template<> struct is_int<int> { enum { value = true }; };
template<typename T> typename llvm::enable_if<is_int<T> >::type if_int(); // expected-note{{candidate template ignored: disabled by 'enable_if' [with T = char]}}
void test_if_int() {
  if_int<char>(); // expected-error{{no matching function}}
}

template<typename T> struct NonTemplateFunction {
  typename boost::enable_if<sizeof(T) == 4, int>::type f(); // expected-error{{no type named 'type' in 'boost::enable_if<false, int>'; 'enable_if' cannot be used to disable this declaration}}
};
NonTemplateFunction<char> NTFC; // expected-note{{here}}
