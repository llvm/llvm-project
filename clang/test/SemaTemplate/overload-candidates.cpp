// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

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
template<typename T> typename boost::enable_if<sizeof(T) == 4, int>::type if_size_4(); // expected-note{{candidate template ignored: requirement 'sizeof(char) == 4' was not satisfied [with T = char]}}
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
  typename boost::enable_if<sizeof(T) == 4, int>::type f(); // expected-error{{failed requirement 'sizeof(char) == 4'; 'enable_if' cannot be used to disable this declaration}}
};
NonTemplateFunction<char> NTFC; // expected-note{{here}}

namespace NS1 {
  template <class A>
  class array {};
}

namespace NS2 {
  template <class A>
  class array {};
}

template <class A>
void foo(NS2::array<A>); // expected-note{{candidate template ignored: could not match 'NS2::array' against 'NS1::array'}}

void test() {
  foo(NS1::array<int>()); // expected-error{{no matching function for call to 'foo'}}
}

namespace std {
  template<bool, typename = void> struct enable_if {};
  template<typename T> struct enable_if<true, T> { typedef T type; };

  template<typename T, T V> struct integral_constant { static const T value = V; };
  typedef integral_constant<bool, false> false_type;
  typedef integral_constant<bool, true> true_type;
};

namespace PR15673 {
  template<typename T>
  struct a_trait : std::false_type {};

  template<typename T,
           typename Requires = typename std::enable_if<a_trait<T>::value>::type>
#if __cplusplus <= 199711L
  // expected-warning@-2 {{default template arguments for a function template are a C++11 extension}}
#endif
  // expected-note@+1 {{candidate template ignored: requirement 'PR15673::a_trait<int>::value' was not satisfied [with T = int]}}
  void foo() {}
  void bar() { foo<int>(); } // expected-error {{no matching function for call to 'foo'}}


  template<typename T>
  struct some_trait : std::false_type {};

  // FIXME: It would be nice to tunnel the 'disabled by enable_if' diagnostic through here.
  template<typename T>
  struct a_pony : std::enable_if<some_trait<T>::value> {};

  template<typename T,
           typename Requires = typename a_pony<T>::type>
#if __cplusplus <= 199711L
  // expected-warning@-2 {{default template arguments for a function template are a C++11 extension}}
#endif
  // FIXME: The source location here is poor.
  void baz() { } // expected-note {{candidate template ignored: substitution failure [with T = int]: no type named 'type' in 'PR15673::a_pony<int>'}}
  void quux() { baz<int>(); } // expected-error {{no matching function for call to 'baz'}}


  // FIXME: This note doesn't make it clear which candidate we rejected.
  template <typename T>
  using unicorns = typename std::enable_if<some_trait<T>::value>::type;
#if __cplusplus <= 199711L
  // expected-warning@-2 {{alias declarations are a C++11 extension}}
#endif
  // expected-note@+7 {{candidate template ignored: requirement 'PR15673::some_trait<int>::value' was not satisfied [with T = int]}}

  template<typename T,
           typename Requires = unicorns<T> >
#if __cplusplus <= 199711L
  // expected-warning@-2 {{default template arguments for a function template are a C++11 extension}}
#endif
  void wibble() {}
  void wobble() { wibble<int>(); } // expected-error {{no matching function for call to 'wibble'}}

  template<typename T>
  struct some_passing_trait : std::true_type {};

#if __cplusplus <= 199711L
  // expected-warning@+4 {{default template arguments for a function template are a C++11 extension}}
  // expected-warning@+4 {{default template arguments for a function template are a C++11 extension}}
#endif
  template<typename T,
           int n = 42,
           typename std::enable_if<n == 43 || (some_passing_trait<T>::value && some_trait<T>::value), int>::type = 0>
  void almost_rangesv3(); // expected-note{{candidate template ignored: requirement '42 == 43 || (PR15673::some_passing_trait<int>::value && PR15673::some_trait<int>::value)' was not satisfied}}
  void test_almost_rangesv3() { almost_rangesv3<int>(); } // expected-error{{no matching function for call to 'almost_rangesv3'}}

  #define CONCEPT_REQUIRES_(...)                                        \
    int x = 42,                                                         \
    typename std::enable_if<(x == 43) || (__VA_ARGS__)>::type = 0

#if __cplusplus <= 199711L
  // expected-warning@+4 {{default template arguments for a function template are a C++11 extension}}
  // expected-warning@+3 {{default template arguments for a function template are a C++11 extension}}
#endif
  template<typename T,
           CONCEPT_REQUIRES_(some_passing_trait<T>::value && some_trait<T>::value)>
  void rangesv3(); // expected-note{{candidate template ignored: requirement 'PR15673::some_trait<int>::value' was not satisfied [with T = int, x = 42]}}
  void test_rangesv3() { rangesv3<int>(); } // expected-error{{no matching function for call to 'rangesv3'}}
}
