// RUN: %clang_cc1 -verify %s -DTEST=1
// RUN: %clang_cc1 -verify %s -DTEST=2
// RUN: %clang_cc1 -verify %s -DTEST=3
// REQUIRES: thread_support

// FIXME: Detection of, or recovery from, stack exhaustion does not work on
// NetBSD at the moment. Since this is a best-effort mitigation for exceeding
// implementation limits, just disable the test.
// UNSUPPORTED: system-netbsd

// asan has own stack-overflow check.
// UNSUPPORTED: asan

// expected-warning@* 0-1{{stack nearly exhausted}}
// expected-note@* 0+{{}}

#if TEST == 1

template<int N> struct X : X<N-1> {};
template<> struct X<0> {};
X<1000> x;

template<typename ...T> struct tuple {};
template<typename ...T> auto f(tuple<T...> t) -> decltype(f(tuple<T...>(t))) {} // expected-error {{exceeded maximum depth}}
void g() { f(tuple<int, int>()); }

int f(X<0>);
template<int N> auto f(X<N>) -> f(X<N-1>());

int k = f(X<1000>());

#elif TEST == 2

namespace template_argument_recursion {
  struct ostream;
  template<typename T> T &&declval();

  namespace mlir {
    template<typename T, typename = decltype(declval<ostream&>() << declval<T&>())>
    ostream &operator<<(ostream& os, const T& obj); // expected-error {{exceeded maximum depth}}
    struct Value;
  }

  void printFunctionalType(ostream &os, mlir::Value &v) { os << v; }
}

#elif TEST == 3

namespace template_parameter_type_recursion {
  struct ostream;
  template<typename T> T &&declval();
  template<bool B, typename T> struct enable_if { using type = T; };

  namespace mlir {
    template<typename T, typename enable_if<declval<ostream&>() << declval<T&>(), void*>::type = nullptr>
    ostream &operator<<(ostream& os, const T& obj); // expected-error {{exceeded maximum depth}}
    struct Value;
  }

  void printFunctionalType(ostream &os, mlir::Value &v) { os << v; }
}

#else
#error unknown test
#endif
