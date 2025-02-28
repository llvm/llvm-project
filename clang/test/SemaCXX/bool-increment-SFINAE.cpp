// RUN: %clang_cc1 %std_cxx98-14 -fsyntax-only -verify=precxx17 %s
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify=expected %s
// RUN: %clang_cc1 -std=c++17 -DFAILED_CXX17 -fsyntax-only -verify=failcxx17 %s
// RUN: %clang_cc1 %std_cxx20- -fsyntax-only -verify=cxx20 %s
// expected-no-diagnostics

template<class T> auto f(T t) -> decltype(++t); // precxx17-warning {{incrementing expression of type bool is deprecated}}

auto f(...) -> void;
void g() { f(true); } // precxx17-note {{while substituting deduced template arguments}}

#ifdef FAILED_CXX17

template<class T> auto f1(T t) -> decltype(++t); // failcxx17-note {{candidate template ignored: substitution failure [with T = bool]: ISO C++17 does not allow incrementing expression of type bool}}
auto f1(void) -> void; // failcxx17-note {{candidate function not viable: requires 0 arguments, but 1 was provided}}
void g1() { f1(true); } // failcxx17-error {{no matching function for call to 'f1'}}

#endif

#if __cplusplus >= 202002L
template <class T>
concept can_increment = requires(T t) {
  ++t;
};

template <class T>
void f() {
  static_assert(requires(T t) { ++t; }); // cxx20-error {{static assertion failed due to requirement 'requires (bool t) { <<error-expression>>; }'}}
}

int main() {
  f<bool>(); // cxx20-note {{in instantiation of function template specialization 'f<bool>' requested here}}
  static_assert(!can_increment<bool>);

  return 0;
}
#endif
