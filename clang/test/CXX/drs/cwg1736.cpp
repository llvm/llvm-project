// RUN: %clang_cc1 -std=c++98 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify=expected,cxx98
// RUN: %clang_cc1 -std=c++11 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify=expected,since-cxx11
// RUN: %clang_cc1 -std=c++14 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify=expected,since-cxx11
// RUN: %clang_cc1 -std=c++17 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify=expected,since-cxx11
// RUN: %clang_cc1 -std=c++20 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify=expected,since-cxx11
// RUN: %clang_cc1 -std=c++23 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify=expected,since-cxx11
// RUN: %clang_cc1 -std=c++2c %s -fexceptions -fcxx-exceptions -pedantic-errors -verify=expected,since-cxx11

// cxx98-no-diagnostics

namespace cwg1736 { // cwg1736: 3.9
#if __cplusplus >= 201103L
struct S {
  template <class T> S(T t) {
    struct L : S {
      using S::S;
    };
    typename T::type value;
    // since-cxx11-error@-1 {{type 'int' cannot be used prior to '::' because it has no members}}
    //   since-cxx11-note@#cwg1736-l {{in instantiation of function template specialization 'cwg1736::S::S<int>' requested here}}
    //   since-cxx11-note@#cwg1736-s {{in instantiation of function template specialization 'cwg1736::S::S<cwg1736::Q>' requested here}}
    L l(value); // #cwg1736-l
  }
};
struct Q { typedef int type; } q;
S s(q); // #cwg1736-s
#endif
} // namespace cwg1736
