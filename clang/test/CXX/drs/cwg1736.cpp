// RUN: %clang_cc1 -std=c++98 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected,cxx98
// RUN: %clang_cc1 -std=c++11 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected,since-cxx11
// RUN: %clang_cc1 -std=c++14 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected,since-cxx11
// RUN: %clang_cc1 -std=c++17 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected,since-cxx11
// RUN: %clang_cc1 -std=c++20 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected,since-cxx11
// RUN: %clang_cc1 -std=c++23 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected,since-cxx11
// RUN: %clang_cc1 -std=c++2c %s -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected,since-cxx11,since-cxx26

// cxx98-no-diagnostics

namespace cwg1736 { // cwg1736: 3.9
#if __cplusplus >= 201103L
struct S {
  template <class T> S(T t) {
    struct L : S {
      using S::S;
    };
    typename T::type value;
    L l(value); // #cwg1736-l
    // since-cxx26-error@-1 {{variable 'value' is uninitialized when used here}}
    //   since-cxx26-note@#cwg1736-s {{in instantiation of function template specialization 'cwg1736::S::S<cwg1736::Q>' requested here}}
    // since-cxx26-note@-4 {{initialize the variable 'value' to silence this warning}}
    // since-cxx11-error@-5 {{type 'int' cannot be used prior to '::' because it has no members}}
    //   since-cxx11-note@#cwg1736-l {{in instantiation of function template specialization 'cwg1736::S::S<int>' requested here}}
    //   since-cxx11-note@#cwg1736-s {{in instantiation of function template specialization 'cwg1736::S::S<cwg1736::Q>' requested here}}
  }
};
struct Q { typedef int type; } q;
S s(q); // #cwg1736-s
#endif
} // namespace cwg1736
