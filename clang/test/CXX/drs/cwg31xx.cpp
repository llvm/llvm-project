// RUN: %clang_cc1 -std=c++98 -fexceptions -fcxx-exceptions -pedantic-errors %s -verify-directives -verify=expected,cxx98-11
// RUN: %clang_cc1 -std=c++11 -fexceptions -fcxx-exceptions -pedantic-errors %s -verify-directives -verify=expected,cxx98-11
// RUN: %clang_cc1 -std=c++14 -fexceptions -fcxx-exceptions -pedantic-errors %s -verify-directives -verify=expected,since-cxx14
// RUN: %clang_cc1 -std=c++17 -fexceptions -fcxx-exceptions -pedantic-errors %s -verify-directives -verify=expected,since-cxx14
// RUN: %clang_cc1 -std=c++20 -fexceptions -fcxx-exceptions -pedantic-errors %s -verify-directives -verify=expected,since-cxx14
// RUN: %clang_cc1 -std=c++23 -fexceptions -fcxx-exceptions -pedantic-errors %s -verify-directives -verify=expected,since-cxx14
// RUN: %clang_cc1 -std=c++2c -fexceptions -fcxx-exceptions -pedantic-errors %s -verify-directives -verify=expected,since-cxx14

// cxx98-11-no-diagnostics

namespace cwg3128 { // cwg3128: 2.7
#if __cplusplus >= 201103L
void f();
static_assert(noexcept(noexcept(f())), "");
#endif
} // namespace cwg3128

namespace cwg3151 { // cwg3151: 2.7
#if __cplusplus >= 201402L
auto lambda = []{};
struct S : decltype(lambda) {};
#endif
} // namespace cwg3151

namespace cwg3156 { // cwg3156: 3.5
#if __cplusplus >= 201402L
struct C { // #cwg3156-C
  C(int) = delete; // #cwg3156-C-int
  C(){};
};

int x = [b = C(3)](){ return 4; }();
// since-cxx14-error@-1 {{functional-style cast from 'int' to 'C' uses deleted function}}
//   since-cxx14-note@#cwg3156-C-int {{candidate constructor has been explicitly deleted}}
//   since-cxx14-note@#cwg3156-C {{candidate constructor (the implicit copy constructor)}}
//   since-cxx14-note@#cwg3156-C {{candidate constructor (the implicit move constructor)}}
#endif
} // namespace cwg3156

// cwg3172: na
