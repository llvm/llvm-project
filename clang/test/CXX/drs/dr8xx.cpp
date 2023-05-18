// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors

#if __cplusplus == 199711L
// expected-no-diagnostics
#endif

namespace dr873 { // dr873: 3.0
#if __cplusplus >= 201103L
template <typename T> void f(T &&);
template <> void f(int &) = delete;  // #dr873-lvalue-ref
template <> void f(int &&) = delete; // #dr873-rvalue-ref
void g(int i) {
  f(i); // calls f<int&>(int&)
  // since-cxx11-error@-1 {{call to deleted function 'f'}}
  //   since-cxx11-note@#dr873-lvalue-ref {{candidate function [with T = int &] has been implicitly deleted}}
  f(0); // calls f<int>(int&&)
  // since-cxx11-error@-1 {{call to deleted function 'f'}}
  //   since-cxx11-note@#dr873-rvalue-ref {{candidate function [with T = int] has been implicitly deleted}}
}
#endif
} // namespace dr873
