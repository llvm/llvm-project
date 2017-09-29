// RUN: %clang_cc1 %s -fcxx-exceptions -fsyntax-only -Wexceptions -verify -std=c++14

struct S {
  //expected-warning@+2 {{attribute 'nothrow' ignored due to conflicting exception specification}}
  //expected-note@+1 {{exception specification declared here}}
  __attribute__((nothrow)) S() noexcept(true);
  //expected-warning@+2 {{attribute 'nothrow' ignored due to conflicting exception specification}}
  //expected-note@+1 {{exception specification declared here}}
  __attribute__((nothrow)) void Func1() noexcept(false);
  __attribute__((nothrow)) void Func3() noexcept;
};

void throwing() noexcept(false);
void non_throwing(bool b = true) noexcept;

template <typename Fn>
struct T {
    __attribute__((nothrow)) void f(Fn) noexcept(Fn());
};

//expected-warning@-3 {{attribute 'nothrow' ignored due to conflicting exception specification}}
//expected-note@-4 {{exception specification declared here}}
template struct T<decltype(throwing)>;
template struct T<decltype(non_throwing)>;
