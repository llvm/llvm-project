// RUN: %clang_cc1 -std=c++14 -fcoroutines -verify %s

struct awaitable {
  bool await_ready();
  void await_suspend(); // FIXME: coroutine_handle
  void await_resume();
} a;

void no_coroutine_traits() {
  co_await a; // expected-error {{need to include <coroutine>}}
}

namespace std {
  template<typename ...T> struct coroutine_traits; // expected-note {{declared here}}
};

void no_specialization() {
  co_await a; // expected-error {{implicit instantiation of undefined template 'std::coroutine_traits<void>'}}
}

template<typename ...T> struct std::coroutine_traits<int, T...> {};

int no_promise_type() {
  co_await a; // expected-error {{this function cannot be a coroutine: 'std::coroutine_traits<int>' has no member named 'promise_type'}}
}

template<> struct std::coroutine_traits<double, double> { typedef int promise_type; };
double bad_promise_type(double) {
  co_await a; // expected-error {{this function cannot be a coroutine: 'std::coroutine_traits<double, double>::promise_type' (aka 'int') is not a class}}
}

struct promise; // expected-note {{forward declaration}}
template<typename ...T> struct std::coroutine_traits<void, T...> { using promise_type = promise; };

  // FIXME: This diagnostic is terrible.
void undefined_promise() { // expected-error {{variable has incomplete type 'promise_type'}}
  co_await a;
}

struct promise {};

void mixed_yield() {
  co_yield 0; // expected-note {{use of 'co_yield'}}
  return; // expected-error {{not allowed in coroutine}}
}

void mixed_await() {
  co_await a; // expected-note {{use of 'co_await'}}
  return; // expected-error {{not allowed in coroutine}}
}

void only_coreturn() {
  co_return; // expected-warning {{'co_return' used in a function that uses neither 'co_await' nor 'co_yield'}}
}

void mixed_coreturn(bool b) {
  if (b)
    // expected-warning@+1 {{'co_return' used in a function that uses neither}}
    co_return; // expected-note {{use of 'co_return'}}
  else
    return; // expected-error {{not allowed in coroutine}}
}

struct CtorDtor {
  CtorDtor() {
    co_yield 0; // expected-error {{'co_yield' cannot be used in a constructor}}
  }
  CtorDtor(awaitable a) {
    // The spec doesn't say this is ill-formed, but it must be.
    co_await a; // expected-error {{'co_await' cannot be used in a constructor}}
  }
  ~CtorDtor() {
    co_return 0; // expected-error {{'co_return' cannot be used in a destructor}}
  }
  // FIXME: The spec says this is ill-formed.
  void operator=(CtorDtor&) {
    co_yield 0;
  }
};

constexpr void constexpr_coroutine() { // expected-error {{never produces a constant expression}}
  co_yield 0; // expected-error {{'co_yield' cannot be used in a constexpr function}} expected-note {{subexpression}}
}

void varargs_coroutine(const char *, ...) {
  co_await a; // expected-error {{'co_await' cannot be used in a varargs function}}
}

struct outer {};

namespace dependent_operator_co_await_lookup {
  template<typename T> void await_template(T t) {
    // no unqualified lookup results
    co_await t; // expected-error {{no member named 'await_ready' in 'dependent_operator_co_await_lookup::not_awaitable'}}
    // expected-error@-1 {{call to function 'operator co_await' that is neither visible in the template definition nor found by argument-dependent lookup}}
  };
  template void await_template(awaitable);

  struct indirectly_awaitable { indirectly_awaitable(outer); };
  awaitable operator co_await(indirectly_awaitable); // expected-note {{should be declared prior to}}
  template void await_template(indirectly_awaitable);

  struct not_awaitable {};
  template void await_template(not_awaitable); // expected-note {{instantiation}}

  template<typename T> void await_template_2(T t) {
    // one unqualified lookup result
    co_await t;
  };
  template void await_template(outer); // expected-note {{instantiation}}
  template void await_template_2(outer);
}
