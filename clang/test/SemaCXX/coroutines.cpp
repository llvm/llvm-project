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

template<> struct std::coroutine_traits<double, int> {
  struct promise_type {};
};
double bad_promise_type_2(int) {
  co_yield 0; // expected-error {{no member named 'yield_value' in 'std::coroutine_traits<double, int>::promise_type'}}
}

struct promise; // expected-note {{forward declaration}}
template<typename ...T> struct std::coroutine_traits<void, T...> { using promise_type = promise; };

  // FIXME: This diagnostic is terrible.
void undefined_promise() { // expected-error {{variable has incomplete type 'promise_type'}}
  co_await a;
}

struct yielded_thing { const char *p; short a, b; };

struct not_awaitable {};

struct promise {
  awaitable yield_value(int); // expected-note 2{{candidate}}
  awaitable yield_value(yielded_thing); // expected-note 2{{candidate}}
  not_awaitable yield_value(void()); // expected-note 2{{candidate}}
  void return_void();
  void return_value(int); // expected-note 2{{here}}
};

void yield() {
  co_yield 0;
  co_yield {"foo", 1, 2};
  co_yield {1e100}; // expected-error {{cannot be narrowed}} expected-note {{explicit cast}} expected-warning {{changes value}} expected-warning {{braces around scalar}}
  co_yield {"foo", __LONG_LONG_MAX__}; // expected-error {{cannot be narrowed}} expected-note {{explicit cast}} expected-warning {{changes value}}
  co_yield {"foo"};
  co_yield "foo"; // expected-error {{no matching}}
  co_yield 1.0;
  co_yield yield; // expected-error {{no member named 'await_ready' in 'not_awaitable'}}
}

void coreturn(int n) {
  co_await a;
  if (n == 0)
    co_return 3;
  if (n == 1)
    co_return {4};
  if (n == 2)
    co_return "foo"; // expected-error {{cannot initialize a parameter of type 'int' with an lvalue of type 'const char [4]'}}
  co_return;
}

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

void unevaluated() {
  decltype(co_await a); // expected-error {{cannot be used in an unevaluated context}}
  sizeof(co_await a); // expected-error {{cannot be used in an unevaluated context}}
  typeid(co_await a); // expected-error {{cannot be used in an unevaluated context}}
  decltype(co_yield a); // expected-error {{cannot be used in an unevaluated context}}
  sizeof(co_yield a); // expected-error {{cannot be used in an unevaluated context}}
  typeid(co_yield a); // expected-error {{cannot be used in an unevaluated context}}
}

constexpr void constexpr_coroutine() {
  co_yield 0; // expected-error {{'co_yield' cannot be used in a constexpr function}}
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

struct yield_fn_tag {};
template<> struct std::coroutine_traits<void, yield_fn_tag> {
  struct promise_type {
    // FIXME: add an await_transform overload for functions
    awaitable yield_value(int());
    void return_value(int());
  };
};

namespace placeholder {
  awaitable f(), f(int); // expected-note 4{{possible target}}
  int g(), g(int); // expected-note 2{{candidate}}
  void x() {
    co_await f; // expected-error {{reference to overloaded function}}
  }
  void y() {
    co_yield g; // expected-error {{no matching member function for call to 'yield_value'}}
  }
  void z() {
    co_await a;
    co_return g; // expected-error {{address of overloaded function 'g' does not match required type 'int'}}
  }

  void x(yield_fn_tag) {
    co_await f; // expected-error {{reference to overloaded function}}
  }
  void y(yield_fn_tag) {
    co_yield g;
  }
  void z(yield_fn_tag) {
    co_await a;
    co_return g;
  }
}
