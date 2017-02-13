// RUN: %clang_cc1 -std=c++14 -fcoroutines-ts -verify %s -fcxx-exceptions

void no_coroutine_traits_bad_arg_await() {
  co_await a; // expected-error {{include <experimental/coroutine>}}
  // expected-error@-1 {{use of undeclared identifier 'a'}}
}

void no_coroutine_traits_bad_arg_yield() {
  co_yield a; // expected-error {{include <experimental/coroutine>}}
  // expected-error@-1 {{use of undeclared identifier 'a'}}
}


void no_coroutine_traits_bad_arg_return() {
  co_return a; // expected-error {{include <experimental/coroutine>}}
  // expected-error@-1 {{use of undeclared identifier 'a'}}
}


struct awaitable {
  bool await_ready();
  void await_suspend(); // FIXME: coroutine_handle
  void await_resume();
} a;

struct suspend_always {
  bool await_ready() { return false; }
  void await_suspend() {}
  void await_resume() {}
};

struct suspend_never {
  bool await_ready() { return true; }
  void await_suspend() {}
  void await_resume() {}
};

void no_coroutine_traits() {
  co_await a; // expected-error {{need to include <experimental/coroutine>}}
}

namespace std {
namespace experimental {
template <typename... T>
struct coroutine_traits; // expected-note {{declared here}}
}
}

template<typename Promise> struct coro {};
template <typename Promise, typename... Ps>
struct std::experimental::coroutine_traits<coro<Promise>, Ps...> {
  using promise_type = Promise;
};

void no_specialization() {
  co_await a; // expected-error {{implicit instantiation of undefined template 'std::experimental::coroutine_traits<void>'}}
}

template <typename... T>
struct std::experimental::coroutine_traits<int, T...> {};

int no_promise_type() {
  co_await a; // expected-error {{this function cannot be a coroutine: 'std::experimental::coroutine_traits<int>' has no member named 'promise_type'}}
}

template <>
struct std::experimental::coroutine_traits<double, double> { typedef int promise_type; };
double bad_promise_type(double) {
  co_await a; // expected-error {{this function cannot be a coroutine: 'experimental::coroutine_traits<double, double>::promise_type' (aka 'int') is not a class}}
}

template <>
struct std::experimental::coroutine_traits<double, int> {
  struct promise_type {};
};
double bad_promise_type_2(int) {
  co_yield 0; // expected-error {{no member named 'yield_value' in 'std::experimental::coroutine_traits<double, int>::promise_type'}}
}

struct promise; // expected-note 2{{forward declaration}}
struct promise_void;
struct void_tag {};
template <typename... T>
struct std::experimental::coroutine_traits<void, T...> { using promise_type = promise; };
template <typename... T>
struct std::experimental::coroutine_traits<void, void_tag, T...>
{ using promise_type = promise_void; };

namespace std {
namespace experimental {
template <typename Promise = void>
struct coroutine_handle;
}
}

// FIXME: This diagnostic is terrible.
void undefined_promise() { // expected-error {{variable has incomplete type 'promise_type'}}
  // FIXME: This diagnostic doesn't make any sense.
  // expected-error@-2 {{incomplete definition of type 'promise'}}
  co_await a;
}

struct yielded_thing { const char *p; short a, b; };

struct not_awaitable {};

struct promise {
  void get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  awaitable yield_value(int); // expected-note 2{{candidate}}
  awaitable yield_value(yielded_thing); // expected-note 2{{candidate}}
  not_awaitable yield_value(void()); // expected-note 2{{candidate}}
  void return_value(int); // expected-note 2{{here}}
};

struct promise_void {
  void get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void return_void();
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
    co_return {4}; // expected-warning {{braces around scalar initializer}}
  if (n == 2)
    co_return "foo"; // expected-error {{cannot initialize a parameter of type 'int' with an lvalue of type 'const char [4]'}}
  co_return 42;
}

void mixed_yield() {
  co_yield 0; // expected-note {{use of 'co_yield'}}
  return; // expected-error {{not allowed in coroutine}}
}

void mixed_await() {
  co_await a; // expected-note {{use of 'co_await'}}
  return; // expected-error {{not allowed in coroutine}}
}

void only_coreturn(void_tag) {
  co_return; // OK
}

void mixed_coreturn(void_tag, bool b) {
  if (b)
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
    co_yield 0; // expected-error {{'co_yield' cannot be used in a copy assignment operator}}
  }
  void operator=(CtorDtor const &) {
    co_yield 0; // expected-error {{'co_yield' cannot be used in a copy assignment operator}}
  }
  void operator=(CtorDtor &&) {
    co_await a; // expected-error {{'co_await' cannot be used in a move assignment operator}}
  }
  void operator=(CtorDtor const &&) {
    co_await a; // expected-error {{'co_await' cannot be used in a move assignment operator}}
  }
  void operator=(int) {
    co_await a; // OK. Not a special member
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

constexpr auto constexpr_deduced_return_coroutine() {
  co_yield 0; // expected-error {{'co_yield' cannot be used in a constexpr function}}
  // expected-error@-1 {{'co_yield' cannot be used in a function with a deduced return type}}
}

void varargs_coroutine(const char *, ...) {
  co_await a; // expected-error {{'co_await' cannot be used in a varargs function}}
}

auto deduced_return_coroutine() {
  co_await a; // expected-error {{'co_await' cannot be used in a function with a deduced return type}}
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
template <>
struct std::experimental::coroutine_traits<void, yield_fn_tag> {
  struct promise_type {
    // FIXME: add an await_transform overload for functions
    awaitable yield_value(int());
    void return_value(int());

    suspend_never initial_suspend();
    suspend_never final_suspend();
    void get_return_object();
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

struct bad_promise_1 {
  suspend_always initial_suspend();
  suspend_always final_suspend();
};
coro<bad_promise_1> missing_get_return_object() { // expected-error {{no member named 'get_return_object' in 'bad_promise_1'}}
  co_await a;
}

struct bad_promise_2 {
  coro<bad_promise_2> get_return_object();
  // FIXME: We shouldn't offer a typo-correction here!
  suspend_always final_suspend(); // expected-note {{here}}
};
coro<bad_promise_2> missing_initial_suspend() { // expected-error {{no member named 'initial_suspend' in 'bad_promise_2'}}
  co_await a;
}

struct bad_promise_3 {
  coro<bad_promise_3> get_return_object();
  // FIXME: We shouldn't offer a typo-correction here!
  suspend_always initial_suspend(); // expected-note {{here}}
};
coro<bad_promise_3> missing_final_suspend() { // expected-error {{no member named 'final_suspend' in 'bad_promise_3'}}
  co_await a;
}

struct bad_promise_4 {
  coro<bad_promise_4> get_return_object();
  not_awaitable initial_suspend();
  suspend_always final_suspend();
};
// FIXME: This diagnostic is terrible.
coro<bad_promise_4> bad_initial_suspend() { // expected-error {{no member named 'await_ready' in 'not_awaitable'}}
  co_await a;
}

struct bad_promise_5 {
  coro<bad_promise_5> get_return_object();
  suspend_always initial_suspend();
  not_awaitable final_suspend();
};
// FIXME: This diagnostic is terrible.
coro<bad_promise_5> bad_final_suspend() { // expected-error {{no member named 'await_ready' in 'not_awaitable'}}
  co_await a;
}

struct bad_promise_6 {
  coro<bad_promise_6> get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void return_void();
  void return_value(int) const;
  void return_value(int);
};
coro<bad_promise_6> bad_implicit_return() { // expected-error {{'bad_promise_6' declares both 'return_value' and 'return_void'}}
  co_await a;
}

struct bad_promise_7 {
  coro<bad_promise_7> get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void return_void();
  void set_exception(int *);
};
coro<bad_promise_7> no_std_current_exc() {
  // expected-error@-1 {{you need to include <exception> before defining a coroutine that implicitly uses 'set_exception'}}
  co_await a;
}

namespace std {
int *current_exception();
}

struct bad_promise_8 {
  coro<bad_promise_8> get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void return_void();
  void set_exception();                                   // expected-note {{function not viable}}
  void set_exception(int *) __attribute__((unavailable)); // expected-note {{explicitly made unavailable}}
  void set_exception(void *);                             // expected-note {{candidate function}}
};
coro<bad_promise_8> calls_set_exception() {
  // expected-error@-1 {{call to unavailable member function 'set_exception'}}
  co_await a;
}

template<> struct std::experimental::coroutine_traits<int, int, const char**>
{ using promise_type = promise; };

int main(int, const char**) {
  co_await a; // expected-error {{'co_await' cannot be used in the 'main' function}}
}
