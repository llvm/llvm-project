// RUN: %clang_cc1 -std=c++14 -fcoroutines-ts -verify %s -fcxx-exceptions -fexceptions -Wunused-result

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

void no_coroutine_traits() {
  co_await 4; // expected-error {{std::experimental::coroutine_traits type was not found; include <experimental/coroutine>}}
}

namespace std {
namespace experimental {

template <class... Args>
struct void_t_imp {
  using type = void;
};
template <class... Args>
using void_t = typename void_t_imp<Args...>::type;

template <class T, class = void>
struct traits_sfinae_base {};

template <class T>
struct traits_sfinae_base<T, void_t<typename T::promise_type>> {
  using promise_type = typename T::promise_type;
};

template <class Ret, class... Args>
struct coroutine_traits : public traits_sfinae_base<Ret> {};
}}  // namespace std::experimental

template<typename Promise> struct coro {};
template <typename Promise, typename... Ps>
struct std::experimental::coroutine_traits<coro<Promise>, Ps...> {
  using promise_type = Promise;
};

struct awaitable {
  bool await_ready();
  template <typename F> void await_suspend(F);
  void await_resume();
} a;

struct suspend_always {
  bool await_ready() { return false; }
  template <typename F> void await_suspend(F);
  void await_resume() {}
};

struct suspend_never {
  bool await_ready() { return true; }
  template <typename F> void await_suspend(F);
  void await_resume() {}
};

struct auto_await_suspend {
  bool await_ready();
  template <typename F> auto await_suspend(F) {}
  void await_resume();
};

struct DummyVoidTag {};
DummyVoidTag no_specialization() { // expected-error {{this function cannot be a coroutine: 'std::experimental::coroutine_traits<DummyVoidTag>' has no member named 'promise_type'}}
  co_await a;
}

template <typename... T>
struct std::experimental::coroutine_traits<int, T...> {};

int no_promise_type() { // expected-error {{this function cannot be a coroutine: 'std::experimental::coroutine_traits<int>' has no member named 'promise_type'}}
  co_await a;
}

template <>
struct std::experimental::coroutine_traits<double, double> { typedef int promise_type; };
double bad_promise_type(double) { // expected-error {{this function cannot be a coroutine: 'experimental::coroutine_traits<double, double>::promise_type' (aka 'int') is not a class}}
  co_await a;
}

template <>
struct std::experimental::coroutine_traits<double, int> {
  struct promise_type {};
};
double bad_promise_type_2(int) { // expected-error {{no member named 'initial_suspend'}}
  co_yield 0; // expected-error {{no member named 'yield_value' in 'std::experimental::coroutine_traits<double, int>::promise_type'}}
}

struct promise; // expected-note {{forward declaration}}
struct promise_void;
struct void_tag {};
template <typename... T>
struct std::experimental::coroutine_traits<void, T...> { using promise_type = promise; };
template <typename... T>
struct std::experimental::coroutine_traits<void, void_tag, T...>
{ using promise_type = promise_void; };

// FIXME: This diagnostic is terrible.
void undefined_promise() { // expected-error {{this function cannot be a coroutine: 'experimental::coroutine_traits<void>::promise_type' (aka 'promise') is an incomplete type}}
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
  void unhandled_exception();
};

struct promise_void {
  void get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void return_void();
  void unhandled_exception();
};

void no_coroutine_handle() { // expected-error {{std::experimental::coroutine_handle type was not found; include <experimental/coroutine> before defining a coroutine}}
  //expected-note@-1 {{call to 'initial_suspend' implicitly required by the initial suspend point}}
  co_return 5; //expected-note {{function is a coroutine due to use of 'co_return' here}}
}

namespace std {
namespace experimental {
template <class PromiseType = void>
struct coroutine_handle {
  static coroutine_handle from_address(void *);
};
template <>
struct coroutine_handle<void> {
  template <class PromiseType>
  coroutine_handle(coroutine_handle<PromiseType>);
  static coroutine_handle from_address(void *);
};
}} // namespace std::experimental

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

void check_auto_await_suspend() {
  co_await auto_await_suspend{}; // Should compile successfully.
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

template <class T>
void co_await_non_dependent_arg(T) {
  co_await a;
}
template void co_await_non_dependent_arg(int);

void mixed_yield() {
  co_yield 0; // expected-note {{use of 'co_yield'}}
  return; // expected-error {{not allowed in coroutine}}
}

void mixed_yield_invalid() {
  co_yield blah; // expected-error {{use of undeclared identifier}}
  // expected-note@-1 {{function is a coroutine due to use of 'co_yield'}}
  return; // expected-error {{return statement not allowed in coroutine}}
}

template <class T>
void mixed_yield_template(T) {
  co_yield blah; // expected-error {{use of undeclared identifier}}
  // expected-note@-1 {{function is a coroutine due to use of 'co_yield'}}
  return; // expected-error {{return statement not allowed in coroutine}}
}

template <class T>
void mixed_yield_template2(T) {
  co_yield 42;
  // expected-note@-1 {{function is a coroutine due to use of 'co_yield'}}
  return; // expected-error {{return statement not allowed in coroutine}}
}

template <class T>
void mixed_yield_template3(T v) {
  co_yield blah(v);
  // expected-note@-1 {{function is a coroutine due to use of 'co_yield'}}
  return; // expected-error {{return statement not allowed in coroutine}}
}

void mixed_await() {
  co_await a; // expected-note {{use of 'co_await'}}
  return; // expected-error {{not allowed in coroutine}}
}

void mixed_await_invalid() {
  co_await 42; // expected-error {{'int' is not a structure or union}}
  // expected-note@-1 {{function is a coroutine due to use of 'co_await'}}
  return; // expected-error {{not allowed in coroutine}}
}

template <class T>
void mixed_await_template(T) {
  co_await 42;
  // expected-note@-1 {{function is a coroutine due to use of 'co_await'}}
  return; // expected-error {{not allowed in coroutine}}
}

template <class T>
void mixed_await_template2(T v) {
  co_await v; // expected-error {{'long' is not a structure or union}}
  // expected-note@-1 {{function is a coroutine due to use of 'co_await'}}
  return; // expected-error {{not allowed in coroutine}}
}
template void mixed_await_template2(long); // expected-note {{requested here}}

void only_coreturn(void_tag) {
  co_return; // OK
}

void mixed_coreturn(void_tag, bool b) {
  if (b)
    co_return; // expected-note {{use of 'co_return'}}
  else
    return; // expected-error {{not allowed in coroutine}}
}

void mixed_coreturn_invalid(bool b) {
  if (b)
    co_return; // expected-note {{use of 'co_return'}}
    // expected-error@-1 {{no member named 'return_void' in 'promise'}}
  else
    return; // expected-error {{not allowed in coroutine}}
}

template <class T>
void mixed_coreturn_template(void_tag, bool b, T v) {
  if (b)
    co_return v; // expected-note {{use of 'co_return'}}
    // expected-error@-1 {{no member named 'return_value' in 'promise_void'}}
  else
    return; // expected-error {{not allowed in coroutine}}
}
template void mixed_coreturn_template(void_tag, bool, int); // expected-note {{requested here}}

template <class T>
void mixed_coreturn_template2(bool b, T) {
  if (b)
    co_return v; // expected-note {{use of 'co_return'}}
    // expected-error@-1 {{use of undeclared identifier 'v'}}
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
struct await_arg_1 {};
struct await_arg_2 {};

namespace adl_ns {
struct coawait_arg_type {};
awaitable operator co_await(coawait_arg_type);
}

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

  struct transform_awaitable {};
  struct transformed {};

  struct transform_promise {
    typedef transform_awaitable await_arg;
    coro<transform_promise> get_return_object();
    transformed initial_suspend();
    ::adl_ns::coawait_arg_type final_suspend();
    transformed await_transform(transform_awaitable);
    void unhandled_exception();
    void return_void();
  };
  template <class AwaitArg>
  struct basic_promise {
    typedef AwaitArg await_arg;
    coro<basic_promise> get_return_object();
    awaitable initial_suspend();
    awaitable final_suspend();
    void unhandled_exception();
    void return_void();
  };

  awaitable operator co_await(await_arg_1);

  template <typename T, typename U>
  coro<T> await_template_3(U t) {
    co_await t;
  }

  template coro<basic_promise<await_arg_1>> await_template_3<basic_promise<await_arg_1>>(await_arg_1);

  template <class T, int I = 0>
  struct dependent_member {
    coro<T> mem_fn() const {
      co_await typename T::await_arg{}; // expected-error {{call to function 'operator co_await'}}}
    }
    template <class U>
    coro<T> dep_mem_fn(U t) {
      co_await t;
    }
  };

  template <>
  struct dependent_member<long> {
    // FIXME this diagnostic is terrible
    coro<transform_promise> mem_fn() const { // expected-error {{no member named 'await_ready' in 'dependent_operator_co_await_lookup::transformed'}}
      // expected-note@-1 {{call to 'initial_suspend' implicitly required by the initial suspend point}}
      // expected-note@+1 {{function is a coroutine due to use of 'co_await' here}}
      co_await transform_awaitable{};
      // expected-error@-1 {{no member named 'await_ready'}}
    }
    template <class R, class U>
    coro<R> dep_mem_fn(U u) { co_await u; }
  };

  awaitable operator co_await(await_arg_2); // expected-note {{'operator co_await' should be declared prior to the call site}}

  template struct dependent_member<basic_promise<await_arg_1>, 0>;
  template struct dependent_member<basic_promise<await_arg_2>, 0>; // expected-note {{in instantiation}}

  template <>
  coro<transform_promise>
      // FIXME this diagnostic is terrible
      dependent_member<long>::dep_mem_fn<transform_promise>(int) { // expected-error {{no member named 'await_ready' in 'dependent_operator_co_await_lookup::transformed'}}
    //expected-note@-1 {{call to 'initial_suspend' implicitly required by the initial suspend point}}
    //expected-note@+1 {{function is a coroutine due to use of 'co_await' here}}
    co_await transform_awaitable{};
    // expected-error@-1 {{no member named 'await_ready'}}
  }

  void operator co_await(transform_awaitable) = delete;
  awaitable operator co_await(transformed);

  template coro<transform_promise>
      dependent_member<long>::dep_mem_fn<transform_promise>(transform_awaitable);

  template <>
  coro<transform_promise> dependent_member<long>::dep_mem_fn<transform_promise>(long) {
    co_await transform_awaitable{};
  }

  template <>
  struct dependent_member<int> {
    coro<transform_promise> mem_fn() const {
      co_await transform_awaitable{};
    }
  };

  template coro<transform_promise> await_template_3<transform_promise>(transform_awaitable);
  template struct dependent_member<transform_promise>;
  template coro<transform_promise> dependent_member<transform_promise>::dep_mem_fn(transform_awaitable);
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
    void unhandled_exception();
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
  void unhandled_exception();
  void return_void();
};
coro<bad_promise_1> missing_get_return_object() { // expected-error {{no member named 'get_return_object' in 'bad_promise_1'}}
  co_await a;
}

struct bad_promise_2 {
  coro<bad_promise_2> get_return_object();
  suspend_always final_suspend();
  void unhandled_exception();
  void return_void();
};
// FIXME: This shouldn't happen twice
coro<bad_promise_2> missing_initial_suspend() { // expected-error {{no member named 'initial_suspend' in 'bad_promise_2'}}
  co_await a;
}

struct bad_promise_3 {
  coro<bad_promise_3> get_return_object();
  suspend_always initial_suspend();
  void unhandled_exception();
  void return_void();
};
coro<bad_promise_3> missing_final_suspend() { // expected-error {{no member named 'final_suspend' in 'bad_promise_3'}}
  co_await a;
}

struct bad_promise_4 {
  coro<bad_promise_4> get_return_object();
  not_awaitable initial_suspend();
  suspend_always final_suspend();
  void return_void();
};
// FIXME: This diagnostic is terrible.
coro<bad_promise_4> bad_initial_suspend() { // expected-error {{no member named 'await_ready' in 'not_awaitable'}}
  // expected-note@-1 {{call to 'initial_suspend' implicitly required by the initial suspend point}}
  co_await a; // expected-note {{function is a coroutine due to use of 'co_await' here}}
}

struct bad_promise_5 {
  coro<bad_promise_5> get_return_object();
  suspend_always initial_suspend();
  not_awaitable final_suspend();
  void return_void();
};
// FIXME: This diagnostic is terrible.
coro<bad_promise_5> bad_final_suspend() { // expected-error {{no member named 'await_ready' in 'not_awaitable'}}
  // expected-note@-1 {{call to 'final_suspend' implicitly required by the final suspend point}}
  co_await a; // expected-note {{function is a coroutine due to use of 'co_await' here}}
}

struct bad_promise_6 {
  coro<bad_promise_6> get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void unhandled_exception();
  void return_void();           // expected-note 2 {{member 'return_void' first declared here}}
  void return_value(int) const; // expected-note 2 {{member 'return_value' first declared here}}
  void return_value(int);
};
coro<bad_promise_6> bad_implicit_return() { // expected-error {{'bad_promise_6' declares both 'return_value' and 'return_void'}}
  co_await a;
}

template <class T>
coro<T> bad_implicit_return_dependent(T) { // expected-error {{'bad_promise_6' declares both 'return_value' and 'return_void'}}
  co_await a;
}
template coro<bad_promise_6> bad_implicit_return_dependent(bad_promise_6); // expected-note {{in instantiation}}

struct bad_promise_7 { // expected-note 2 {{defined here}}
  coro<bad_promise_7> get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void return_void();
};
coro<bad_promise_7> no_unhandled_exception() { // expected-error {{'bad_promise_7' is required to declare the member 'unhandled_exception()'}}
  co_await a;
}

template <class T>
coro<T> no_unhandled_exception_dependent(T) { // expected-error {{'bad_promise_7' is required to declare the member 'unhandled_exception()'}}
  co_await a;
}
template coro<bad_promise_7> no_unhandled_exception_dependent(bad_promise_7); // expected-note {{in instantiation}}

struct bad_promise_base {
private:
  void return_void();
};
struct bad_promise_8 : bad_promise_base {
  coro<bad_promise_8> get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void unhandled_exception() __attribute__((unavailable)); // expected-note 2 {{made unavailable}}
  void unhandled_exception() const;                        // expected-note 2 {{candidate}}
  void unhandled_exception(void *) const;                  // expected-note 2 {{requires 1 argument, but 0 were provided}}
};
coro<bad_promise_8> calls_unhandled_exception() {
  // expected-error@-1 {{call to unavailable member function 'unhandled_exception'}}
  // FIXME: also warn about private 'return_void' here. Even though building
  // the call to unhandled_exception has already failed.
  co_await a;
}

template <class T>
coro<T> calls_unhandled_exception_dependent(T) {
  // expected-error@-1 {{call to unavailable member function 'unhandled_exception'}}
  co_await a;
}
template coro<bad_promise_8> calls_unhandled_exception_dependent(bad_promise_8); // expected-note {{in instantiation}}

struct bad_promise_9 {
  coro<bad_promise_9> get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void await_transform(void *);                                // expected-note {{candidate}}
  awaitable await_transform(int) __attribute__((unavailable)); // expected-note {{explicitly made unavailable}}
  void return_void();
  void unhandled_exception();
};
coro<bad_promise_9> calls_await_transform() {
  co_await 42; // expected-error {{call to unavailable member function 'await_transform'}}
  // expected-note@-1 {{call to 'await_transform' implicitly required by 'co_await' here}}
}

struct bad_promise_10 {
  coro<bad_promise_10> get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  int await_transform;
  void return_void();
  void unhandled_exception();
};
coro<bad_promise_10> bad_coawait() {
  // FIXME this diagnostic is terrible
  co_await 42; // expected-error {{called object type 'int' is not a function or function pointer}}
  // expected-note@-1 {{call to 'await_transform' implicitly required by 'co_await' here}}
}

struct call_operator {
  template <class... Args>
  awaitable operator()(Args...) const { return a; }
};
void ret_void();
struct good_promise_1 {
  coro<good_promise_1> get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void unhandled_exception();
  static const call_operator await_transform;
  using Fn = void (*)();
  Fn return_void = ret_void;
};
const call_operator good_promise_1::await_transform;
coro<good_promise_1> ok_static_coawait() {
  // FIXME this diagnostic is terrible
  co_await 42;
}

template<> struct std::experimental::coroutine_traits<int, int, const char**>
{ using promise_type = promise; };

int main(int, const char**) {
  co_await a; // expected-error {{'co_await' cannot be used in the 'main' function}}
}

struct good_promise_2 {
  float get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void return_void();
  void unhandled_exception();
};
template<> struct std::experimental::coroutine_handle<good_promise_2> {};

template<> struct std::experimental::coroutine_traits<float>
{ using promise_type = good_promise_2; };

float badly_specialized_coro_handle() { // expected-error {{std::experimental::coroutine_handle missing a member named 'from_address'}}
  //expected-note@-1 {{call to 'initial_suspend' implicitly required by the initial suspend point}}
  co_return; //expected-note {{function is a coroutine due to use of 'co_return' here}}
}

namespace std {
  struct nothrow_t {};
  constexpr nothrow_t nothrow = {};
}

using SizeT = decltype(sizeof(int));

void* operator new(SizeT __sz, const std::nothrow_t&) noexcept;
void  operator delete(void* __p, const std::nothrow_t&) noexcept;



struct promise_on_alloc_failure_tag {};

template<>
struct std::experimental::coroutine_traits<int, promise_on_alloc_failure_tag> {
  struct promise_type {
    int get_return_object() {}
    suspend_always initial_suspend() { return {}; }
    suspend_always final_suspend() { return {}; }
    void return_void() {}
    int get_return_object_on_allocation_failure(); // expected-error{{'promise_type': 'get_return_object_on_allocation_failure()' must be a static member function}}
    void unhandled_exception();
  };
};

extern "C" int f(promise_on_alloc_failure_tag) {
  co_return; //expected-note {{function is a coroutine due to use of 'co_return' here}}
}

struct bad_promise_11 {
  coro<bad_promise_11> get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void unhandled_exception();
  void return_void();

private:
  static coro<bad_promise_11> get_return_object_on_allocation_failure(); // expected-note 2 {{declared private here}}
};
coro<bad_promise_11> private_alloc_failure_handler() {
  // expected-error@-1 {{'get_return_object_on_allocation_failure' is a private member of 'bad_promise_11'}}
  co_return; // FIXME: Add a "declared coroutine here" note.
}

template <class T>
coro<T> dependent_private_alloc_failure_handler(T) {
  // expected-error@-1 {{'get_return_object_on_allocation_failure' is a private member of 'bad_promise_11'}}
  co_return; // FIXME: Add a "declared coroutine here" note.
}
template coro<bad_promise_11> dependent_private_alloc_failure_handler(bad_promise_11);
// expected-note@-1 {{requested here}}

struct bad_promise_12 {
  coro<bad_promise_12> get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void unhandled_exception();
  void return_void();
  static coro<bad_promise_12> get_return_object_on_allocation_failure();

  static void* operator new(SizeT);
  // expected-error@-1 2 {{'operator new' is required to have a non-throwing noexcept specification when the promise type declares 'get_return_object_on_allocation_failure()'}}
};
coro<bad_promise_12> throwing_in_class_new() { // expected-note {{call to 'operator new' implicitly required by coroutine function here}}
  co_return;
}

template <class T>
coro<T> dependent_throwing_in_class_new(T) { // expected-note {{call to 'operator new' implicitly required by coroutine function here}}
   co_return;
}
template coro<bad_promise_12> dependent_throwing_in_class_new(bad_promise_12); // expected-note {{requested here}}


struct good_promise_13 {
  coro<good_promise_13> get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void unhandled_exception();
  void return_void();
  static coro<good_promise_13> get_return_object_on_allocation_failure();
};
coro<good_promise_13> uses_nothrow_new() {
  co_return;
}

template <class T>
coro<T> dependent_uses_nothrow_new(T) {
   co_return;
}
template coro<good_promise_13> dependent_uses_nothrow_new(good_promise_13);

struct good_promise_custom_new_operator {
  coro<good_promise_custom_new_operator> get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void return_void();
  void unhandled_exception();
  void *operator new(SizeT, double, float, int);
};

coro<good_promise_custom_new_operator>
good_coroutine_calls_custom_new_operator(double, float, int) {
  co_return;
}

struct coroutine_nonstatic_member_struct;

struct good_promise_nonstatic_member_custom_new_operator {
  coro<good_promise_nonstatic_member_custom_new_operator> get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void return_void();
  void unhandled_exception();
  void *operator new(SizeT, coroutine_nonstatic_member_struct &, double);
};

struct good_promise_noexcept_custom_new_operator {
  static coro<good_promise_noexcept_custom_new_operator> get_return_object_on_allocation_failure();
  coro<good_promise_noexcept_custom_new_operator> get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void return_void();
  void unhandled_exception();
  void *operator new(SizeT, double, float, int) noexcept;
};

coro<good_promise_noexcept_custom_new_operator>
good_coroutine_calls_noexcept_custom_new_operator(double, float, int) {
  co_return;
}

struct mismatch_gro_type_tag1 {};
template<>
struct std::experimental::coroutine_traits<int, mismatch_gro_type_tag1> {
  struct promise_type {
    void get_return_object() {} //expected-note {{member 'get_return_object' declared here}}
    suspend_always initial_suspend() { return {}; }
    suspend_always final_suspend() { return {}; }
    void return_void() {}
    void unhandled_exception();
  };
};

extern "C" int f(mismatch_gro_type_tag1) { 
  // expected-error@-1 {{cannot initialize return object of type 'int' with an rvalue of type 'void'}}
  co_return; //expected-note {{function is a coroutine due to use of 'co_return' here}}
}

struct mismatch_gro_type_tag2 {};
template<>
struct std::experimental::coroutine_traits<int, mismatch_gro_type_tag2> {
  struct promise_type {
    void *get_return_object() {} //expected-note {{member 'get_return_object' declared here}}
    suspend_always initial_suspend() { return {}; }
    suspend_always final_suspend() { return {}; }
    void return_void() {}
    void unhandled_exception();
  };
};

extern "C" int f(mismatch_gro_type_tag2) { 
  // expected-error@-1 {{cannot initialize return object of type 'int' with an lvalue of type 'void *'}}
  co_return; //expected-note {{function is a coroutine due to use of 'co_return' here}}
}

struct mismatch_gro_type_tag3 {};
template<>
struct std::experimental::coroutine_traits<int, mismatch_gro_type_tag3> {
  struct promise_type {
    int get_return_object() {}
    static void get_return_object_on_allocation_failure() {} //expected-note {{member 'get_return_object_on_allocation_failure' declared here}}
    suspend_always initial_suspend() { return {}; }
    suspend_always final_suspend() { return {}; }
    void return_void() {}
    void unhandled_exception();
  };
};

extern "C" int f(mismatch_gro_type_tag3) { 
  // expected-error@-1 {{cannot initialize return object of type 'int' with an rvalue of type 'void'}}
  co_return; //expected-note {{function is a coroutine due to use of 'co_return' here}}
}


struct mismatch_gro_type_tag4 {};
template<>
struct std::experimental::coroutine_traits<int, mismatch_gro_type_tag4> {
  struct promise_type {
    int get_return_object() {}
    static char *get_return_object_on_allocation_failure() {} //expected-note {{member 'get_return_object_on_allocation_failure' declared}}
    suspend_always initial_suspend() { return {}; }
    suspend_always final_suspend() { return {}; }
    void return_void() {}
    void unhandled_exception();
  };
};

extern "C" int f(mismatch_gro_type_tag4) { 
  // expected-error@-1 {{cannot initialize return object of type 'int' with an rvalue of type 'char *'}}
  co_return; //expected-note {{function is a coroutine due to use of 'co_return' here}}
}

struct bad_promise_no_return_func { // expected-note {{'bad_promise_no_return_func' defined here}}
  coro<bad_promise_no_return_func> get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void unhandled_exception();
};
// FIXME: The PDTS currently specifies this as UB, technically forbidding a
// diagnostic.
coro<bad_promise_no_return_func> no_return_value_or_return_void() {
  // expected-error@-1 {{'bad_promise_no_return_func' must declare either 'return_value' or 'return_void'}}
  co_await a;
}

struct bad_await_suspend_return {
  bool await_ready();
  // expected-error@+1 {{return type of 'await_suspend' is required to be 'void' or 'bool' (have 'char')}}
  char await_suspend(std::experimental::coroutine_handle<>);
  void await_resume();
};
struct bad_await_ready_return {
  // expected-note@+1 {{return type of 'await_ready' is required to be contextually convertible to 'bool'}}
  void await_ready();
  bool await_suspend(std::experimental::coroutine_handle<>);
  void await_resume();
};
struct await_ready_explicit_bool {
  struct BoolT {
    explicit operator bool() const;
  };
  BoolT await_ready();
  void await_suspend(std::experimental::coroutine_handle<>);
  void await_resume();
};
template <class SuspendTy>
struct await_suspend_type_test {
  bool await_ready();
  // expected-error@+2 {{return type of 'await_suspend' is required to be 'void' or 'bool' (have 'bool &')}}
  // expected-error@+1 {{return type of 'await_suspend' is required to be 'void' or 'bool' (have 'bool &&')}}
  SuspendTy await_suspend(std::experimental::coroutine_handle<>);
  void await_resume();
};
void test_bad_suspend() {
  {
    // FIXME: The actual error emitted here is terrible, and no number of notes can save it.
    bad_await_ready_return a;
    // expected-error@+1 {{value of type 'void' is not contextually convertible to 'bool'}}
    co_await a; // expected-note {{call to 'await_ready' implicitly required by coroutine function here}}
  }
  {
    bad_await_suspend_return b;
    co_await b; // expected-note {{call to 'await_suspend' implicitly required by coroutine function here}}
  }
  {
    await_ready_explicit_bool c;
    co_await c; // OK
  }
  {
    await_suspend_type_test<bool &&> a;
    await_suspend_type_test<bool &> b;
    await_suspend_type_test<const void> c;
    await_suspend_type_test<const volatile bool> d;
    co_await a; // expected-note {{call to 'await_suspend' implicitly required by coroutine function here}}
    co_await b; // expected-note {{call to 'await_suspend' implicitly required by coroutine function here}}
    co_await c; // OK
    co_await d; // OK
  }
}

template <int ID = 0>
struct NoCopy {
  NoCopy(NoCopy const&) = delete; // expected-note 2 {{deleted here}}
};
template <class T, class U>
void test_dependent_param(T t, U) {
  // expected-error@-1 {{call to deleted constructor of 'NoCopy<0>'}}
  // expected-error@-2 {{call to deleted constructor of 'NoCopy<1>'}}
  ((void)t);
  co_return 42;
}
template void test_dependent_param(NoCopy<0>, NoCopy<1>); // expected-note {{requested here}}

namespace CoroHandleMemberFunctionTest {
struct CoroMemberTag {};
struct BadCoroMemberTag {};

template <class T, class U>
constexpr bool IsSameV = false;
template <class T>
constexpr bool IsSameV<T, T> = true;

template <class T>
struct TypeTest {
  template <class U>
  static constexpr bool IsSame = IsSameV<T, U>;

  template <class... Args>
  static constexpr bool MatchesArgs = IsSameV<T,
                                              std::experimental::coroutine_traits<CoroMemberTag, Args...>>;
};

template <class T>
struct AwaitReturnsType {
  bool await_ready() const;
  void await_suspend(...) const;
  T await_resume() const;
};

template <class... CoroTraitsArgs>
struct CoroMemberPromise {
  using TraitsT = std::experimental::coroutine_traits<CoroTraitsArgs...>;
  using TypeTestT = TypeTest<TraitsT>;
  using AwaitTestT = AwaitReturnsType<TypeTestT>;

  CoroMemberTag get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();

  AwaitTestT yield_value(int);

  void return_void();
  void unhandled_exception();
};

} // namespace CoroHandleMemberFunctionTest

template <class... Args>
struct ::std::experimental::coroutine_traits<CoroHandleMemberFunctionTest::CoroMemberTag, Args...> {
  using promise_type = CoroHandleMemberFunctionTest::CoroMemberPromise<CoroHandleMemberFunctionTest::CoroMemberTag, Args...>;
};

namespace CoroHandleMemberFunctionTest {
struct TestType {

  CoroMemberTag test_qual() {
    auto TC = co_yield 0;
    static_assert(TC.MatchesArgs<TestType &>, "");
    static_assert(!TC.MatchesArgs<TestType>, "");
    static_assert(!TC.MatchesArgs<TestType *>, "");
  }

  CoroMemberTag test_sanity(int *) const {
    auto TC = co_yield 0;
    static_assert(TC.MatchesArgs<const TestType &>, ""); // expected-error {{static_assert failed}}
    static_assert(TC.MatchesArgs<const TestType &>, ""); // expected-error {{static_assert failed}}
    static_assert(TC.MatchesArgs<const TestType &, int *>, "");
  }

  CoroMemberTag test_qual(int *, const float &&, volatile void *volatile) const {
    auto TC = co_yield 0;
    static_assert(TC.MatchesArgs<const TestType &, int *, const float &&, volatile void *volatile>, "");
  }

  CoroMemberTag test_qual() const volatile {
    auto TC = co_yield 0;
    static_assert(TC.MatchesArgs<const volatile TestType &>, "");
  }

  CoroMemberTag test_ref_qual() & {
    auto TC = co_yield 0;
    static_assert(TC.MatchesArgs<TestType &>, "");
  }
  CoroMemberTag test_ref_qual() const & {
    auto TC = co_yield 0;
    static_assert(TC.MatchesArgs<TestType const &>, "");
  }
  CoroMemberTag test_ref_qual() && {
    auto TC = co_yield 0;
    static_assert(TC.MatchesArgs<TestType &&>, "");
  }
  CoroMemberTag test_ref_qual(const char *&) const volatile && {
    auto TC = co_yield 0;
    static_assert(TC.MatchesArgs<TestType const volatile &&, const char *&>, "");
  }

  CoroMemberTag test_args(int) {
    auto TC = co_yield 0;
    static_assert(TC.MatchesArgs<TestType &, int>, "");
  }
  CoroMemberTag test_args(int, long &, void *) const {
    auto TC = co_yield 0;
    static_assert(TC.MatchesArgs<TestType const &, int, long &, void *>, "");
  }

  template <class... Args>
  CoroMemberTag test_member_template(Args...) const && {
    auto TC = co_yield 0;
    static_assert(TC.template MatchesArgs<TestType const &&, Args...>, "");
  }

  static CoroMemberTag test_static() {
    auto TC = co_yield 0;
    static_assert(TC.MatchesArgs<>, "");
    static_assert(!TC.MatchesArgs<TestType>, "");
    static_assert(!TC.MatchesArgs<TestType &>, "");
    static_assert(!TC.MatchesArgs<TestType *>, "");
  }

  static CoroMemberTag test_static(volatile void *const, char &&) {
    auto TC = co_yield 0;
    static_assert(TC.MatchesArgs<volatile void *const, char &&>, "");
  }

  template <class Dummy>
  static CoroMemberTag test_static_template(const char *volatile &, unsigned) {
    auto TC = co_yield 0;
    using TCT = decltype(TC);
    static_assert(TCT::MatchesArgs<const char *volatile &, unsigned>, "");
    static_assert(!TCT::MatchesArgs<TestType &, const char *volatile &, unsigned>, "");
  }

  BadCoroMemberTag test_diagnostics() {
    // expected-error@-1 {{this function cannot be a coroutine: 'std::experimental::coroutine_traits<CoroHandleMemberFunctionTest::BadCoroMemberTag, CoroHandleMemberFunctionTest::TestType &>' has no member named 'promise_type'}}
    co_return;
  }
  BadCoroMemberTag test_diagnostics(int) const && {
    // expected-error@-1 {{this function cannot be a coroutine: 'std::experimental::coroutine_traits<CoroHandleMemberFunctionTest::BadCoroMemberTag, const CoroHandleMemberFunctionTest::TestType &&, int>' has no member named 'promise_type'}}
    co_return;
  }

  static BadCoroMemberTag test_static_diagnostics(long *) {
    // expected-error@-1 {{this function cannot be a coroutine: 'std::experimental::coroutine_traits<CoroHandleMemberFunctionTest::BadCoroMemberTag, long *>' has no member named 'promise_type'}}
    co_return;
  }
};

template CoroMemberTag TestType::test_member_template(long, const char *) const &&;
template CoroMemberTag TestType::test_static_template<void>(const char *volatile &, unsigned);

template <class... Args>
struct DepTestType {

  CoroMemberTag test_sanity(int *) const {
    auto TC = co_yield 0;
    static_assert(TC.template MatchesArgs<const DepTestType &>, ""); // expected-error {{static_assert failed}}
    static_assert(TC.template MatchesArgs<>, ""); // expected-error {{static_assert failed}}
    static_assert(TC.template MatchesArgs<const DepTestType &, int *>, "");
  }

  CoroMemberTag test_qual() {
    auto TC = co_yield 0;
    static_assert(TC.template MatchesArgs<DepTestType &>, "");
    static_assert(!TC.template MatchesArgs<DepTestType>, "");
    static_assert(!TC.template MatchesArgs<DepTestType *>, "");
  }

  CoroMemberTag test_qual(int *, const float &&, volatile void *volatile) const {
    auto TC = co_yield 0;
    static_assert(TC.template MatchesArgs<const DepTestType &, int *, const float &&, volatile void *volatile>, "");
  }

  CoroMemberTag test_qual() const volatile {
    auto TC = co_yield 0;
    static_assert(TC.template MatchesArgs<const volatile DepTestType &>, "");
  }

  CoroMemberTag test_ref_qual() & {
    auto TC = co_yield 0;
    static_assert(TC.template MatchesArgs<DepTestType &>, "");
  }
  CoroMemberTag test_ref_qual() const & {
    auto TC = co_yield 0;
    static_assert(TC.template MatchesArgs<DepTestType const &>, "");
  }
  CoroMemberTag test_ref_qual() && {
    auto TC = co_yield 0;
    static_assert(TC.template MatchesArgs<DepTestType &&>, "");
  }
  CoroMemberTag test_ref_qual(const char *&) const volatile && {
    auto TC = co_yield 0;
    static_assert(TC.template MatchesArgs<DepTestType const volatile &&, const char *&>, "");
  }

  CoroMemberTag test_args(int) {
    auto TC = co_yield 0;
    static_assert(TC.template MatchesArgs<DepTestType &, int>, "");
  }
  CoroMemberTag test_args(int, long &, void *) const {
    auto TC = co_yield 0;
    static_assert(TC.template MatchesArgs<DepTestType const &, int, long &, void *>, "");
  }

  template <class... UArgs>
  CoroMemberTag test_member_template(UArgs...) const && {
    auto TC = co_yield 0;
    static_assert(TC.template MatchesArgs<DepTestType const &&, UArgs...>, "");
  }

  static CoroMemberTag test_static() {
    auto TC = co_yield 0;
    using TCT = decltype(TC);
    static_assert(TCT::MatchesArgs<>, "");
    static_assert(!TCT::MatchesArgs<DepTestType>, "");
    static_assert(!TCT::MatchesArgs<DepTestType &>, "");
    static_assert(!TCT::MatchesArgs<DepTestType *>, "");

    // Ensure diagnostics are actually being generated here
    static_assert(TCT::MatchesArgs<int>, ""); // expected-error {{static_assert failed}}
  }

  static CoroMemberTag test_static(volatile void *const, char &&) {
    auto TC = co_yield 0;
    using TCT = decltype(TC);
    static_assert(TCT::MatchesArgs<volatile void *const, char &&>, "");
  }

  template <class Dummy>
  static CoroMemberTag test_static_template(const char *volatile &, unsigned) {
    auto TC = co_yield 0;
    using TCT = decltype(TC);
    static_assert(TCT::MatchesArgs<const char *volatile &, unsigned>, "");
    static_assert(!TCT::MatchesArgs<DepTestType &, const char *volatile &, unsigned>, "");
  }
};

template struct DepTestType<int>; // expected-note {{requested here}}
template CoroMemberTag DepTestType<int>::test_member_template(long, const char *) const &&;

template CoroMemberTag DepTestType<int>::test_static_template<void>(const char *volatile &, unsigned);

struct bad_promise_deleted_constructor {
  // expected-note@+1 {{'bad_promise_deleted_constructor' has been explicitly marked deleted here}}
  bad_promise_deleted_constructor() = delete;
  coro<bad_promise_deleted_constructor> get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void return_void();
  void unhandled_exception();
};

coro<bad_promise_deleted_constructor>
bad_coroutine_calls_deleted_promise_constructor() {
  // expected-error@-1 {{call to deleted constructor of 'std::experimental::coroutine_traits<coro<CoroHandleMemberFunctionTest::bad_promise_deleted_constructor>>::promise_type' (aka 'CoroHandleMemberFunctionTest::bad_promise_deleted_constructor')}}
  co_return;
}

// Test that, when the promise type has a constructor whose signature matches
// that of the coroutine function, that constructor is used. If no matching
// constructor exists, the default constructor is used as a fallback. If no
// matching constructors exist at all, an error is emitted. This is an
// experimental feature that will be proposed for the Coroutines TS.

struct good_promise_default_constructor {
  good_promise_default_constructor(double, float, int);
  good_promise_default_constructor() = default;
  coro<good_promise_default_constructor> get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void return_void();
  void unhandled_exception();
};

coro<good_promise_default_constructor>
good_coroutine_calls_default_constructor() {
  co_return;
}

struct good_promise_custom_constructor {
  good_promise_custom_constructor(double, float, int);
  good_promise_custom_constructor() = delete;
  coro<good_promise_custom_constructor> get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void return_void();
  void unhandled_exception();
};

coro<good_promise_custom_constructor>
good_coroutine_calls_custom_constructor(double, float, int) {
  co_return;
}

struct bad_promise_no_matching_constructor {
  bad_promise_no_matching_constructor(int, int, int);
  // expected-note@+1 {{'bad_promise_no_matching_constructor' has been explicitly marked deleted here}}
  bad_promise_no_matching_constructor() = delete;
  coro<bad_promise_no_matching_constructor> get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void return_void();
  void unhandled_exception();
};

coro<bad_promise_no_matching_constructor>
bad_coroutine_calls_with_no_matching_constructor(int, int) {
  // expected-error@-1 {{call to deleted constructor of 'std::experimental::coroutine_traits<coro<CoroHandleMemberFunctionTest::bad_promise_no_matching_constructor>, int, int>::promise_type' (aka 'CoroHandleMemberFunctionTest::bad_promise_no_matching_constructor')}}
  co_return;
}

} // namespace CoroHandleMemberFunctionTest

class awaitable_no_unused_warn {
public:
  using handle_type = std::experimental::coroutine_handle<>;
  constexpr bool await_ready()  { return false; }
  void await_suspend(handle_type) noexcept {}
  int await_resume() { return 1; }
};


class awaitable_unused_warn {
public:
  using handle_type = std::experimental::coroutine_handle<>;
  constexpr bool await_ready()  { return false; }
  void await_suspend(handle_type) noexcept {}
  [[nodiscard]]
  int await_resume() { return 1; }
};

template <class Await>
struct check_warning_promise {
  coro<check_warning_promise> get_return_object();
  Await initial_suspend();
  Await final_suspend();
  Await yield_value(int);
  void return_void();
  void unhandled_exception();
};


coro<check_warning_promise<awaitable_no_unused_warn>>
test_no_unused_warning() {
  co_await awaitable_no_unused_warn();
  co_yield 42;
}

coro<check_warning_promise<awaitable_unused_warn>>
test_unused_warning() {
  co_await awaitable_unused_warn(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  co_yield 42; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}

struct missing_await_ready {
  void await_suspend(std::experimental::coroutine_handle<>);
  void await_resume();
};
struct missing_await_suspend {
  bool await_ready();
  void await_resume();
};
struct missing_await_resume {
  bool await_ready();
  void await_suspend(std::experimental::coroutine_handle<>);
};

void test_missing_awaitable_members() {
  co_await missing_await_ready{}; // expected-error {{no member named 'await_ready' in 'missing_await_ready'}}
  co_await missing_await_suspend{}; // expected-error {{no member named 'await_suspend' in 'missing_await_suspend'}}
  co_await missing_await_resume{}; // expected-error {{no member named 'await_resume' in 'missing_await_resume'}}
}
