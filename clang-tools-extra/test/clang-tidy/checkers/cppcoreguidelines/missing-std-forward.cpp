// RUN: %check_clang_tidy -std=c++11 %s cppcoreguidelines-missing-std-forward %t -- -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -std=c++14-or-later -check-suffixes=,CXX14 %s cppcoreguidelines-missing-std-forward %t -- -- -fno-delayed-template-parsing

#include <utility>

struct S {
  S();
  S(const S&);
  S(S&&) noexcept;
  S& operator=(const S&);
  S& operator=(S&&) noexcept;
};

template <class... Ts>
void consumes_all(Ts&&...);

namespace positive_cases {

template <class T>
void does_not_forward(T&& t) {
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: forwarding reference parameter 't' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
  T other = t;
}

template <class T>
void does_not_forward_invoked(T&& t) {
  // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: forwarding reference parameter 't' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
  T other = t();
}

template <class T>
void forwards_pairwise(T&& t) {
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: forwarding reference parameter 't' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
  auto first = std::forward<T>(t.first);
  auto second = std::forward<T>(t.second);
}

template <class... Ts>
void does_not_forward_pack(Ts&&... ts) {
  // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: forwarding reference parameter 'ts' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
  consumes_all(ts...);
}

template <class T>
class AClass {

  template <class U>
  AClass(U&& u) : data(u) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: forwarding reference parameter 'u' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]

  template <class U>
  AClass& operator=(U&& u) { }
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: forwarding reference parameter 'u' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]

  template <class U>
  void mixed_params(T&& t, U&& u) {
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: forwarding reference parameter 'u' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
    T other1 = std::move(t);
    U other2 = std::move(u);
  }

  T data;
};

template <class T>
void does_not_forward_in_evaluated_code(T&& t) {
  // CHECK-MESSAGES: :[[@LINE-1]]:45: warning: forwarding reference parameter 't' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
  using result_t = decltype(std::forward<T>(t));
  unsigned len = sizeof(std::forward<T>(t));
  T other = t;
}

template <class T>
void lambda_value_capture(T&& t) {
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: forwarding reference parameter 't' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
  [=]() { T other = std::forward<T>(t); };
}

template <class T>
void lambda_value_capture_copy(T&& t) {
  // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: forwarding reference parameter 't' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
  [&,t]() { T other = std::forward<T>(t); };
}

template <class T>
void lambda_capture_list_brace_init_no_forward(T&& t) {
  // CHECK-MESSAGES: :[[@LINE-1]]:52: warning: forwarding reference parameter 't' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
  [t2{t}] { t2(); };
}

template <class T>
void lambda_capture_list_paren_init_no_forward(T&& t) {
  // CHECK-MESSAGES: :[[@LINE-1]]:52: warning: forwarding reference parameter 't' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
  [t2(t)] { t2(); };
}

template <typename X>
void use(const X &x) {}

template <typename X, typename Y>
void foo(X &&x, Y &&y) {
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: forwarding reference parameter 'y' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
    use(std::forward<X>(x));
    use(y);
}

template <typename T>
void nested_but_no_forward(T &&arg) {
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: forwarding reference parameter 'arg' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
	[&]()
	{
		[&]()
		{ consumes_all(arg); }();
	}();
}

template <typename T, typename U>
void nested_forward_only_one(T &&arg1, U &&arg2) {
  // CHECK-MESSAGES: :[[@LINE-1]]:44: warning: forwarding reference parameter 'arg2' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
	[&]()
	{
		[&]()
		{ consumes_all(std::forward<T>(arg1)); }();
	}();
}

template <typename T>
void nested_rename_no_forward(T &&t) {
  // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: forwarding reference parameter 't' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
  [&x = t]() {
    [&y = x]() {
      (void)y;
    }();
  }();
}

} // namespace positive_cases

namespace negative_cases {

template <class T>
void just_a_decl(T&&t);

template <class T>
void does_forward(T&& t) {
  T other = std::forward<T>(t);
}

template <class... Ts>
void does_forward_pack(Ts&&... ts) {
  consumes_all(std::forward<Ts>(ts)...);
}

void pass_by_value(S s) {
  S other = std::move(s);
}

void lvalue_ref(S& s) {
  S other = std::move(s);
}

void rvalue_ref(S&& s) {
  S other = std::move(s);
}

template <class T>
void templated_rvalue_ref(std::remove_reference_t<T>&& t) {
  T other = std::move(t);
}

template <class T>
class AClass {

  template <class U>
  AClass(U&& u) : data(std::forward<U>(u)) {}

  template <class U>
  AClass& operator=(U&& u) {
    data = std::forward<U>(u);
  }

  void rvalue_ref(T&& t) {
    T other = std::move(t);
  }

  T data;
};

template <class T>
void lambda_value_reference(T&& t) {
  [&]() { T other = std::forward<T>(t); };
}

template<typename T>
void lambda_value_reference_capture_list_ref_1(T&& t) {
    [=, &t] { T other = std::forward<T>(t); };
}

template<typename T>
void lambda_value_reference_capture_list_ref_2(T&& t) {
    [&t] { T other = std::forward<T>(t); };
}

template<typename T>
void lambda_value_reference_capture_list(T&& t) {
    [t = std::forward<T>(t)] { t(); };
}

template <class T>
void lambda_value_reference_auxiliary_var(T&& t) {
  [&x = t]() { T other = std::forward<T>(x); };
}

template <class T>
void lambda_multi_level_rename(T &&t) {
  [&x = t]() {
    [&y = x]() {
      T other = std::forward<T>(y);
    }();
  }();
}

template <class T>
void lambda_implicit_and_rename(T &&t) {
  [&]() {
    [&y = t]() {
      T other = std::forward<T>(y);
    }();
  }();
}

template <typename T>
void nested_forward(T &&arg) {
  [&]() { [&]() { consumes_all(std::forward<T>(arg)); }(); }();
}

template <typename T>
void triple_nested_forward(T &&arg) {
  [&]() {
    [&]() { [&]() { consumes_all(std::forward<T>(arg)); }(); }();
  }();
}

template <typename T>
void lambda_value_reference_capture_list_brace_init(T&& t) {
  [t2{std::forward<T>(t)}] { t2(); };
}

template <typename T>
void lambda_value_reference_capture_list_paren_init(T&& t) {
  [t2(std::forward<T>(t))] { t2(); };
}

} // namespace negative_cases

namespace deleted_functions {

template <typename T>
void f(T &&t) = delete;

struct S {
    template <typename T>
    S(T &&t) = delete;

    template <typename T>
    void operator&(T &&t) = delete;
};

} // namespace deleted_functions

namespace unused_arguments {

template<typename F>
void unused_argument1(F&&) {}

template<typename F>
void unused_argument2([[maybe_unused]] F&& f) {}

template<typename F>
void unused_argument3(F&& _) {}

} // namespace unused_arguments

#if __cplusplus >= 201402L

namespace generic_lambdas {

void forwards_parameter() {
  auto lambda = [](auto &&x) {
    (void)std::forward<decltype(x)>(x);
  };
}

void does_not_forward_parameter() {
  auto lambda = [](auto &&x) {
    // CHECK-MESSAGES-CXX14: :[[@LINE-1]]:27: warning: forwarding reference parameter 'x' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
    (void)x;
  };
}

void forwards_parameter_by_reference_capture() {
  auto lambda = [](auto &&x) {
    [&x]() {
      (void)std::forward<decltype(x)>(x);
    }();
  };
}

void does_not_forward_parameter_by_value_capture() {
  auto lambda = [](auto &&x) {
    // CHECK-MESSAGES-CXX14: :[[@LINE-1]]:27: warning: forwarding reference parameter 'x' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
    [x]() {
      (void)std::forward<decltype(x)>(x);
    }();
  };
}

template <typename T>
void forwards_parameter_in_template() {
  auto lambda = [](auto &&x) {
    (void)std::forward<decltype(x)>(x);
  };
}

void forwards_only_one_parameter() {
  auto lambda = [](auto &&x, auto &&y) {
    // CHECK-MESSAGES-CXX14: :[[@LINE-1]]:37: warning: forwarding reference parameter 'y' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
    (void)std::forward<decltype(x)>(x);
    (void)y;
  };
}

void forwards_only_inner_lambda_parameter() {
  auto outer = [](auto &&x) {
    // CHECK-MESSAGES-CXX14: :[[@LINE-1]]:26: warning: forwarding reference parameter 'x' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
    (void)x;
    auto inner = [](auto &&x) {
      (void)std::forward<decltype(x)>(x);
    };
  };
}

// decltype examines a type; it does not execute the forwarding expression.
void does_not_forward_in_decltype() {
  auto lambda = [](auto &&x) {
    // CHECK-MESSAGES-CXX14: :[[@LINE-1]]:27: warning: forwarding reference parameter 'x' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
    using ForwardedType = decltype(std::forward<decltype(x)>(x));
  };
}

// sizeof does not execute this forwarding expression either.
void does_not_forward_in_sizeof() {
  auto lambda = [](auto &&x) {
    // CHECK-MESSAGES-CXX14: :[[@LINE-1]]:27: warning: forwarding reference parameter 'x' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
    (void)sizeof(std::forward<decltype(x)>(x));
  };
}

// Instantiating the lambda for lvalues and rvalues must not introduce warnings.
void forwards_instantiated_parameter() {
  auto lambda = [](auto &&x) {
    (void)std::forward<decltype(x)>(x);
  };

  int value = 0;
  lambda(value);
  lambda(0);
}

// const auto&& is not a forwarding reference.
void const_rvalue_reference_parameter() {
  auto lambda = [](const auto &&x) {
    (void)x;
  };
}

template <typename T>
void forwards_outer_and_inner_parameters(T &&x) {
  auto inner = [](auto &&y) {
    (void)std::forward<decltype(y)>(y);
  };

  (void)std::forward<T>(x);
}

} // namespace generic_lambdas

#endif // __cplusplus >= 201402L
