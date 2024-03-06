// RUN: %check_clang_tidy %s cppcoreguidelines-missing-std-forward %t -- -- -fno-delayed-template-parsing

// NOLINTBEGIN
namespace std {

template <typename T> struct remove_reference      { using type = T; };
template <typename T> struct remove_reference<T&>  { using type = T; };
template <typename T> struct remove_reference<T&&> { using type = T; };

template <typename T> using remove_reference_t = typename remove_reference<T>::type;

template <typename T> constexpr T &&forward(remove_reference_t<T> &t) noexcept;
template <typename T> constexpr T &&forward(remove_reference_t<T> &&t) noexcept;
template <typename T> constexpr remove_reference_t<T> &&move(T &&x);

} // namespace std
// NOLINTEND

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

template <typename X>
void use(const X &x) {}

template <typename X, typename Y>
void foo(X &&x, Y &&y) {
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: forwarding reference parameter 'y' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
    use(std::forward<X>(x));
    use(y);
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

} // namespace negative_cases

namespace deleted_functions {

template <typename T>
void f(T &&) = delete;

struct S {
    template <typename T>
    S(T &&) = delete;

    template <typename T>
    void operator&(T &&) = delete;
};

} // namespace deleted_functions
