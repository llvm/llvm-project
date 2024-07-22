// RUN: %check_clang_tidy %s cppcoreguidelines-missing-std-forward %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:     {cppcoreguidelines-missing-std-forward.IgnoreStaticCasts: true }}" \
// RUN: -- -fno-delayed-template-parsing

// NOLINTBEGIN
namespace std {

template <typename T> struct remove_reference      { using type = T; };
template <typename T> struct remove_reference<T&>  { using type = T; };
template <typename T> struct remove_reference<T&&> { using type = T; };

template <typename T> using remove_reference_t = typename remove_reference<T>::type;

template <typename T> constexpr T &&forward(remove_reference_t<T> &t) noexcept;
template <typename T> constexpr T &&forward(remove_reference_t<T> &&t) noexcept;

template<typename T> using add_lvalue_reference_t = __add_lvalue_reference(T);

} // namespace std
// NOLINTEND

namespace in_static_cast {

template<typename T>
void to_lvalue_ref(T&& t) {
  static_cast<T&>(t);
}

template<typename T>
void to_const_lvalue_ref(T&& t) {
  static_cast<const T&>(t);
}

template<typename T>
void to_rvalue_ref(T&& t) {
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: forwarding reference parameter 't' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
  static_cast<T&&>(t);
}

template<typename T>
void to_value(T&& t) {
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: forwarding reference parameter 't' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
  static_cast<T>(t);
}

template<typename T>
void to_const_float_lvalue_ref(T&& t) {
  // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: forwarding reference parameter 't' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
  static_cast<float&>(t);
}

template<typename T>
void to_float(T&& t) {
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: forwarding reference parameter 't' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
  static_cast<float>(t);
}

template<typename T>
void to_dependent(T&& t) {
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: forwarding reference parameter 't' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
  static_cast<std::add_lvalue_reference_t<T>>(t);
}

template<typename... T>
void to_float_expanded(T&&... t) {
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: forwarding reference parameter 't' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
  (static_cast<float>(t), ...);
}

template<typename... T>
void to_lvalue_ref_expanded(T&&... t) {
  (static_cast<T&>(t), ...);
}

} // namespace in_static_cast

