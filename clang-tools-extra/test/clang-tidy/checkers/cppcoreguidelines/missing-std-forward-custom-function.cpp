// RUN: %check_clang_tidy -std=c++14 %s cppcoreguidelines-missing-std-forward %t -- \
// RUN: -config="{CheckOptions: {cppcoreguidelines-missing-std-forward.ForwardFunction: custom_forward}}" -- -fno-delayed-template-parsing

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

template<class T>
constexpr decltype(auto) custom_forward(std::remove_reference_t<T>& tmp) noexcept
{
  return static_cast<T&&>(tmp);
}

template<class T>
constexpr decltype(auto) custom_forward(std::remove_reference_t<T>&& tmp) noexcept
{
  return static_cast<T&&>(tmp);
}

template<class T>
void forward_with_std(T&& t) {
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: forwarding reference parameter 't' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]

  T other{std::forward<T>(t)};
}

template<class T>
void move_with_custom(T&& t) {
  T other{custom_forward<T>(t)};
}
