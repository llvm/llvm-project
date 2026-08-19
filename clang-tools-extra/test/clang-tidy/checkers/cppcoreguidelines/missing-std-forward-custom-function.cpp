// RUN: %check_clang_tidy -std=c++14-or-later %s cppcoreguidelines-missing-std-forward %t -- \
// RUN: -config="{CheckOptions: {cppcoreguidelines-missing-std-forward.ForwardFunction: custom_forward}}" -- -fno-delayed-template-parsing

#include <utility>

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
