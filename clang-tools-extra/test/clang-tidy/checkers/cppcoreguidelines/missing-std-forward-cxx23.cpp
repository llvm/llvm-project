// RUN: %check_clang_tidy -std=c++23 %s cppcoreguidelines-missing-std-forward %t -- -- -fno-delayed-template-parsing

// NOLINTBEGIN
namespace std {

template <typename T> struct remove_reference      { using type = T; };
template <typename T> struct remove_reference<T&>  { using type = T; };
template <typename T> struct remove_reference<T&&> { using type = T; };

template <typename T> using remove_reference_t = typename remove_reference<T>::type;

template <typename T> constexpr T &&forward(remove_reference_t<T> &t) noexcept;
template <typename T> constexpr T &&forward(remove_reference_t<T> &&t) noexcept;
template <typename T> constexpr remove_reference_t<T> &&move(T &&x);

template <class T, class U>
concept derived_from = __is_base_of(U, T);

template <class T>
concept integral = __is_integral(T);

} // namespace std
// NOLINTEND

// Tests for constrained template parameters (GH#180362).

class GH180362_A {
public:
  template <std::derived_from<GH180362_A> Self>
  auto operator|(this Self &&self, int) -> void {}
};

template <std::integral T>
void gh180362_takes_integral(T &&t) {}

template <typename T>
void gh180362_unconstrained(T &&t) {}
// CHECK-MESSAGES: :[[@LINE-1]]:{{.*}}: warning: forwarding reference parameter 't' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]

template <typename T>
requires std::integral<T>
void gh180362_requires_clause(T &&t) {}
// CHECK-MESSAGES: :[[@LINE-1]]:{{.*}}: warning: forwarding reference parameter 't' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]

struct GH180362_B {
  template <typename Self>
  void foo(this Self &&self) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:{{.*}}: warning: forwarding reference parameter 'self' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
};
