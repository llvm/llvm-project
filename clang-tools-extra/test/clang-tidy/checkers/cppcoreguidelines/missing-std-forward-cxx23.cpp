// RUN: %check_clang_tidy -std=c++23-or-later %s cppcoreguidelines-missing-std-forward %t -- -- -fno-delayed-template-parsing

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
concept derived_from = true;

template <class T>
concept integral = true;

} // namespace std
// NOLINTEND

// Tests for constrained explicit object parameters (GH#180362).

// Constrained explicit object parameter is not a forwarding reference.
class GH180362_A {
public:
  template <std::derived_from<GH180362_A> Self>
  auto operator|(this Self &&self, int) -> void {}
};

// Another constrained explicit object parameter — no warning.
struct GH180362_C {
  template <std::integral Self>
  void bar(this Self &&self) {}
};

// Unconstrained explicit object parameter IS a forwarding reference.
struct GH180362_B {
  template <typename Self>
  void foo(this Self &&self) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: forwarding reference parameter 'self' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
};

// Requires-clause on explicit object parameter is NOT a type constraint.
struct GH180362_D {
  template <typename Self>
    requires std::derived_from<Self, GH180362_D>
  void baz(this Self &&self) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: forwarding reference parameter 'self' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
};

// Constrained non-explicit-object parameter IS still a forwarding reference.
template <std::integral T>
void gh180362_takes_integral(T &&t) {}
// CHECK-MESSAGES: :[[@LINE-1]]:34: warning: forwarding reference parameter 't' is never forwarded inside the function body [cppcoreguidelines-missing-std-forward]
