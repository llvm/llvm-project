// RUN: %check_clang_tidy -std=c++20-or-later %s modernize-type-traits %t

namespace std {
  typedef decltype(sizeof(0)) size_t;

  template <typename T>
  struct tuple_size;

  template <typename T>
  inline constexpr size_t tuple_size_v = tuple_size<T>::value; // NOLINT

  template <typename T>
  struct variant_size;

  template <typename T>
  inline constexpr size_t variant_size_v = variant_size<T>::value; // NOLINT

  template <typename>
  struct is_const {
    static constexpr bool value = true;
  };

  template <typename T>
  inline constexpr bool is_const_v = is_const<T>::value; // NOLINT
} // namespace std

// `std::tuple_size_v<T>` and `std::variant_size_v<T>` are hard errors when
// `std::tuple_size<T>::value` would be a substitution failure, so they must
// not be suggested inside constraints.

template <typename T>
constexpr bool has_tuple_size() {
  if constexpr (requires { std::tuple_size<T>::value; })
    return true;
  return false;
}

template <typename T>
concept is_pair = std::tuple_size<T>::value == 2;

template <typename T>
concept has_variant_size = requires {
  requires std::variant_size<T>::value > 0;
};

template <typename T>
  requires(std::tuple_size<T>::value == 2)
void requiresClause();

template <typename T>
void trailingRequiresClause()
  requires(std::tuple_size<T>::value == 2);

template <typename T>
struct S {
  void method()
    requires(std::tuple_size<T>::value == 2);
};

template <typename T>
struct Partial;

template <typename T>
  requires(std::tuple_size<T>::value == 2)
struct Partial<T *> {};

template <typename T>
void lambdas() {
  auto L1 = []<typename U>(U)
    requires(std::tuple_size<U>::value == 2)
  {};
  auto L2 = [](auto U)
    requires(std::tuple_size<decltype(U)>::value == 2)
  {};
}

// Other traits are always defined, so they are still diagnosed in constraints.
template <typename T>
concept is_const_type = std::is_const<T>::value;
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: use c++17 style variable templates
// CHECK-FIXES: concept is_const_type = std::is_const_v<T>;

// Outside of constraints, the replacement is still suggested.
template <typename T>
constexpr std::size_t tupleSize() {
  return std::tuple_size<T>::value;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use c++17 style variable templates
  // CHECK-FIXES: return std::tuple_size_v<T>;
}

template <typename T>
constexpr std::size_t variantSize = std::variant_size<T>::value;
// CHECK-MESSAGES: :[[@LINE-1]]:37: warning: use c++17 style variable templates
// CHECK-FIXES: constexpr std::size_t variantSize = std::variant_size_v<T>;
