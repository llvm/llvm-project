#ifndef LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_ABSL_TYPES_OPTIONAL_H_
#define LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_ABSL_TYPES_OPTIONAL_H_

/// Mock of `absl::optional`.
namespace absl {

// clang-format off
template <typename T> struct remove_reference      { using type = T; };
template <typename T> struct remove_reference<T&>  { using type = T; };
template <typename T> struct remove_reference<T&&> { using type = T; };
// clang-format on

template <typename T>
using remove_reference_t = typename remove_reference<T>::type;

template <typename T>
constexpr T &&forward(remove_reference_t<T> &t) noexcept;

template <typename T>
constexpr T &&forward(remove_reference_t<T> &&t) noexcept;

template <typename T>
constexpr remove_reference_t<T> &&move(T &&x);

struct nullopt_t {
  constexpr explicit nullopt_t() {}
};

constexpr nullopt_t nullopt;

template <typename T>
class optional {
public:
  constexpr optional() noexcept;

  constexpr optional(nullopt_t) noexcept;

  optional(const optional &) = default;

  optional(optional &&) = default;

  const T &operator*() const &;
  T &operator*() &;
  const T &&operator*() const &&;
  T &&operator*() &&;

  const T *operator->() const;
  T *operator->();

  const T &value() const &;
  T &value() &;
  const T &&value() const &&;
  T &&value() &&;

  constexpr explicit operator bool() const noexcept;
  constexpr bool has_value() const noexcept;

  template <typename U>
  constexpr T value_or(U &&v) const &;
  template <typename U>
  T value_or(U &&v) &&;

  template <typename... Args>
  T &emplace(Args &&...args);

  void reset() noexcept;

  void swap(optional &rhs) noexcept;

  template <typename U> optional &operator=(const U &u);
};

} // namespace absl

#endif // LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_ABSL_TYPES_OPTIONAL_H_
