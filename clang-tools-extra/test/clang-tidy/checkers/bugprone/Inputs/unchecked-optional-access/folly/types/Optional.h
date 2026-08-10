#ifndef LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_BUGPRONE_INPUTS_UNCHECKED_OPTIONAL_ACCESS_FOLLY_TYPES_OPTIONAL_H_
#define LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_BUGPRONE_INPUTS_UNCHECKED_OPTIONAL_ACCESS_FOLLY_TYPES_OPTIONAL_H_

/// Mock of `folly::Optional`.
namespace folly {

struct None {
  constexpr explicit None() {}
};

constexpr None none;

template <typename T>
class Optional {
public:
  constexpr Optional() noexcept;

  constexpr Optional(None) noexcept;

  Optional(const Optional &) = default;

  Optional(Optional &&) = default;

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
  constexpr bool hasValue() const noexcept;

  template <typename U>
  constexpr T value_or(U &&v) const &;
  template <typename U>
  T value_or(U &&v) &&;

  template <typename... Args>
  T &emplace(Args &&...args);

  void reset() noexcept;

  void swap(Optional &rhs) noexcept;
};

} // namespace folly

#endif // LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_BUGPRONE_INPUTS_UNCHECKED_OPTIONAL_ACCESS_FOLLY_TYPES_OPTIONAL_H_
