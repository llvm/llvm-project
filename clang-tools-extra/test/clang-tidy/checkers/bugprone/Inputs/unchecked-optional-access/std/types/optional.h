#ifndef LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_STD_TYPES_OPTIONAL_H_
#define LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_STD_TYPES_OPTIONAL_H_

namespace std {

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

} // namespace std


#endif // LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_STD_TYPES_OPTIONAL_H_