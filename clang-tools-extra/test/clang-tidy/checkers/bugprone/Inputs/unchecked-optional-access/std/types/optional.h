#ifndef LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_STD_TYPES_OPTIONAL_H_
#define LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_STD_TYPES_OPTIONAL_H_

/// Mock of `std::optional`.
namespace std {

struct nullopt_t {
  constexpr explicit nullopt_t() {}
};

constexpr nullopt_t nullopt;

template <class T> struct __optional_destruct_base {
  constexpr void reset() noexcept;
};

template <class T>
struct __optional_storage_base : __optional_destruct_base<T> {
  constexpr bool has_value() const noexcept;

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
};

// Note: the inheritance may or may not be private:
// https://github.com/llvm/llvm-project/issues/187788
template <typename T> class optional : public __optional_storage_base<T> {
  using base = __optional_storage_base<T>;

public:
  constexpr optional() noexcept;

  constexpr optional(nullopt_t) noexcept;

  optional(const optional &) = default;

  optional(optional &&) = default;

  using base::operator*;
  using base::operator->;
  using base::value;

  constexpr explicit operator bool() const noexcept;
  using base::has_value;

  template <typename U>
  constexpr T value_or(U &&v) const &;
  template <typename U>
  T value_or(U &&v) &&;

  template <typename... Args>
  T &emplace(Args &&...args);

  using base::reset;

  void swap(optional &rhs) noexcept;

  template <typename U> optional &operator=(const U &u);
};

} // namespace std


#endif // LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_STD_TYPES_OPTIONAL_H_
