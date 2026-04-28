#ifndef HICKETTS_OPTIONAL_H_
#define HICKETTS_OPTIONAL_H_

/// A custom optional-like type with differently named functions.
/// Mirrors std::optional semantics but uses its own vocabulary
/// In order to test implementation of attributes for clang-tidy
namespace mylib {

struct nothing_t {
  constexpr explicit nothing_t() {}
};

constexpr nothing_t nothing;

template <typename T>
class [[clang::analyse_as_class("std::optional")]] HickettsOptional {
  T *storage_ = nullptr;

public:
  constexpr HickettsOptional() noexcept {}

  constexpr HickettsOptional(nothing_t) noexcept {}

  HickettsOptional(const HickettsOptional &) = default;

  HickettsOptional(HickettsOptional &&) = default;

  // Equivalent to std::optional::value()
  [[clang::analyse_as_method("std::optional::value")]] const T &unwrap() const & { return *storage_; }
  [[clang::analyse_as_method("std::optional::value")]] T &unwrap() & { return *storage_; }
  [[clang::analyse_as_method("std::optional::value")]] const T &&unwrap() const && { return static_cast<const T &&>(*storage_); }
  [[clang::analyse_as_method("std::optional::value")]] T &&unwrap() && { return static_cast<T &&>(*storage_); }

  const T &value() const & { return *storage_; }
  T &value() & { return *storage_; }
  const T &&value() const && { return static_cast<const T &&>(*storage_); }
  T &&value() && { return static_cast<T &&>(*storage_); }

  // Equivalent to std::optional::operator*()
  [[clang::analyse_as_method("std::optional::operator*")]] const T &deref() const & { return *storage_; }
  [[clang::analyse_as_method("std::optional::operator*")]] T &deref() & { return *storage_; }

  // Equivalent to std::optional::operator->()
  const T* operator ->() const { return storage_; }
  T* operator ->() { return storage_; }
  const T *arrow() const { return storage_; }
  T *arrow() { return storage_; }

  // Equivalent to std::optional::operator bool / hasValue()
  constexpr bool hasValue() const noexcept { return storage_ != nullptr; }
  constexpr explicit operator bool() const noexcept { return storage_ != nullptr; }
  [[clang::analyse_as_method("std::optional::hasValue")]] constexpr bool isPresent() const noexcept { return storage_ != nullptr; }
  constexpr bool isEmpty() const noexcept { return storage_ == nullptr; }

  // Equivalent to std::optional::value_or()
  template <typename U>
  constexpr T unwrapOr(U &&fallback) const & {
    return storage_ ? *storage_ : static_cast<T>(fallback);
  }

  // Equivalent to std::optional::emplace()
  template <typename... Args>
  T &construct(Args &&...args) { return *storage_; }

  // Equivalent to std::optional::reset()
  void clear() noexcept { storage_ = nullptr; }

  // Equivalent to std::optional::swap()
  void exchange(HickettsOptional &other) noexcept {
    T *tmp = storage_;
    storage_ = other.storage_;
    other.storage_ = tmp;
  }

  // Assignment
  template <typename U>
  HickettsOptional &operator=(const U &u) { return *this; }
};

} // namespace mylib

#endif // HICKETTS_OPTIONAL_H_
