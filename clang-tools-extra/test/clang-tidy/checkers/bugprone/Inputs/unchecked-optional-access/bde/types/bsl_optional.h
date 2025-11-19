#ifndef LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_BDE_TYPES_OPTIONAL_H_
#define LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_BDE_TYPES_OPTIONAL_H_

#include "../../std/types/optional.h"

namespace bsl {
  class string {};
}

/// Mock of `BloombergLP::bslstl::Optional_Base`
namespace BloombergLP::bslstl {

template <class T>
constexpr bool isAllocatorAware() {
  return false;
}

template <>
constexpr bool isAllocatorAware<bsl::string>() {
  return true;
}

// Note: in reality `Optional_Base` checks if type uses bsl::allocator<>
// This is simplified mock to illustrate similar behaviour
template <class T, bool AA = isAllocatorAware<T>()>
class Optional_Base {

};

template <class T>
class Optional_Base<T, false> : public std::optional<T> {
};

template <class T>
class Optional_Base<T, true> {
public:
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

  void swap(Optional_Base &rhs) noexcept;

  template <typename U> Optional_Base &operator=(const U &u);
};

} // namespace BloombergLP::bslstl


/// Mock of `bsl::optional`.
namespace bsl {

struct nullopt_t {
  constexpr explicit nullopt_t() {}
};

constexpr nullopt_t nullopt;

template <typename T>
class optional : public BloombergLP::bslstl::Optional_Base<T> {
public:
  constexpr optional() noexcept;

  constexpr optional(nullopt_t) noexcept;

  optional(const optional &) = default;

  optional(optional &&) = default;
};

} // namespace bsl

#endif // LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_BDE_TYPES_OPTIONAL_H_
