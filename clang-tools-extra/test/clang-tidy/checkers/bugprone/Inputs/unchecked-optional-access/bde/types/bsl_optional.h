#ifndef LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_BDE_TYPES_OPTIONAL_H_
#define LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_BDE_TYPES_OPTIONAL_H_

#include "../../std/types/optional.h"

namespace bsl {
  class string {};

  template <typename T> class optional;

  struct nullopt_t {
    constexpr explicit nullopt_t() {}
  };

  constexpr nullopt_t nullopt;

  struct in_place_t {
    constexpr explicit in_place_t() {}
  };

  constexpr in_place_t in_place;

  struct allocator_arg_t {
    constexpr explicit allocator_arg_t() {}
  };

  constexpr allocator_arg_t allocator_arg;

  /// Mock of the allocator type taken by the allocator-extended constructors.
  class allocator {};

  template <bool B, class T> struct enable_if {};
  template <class T> struct enable_if<true, T> { using type = T; };
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

/// Mock of `BloombergLP::bslstl::Optional_OptNoSuchType`
struct Optional_OptNoSuchType {
  explicit Optional_OptNoSuchType(int) noexcept {}
};

template <class T> struct Optional_RemoveCVRef { using type = T; };
template <class T> struct Optional_RemoveCVRef<T &> : Optional_RemoveCVRef<T> {};
template <class T> struct Optional_RemoveCVRef<T &&> : Optional_RemoveCVRef<T> {};
template <class T> struct Optional_RemoveCVRef<const T> : Optional_RemoveCVRef<T> {};

template <class T> struct Optional_IsTagType {
  static constexpr bool value = false;
};
template <> struct Optional_IsTagType<bsl::nullopt_t> {
  static constexpr bool value = true;
};
template <> struct Optional_IsTagType<bsl::in_place_t> {
  static constexpr bool value = true;
};
template <> struct Optional_IsTagType<bsl::allocator_arg_t> {
  static constexpr bool value = true;
};
template <> struct Optional_IsTagType<bsl::allocator> {
  static constexpr bool value = true;
};

template <class T> struct Optional_IsStdOptional {
  static constexpr bool value = false;
};
template <class T> struct Optional_IsStdOptional<std::optional<T>> {
  static constexpr bool value = true;
};

/// Mock of `BloombergLP::bslstl::Optional_ConstructsFromType`.
template <class TYPE, class ANY_TYPE>
struct Optional_ConstructsFromType {
private:
  using Any = typename Optional_RemoveCVRef<ANY_TYPE>::type;

public:
  static constexpr bool value =
      !Optional_IsTagType<Any>::value && !Optional_IsStdOptional<Any>::value;
};

/// Mock of the trait behind `BSLSTL_OPTIONAL_DEFINE_IF_NOT_DERIVED_FROM_OPTIONAL`.
template <class TYPE, class ANY_TYPE>
struct Optional_IsNotDerivedFromOptional {
  static constexpr bool value = !__is_base_of(
      bsl::optional<TYPE>, typename Optional_RemoveCVRef<ANY_TYPE>::type);
};

// Note: real `Optional_Base` uses `BloombergLP::bslma::UsesBslmaAllocator`
// to check if type is allocator-aware.
// This is simplified mock to illustrate similar behaviour.
template <class T, bool AA = isAllocatorAware<T>()>
class Optional_Base {
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

template <class T>
class Optional_Base<T, false> : public std::optional<T> {
};

} // namespace BloombergLP::bslstl


/// Mock of `bsl::optional`.
namespace bsl {

/// Mocks of the `BSLSTL_OPTIONAL_DECLARE_IF_*` macros.
template <class TYPE, class ANY_TYPE>
using Optional_IfConstructsFrom = typename enable_if<
    BloombergLP::bslstl::Optional_ConstructsFromType<TYPE, ANY_TYPE>::value,
    BloombergLP::bslstl::Optional_OptNoSuchType>::type;

template <class TYPE, class ANY_TYPE>
using Optional_IfNotDerivedFromOptional = typename enable_if<
    BloombergLP::bslstl::Optional_IsNotDerivedFromOptional<TYPE,
                                                           ANY_TYPE>::value,
    BloombergLP::bslstl::Optional_OptNoSuchType>::type;

using Optional_NoSuchType = BloombergLP::bslstl::Optional_OptNoSuchType;

template <typename T>
class optional : public BloombergLP::bslstl::Optional_Base<T> {
public:
  constexpr optional() noexcept;

  constexpr optional(nullopt_t) noexcept;

  template <typename ANY_TYPE = T>
  optional(ANY_TYPE &&v,
           Optional_IfConstructsFrom<T, ANY_TYPE> = Optional_NoSuchType(0),
           Optional_IfNotDerivedFromOptional<T, ANY_TYPE> = Optional_NoSuchType(0));

  template <typename ANY_TYPE>
  optional(const std::optional<ANY_TYPE> &v,
           Optional_IfConstructsFrom<T, ANY_TYPE> = Optional_NoSuchType(0),
           Optional_IfNotDerivedFromOptional<T, ANY_TYPE> = Optional_NoSuchType(0));

  template <typename... ARGS>
  explicit optional(in_place_t, ARGS &&...args);

  optional(allocator_arg_t, allocator);

  template <typename ANY_TYPE = T>
  optional(allocator_arg_t, allocator, ANY_TYPE &&v,
           Optional_IfConstructsFrom<T, ANY_TYPE> = Optional_NoSuchType(0),
           Optional_IfNotDerivedFromOptional<T, ANY_TYPE> = Optional_NoSuchType(0));

  template <typename... ARGS>
  explicit optional(allocator_arg_t, allocator, in_place_t, ARGS &&...args);

  optional(const optional &) = default;

  optional(optional &&) = default;
};

template <typename T, typename... ARGS>
optional<T> make_optional(ARGS &&...args);

} // namespace bsl

#endif // LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_BDE_TYPES_OPTIONAL_H_
