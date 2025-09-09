#ifndef LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_BDE_TYPES_NULLABLEVALUE_H_
#define LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_BDE_TYPES_NULLABLEVALUE_H_

#include "bsl_optional.h"

/// Mock of `bdlb::NullableValue`.
namespace BloombergLP::bdlb {

template <typename T>
class NullableValue : public bsl::optional<T> {
public:
  constexpr NullableValue() noexcept;

  constexpr NullableValue(bsl::nullopt_t) noexcept;

  NullableValue(const NullableValue &) = default;

  NullableValue(NullableValue &&) = default;

  const T &value() const &;
  T &value() &;

  constexpr T &makeValue();

  template <typename U>
  constexpr T &makeValue(U&& v);

  template <typename... ARGS>
  constexpr T &makeValueInplace(ARGS &&... args);

  // 'operator bool' is inherited from bsl::optional

  constexpr bool isNull() const noexcept;

  template <typename U>
  constexpr T valueOr(U &&v) const &;

  // 'reset' is inherited from bsl::optional

  template <typename U> NullableValue &operator=(const U &u);
};


} // namespace BloombergLP::bdlb

#endif // LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_BDE_TYPES_NULLABLEVALUE_H_
