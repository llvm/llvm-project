//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL 2020 half scalar data type
/// (4.14.1.).
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_HALF_HPP
#define _LIBSYCL___IMPL_HALF_HPP

#include <sycl/__impl/detail/config.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <limits>
#include <type_traits>

#ifndef __FLT16_MANT_DIG__
#  error "Libsycl requires a target with _Float16 support for sycl::half."
#endif

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

// Type in which a mixed half/T operation is computed: T if it is a wider
// floating-point type, NarrowT otherwise.
template <typename T, typename NarrowT>
using HalfMixedT =
    std::conditional_t<std::is_floating_point_v<T> && (sizeof(T) > 2), T,
                       NarrowT>;

// half is defined in its own namespace so that argument-dependent lookup for
// half finds only its own operators, as for fundamental floating-point types.
namespace half_impl {

// TODO: restrict device usage to devices with aspect::fp16 once clang supports
// an attribute to attach aspects to types.

/// A 16-bit floating-point type conforming to the IEEE 754-2008 half precision
/// storage format.
///
/// This type is only supported on devices that have aspect::fp16.
///
/// \sa SYCL 2020 4.14.1. Scalar data types.
class half {
public:
  half() = default;
  constexpr half(const half &) = default;
  constexpr half(half &&) = default;
  constexpr half &operator=(const half &) = default;
  constexpr half &operator=(half &&) = default;

  /// Constructs a half from \p Value rounded to the nearest representable
  /// half value.
  constexpr half(float Value) : MData(static_cast<_Float16>(Value)) {}

  /// \return the value of this half converted to float without precision loss.
  constexpr operator float() const { return static_cast<float>(MData); }

  constexpr half &operator++() {
    ++MData;
    return *this;
  }
  constexpr half operator++(int) {
    half Old(*this);
    ++MData;
    return Old;
  }
  constexpr half &operator--() {
    --MData;
    return *this;
  }
  constexpr half operator--(int) {
    half Old(*this);
    --MData;
    return Old;
  }

  friend constexpr half operator+(half rhs) { return rhs; }
  friend constexpr half operator-(half rhs) { return fromNative(-rhs.MData); }

  // The mixed operators are overloads for exact types rather than a template,
  // so that operands that convert implicitly to these types (enumerations,
  // sycl::id<1>, ...) are accepted, as by the built-in operators.
#define _LIBSYCL_HALF_MIXED_TYPES(X, op)                                       \
  X(op, int)                                                                   \
  X(op, unsigned int)                                                          \
  X(op, long)                                                                  \
  X(op, unsigned long)                                                         \
  X(op, long long)                                                             \
  X(op, unsigned long long)                                                    \
  X(op, float)                                                                 \
  X(op, double)                                                                \
  X(op, long double)

#define _LIBSYCL_GEN_MIXED_OPT(op, T)                                          \
  friend constexpr detail::HalfMixedT<T, half> operator op(half lhs, T rhs) {  \
    using OpT = detail::HalfMixedT<T, _Float16>;                               \
    return fromNative(static_cast<OpT>(lhs.MData) op static_cast<OpT>(rhs));   \
  }                                                                            \
  friend constexpr detail::HalfMixedT<T, half> operator op(T lhs, half rhs) {  \
    using OpT = detail::HalfMixedT<T, _Float16>;                               \
    return fromNative(static_cast<OpT>(lhs) op static_cast<OpT>(rhs.MData));   \
  }

#define _LIBSYCL_GEN_OPT(op)                                                   \
  friend constexpr half operator op(half lhs, half rhs) {                      \
    return fromNative(lhs.MData op rhs.MData);                                 \
  }                                                                            \
  _LIBSYCL_HALF_MIXED_TYPES(_LIBSYCL_GEN_MIXED_OPT, op)

  _LIBSYCL_GEN_OPT(+)
  _LIBSYCL_GEN_OPT(-)
  _LIBSYCL_GEN_OPT(*)
  _LIBSYCL_GEN_OPT(/)

#undef _LIBSYCL_GEN_OPT
#undef _LIBSYCL_GEN_MIXED_OPT

  // Compound assignment converts the other operand to half first, so that
  // double operands such as literals don't make device code depend on
  // aspect::fp64.
#define _LIBSYCL_GEN_MIXED_OPT(op, T)                                          \
  constexpr half &operator op(T rhs) {                                         \
    MData op static_cast<_Float16>(rhs);                                       \
    return *this;                                                              \
  }

#define _LIBSYCL_GEN_OPT(op)                                                   \
  constexpr half &operator op(half rhs) {                                      \
    MData op rhs.MData;                                                        \
    return *this;                                                              \
  }                                                                            \
  _LIBSYCL_HALF_MIXED_TYPES(_LIBSYCL_GEN_MIXED_OPT, op)

  _LIBSYCL_GEN_OPT(+=)
  _LIBSYCL_GEN_OPT(-=)
  _LIBSYCL_GEN_OPT(*=)
  _LIBSYCL_GEN_OPT(/=)

#undef _LIBSYCL_GEN_OPT
#undef _LIBSYCL_GEN_MIXED_OPT

  // Integers compare as float, so that values outside the half range don't
  // round to infinity.
#define _LIBSYCL_GEN_MIXED_OPT(op, T)                                          \
  friend constexpr bool operator op(half lhs, T rhs) {                         \
    using OpT = detail::HalfMixedT<T, float>;                                  \
    return static_cast<OpT>(lhs.MData) op static_cast<OpT>(rhs);               \
  }                                                                            \
  friend constexpr bool operator op(T lhs, half rhs) {                         \
    using OpT = detail::HalfMixedT<T, float>;                                  \
    return static_cast<OpT>(lhs) op static_cast<OpT>(rhs.MData);               \
  }

#define _LIBSYCL_GEN_OPT(op)                                                   \
  friend constexpr bool operator op(half lhs, half rhs) {                      \
    return lhs.MData op rhs.MData;                                             \
  }                                                                            \
  _LIBSYCL_HALF_MIXED_TYPES(_LIBSYCL_GEN_MIXED_OPT, op)

  _LIBSYCL_GEN_OPT(==)
  _LIBSYCL_GEN_OPT(!=)
  _LIBSYCL_GEN_OPT(<)
  _LIBSYCL_GEN_OPT(>)
  _LIBSYCL_GEN_OPT(<=)
  _LIBSYCL_GEN_OPT(>=)

#undef _LIBSYCL_GEN_OPT
#undef _LIBSYCL_GEN_MIXED_OPT
#undef _LIBSYCL_HALF_MIXED_TYPES

  template <typename CharT, typename Traits>
  friend std::basic_ostream<CharT, Traits> &
  operator<<(std::basic_ostream<CharT, Traits> &lhs, half rhs) {
    return lhs << static_cast<float>(rhs);
  }

  template <typename CharT, typename Traits>
  friend std::basic_istream<CharT, Traits> &
  operator>>(std::basic_istream<CharT, Traits> &lhs, half &rhs) {
    float Value = 0.0f;
    lhs >> Value;
    rhs = Value;
    return lhs;
  }

private:
  // Wraps a _Float16 result without a round trip through float.
  template <typename T>
  static constexpr std::conditional_t<std::is_same_v<T, _Float16>, half, T>
  fromNative(T Value) {
    if constexpr (std::is_same_v<T, _Float16>) {
      half Result{};
      Result.MData = Value;
      return Result;
    } else {
      return Value;
    }
  }

  friend std::numeric_limits<half>;

  _Float16 MData;
};

} // namespace half_impl
} // namespace detail

/// \copydoc detail::half_impl::half
using half = detail::half_impl::half;

static_assert(sizeof(half) == 2 && alignof(half) == 2);
static_assert(std::is_trivially_copyable_v<half> &&
              std::is_standard_layout_v<half>);

_LIBSYCL_END_NAMESPACE_SYCL

namespace std {
template <> struct numeric_limits<sycl::half> {
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr float_denorm_style has_denorm = denorm_present;
  static constexpr bool has_denorm_loss = false;
  static constexpr float_round_style round_style = round_to_nearest;
  static constexpr bool is_iec559 = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = __FLT16_MANT_DIG__;
  static constexpr int digits10 = __FLT16_DIG__;
  static constexpr int max_digits10 = __FLT16_DECIMAL_DIG__;
  static constexpr int radix = 2;
  static constexpr int min_exponent = __FLT16_MIN_EXP__;
  static constexpr int min_exponent10 = __FLT16_MIN_10_EXP__;
  static constexpr int max_exponent = __FLT16_MAX_EXP__;
  static constexpr int max_exponent10 = __FLT16_MAX_10_EXP__;
  static constexpr bool traps = false;
  static constexpr bool tinyness_before = false;

  // The values are built from _Float16 directly: a signaling NaN converted
  // from float would become quiet.
  static constexpr sycl::half(min)() noexcept {
    return sycl::half::fromNative(__FLT16_MIN__);
  }
  static constexpr sycl::half lowest() noexcept {
    return sycl::half::fromNative(-__FLT16_MAX__);
  }
  static constexpr sycl::half(max)() noexcept {
    return sycl::half::fromNative(__FLT16_MAX__);
  }
  static constexpr sycl::half epsilon() noexcept {
    return sycl::half::fromNative(__FLT16_EPSILON__);
  }
  static constexpr sycl::half round_error() noexcept {
    return sycl::half::fromNative(0.5F16);
  }
  static constexpr sycl::half infinity() noexcept {
    return sycl::half::fromNative(__builtin_inff16());
  }
  static constexpr sycl::half quiet_NaN() noexcept {
    return sycl::half::fromNative(__builtin_nanf16(""));
  }
  static constexpr sycl::half signaling_NaN() noexcept {
    return sycl::half::fromNative(__builtin_nansf16(""));
  }
  static constexpr sycl::half denorm_min() noexcept {
    return sycl::half::fromNative(__FLT16_DENORM_MIN__);
  }
};

template <> struct hash<sycl::half> {
  size_t operator()(sycl::half Key) const noexcept {
    // -0 and +0 compare equal, so they must hash equally.
    return hash<uint16_t>{}(Key == 0 ? 0 : __builtin_bit_cast(uint16_t, Key));
  }
};
} // namespace std

#endif // _LIBSYCL___IMPL_HALF_HPP
