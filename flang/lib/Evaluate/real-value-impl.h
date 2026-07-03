//===-- lib/Evaluate/real-value-impl.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_REAL_VALUE_IMPL_H_
#define FORTRAN_EVALUATE_REAL_VALUE_IMPL_H_

#include "flang/Evaluate/real.h"
#include "llvm/Support/ErrorHandling.h"
#include <type_traits>
#include <utility>
#include <variant>

// Some environments, viz. glibc 2.17 and *BSD, allow the macro HUGE
// to leak out of <math.h>.
#undef HUGE

namespace llvm {
class raw_ostream;
}

namespace Fortran::evaluate::value {

// RealValueImpl: runtime-kind IEEE floating-point value (variant-backed).
class RealValueImpl {
public:
  // Per-KIND fixed-width backing formats.
  using R2 = Real<Integer<16>, 11>; // IEEE half
  using R3 = Real<Integer<16>, 8>; // bfloat16
  using R4 = Real<Integer<32>, 24>; // IEEE single
  using R8 = Real<Integer<64>, 53>; // IEEE double
  using R10 = Real<X87IntegerContainer, 64>; // 80387 extended precision
  using R16 = Real<Integer<128>, 113>; // IEEE quad
  // Widest representation: used for raw access and as the common type when
  // converting between kinds.
  using Word = IntegerValue;
  // The value is held in its actual fixed-width value::Real format; the active
  // std::variant alternative carries the runtime REAL kind.  std::monostate
  // models a default-initialized zero of an as-yet-unknown kind.  All
  // operations are expressed by forwarding to the stored value::Real.
  using Storage = std::variant<std::monostate, R2, R3, R4, R8, R10, R16>;

  RealValueImpl() = default;
  RealValueImpl(const RealValueImpl &) = default;
  RealValueImpl(RealValueImpl &&) = default;
  RealValueImpl &operator=(const RealValueImpl &) = default;
  RealValueImpl &operator=(RealValueImpl &&) = default;

  // Interpret w as the raw bit pattern of a value of the given runtime kind.
  RealValueImpl(const Word &w, int kind);

  // Comparison operators
  bool operator==(const RealValueImpl &y) const;

  // Kind-property inquiries, formerly compile-time constants derived from the
  // PREC template parameter; now selected by the runtime KIND.
  static int binaryPrecision(int kind);
  static int isImplicitMSB(int kind);
  static int DIGITS(int kind);
  static int PRECISION(int kind);
  static int RANGE(int kind);
  static int MAXEXPONENT(int kind);
  static int MINEXPONENT(int kind);

  static RealValueImpl HUGE(int kind);
  static RealValueImpl EPSILON(int kind);
  static RealValueImpl TINY(int kind);
  static RealValueImpl NotANumber(int kind);

  // Runtime kind / width accessors
  int kind() const;
  int bits() const;
  bool IsMonostate() const { return storage_.index() == 0; }
  bool IsZero() const;
  bool IsNegative() const;
  bool IsNotANumber() const;
  bool IsQuietNaN() const;
  bool IsSignalingNaN() const;
  bool IsInfinite() const;
  bool IsFinite() const;
  bool IsNormal() const;
  int Exponent() const;
  bool StoreRawBytes(void *to, std::size_t bytes) const;
  static RealValueImpl FromRawBytes(const void *raw, std::size_t bytes);
  static RealValueImpl Zero(int kind);

  // The raw bit pattern at the value's runtime width.
  IntegerValue RawBits() const;

  // Comparisons
  Relation Compare(const RealValueImpl &y) const;

  // Unary operations
  RealValueImpl ABS() const;
  RealValueImpl Negate() const;
  RealValueImpl SIGN(const RealValueImpl &x) const;
  RealValueImpl SetSign(bool toNegative) const;
  RealValueImpl FlushSubnormalToZero() const;

  // Binary arithmetic
  ValueWithRealFlags<RealValueImpl> Add(const RealValueImpl &y,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;
  ValueWithRealFlags<RealValueImpl> Subtract(const RealValueImpl &y,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;
  ValueWithRealFlags<RealValueImpl> Multiply(const RealValueImpl &y,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;
  ValueWithRealFlags<RealValueImpl> Divide(const RealValueImpl &y,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;
  ValueWithRealFlags<RealValueImpl> SQRT(
      Rounding rounding = TargetCharacteristics::defaultRounding) const;
  ValueWithRealFlags<RealValueImpl> HYPOT(const RealValueImpl &y,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;
  ValueWithRealFlags<RealValueImpl> MOD(const RealValueImpl &y,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;
  ValueWithRealFlags<RealValueImpl> MODULO(const RealValueImpl &y,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;
  ValueWithRealFlags<RealValueImpl> DIM(const RealValueImpl &y,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;

  // Convert this value to a different real kind (single rounding).
  RealValueImpl ConvertToKind(int kind) const {
    return Convert(*this, kind).value;
  }

  RealValueImpl FRACTION() const;
  RealValueImpl RRSPACING() const;
  RealValueImpl SPACING() const;
  RealValueImpl SET_EXPONENT(std::int64_t e) const;

  ValueWithRealFlags<RealValueImpl> NEAREST(bool upward) const;
  ValueWithRealFlags<RealValueImpl> ToWholeNumber(
      common::RoundingMode mode = common::RoundingMode::ToZero) const;
  // Convert this real to an integer of the given bit width.
  ValueWithRealFlags<IntegerValue> ToInteger(
      common::RoundingMode mode = common::RoundingMode::ToZero,
      int toBits = 0) const;

  ValueWithRealFlags<RealValueImpl> SCALE(const IntegerValue &by,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;

  ValueWithRealFlags<RealValueImpl> KahanSummation(const RealValueImpl &y,
      RealValueImpl &correction,
      Rounding rounding = TargetCharacteristics::defaultRounding) const;

  IntegerValue EXPONENT() const;

  // Conversion from an integer facade (REAL()).
  static ValueWithRealFlags<RealValueImpl> FromInteger(const IntegerValue &n,
      int kind, bool isUnsigned = false,
      Rounding rounding = TargetCharacteristics::defaultRounding);

  // Conversion between real kinds.
  static ValueWithRealFlags<RealValueImpl> Convert(const RealValueImpl &from,
      int kind, Rounding rounding = TargetCharacteristics::defaultRounding);

  static ValueWithRealFlags<RealValueImpl> Read(const char *&pp, int kind,
      Rounding rounding = TargetCharacteristics::defaultRounding);

  std::string DumpHexadecimal() const;
  llvm::raw_ostream &AsFortran(
      llvm::raw_ostream &o, int kind, bool minimal = false) const;

  template <typename F>
  auto WithReal(F &&f) const
      -> decltype(std::declval<F>()(std::declval<R8>())) {
    switch (storage_.index()) {
    case 1:
      return f(std::get<R2>(storage_));
    case 2:
      return f(std::get<R3>(storage_));
    case 3:
      return f(std::get<R4>(storage_));
    case 4:
      return f(std::get<R8>(storage_));
    case 5:
      return f(std::get<R10>(storage_));
    case 6:
      return f(std::get<R16>(storage_));
    default:
      llvm_unreachable("operation on uninitialized RealValueImpl");
    }
  }

  template <typename T> static RealValueImpl Wrap(const T &r) {
    RealValueImpl v;
    v.storage_ = r;
    return v;
  }

  template <typename T>
  static ValueWithRealFlags<RealValueImpl> Wrap(
      const ValueWithRealFlags<T> &x) {
    ValueWithRealFlags<RealValueImpl> r;
    r.value = Wrap(x.value);
    r.flags = x.flags;
    return r;
  }

  template <typename V> static std::decay_t<V> As(const RealValueImpl &y) {
    using R = std::decay_t<V>;
    if (y.IsMonostate()) {
      return R{};
    }
    return y.WithReal([](const auto &yv) -> R {
      using YR = std::decay_t<decltype(yv)>;
      if constexpr (std::is_same_v<YR, R>) {
        return yv;
      } else {
        return R::Convert(yv).value;
      }
    });
  }

  template <typename R> static constexpr int KindOf() {
    if constexpr (std::is_same_v<R, R2>) {
      return 2;
    } else if constexpr (std::is_same_v<R, R3>) {
      return 3;
    } else if constexpr (std::is_same_v<R, R4>) {
      return 4;
    } else if constexpr (std::is_same_v<R, R8>) {
      return 8;
    } else if constexpr (std::is_same_v<R, R10>) {
      return 10;
    } else if constexpr (std::is_same_v<R, R16>) {
      return 16;
    }
    llvm_unreachable(
        "uninitialized value has not a defined kind or unsupported width");
  }

  Storage storage_;
};

} // namespace Fortran::evaluate::value
#endif // FORTRAN_EVALUATE_REAL_VALUE_IMPL_H_
