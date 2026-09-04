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
class IntegerValue;

class RealValueImpl {
public:
  using R2 = Real<Integer<16>, 11>; // IEEE half
  using R3 = Real<Integer<16>, 8>; // bfloat16
  using R4 = Real<Integer<32>, 24>; // IEEE single
  using R8 = Real<Integer<64>, 53>; // IEEE double
  using R10 = Real<X87IntegerContainer, 64>; // 80387 extended precision
  using R16 = Real<Integer<128>, 113>; // IEEE quad
  using Storage = std::variant<std::monostate, R2, R3, R4, R8, R10, R16>;
  using Word = IntegerValue;

  // rule-of-five
  ~RealValueImpl() = default;
  RealValueImpl(const RealValueImpl &) = default;
  RealValueImpl(RealValueImpl &&) = default;
  RealValueImpl &operator=(const RealValueImpl &) = default;
  RealValueImpl &operator=(RealValueImpl &&) = default;

  RealValueImpl() = default;

  // Interpret w as the raw bit pattern of a value of the given runtime kind.
  RealValueImpl(int kind, const Word &w);

  RealValueImpl(int kind, double x);

  static RealValueImpl Zero(int kind);

  template <typename T> static RealValueImpl FromWord(const T &r) {
    RealValueImpl v;
    v.storage_ = r;
    return v;
  }

  template <typename T>
  static ValueWithRealFlags<RealValueImpl> FromWord(
      const ValueWithRealFlags<T> &x) {
    ValueWithRealFlags<RealValueImpl> r;
    r.value = FromWord(x.value);
    r.flags = x.flags;
    return r;
  }

  static RealValueImpl FromRawBytes(
      int kind, const void *raw, std::size_t expectedSize);

  void print(llvm::raw_ostream &os) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif

  bool IsMonostate() const { return storage_.index() == 0; }
  int kind() const;

  int bits() const;

  std::size_t bytesStored() const { return bytesStored(kind()); }
  static constexpr std::size_t bytesStored(int kind) {
    switch (kind) {
    case 3:
      return 2;
    case 10:
      return 16;
    default:
      return kind;
    }
  }

  bool IsZero() const;

  // Comparison operators
  bool operator==(const RealValueImpl &y) const;
  bool operator!=(const RealValueImpl &y) const { return !(*this == y); }

  // Kind-property inquiries, formerly compile-time constants derived from the
  // PREC template parameter; now selected by the runtime KIND.
  static int DIGITS(int kind);
  static int PRECISION(int kind);
  static int RANGE(int kind);
  static int MAXEXPONENT(int kind);
  static int MINEXPONENT(int kind);

  static RealValueImpl HUGE(int kind);
  static RealValueImpl EPSILON(int kind);
  static RealValueImpl TINY(int kind);
  static RealValueImpl NotANumber(int kind);
  static RealValueImpl Infinity(int kind, bool negative = false);
  static RealValueImpl NegativeZero(int kind);

  // Runtime kind / width accessors
  bool IsNegative() const;
  bool IsNotANumber() const;
  bool IsSignalingNaN() const;
  bool IsInfinite() const;
  bool IsFinite() const;
  bool IsNormal() const;
  int Exponent() const;
  void StoreRawBytes(void *dst, size_t size, bool *changed) const;

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
  static ValueWithRealFlags<RealValueImpl> FromInteger(int kind,
      const IntegerValue &n, bool isUnsigned = false,
      Rounding rounding = TargetCharacteristics::defaultRounding);

  // Conversion between real kinds.
  static ValueWithRealFlags<RealValueImpl> Convert(int kind,
      const RealValueImpl &from,
      Rounding rounding = TargetCharacteristics::defaultRounding);

  static ValueWithRealFlags<RealValueImpl> Read(int kind, const char *&pp,
      Rounding rounding = TargetCharacteristics::defaultRounding);

  std::string DumpHexadecimal() const;
  llvm::raw_ostream &AsFortran(
      llvm::raw_ostream &o, int kind, bool minimal = false) const;

  template <typename V> static std::decay_t<V> AsWord(const RealValueImpl &y) {
    using R = std::decay_t<V>;
    if (y.IsMonostate()) {
      return R{};
    }

    return y.withWord([](const auto &yv) -> R {
      using YR = std::decay_t<decltype(yv)>;
      if constexpr (std::is_same_v<YR, R>) {
        return yv;
      } else {
        return R::Convert(yv).value;
      }
    });
  }

  // Compile-time dispatchers to current/specified kind

  template <typename F> static inline auto withWordProto(int kind, F &&f) {
    using namespace Fortran::evaluate::value;
    switch (kind) {
    case 2:
      return f(RealValueImpl::R2{});
    case 3:
      return f(RealValueImpl::R3{});
    case 4:
      return f(RealValueImpl::R4{});
    case 8:
      return f(RealValueImpl::R8{});
    case 10:
      return f(RealValueImpl::R10{});
    case 16:
      return f(RealValueImpl::R16{});
    default:
      DIE("arbitrary bits not yet supported");
    }
  }

  template <typename F> auto withWord(F &&f) const {
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
      DIE("operation on uninitialized RealValueImpl");
    }
  }

private:
  Storage storage_;
};

} // namespace Fortran::evaluate::value

namespace llvm {
/// For pretty printing in GTest
inline raw_ostream &operator<<(
    raw_ostream &os, const Fortran::evaluate::value::RealValueImpl &v) {
  v.print(os);
  return os;
}
} // namespace llvm

#endif // FORTRAN_EVALUATE_REAL_VALUE_IMPL_H_
