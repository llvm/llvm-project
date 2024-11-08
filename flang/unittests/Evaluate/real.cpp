#include "fp-testing.h"
#include "testing.h"
#include "flang/Evaluate/type.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

using namespace Fortran::evaluate;
using namespace Fortran::common;

using Real2 = Scalar<Type<TypeCategory::Real, 2>>;
using Real3 = Scalar<Type<TypeCategory::Real, 3>>;
using Real4 = Scalar<Type<TypeCategory::Real, 4>>;
using Real8 = Scalar<Type<TypeCategory::Real, 8>>;
#ifdef __x86_64__
using Real10 = Scalar<Type<TypeCategory::Real, 10>>;
#endif
using Real16 = Scalar<Type<TypeCategory::Real, 16>>;
using Integer4 = Scalar<Type<TypeCategory::Integer, 4>>;
using Integer8 = Scalar<Type<TypeCategory::Integer, 8>>;

void dumpTest() {
  struct {
    std::uint64_t raw;
    const char *expected;
  } table[] = {
      {0x7f876543, "NaN0x7f876543"},
      {0x7f800000, "Inf"},
      {0xff800000, "-Inf"},
      {0x00000000, "0.0"},
      {0x80000000, "-0.0"},
      {0x3f800000, "0x1.0p0"},
      {0xbf800000, "-0x1.0p0"},
      {0x40000000, "0x1.0p1"},
      {0x3f000000, "0x1.0p-1"},
      {0x7f7fffff, "0x1.fffffep127"},
      {0x00800000, "0x1.0p-126"},
      {0x00400000, "0x0.8p-126"},
      {0x00000001, "0x0.000002p-126"},
      {0, nullptr},
  };
  for (int j{0}; table[j].expected != nullptr; ++j) {
    TEST(Real4{Integer4{table[j].raw}}.DumpHexadecimal() == table[j].expected)
    ("%d", j);
  }
}

template <typename R> void basicTests(int rm, Rounding rounding) {
  static constexpr int kind{R::bits / 8};
  char desc[64];
  using Word = typename R::Word;
  std::snprintf(desc, sizeof desc, "bits=%d, le=%d, kind=%d", R::bits,
      Word::littleEndian, kind);
  R zero;
  TEST(!zero.IsNegative())(desc);
  TEST(!zero.IsNotANumber())(desc);
  TEST(!zero.IsInfinite())(desc);
  TEST(zero.IsZero())(desc);
  MATCH(0, zero.Exponent())(desc);
  TEST(zero.RawBits().IsZero())(desc);
  MATCH(0, zero.RawBits().ToUInt64())(desc);
  TEST(zero.ABS().RawBits().IsZero())(desc);
  TEST(zero.Negate().RawBits().IEOR(Word::MASKL(1)).IsZero())(desc);
  TEST(zero.Compare(zero) == Relation::Equal)(desc);
  R minusZero{Word{std::uint64_t{1}}.SHIFTL(R::bits - 1)};
  TEST(minusZero.IsNegative())(desc);
  TEST(!minusZero.IsNotANumber())(desc);
  TEST(!minusZero.IsInfinite())(desc);
  TEST(minusZero.IsZero())(desc);
  TEST(minusZero.ABS().RawBits().IsZero())(desc);
  TEST(minusZero.Negate().RawBits().IsZero())(desc);
  MATCH(0, minusZero.Exponent())(desc);
  MATCH(0, minusZero.RawBits().LEADZ())(desc);
  MATCH(1, minusZero.RawBits().POPCNT())(desc);
  TEST(minusZero.Compare(minusZero) == Relation::Equal)(desc);
  TEST(zero.Compare(minusZero) == Relation::Equal)(desc);
  ValueWithRealFlags<R> vr;
  MATCH(0, vr.value.RawBits().ToUInt64())(desc);
  TEST(vr.flags.empty())(desc);
  R nan{Word{std::uint64_t{1}}
            .SHIFTL(R::bits)
            .SubtractSigned(Word{std::uint64_t{1}})
            .value};
  MATCH(R::bits, nan.RawBits().POPCNT())(desc);
  TEST(!nan.IsNegative())(desc);
  TEST(nan.IsNotANumber())(desc);
  TEST(!nan.IsInfinite())(desc);
  TEST(!nan.IsZero())(desc);
  TEST(zero.Compare(nan) == Relation::Unordered)(desc);
  TEST(minusZero.Compare(nan) == Relation::Unordered)(desc);
  TEST(nan.Compare(zero) == Relation::Unordered)(desc);
  TEST(nan.Compare(minusZero) == Relation::Unordered)(desc);
  TEST(nan.Compare(nan) == Relation::Unordered)(desc);
  int significandBits{R::binaryPrecision - R::isImplicitMSB};
  int exponentBits{R::bits - significandBits - 1};
  std::uint64_t maxExponent{(std::uint64_t{1} << exponentBits) - 1};
  MATCH(nan.Exponent(), maxExponent)(desc);
  Word infWord{Word{maxExponent}.SHIFTL(significandBits)};
  Word negInfWord{
      Word{maxExponent}.SHIFTL(significandBits).IOR(Word::MASKL(1))};
  if constexpr (kind == 10) { // x87
    infWord = infWord.IBSET(63);
    negInfWord = negInfWord.IBSET(63);
  }
  R inf{infWord};
  R negInf{negInfWord};
  TEST(!inf.IsNegative())(desc);
  TEST(!inf.IsNotANumber())(desc);
  TEST(inf.IsInfinite())(desc);
  TEST(!inf.IsZero())(desc);
  TEST(inf.RawBits().CompareUnsigned(inf.ABS().RawBits()) == Ordering::Equal)
  (desc);
  TEST(zero.Compare(inf) == Relation::Less)(desc);
  TEST(minusZero.Compare(inf) == Relation::Less)(desc);
  TEST(nan.Compare(inf) == Relation::Unordered)(desc);
  TEST(inf.Compare(inf) == Relation::Equal)(desc);
  TEST(negInf.IsNegative())(desc);
  TEST(!negInf.IsNotANumber())(desc);
  TEST(negInf.IsInfinite())(desc);
  TEST(!negInf.IsZero())(desc);
  TEST(inf.RawBits().CompareUnsigned(negInf.ABS().RawBits()) == Ordering::Equal)
  (desc);
  TEST(inf.RawBits().CompareUnsigned(negInf.Negate().RawBits()) ==
      Ordering::Equal)
  (desc);
  TEST(inf.Negate().RawBits().CompareUnsigned(negInf.RawBits()) ==
      Ordering::Equal)
  (desc);
  TEST(zero.Compare(negInf) == Relation::Greater)(desc);
  TEST(minusZero.Compare(negInf) == Relation::Greater)(desc);
  TEST(nan.Compare(negInf) == Relation::Unordered)(desc);
  TEST(inf.Compare(negInf) == Relation::Greater)(desc);
  TEST(negInf.Compare(negInf) == Relation::Equal)(desc);
  for (std::uint64_t j{0}; j < 63; ++j) {
    char ldesc[128];
    std::uint64_t x{1};
    x <<= j;
    std::snprintf(ldesc, sizeof ldesc, "%s j=%d x=0x%jx rm=%d", desc,
        static_cast<int>(j), static_cast<std::intmax_t>(x), rm);
    Integer8 ix{x};
    TEST(!ix.IsNegative())(ldesc);
    MATCH(x, ix.ToUInt64())(ldesc);
    vr = R::FromInteger(ix, rounding);
    TEST(!vr.value.IsNegative())(ldesc);
    TEST(!vr.value.IsNotANumber())(ldesc);
    TEST(!vr.value.IsZero())(ldesc);
    auto ivf = vr.value.template ToInteger<Integer8>();
    if (j > (maxExponent / 2)) {
      TEST(vr.flags.test(RealFlag::Overflow))(ldesc);
      TEST(vr.value.IsInfinite())(ldesc);
      TEST(ivf.flags.test(RealFlag::Overflow))(ldesc);
      MATCH(0x7fffffffffffffff, ivf.value.ToUInt64())(ldesc);
    } else {
      TEST(vr.flags.empty())(ldesc);
      TEST(!vr.value.IsInfinite())(ldesc);
      TEST(ivf.flags.empty())(ldesc);
      MATCH(x, ivf.value.ToUInt64())(ldesc);
      if (rounding.mode == RoundingMode::TiesToEven) { // to match stold()
        std::string decimal;
        llvm::raw_string_ostream ss{decimal};
        vr.value.AsFortran(ss, kind, false /*exact*/);
        const char *p{decimal.data()};
        MATCH(x, static_cast<std::uint64_t>(std::stold(decimal)))
        ("%s %s", ldesc, p);
        auto check{R::Read(p, rounding)};
        auto icheck{check.value.template ToInteger<Integer8>()};
        MATCH(x, icheck.value.ToUInt64())(ldesc);
        TEST(vr.value.Compare(check.value) == Relation::Equal)(ldesc);
      }
    }
    TEST(vr.value.ToWholeNumber().value.Compare(vr.value) == Relation::Equal)
    (ldesc);
    ix = ix.Negate().value;
    TEST(ix.IsNegative())(ldesc);
    x = -x;
    std::int64_t nx = x;
    MATCH(x, ix.ToUInt64())(ldesc);
    MATCH(nx, ix.ToInt64())(ldesc);
    vr = R::FromInteger(ix);
    TEST(vr.value.IsNegative())(ldesc);
    TEST(!vr.value.IsNotANumber())(ldesc);
    TEST(!vr.value.IsZero())(ldesc);
    ivf = vr.value.template ToInteger<Integer8>();
    if (j > (maxExponent / 2)) {
      TEST(vr.flags.test(RealFlag::Overflow))(ldesc);
      TEST(vr.value.IsInfinite())(ldesc);
      TEST(ivf.flags.test(RealFlag::Overflow))(ldesc);
      MATCH(0x8000000000000000, ivf.value.ToUInt64())(ldesc);
    } else {
      TEST(vr.flags.empty())(ldesc);
      TEST(!vr.value.IsInfinite())(ldesc);
      TEST(ivf.flags.empty())(ldesc);
      MATCH(x, ivf.value.ToUInt64())(ldesc);
      MATCH(nx, ivf.value.ToInt64())(ldesc);
    }
    TEST(vr.value.ToWholeNumber().value.Compare(vr.value) == Relation::Equal)
    (ldesc);
  }
}

// Takes an integer and distributes its bits across a floating
// point value.  The LSB is used to complement the result.
std::uint32_t MakeReal(std::uint32_t n) {
  int shifts[] = {-1, 31, 23, 30, 22, 0, 24, 29, 25, 28, 26, 1, 16, 21, 2, -1};
  std::uint32_t x{0};
  for (int j{1}; shifts[j] >= 0; ++j) {
    x |= ((n >> j) & 1) << shifts[j];
  }
  x ^= -(n & 1);
  return x;
}

std::uint64_t MakeReal(std::uint64_t n) {
  int shifts[] = {
      -1, 63, 52, 62, 51, 0, 53, 61, 54, 60, 55, 59, 1, 16, 50, 2, -1};
  std::uint64_t x{0};
  for (int j{1}; shifts[j] >= 0; ++j) {
    x |= ((n >> j) & 1) << shifts[j];
  }
  x ^= -(n & 1);
  return x;
}

inline bool IsNaN(std::uint32_t x) {
  return (x & 0x7f800000) == 0x7f800000 && (x & 0x007fffff) != 0;
}

inline bool IsNaN(std::uint64_t x) {
  return (x & 0x7ff0000000000000) == 0x7ff0000000000000 &&
      (x & 0x000fffffffffffff) != 0;
}

inline bool IsInfinite(std::uint32_t x) {
  return (x & 0x7fffffff) == 0x7f800000;
}

inline bool IsInfinite(std::uint64_t x) {
  return (x & 0x7fffffffffffffff) == 0x7ff0000000000000;
}

inline bool IsNegative(std::uint32_t x) { return (x & 0x80000000) != 0; }

inline bool IsNegative(std::uint64_t x) {
  return (x & 0x8000000000000000) != 0;
}

inline std::uint32_t NormalizeNaN(std::uint32_t x) {
  if (IsNaN(x)) {
    x = 0x7fe00000;
  }
  return x;
}

inline std::uint64_t NormalizeNaN(std::uint64_t x) {
  if (IsNaN(x)) {
    x = 0x7ffc000000000000;
  }
  return x;
}

enum FlagBits {
  Overflow = 1,
  DivideByZero = 2,
  InvalidArgument = 4,
  Underflow = 8,
  Inexact = 16,
};

#ifdef __clang__
// clang support for fenv.h is broken, so tests of flag settings
// are disabled.
inline std::uint32_t FlagsToBits(const RealFlags &) { return 0; }
#else
inline std::uint32_t FlagsToBits(const RealFlags &flags) {
  std::uint32_t bits{0};
  if (flags.test(RealFlag::Overflow)) {
    bits |= Overflow;
  }
  if (flags.test(RealFlag::DivideByZero)) {
    bits |= DivideByZero;
  }
  if (flags.test(RealFlag::InvalidArgument)) {
    bits |= InvalidArgument;
  }
  if (flags.test(RealFlag::Underflow)) {
    bits |= Underflow;
  }
  if (flags.test(RealFlag::Inexact)) {
    bits |= Inexact;
  }
  return bits;
}
#endif // __clang__

template <typename UINT = std::uint32_t, typename FLT = float, typename REAL>
void inttest(std::int64_t x, int pass, Rounding rounding) {
  union {
    UINT ui;
    FLT f;
  } u;
  ScopedHostFloatingPointEnvironment fpenv;
  Integer8 ix{x};
  ValueWithRealFlags<REAL> real;
  real = real.value.FromInteger(ix, rounding);
#ifndef __clang__ // broken and also slow
  fpenv.ClearFlags();
#endif
  FLT fcheck = x; // TODO unsigned too
  auto actualFlags{FlagsToBits(fpenv.CurrentFlags())};
  u.f = fcheck;
  UINT rcheck{NormalizeNaN(u.ui)};
  UINT check = real.value.RawBits().ToUInt64();
  MATCH(rcheck, check)("%d 0x%llx", pass, x);
  MATCH(actualFlags, FlagsToBits(real.flags))("%d 0x%llx", pass, x);
}

template <typename FLT = float> FLT ToIntPower(FLT x, int power) {
  if (power == 0) {
    return x / x;
  }
  bool negative{power < 0};
  if (negative) {
    power = -power;
  }
  FLT result{1};
  while (power > 0) {
    if (power & 1) {
      result *= x;
    }
    x *= x;
    power >>= 1;
  }
  if (negative) {
    result = 1.0 / result;
  }
  return result;
}

template <typename FLT, int decimalDigits>
FLT TimesIntPowerOfTen(FLT x, int power) {
  if (power > decimalDigits || power < -decimalDigits) {
    auto maxExactPowerOfTen{
        TimesIntPowerOfTen<FLT, decimalDigits>(1, decimalDigits)};
    auto big{ToIntPower<FLT>(maxExactPowerOfTen, power / decimalDigits)};
    auto small{
        TimesIntPowerOfTen<FLT, decimalDigits>(1, power % decimalDigits)};
    return (x * big) * small;
  }
  return x * ToIntPower<FLT>(10.0, power);
}

template <typename UINT = std::uint32_t, typename FLT = float,
    typename REAL = Real4>
void subsetTests(int pass, Rounding rounding, std::uint32_t opds) {
  for (int j{0}; j < 63; ++j) {
    std::int64_t x{1};
    x <<= j;
    inttest<UINT, FLT, REAL>(x, pass, rounding);
    inttest<UINT, FLT, REAL>(-x, pass, rounding);
  }
  inttest<UINT, FLT, REAL>(0, pass, rounding);
  inttest<UINT, FLT, REAL>(
      static_cast<std::int64_t>(0x8000000000000000), pass, rounding);

  union {
    UINT ui;
    FLT f;
  } u;
  ScopedHostFloatingPointEnvironment fpenv;

  for (UINT j{0}; j < opds; ++j) {

    UINT rj{MakeReal(j)};
    u.ui = rj;
    FLT fj{u.f};
    REAL x{typename REAL::Word{std::uint64_t{rj}}};

    // unary operations
    {
      ValueWithRealFlags<REAL> aint{x.ToWholeNumber()};
#ifndef __clang__ // broken and also slow
      fpenv.ClearFlags();
#endif
      FLT fcheck{std::trunc(fj)};
      auto actualFlags{FlagsToBits(fpenv.CurrentFlags())};
      actualFlags &= ~Inexact; // x86 std::trunc can set Inexact; AINT ain't
      u.f = fcheck;
#ifndef __clang__
      if (IsNaN(u.ui)) {
        actualFlags |= InvalidArgument; // x86 std::trunc(NaN) workaround
      }
#endif
      UINT rcheck{NormalizeNaN(u.ui)};
      UINT check = aint.value.RawBits().ToUInt64();
      MATCH(rcheck, check)
      ("%d AINT(0x%jx)", pass, static_cast<std::intmax_t>(rj));
      MATCH(actualFlags, FlagsToBits(aint.flags))
      ("%d AINT(0x%jx)", pass, static_cast<std::intmax_t>(rj));
    }

    {
      ValueWithRealFlags<REAL> root{x.SQRT(rounding)};
#ifndef __clang__ // broken and also slow
      fpenv.ClearFlags();
#endif
      FLT fcheck{std::sqrt(fj)};
      auto actualFlags{FlagsToBits(fpenv.CurrentFlags())};
      u.f = fcheck;
      UINT rcheck{NormalizeNaN(u.ui)};
      UINT check = root.value.RawBits().ToUInt64();
      MATCH(rcheck, check)
      ("%d SQRT(0x%jx)", pass, static_cast<std::intmax_t>(rj));
      MATCH(actualFlags, FlagsToBits(root.flags))
      ("%d SQRT(0x%jx)", pass, static_cast<std::intmax_t>(rj));
    }

    {
      MATCH(IsNaN(rj), x.IsNotANumber())
      ("%d IsNaN(0x%jx)", pass, static_cast<std::intmax_t>(rj));
      MATCH(IsInfinite(rj), x.IsInfinite())
      ("%d IsInfinite(0x%jx)", pass, static_cast<std::intmax_t>(rj));

      static constexpr int kind{REAL::bits / 8};
      std::string s, cssBuf;
      llvm::raw_string_ostream ss{s};
      llvm::raw_string_ostream css{cssBuf};
      x.AsFortran(ss, kind, false /*exact*/);
      if (IsNaN(rj)) {
        css << "(0._" << kind << "/0.)";
        MATCH(cssBuf, s)
        ("%d invalid(0x%jx)", pass, static_cast<std::intmax_t>(rj));
      } else if (IsInfinite(rj)) {
        css << '(';
        if (IsNegative(rj)) {
          css << '-';
        }
        css << "1._" << kind << "/0.)";
        MATCH(cssBuf, s)
        ("%d overflow(0x%jx)", pass, static_cast<std::intmax_t>(rj));
      } else {
        const char *p = s.data();
        if (*p == '(') {
          ++p;
        }
        auto readBack{REAL::Read(p, rounding)};
        MATCH(rj, readBack.value.RawBits().ToUInt64())
        ("%d Read(AsFortran()) 0x%jx %s %g", pass,
            static_cast<std::intmax_t>(rj), s.data(), static_cast<double>(fj));
        MATCH('_', *p)
        ("%d Read(AsFortran()) 0x%jx %s %d", pass,
            static_cast<std::intmax_t>(rj), s.data(),
            static_cast<int>(p - s.data()));
      }
    }

    // dyadic operations
    for (UINT k{0}; k < opds; ++k) {
      UINT rk{MakeReal(k)};
      u.ui = rk;
      FLT fk{u.f};
      REAL y{typename REAL::Word{std::uint64_t{rk}}};
      {
        ValueWithRealFlags<REAL> sum{x.Add(y, rounding)};
#ifndef __clang__ // broken and also slow
        fpenv.ClearFlags();
#endif
        FLT fcheck{fj + fk};
        auto actualFlags{FlagsToBits(fpenv.CurrentFlags())};
        u.f = fcheck;
        UINT rcheck{NormalizeNaN(u.ui)};
        UINT check = sum.value.RawBits().ToUInt64();
        MATCH(rcheck, check)
        ("%d 0x%jx + 0x%jx", pass, static_cast<std::intmax_t>(rj),
            static_cast<std::intmax_t>(rk));
        MATCH(actualFlags, FlagsToBits(sum.flags))
        ("%d 0x%jx + 0x%jx", pass, static_cast<std::intmax_t>(rj),
            static_cast<std::intmax_t>(rk));
      }
      {
        ValueWithRealFlags<REAL> diff{x.Subtract(y, rounding)};
#ifndef __clang__ // broken and also slow
        fpenv.ClearFlags();
#endif
        FLT fcheck{fj - fk};
        auto actualFlags{FlagsToBits(fpenv.CurrentFlags())};
        u.f = fcheck;
        UINT rcheck{NormalizeNaN(u.ui)};
        UINT check = diff.value.RawBits().ToUInt64();
        MATCH(rcheck, check)
        ("%d 0x%jx - 0x%jx", pass, static_cast<std::intmax_t>(rj),
            static_cast<std::intmax_t>(rk));
        MATCH(actualFlags, FlagsToBits(diff.flags))
        ("%d 0x%jx - 0x%jx", pass, static_cast<std::intmax_t>(rj),
            static_cast<std::intmax_t>(rk));
      }
      {
        ValueWithRealFlags<REAL> prod{x.Multiply(y, rounding)};
#ifndef __clang__ // broken and also slow
        fpenv.ClearFlags();
#endif
        FLT fcheck{fj * fk};
        auto actualFlags{FlagsToBits(fpenv.CurrentFlags())};
        u.f = fcheck;
        UINT rcheck{NormalizeNaN(u.ui)};
        UINT check = prod.value.RawBits().ToUInt64();
        MATCH(rcheck, check)
        ("%d 0x%jx * 0x%jx", pass, static_cast<std::intmax_t>(rj),
            static_cast<std::intmax_t>(rk));
        MATCH(actualFlags, FlagsToBits(prod.flags))
        ("%d 0x%jx * 0x%jx", pass, static_cast<std::intmax_t>(rj),
            static_cast<std::intmax_t>(rk));
      }
      {
        ValueWithRealFlags<REAL> quot{x.Divide(y, rounding)};
#ifndef __clang__ // broken and also slow
        fpenv.ClearFlags();
#endif
        FLT fcheck{fj / fk};
        auto actualFlags{FlagsToBits(fpenv.CurrentFlags())};
        u.f = fcheck;
        UINT rcheck{NormalizeNaN(u.ui)};
        UINT check = quot.value.RawBits().ToUInt64();
        MATCH(rcheck, check)
        ("%d 0x%jx / 0x%jx", pass, static_cast<std::intmax_t>(rj),
            static_cast<std::intmax_t>(rk));
        MATCH(actualFlags, FlagsToBits(quot.flags))
        ("%d 0x%jx / 0x%jx", pass, static_cast<std::intmax_t>(rj),
            static_cast<std::intmax_t>(rk));
      }
    }
  }
}

void roundTest(int rm, Rounding rounding, std::uint32_t opds) {
  basicTests<Real2>(rm, rounding);
  basicTests<Real3>(rm, rounding);
  basicTests<Real4>(rm, rounding);
  basicTests<Real8>(rm, rounding);
#ifdef __x86_64__
  basicTests<Real10>(rm, rounding);
#endif
  basicTests<Real16>(rm, rounding);
  ScopedHostFloatingPointEnvironment::SetRounding(rounding);
  subsetTests<std::uint32_t, float, Real4>(rm, rounding, opds);
  subsetTests<std::uint64_t, double, Real8>(rm, rounding, opds);
}

int main() {
  dumpTest();
  std::uint32_t opds{512}; // for quick testing by default
  if (const char *p{std::getenv("REAL_TEST_OPERANDS")}) {
    // Use 8192 or 16384 for more exhaustive testing.
    opds = std::atol(p);
  }
  roundTest(0, Rounding{RoundingMode::TiesToEven}, opds);
  roundTest(1, Rounding{RoundingMode::ToZero}, opds);
  roundTest(2, Rounding{RoundingMode::Up}, opds);
  roundTest(3, Rounding{RoundingMode::Down}, opds);
  // TODO: how to test Rounding::TiesAwayFromZero on x86?
  return testing::Complete();
}
