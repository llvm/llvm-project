//===-- runtime/sum.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements SUM for all required operand types and shapes.
//
// Real and complex SUM reductions attempt to reduce floating-point
// cancellation on intermediate results by using "Kahan summation"
// (basically the same as manual "double-double").

#include "reduction-templates.h"
#include "flang/Common/float128.h"
#include "flang/Runtime/reduction.h"
#include <cfloat>
#include <cinttypes>
#include <complex>

namespace Fortran::runtime {

template <typename INTERMEDIATE> class IntegerSumAccumulator {
public:
  explicit RT_API_ATTRS IntegerSumAccumulator(const Descriptor &array)
      : array_{array} {}
  void RT_API_ATTRS Reinitialize() { sum_ = 0; }
  template <typename A>
  RT_API_ATTRS void GetResult(A *p, int /*zeroBasedDim*/ = -1) const {
    *p = static_cast<A>(sum_);
  }
  template <typename A>
  RT_API_ATTRS bool AccumulateAt(const SubscriptValue at[]) {
    sum_ += *array_.Element<A>(at);
    return true;
  }

private:
  const Descriptor &array_;
  INTERMEDIATE sum_{0};
};

template <typename INTERMEDIATE> class RealSumAccumulator {
public:
  explicit RT_API_ATTRS RealSumAccumulator(const Descriptor &array)
      : array_{array} {}
  void RT_API_ATTRS Reinitialize() { sum_ = correction_ = 0; }
  template <typename A> RT_API_ATTRS A Result() const { return sum_; }
  template <typename A>
  RT_API_ATTRS void GetResult(A *p, int /*zeroBasedDim*/ = -1) const {
    *p = Result<A>();
  }
  template <typename A> RT_API_ATTRS bool Accumulate(A x) {
    // Kahan summation
    auto next{x + correction_};
    auto oldSum{sum_};
    sum_ += next;
    correction_ = (sum_ - oldSum) - next; // algebraically zero
    return true;
  }
  template <typename A>
  RT_API_ATTRS bool AccumulateAt(const SubscriptValue at[]) {
    return Accumulate(*array_.Element<A>(at));
  }

private:
  const Descriptor &array_;
  INTERMEDIATE sum_{0.0}, correction_{0.0};
};

template <typename PART> class ComplexSumAccumulator {
public:
  explicit RT_API_ATTRS ComplexSumAccumulator(const Descriptor &array)
      : array_{array} {}
  void RT_API_ATTRS Reinitialize() {
    reals_.Reinitialize();
    imaginaries_.Reinitialize();
  }
  template <typename A>
  RT_API_ATTRS void GetResult(A *p, int /*zeroBasedDim*/ = -1) const {
    using ResultPart = typename A::value_type;
    *p = {reals_.template Result<ResultPart>(),
        imaginaries_.template Result<ResultPart>()};
  }
  template <typename A> RT_API_ATTRS bool Accumulate(const A &z) {
    reals_.Accumulate(z.real());
    imaginaries_.Accumulate(z.imag());
    return true;
  }
  template <typename A>
  RT_API_ATTRS bool AccumulateAt(const SubscriptValue at[]) {
    return Accumulate(*array_.Element<A>(at));
  }

private:
  const Descriptor &array_;
  RealSumAccumulator<PART> reals_{array_}, imaginaries_{array_};
};

extern "C" {
RT_EXT_API_GROUP_BEGIN

CppTypeFor<TypeCategory::Integer, 1> RTDEF(SumInteger1)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 1>(x, source, line, dim, mask,
      IntegerSumAccumulator<CppTypeFor<TypeCategory::Integer, 4>>{x}, "SUM");
}
CppTypeFor<TypeCategory::Integer, 2> RTDEF(SumInteger2)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 2>(x, source, line, dim, mask,
      IntegerSumAccumulator<CppTypeFor<TypeCategory::Integer, 4>>{x}, "SUM");
}
CppTypeFor<TypeCategory::Integer, 4> RTDEF(SumInteger4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 4>(x, source, line, dim, mask,
      IntegerSumAccumulator<CppTypeFor<TypeCategory::Integer, 4>>{x}, "SUM");
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(SumInteger8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 8>(x, source, line, dim, mask,
      IntegerSumAccumulator<CppTypeFor<TypeCategory::Integer, 8>>{x}, "SUM");
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTDEF(SumInteger16)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 16>(x, source, line, dim,
      mask, IntegerSumAccumulator<CppTypeFor<TypeCategory::Integer, 16>>{x},
      "SUM");
}
#endif

// TODO: real/complex(2 & 3)
CppTypeFor<TypeCategory::Real, 4> RTDEF(SumReal4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Real, 4>(
      x, source, line, dim, mask, RealSumAccumulator<double>{x}, "SUM");
}
CppTypeFor<TypeCategory::Real, 8> RTDEF(SumReal8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Real, 8>(
      x, source, line, dim, mask, RealSumAccumulator<double>{x}, "SUM");
}
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTDEF(SumReal10)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Real, 10>(
      x, source, line, dim, mask, RealSumAccumulator<long double>{x}, "SUM");
}
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTDEF(SumReal16)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Real, 16>(
      x, source, line, dim, mask, RealSumAccumulator<long double>{x}, "SUM");
}
#endif

void RTDEF(CppSumComplex4)(CppTypeFor<TypeCategory::Complex, 4> &result,
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  result = GetTotalReduction<TypeCategory::Complex, 4>(
      x, source, line, dim, mask, ComplexSumAccumulator<double>{x}, "SUM");
}
void RTDEF(CppSumComplex8)(CppTypeFor<TypeCategory::Complex, 8> &result,
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  result = GetTotalReduction<TypeCategory::Complex, 8>(
      x, source, line, dim, mask, ComplexSumAccumulator<double>{x}, "SUM");
}
#if LDBL_MANT_DIG == 64
void RTDEF(CppSumComplex10)(CppTypeFor<TypeCategory::Complex, 10> &result,
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  result = GetTotalReduction<TypeCategory::Complex, 10>(
      x, source, line, dim, mask, ComplexSumAccumulator<long double>{x}, "SUM");
}
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
void RTDEF(CppSumComplex16)(CppTypeFor<TypeCategory::Complex, 16> &result,
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  result = GetTotalReduction<TypeCategory::Complex, 16>(
      x, source, line, dim, mask, ComplexSumAccumulator<long double>{x}, "SUM");
}
#endif

void RTDEF(SumDim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line, const Descriptor *mask) {
  TypedPartialNumericReduction<IntegerSumAccumulator, RealSumAccumulator,
      ComplexSumAccumulator>(result, x, dim, source, line, mask, "SUM");
}

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime
