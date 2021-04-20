//===-- runtime/reduction.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements ALL, ANY, COUNT, FINDLOC, IPARITY, MAXLOC, MAXVAL, MINLOC, MINVAL,
// PARITY, PRODUCT, and SUM for all required operand types and shapes and,
// for FINDLOC, MAXLOC, & MINLOC, kinds of results.
//
// * Real and complex SUM reductions attempt to reduce floating-point
//   cancellation on intermediate results by adding up partial sums
//   for positive and negative elements independently.
// * Partial reductions (i.e., those with DIM= arguments that are not
//   required to be 1 by the rank of the argument) return arrays that
//   are dynamically allocated in a caller-supplied descriptor.
// * Total reductions (i.e., no DIM= argument) with FINDLOC, MAXLOC, & MINLOC
//   return integer vectors of some kind, not scalars; a caller-supplied
//   descriptor is used
// * Character-valued reductions (MAXVAL & MINVAL) return arbitrary
//   length results, dynamically allocated in a caller-supplied descriptor

#include "reduction.h"
#include "character.h"
#include "cpp-type.h"
#include "terminator.h"
#include "tools.h"
#include "flang/Common/long-double.h"
#include <cinttypes>
#include <complex>
#include <limits>
#include <type_traits>

namespace Fortran::runtime {

// Generic reduction templates

// Reductions are implemented with *accumulators*, which are instances of
// classes that incrementally build up the result (or an element thereof) during
// a traversal of the unmasked elements of an array.  Each accumulator class
// supports a constructor (which captures a reference to the array), an
// AccumulateAt() member function that applies supplied subscripts to the
// array and does something with a scalar element, and a GetResult()
// member function that copies a final result into its destination.

// Total reduction of the array argument to a scalar (or to a vector in the
// cases of FINDLOC, MAXLOC, & MINLOC).  These are the cases without DIM= or
// cases where the argument has rank 1 and DIM=, if present, must be 1.
template <typename TYPE, typename ACCUMULATOR>
inline void DoTotalReduction(const Descriptor &x, int dim,
    const Descriptor *mask, ACCUMULATOR &accumulator, const char *intrinsic,
    Terminator &terminator) {
  if (dim < 0 || dim > 1) {
    terminator.Crash(
        "%s: bad DIM=%d for argument with rank %d", intrinsic, dim, x.rank());
  }
  SubscriptValue xAt[maxRank];
  x.GetLowerBounds(xAt);
  if (mask) {
    CheckConformability(x, *mask, terminator, intrinsic, "ARRAY", "MASK");
    SubscriptValue maskAt[maxRank];
    mask->GetLowerBounds(maskAt);
    if (mask->rank() > 0) {
      for (auto elements{x.Elements()}; elements--;
           x.IncrementSubscripts(xAt), mask->IncrementSubscripts(maskAt)) {
        if (IsLogicalElementTrue(*mask, maskAt)) {
          accumulator.template AccumulateAt<TYPE>(xAt);
        }
      }
      return;
    } else if (!IsLogicalElementTrue(*mask, maskAt)) {
      // scalar MASK=.FALSE.: return identity value
      return;
    }
  }
  // No MASK=, or scalar MASK=.TRUE.
  for (auto elements{x.Elements()}; elements--; x.IncrementSubscripts(xAt)) {
    if (!accumulator.template AccumulateAt<TYPE>(xAt)) {
      break; // cut short, result is known
    }
  }
}

template <TypeCategory CAT, int KIND, typename ACCUMULATOR>
inline CppTypeFor<CAT, KIND> GetTotalReduction(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask,
    ACCUMULATOR &&accumulator, const char *intrinsic) {
  Terminator terminator{source, line};
  RUNTIME_CHECK(terminator, TypeCode(CAT, KIND) == x.type());
  using CppType = CppTypeFor<CAT, KIND>;
  DoTotalReduction<CppType>(x, dim, mask, accumulator, intrinsic, terminator);
  CppType result;
#ifdef _MSC_VER // work around MSVC spurious error
  accumulator.GetResult(&result);
#else
  accumulator.template GetResult(&result);
#endif
  return result;
}

// For reductions on a dimension, e.g. SUM(array,DIM=2) where the shape
// of the array is [2,3,5], the shape of the result is [2,5] and
// result(j,k) = SUM(array(j,:,k)), possibly modified if the array has
// lower bounds other than one.  This utility subroutine creates an
// array of subscripts [j,_,k] for result subscripts [j,k] so that the
// elemets of array(j,:,k) can be reduced.
inline void GetExpandedSubscripts(SubscriptValue at[],
    const Descriptor &descriptor, int zeroBasedDim,
    const SubscriptValue from[]) {
  descriptor.GetLowerBounds(at);
  int rank{descriptor.rank()};
  int j{0};
  for (; j < zeroBasedDim; ++j) {
    at[j] += from[j] - 1 /*lower bound*/;
  }
  for (++j; j < rank; ++j) {
    at[j] += from[j - 1] - 1;
  }
}

template <typename TYPE, typename ACCUMULATOR>
inline void ReduceDimToScalar(const Descriptor &x, int zeroBasedDim,
    SubscriptValue subscripts[], TYPE *result, ACCUMULATOR &accumulator) {
  SubscriptValue xAt[maxRank];
  GetExpandedSubscripts(xAt, x, zeroBasedDim, subscripts);
  const auto &dim{x.GetDimension(zeroBasedDim)};
  SubscriptValue at{dim.LowerBound()};
  for (auto n{dim.Extent()}; n-- > 0; ++at) {
    xAt[zeroBasedDim] = at;
    if (!accumulator.template AccumulateAt<TYPE>(xAt)) {
      break;
    }
  }
#ifdef _MSC_VER // work around MSVC spurious error
  accumulator.GetResult(result, zeroBasedDim);
#else
  accumulator.template GetResult(result, zeroBasedDim);
#endif
}

template <typename TYPE, typename ACCUMULATOR>
inline void ReduceDimMaskToScalar(const Descriptor &x, int zeroBasedDim,
    SubscriptValue subscripts[], const Descriptor &mask, TYPE *result,
    ACCUMULATOR &accumulator) {
  SubscriptValue xAt[maxRank], maskAt[maxRank];
  GetExpandedSubscripts(xAt, x, zeroBasedDim, subscripts);
  GetExpandedSubscripts(maskAt, mask, zeroBasedDim, subscripts);
  const auto &xDim{x.GetDimension(zeroBasedDim)};
  SubscriptValue xPos{xDim.LowerBound()};
  const auto &maskDim{mask.GetDimension(zeroBasedDim)};
  SubscriptValue maskPos{maskDim.LowerBound()};
  for (auto n{x.GetDimension(zeroBasedDim).Extent()}; n-- > 0;
       ++xPos, ++maskPos) {
    maskAt[zeroBasedDim] = maskPos;
    if (IsLogicalElementTrue(mask, maskAt)) {
      xAt[zeroBasedDim] = xPos;
      if (!accumulator.template AccumulateAt<TYPE>(xAt)) {
        break;
      }
    }
  }
#ifdef _MSC_VER // work around MSVC spurious error
  accumulator.GetResult(result, zeroBasedDim);
#else
  accumulator.template GetResult(result, zeroBasedDim);
#endif
}

// Utility: establishes & allocates the result array for a partial
// reduction (i.e., one with DIM=).
static void CreatePartialReductionResult(Descriptor &result,
    const Descriptor &x, int dim, Terminator &terminator, const char *intrinsic,
    TypeCode typeCode) {
  int xRank{x.rank()};
  if (dim < 1 || dim > xRank) {
    terminator.Crash("%s: bad DIM=%d for rank %d", intrinsic, dim, xRank);
  }
  int zeroBasedDim{dim - 1};
  SubscriptValue resultExtent[maxRank];
  for (int j{0}; j < zeroBasedDim; ++j) {
    resultExtent[j] = x.GetDimension(j).Extent();
  }
  for (int j{zeroBasedDim + 1}; j < xRank; ++j) {
    resultExtent[j - 1] = x.GetDimension(j).Extent();
  }
  result.Establish(typeCode, x.ElementBytes(), nullptr, xRank - 1, resultExtent,
      CFI_attribute_allocatable);
  for (int j{0}; j + 1 < xRank; ++j) {
    result.GetDimension(j).SetBounds(1, resultExtent[j]);
  }
  if (int stat{result.Allocate()}) {
    terminator.Crash(
        "%s: could not allocate memory for result; STAT=%d", intrinsic, stat);
  }
}

// Partial reductions with DIM=

template <typename ACCUMULATOR, TypeCategory CAT, int KIND>
inline void PartialReduction(Descriptor &result, const Descriptor &x, int dim,
    const Descriptor *mask, Terminator &terminator, const char *intrinsic,
    ACCUMULATOR &accumulator) {
  CreatePartialReductionResult(
      result, x, dim, terminator, intrinsic, TypeCode{CAT, KIND});
  SubscriptValue at[maxRank];
  result.GetLowerBounds(at);
  INTERNAL_CHECK(at[0] == 1);
  using CppType = CppTypeFor<CAT, KIND>;
  if (mask) {
    CheckConformability(x, *mask, terminator, intrinsic, "ARRAY", "MASK");
    SubscriptValue maskAt[maxRank]; // contents unused
    if (mask->rank() > 0) {
      for (auto n{result.Elements()}; n-- > 0; result.IncrementSubscripts(at)) {
        accumulator.Reinitialize();
        ReduceDimMaskToScalar<CppType, ACCUMULATOR>(
            x, dim - 1, at, *mask, result.Element<CppType>(at), accumulator);
      }
      return;
    } else if (!IsLogicalElementTrue(*mask, maskAt)) {
      // scalar MASK=.FALSE.
      accumulator.Reinitialize();
      for (auto n{result.Elements()}; n-- > 0; result.IncrementSubscripts(at)) {
        accumulator.GetResult(result.Element<CppType>(at));
      }
      return;
    }
  }
  // No MASK= or scalar MASK=.TRUE.
  for (auto n{result.Elements()}; n-- > 0; result.IncrementSubscripts(at)) {
    accumulator.Reinitialize();
    ReduceDimToScalar<CppType, ACCUMULATOR>(
        x, dim - 1, at, result.Element<CppType>(at), accumulator);
  }
}

template <template <typename> class ACCUM>
struct PartialIntegerReductionHelper {
  template <int KIND> struct Functor {
    static constexpr int Intermediate{
        std::max(KIND, 4)}; // use at least "int" for intermediate results
    void operator()(Descriptor &result, const Descriptor &x, int dim,
        const Descriptor *mask, Terminator &terminator,
        const char *intrinsic) const {
      using Accumulator =
          ACCUM<CppTypeFor<TypeCategory::Integer, Intermediate>>;
      Accumulator accumulator{x};
      PartialReduction<Accumulator, TypeCategory::Integer, KIND>(
          result, x, dim, mask, terminator, intrinsic, accumulator);
    }
  };
};

template <template <typename> class INTEGER_ACCUM>
inline void PartialIntegerReduction(Descriptor &result, const Descriptor &x,
    int dim, int kind, const Descriptor *mask, const char *intrinsic,
    Terminator &terminator) {
  ApplyIntegerKind<
      PartialIntegerReductionHelper<INTEGER_ACCUM>::template Functor, void>(
      kind, terminator, result, x, dim, mask, terminator, intrinsic);
}

template <TypeCategory CAT, template <typename> class ACCUM>
struct PartialFloatingReductionHelper {
  template <int KIND> struct Functor {
    static constexpr int Intermediate{
        std::max(KIND, 8)}; // use at least "double" for intermediate results
    void operator()(Descriptor &result, const Descriptor &x, int dim,
        const Descriptor *mask, Terminator &terminator,
        const char *intrinsic) const {
      using Accumulator = ACCUM<CppTypeFor<TypeCategory::Real, Intermediate>>;
      Accumulator accumulator{x};
      PartialReduction<Accumulator, CAT, KIND>(
          result, x, dim, mask, terminator, intrinsic, accumulator);
    }
  };
};

template <template <typename> class INTEGER_ACCUM,
    template <typename> class REAL_ACCUM,
    template <typename> class COMPLEX_ACCUM>
inline void TypedPartialNumericReduction(Descriptor &result,
    const Descriptor &x, int dim, const char *source, int line,
    const Descriptor *mask, const char *intrinsic) {
  Terminator terminator{source, line};
  auto catKind{x.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, catKind.has_value());
  switch (catKind->first) {
  case TypeCategory::Integer:
    PartialIntegerReduction<INTEGER_ACCUM>(
        result, x, dim, catKind->second, mask, intrinsic, terminator);
    break;
  case TypeCategory::Real:
    ApplyFloatingPointKind<PartialFloatingReductionHelper<TypeCategory::Real,
                               REAL_ACCUM>::template Functor,
        void>(catKind->second, terminator, result, x, dim, mask, terminator,
        intrinsic);
    break;
  case TypeCategory::Complex:
    ApplyFloatingPointKind<PartialFloatingReductionHelper<TypeCategory::Complex,
                               COMPLEX_ACCUM>::template Functor,
        void>(catKind->second, terminator, result, x, dim, mask, terminator,
        intrinsic);
    break;
  default:
    terminator.Crash("%s: invalid type code %d", intrinsic, x.type().raw());
  }
}

// SUM()

template <typename INTERMEDIATE> class IntegerSumAccumulator {
public:
  explicit IntegerSumAccumulator(const Descriptor &array) : array_{array} {}
  void Reinitialize() { sum_ = 0; }
  template <typename A> void GetResult(A *p, int /*zeroBasedDim*/ = -1) const {
    *p = static_cast<A>(sum_);
  }
  template <typename A> bool AccumulateAt(const SubscriptValue at[]) {
    sum_ += *array_.Element<A>(at);
    return true;
  }

private:
  const Descriptor &array_;
  INTERMEDIATE sum_{0};
};

template <typename INTERMEDIATE> class RealSumAccumulator {
public:
  explicit RealSumAccumulator(const Descriptor &array) : array_{array} {}
  void Reinitialize() { positives_ = negatives_ = inOrder_ = 0; }
  template <typename A> A Result() const {
    auto sum{static_cast<A>(positives_ + negatives_)};
    return std::isfinite(sum) ? sum : static_cast<A>(inOrder_);
  }
  template <typename A> void GetResult(A *p, int /*zeroBasedDim*/ = -1) const {
    *p = Result<A>();
  }
  template <typename A> bool Accumulate(A x) {
    // Accumulate the nonnegative and negative elements independently
    // to reduce cancellation; also record an in-order sum for use
    // in case of overflow.
    if (x >= 0) {
      positives_ += x;
    } else {
      negatives_ += x;
    }
    inOrder_ += x;
    return true;
  }
  template <typename A> bool AccumulateAt(const SubscriptValue at[]) {
    return Accumulate(*array_.Element<A>(at));
  }

private:
  const Descriptor &array_;
  INTERMEDIATE positives_{0.0}, negatives_{0.0}, inOrder_{0.0};
};

template <typename PART> class ComplexSumAccumulator {
public:
  explicit ComplexSumAccumulator(const Descriptor &array) : array_{array} {}
  void Reinitialize() {
    reals_.Reinitialize();
    imaginaries_.Reinitialize();
  }
  template <typename A> void GetResult(A *p, int /*zeroBasedDim*/ = -1) const {
    using ResultPart = typename A::value_type;
    *p = {reals_.template Result<ResultPart>(),
        imaginaries_.template Result<ResultPart>()};
  }
  template <typename A> bool Accumulate(const A &z) {
    reals_.Accumulate(z.real());
    imaginaries_.Accumulate(z.imag());
    return true;
  }
  template <typename A> bool AccumulateAt(const SubscriptValue at[]) {
    return Accumulate(*array_.Element<A>(at));
  }

private:
  const Descriptor &array_;
  RealSumAccumulator<PART> reals_{array_}, imaginaries_{array_};
};

extern "C" {
CppTypeFor<TypeCategory::Integer, 1> RTNAME(SumInteger1)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 1>(x, source, line, dim, mask,
      IntegerSumAccumulator<CppTypeFor<TypeCategory::Integer, 4>>{x}, "SUM");
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(SumInteger2)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 2>(x, source, line, dim, mask,
      IntegerSumAccumulator<CppTypeFor<TypeCategory::Integer, 4>>{x}, "SUM");
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(SumInteger4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 4>(x, source, line, dim, mask,
      IntegerSumAccumulator<CppTypeFor<TypeCategory::Integer, 4>>{x}, "SUM");
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(SumInteger8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 8>(x, source, line, dim, mask,
      IntegerSumAccumulator<CppTypeFor<TypeCategory::Integer, 8>>{x}, "SUM");
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(SumInteger16)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 16>(x, source, line, dim,
      mask, IntegerSumAccumulator<CppTypeFor<TypeCategory::Integer, 16>>{x},
      "SUM");
}
#endif

// TODO: real/complex(2 & 3)
CppTypeFor<TypeCategory::Real, 4> RTNAME(SumReal4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Real, 4>(
      x, source, line, dim, mask, RealSumAccumulator<double>{x}, "SUM");
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(SumReal8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Real, 8>(
      x, source, line, dim, mask, RealSumAccumulator<double>{x}, "SUM");
}
#if LONG_DOUBLE == 80
CppTypeFor<TypeCategory::Real, 10> RTNAME(SumReal10)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Real, 10>(
      x, source, line, dim, mask, RealSumAccumulator<long double>{x}, "SUM");
}
#elif LONG_DOUBLE == 128
CppTypeFor<TypeCategory::Real, 16> RTNAME(SumReal16)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Real, 16>(
      x, source, line, dim, mask, RealSumAccumulator<long double>{x}, "SUM");
}
#endif

void RTNAME(CppSumComplex4)(CppTypeFor<TypeCategory::Complex, 4> &result,
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  result = GetTotalReduction<TypeCategory::Complex, 4>(
      x, source, line, dim, mask, ComplexSumAccumulator<double>{x}, "SUM");
}
void RTNAME(CppSumComplex8)(CppTypeFor<TypeCategory::Complex, 8> &result,
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  result = GetTotalReduction<TypeCategory::Complex, 8>(
      x, source, line, dim, mask, ComplexSumAccumulator<double>{x}, "SUM");
}
#if LONG_DOUBLE == 80
void RTNAME(CppSumComplex10)(CppTypeFor<TypeCategory::Complex, 10> &result,
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  result = GetTotalReduction<TypeCategory::Complex, 10>(
      x, source, line, dim, mask, ComplexSumAccumulator<long double>{x}, "SUM");
}
#elif LONG_DOUBLE == 128
void RTNAME(CppSumComplex16)(CppTypeFor<TypeCategory::Complex, 16> &result,
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  result = GetTotalReduction<TypeCategory::Complex, 16>(
      x, source, line, dim, mask, ComplexSumAccumulator<long double>{x}, "SUM");
}
#endif

void RTNAME(SumDim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line, const Descriptor *mask) {
  TypedPartialNumericReduction<IntegerSumAccumulator, RealSumAccumulator,
      ComplexSumAccumulator>(result, x, dim, source, line, mask, "SUM");
}
} // extern "C"

// PRODUCT()

template <typename INTERMEDIATE> class NonComplexProductAccumulator {
public:
  explicit NonComplexProductAccumulator(const Descriptor &array)
      : array_{array} {}
  void Reinitialize() { product_ = 1; }
  template <typename A> void GetResult(A *p, int /*zeroBasedDim*/ = -1) const {
    *p = static_cast<A>(product_);
  }
  template <typename A> bool AccumulateAt(const SubscriptValue at[]) {
    product_ *= *array_.Element<A>(at);
    return product_ != 0;
  }

private:
  const Descriptor &array_;
  INTERMEDIATE product_{1};
};

template <typename PART> class ComplexProductAccumulator {
public:
  explicit ComplexProductAccumulator(const Descriptor &array) : array_{array} {}
  void Reinitialize() { product_ = std::complex<PART>{1, 0}; }
  template <typename A> void GetResult(A *p, int /*zeroBasedDim*/ = -1) const {
    using ResultPart = typename A::value_type;
    *p = {static_cast<ResultPart>(product_.real()),
        static_cast<ResultPart>(product_.imag())};
  }
  template <typename A> bool AccumulateAt(const SubscriptValue at[]) {
    product_ *= *array_.Element<A>(at);
    return true;
  }

private:
  const Descriptor &array_;
  std::complex<PART> product_{1, 0};
};

extern "C" {
CppTypeFor<TypeCategory::Integer, 1> RTNAME(ProductInteger1)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 1>(x, source, line, dim, mask,
      NonComplexProductAccumulator<CppTypeFor<TypeCategory::Integer, 4>>{x},
      "PRODUCT");
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(ProductInteger2)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 2>(x, source, line, dim, mask,
      NonComplexProductAccumulator<CppTypeFor<TypeCategory::Integer, 4>>{x},
      "PRODUCT");
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(ProductInteger4)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 4>(x, source, line, dim, mask,
      NonComplexProductAccumulator<CppTypeFor<TypeCategory::Integer, 4>>{x},
      "PRODUCT");
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(ProductInteger8)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 8>(x, source, line, dim, mask,
      NonComplexProductAccumulator<CppTypeFor<TypeCategory::Integer, 8>>{x},
      "PRODUCT");
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(ProductInteger16)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 16>(x, source, line, dim,
      mask,
      NonComplexProductAccumulator<CppTypeFor<TypeCategory::Integer, 16>>{x},
      "PRODUCT");
}
#endif

// TODO: real/complex(2 & 3)
CppTypeFor<TypeCategory::Real, 4> RTNAME(ProductReal4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Real, 4>(x, source, line, dim, mask,
      NonComplexProductAccumulator<CppTypeFor<TypeCategory::Real, 8>>{x},
      "PRODUCT");
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(ProductReal8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Real, 8>(x, source, line, dim, mask,
      NonComplexProductAccumulator<CppTypeFor<TypeCategory::Real, 8>>{x},
      "PRODUCT");
}
#if LONG_DOUBLE == 80
CppTypeFor<TypeCategory::Real, 10> RTNAME(ProductReal10)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Real, 10>(x, source, line, dim, mask,
      NonComplexProductAccumulator<CppTypeFor<TypeCategory::Real, 10>>{x},
      "PRODUCT");
}
#elif LONG_DOUBLE == 128
CppTypeFor<TypeCategory::Real, 16> RTNAME(ProductReal16)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Real, 16>(x, source, line, dim, mask,
      NonComplexProductAccumulator<CppTypeFor<TypeCategory::Real, 16>>{x},
      "PRODUCT");
}
#endif

void RTNAME(CppProductComplex4)(CppTypeFor<TypeCategory::Complex, 4> &result,
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  result = GetTotalReduction<TypeCategory::Complex, 4>(x, source, line, dim,
      mask, ComplexProductAccumulator<CppTypeFor<TypeCategory::Real, 8>>{x},
      "PRODUCT");
}
void RTNAME(CppProductComplex8)(CppTypeFor<TypeCategory::Complex, 8> &result,
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  result = GetTotalReduction<TypeCategory::Complex, 8>(x, source, line, dim,
      mask, ComplexProductAccumulator<CppTypeFor<TypeCategory::Real, 8>>{x},
      "PRODUCT");
}
#if LONG_DOUBLE == 80
void RTNAME(CppProductComplex10)(CppTypeFor<TypeCategory::Complex, 10> &result,
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  result = GetTotalReduction<TypeCategory::Complex, 10>(x, source, line, dim,
      mask, ComplexProductAccumulator<CppTypeFor<TypeCategory::Real, 10>>{x},
      "PRODUCT");
}
#elif LONG_DOUBLE == 128
void RTNAME(CppProductComplex16)(CppTypeFor<TypeCategory::Complex, 16> &result,
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  result = GetTotalReduction<TypeCategory::Complex, 16>(x, source, line, dim,
      mask, ComplexProductAccumulator<CppTypeFor<TypeCategory::Real, 16>>{x},
      "PRODUCT");
}
#endif

void RTNAME(ProductDim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line, const Descriptor *mask) {
  TypedPartialNumericReduction<NonComplexProductAccumulator,
      NonComplexProductAccumulator, ComplexProductAccumulator>(
      result, x, dim, source, line, mask, "PRODUCT");
}
} // extern "C"

// IPARITY()

template <typename INTERMEDIATE> class IntegerXorAccumulator {
public:
  explicit IntegerXorAccumulator(const Descriptor &array) : array_{array} {}
  void Reinitialize() { xor_ = 0; }
  template <typename A> void GetResult(A *p, int /*zeroBasedDim*/ = -1) const {
    *p = static_cast<A>(xor_);
  }
  template <typename A> bool AccumulateAt(const SubscriptValue at[]) {
    xor_ ^= *array_.Element<A>(at);
    return true;
  }

private:
  const Descriptor &array_;
  INTERMEDIATE xor_{0};
};

extern "C" {
CppTypeFor<TypeCategory::Integer, 1> RTNAME(IParity1)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 1>(x, source, line, dim, mask,
      IntegerXorAccumulator<CppTypeFor<TypeCategory::Integer, 4>>{x},
      "IPARITY");
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(IParity2)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 2>(x, source, line, dim, mask,
      IntegerXorAccumulator<CppTypeFor<TypeCategory::Integer, 4>>{x},
      "IPARITY");
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(IParity4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 4>(x, source, line, dim, mask,
      IntegerXorAccumulator<CppTypeFor<TypeCategory::Integer, 4>>{x},
      "IPARITY");
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(IParity8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 8>(x, source, line, dim, mask,
      IntegerXorAccumulator<CppTypeFor<TypeCategory::Integer, 8>>{x},
      "IPARITY");
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(IParity16)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 16>(x, source, line, dim,
      mask, IntegerXorAccumulator<CppTypeFor<TypeCategory::Integer, 16>>{x},
      "IPARITY");
}
#endif
void RTNAME(IParityDim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line, const Descriptor *mask) {
  Terminator terminator{source, line};
  auto catKind{x.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator,
      catKind.has_value() && catKind->first == TypeCategory::Integer);
  PartialIntegerReduction<IntegerXorAccumulator>(
      result, x, dim, catKind->second, mask, "IPARITY", terminator);
}
}

// MAXLOC & MINLOC

template <typename T, bool IS_MAX, bool BACK> struct NumericCompare {
  using Type = T;
  explicit NumericCompare(std::size_t /*elemLen; ignored*/) {}
  bool operator()(const T &value, const T &previous) const {
    if (value == previous) {
      return BACK;
    } else if constexpr (IS_MAX) {
      return value > previous;
    } else {
      return value < previous;
    }
  }
};

template <typename T, bool IS_MAX, bool BACK> class CharacterCompare {
public:
  using Type = T;
  explicit CharacterCompare(std::size_t elemLen)
      : chars_{elemLen / sizeof(T)} {}
  bool operator()(const T &value, const T &previous) const {
    int cmp{CharacterScalarCompare<T>(&value, &previous, chars_, chars_)};
    if (cmp == 0) {
      return BACK;
    } else if constexpr (IS_MAX) {
      return cmp > 0;
    } else {
      return cmp < 0;
    }
  }

private:
  std::size_t chars_;
};

template <typename COMPARE> class ExtremumLocAccumulator {
public:
  using Type = typename COMPARE::Type;
  ExtremumLocAccumulator(const Descriptor &array, std::size_t chars = 0)
      : array_{array}, argRank_{array.rank()}, compare_{array.ElementBytes()} {
    Reinitialize();
  }
  void Reinitialize() {
    // per standard: result indices are all zero if no data
    for (int j{0}; j < argRank_; ++j) {
      extremumLoc_[j] = 0;
    }
    previous_ = nullptr;
  }
  int argRank() const { return argRank_; }
  template <typename A> void GetResult(A *p, int zeroBasedDim = -1) {
    if (zeroBasedDim >= 0) {
      *p = extremumLoc_[zeroBasedDim];
    } else {
      for (int j{0}; j < argRank_; ++j) {
        p[j] = extremumLoc_[j];
      }
    }
  }
  template <typename IGNORED> bool AccumulateAt(const SubscriptValue at[]) {
    const auto &value{*array_.Element<Type>(at)};
    if (!previous_ || compare_(value, *previous_)) {
      previous_ = &value;
      for (int j{0}; j < argRank_; ++j) {
        extremumLoc_[j] = at[j];
      }
    }
    return true;
  }

private:
  const Descriptor &array_;
  int argRank_;
  SubscriptValue extremumLoc_[maxRank];
  const Type *previous_{nullptr};
  COMPARE compare_;
};

template <typename ACCUMULATOR> struct LocationResultHelper {
  template <int KIND> struct Functor {
    void operator()(ACCUMULATOR &accumulator, const Descriptor &result) const {
      accumulator.GetResult(
          result.OffsetElement<CppTypeFor<TypeCategory::Integer, KIND>>());
    }
  };
};

template <typename ACCUMULATOR, typename CPPTYPE>
static void LocationHelper(const char *intrinsic, Descriptor &result,
    const Descriptor &x, int kind, const Descriptor *mask,
    Terminator &terminator) {
  ACCUMULATOR accumulator{x};
  DoTotalReduction<CPPTYPE>(x, 0, mask, accumulator, intrinsic, terminator);
  ApplyIntegerKind<LocationResultHelper<ACCUMULATOR>::template Functor, void>(
      kind, terminator, accumulator, result);
}

template <TypeCategory CAT, int KIND, bool IS_MAX,
    template <typename, bool, bool> class COMPARE>
inline void DoMaxOrMinLoc(const char *intrinsic, Descriptor &result,
    const Descriptor &x, int kind, const char *source, int line,
    const Descriptor *mask, bool back) {
  using CppType = CppTypeFor<CAT, KIND>;
  Terminator terminator{source, line};
  if (back) {
    LocationHelper<ExtremumLocAccumulator<COMPARE<CppType, IS_MAX, true>>,
        CppType>(intrinsic, result, x, kind, mask, terminator);
  } else {
    LocationHelper<ExtremumLocAccumulator<COMPARE<CppType, IS_MAX, false>>,
        CppType>(intrinsic, result, x, kind, mask, terminator);
  }
}

template <TypeCategory CAT, bool IS_MAX> struct TypedMaxOrMinLocHelper {
  template <int KIND> struct Functor {
    void operator()(const char *intrinsic, Descriptor &result,
        const Descriptor &x, int kind, const char *source, int line,
        const Descriptor *mask, bool back) const {
      DoMaxOrMinLoc<TypeCategory::Integer, KIND, IS_MAX, NumericCompare>(
          intrinsic, result, x, kind, source, line, mask, back);
    }
  };
};

template <bool IS_MAX>
inline void TypedMaxOrMinLoc(const char *intrinsic, Descriptor &result,
    const Descriptor &x, int kind, const char *source, int line,
    const Descriptor *mask, bool back) {
  int rank{x.rank()};
  SubscriptValue extent[1]{rank};
  result.Establish(TypeCategory::Integer, kind, nullptr, 1, extent,
      CFI_attribute_allocatable);
  result.GetDimension(0).SetBounds(1, extent[0]);
  Terminator terminator{source, line};
  if (int stat{result.Allocate()}) {
    terminator.Crash(
        "%s: could not allocate memory for result; STAT=%d", intrinsic, stat);
  }
  CheckIntegerKind(terminator, kind, intrinsic);
  auto catKind{x.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, catKind.has_value());
  switch (catKind->first) {
  case TypeCategory::Integer:
    ApplyIntegerKind<
        TypedMaxOrMinLocHelper<TypeCategory::Integer, IS_MAX>::template Functor,
        void>(catKind->second, terminator, intrinsic, result, x, kind, source,
        line, mask, back);
    break;
  case TypeCategory::Real:
    ApplyFloatingPointKind<
        TypedMaxOrMinLocHelper<TypeCategory::Real, IS_MAX>::template Functor,
        void>(catKind->second, terminator, intrinsic, result, x, kind, source,
        line, mask, back);
    break;
  case TypeCategory::Character:
    ApplyCharacterKind<TypedMaxOrMinLocHelper<TypeCategory::Character,
                           IS_MAX>::template Functor,
        void>(catKind->second, terminator, intrinsic, result, x, kind, source,
        line, mask, back);
    break;
  default:
    terminator.Crash(
        "%s: Bad data type code (%d) for array", intrinsic, x.type().raw());
  }
}

extern "C" {
void RTNAME(Maxloc)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TypedMaxOrMinLoc<true>("MAXLOC", result, x, kind, source, line, mask, back);
}
void RTNAME(Minloc)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TypedMaxOrMinLoc<false>("MINLOC", result, x, kind, source, line, mask, back);
}
} // extern "C"

// MAXLOC/MINLOC with DIM=

template <typename ACCUMULATOR> struct PartialLocationHelper {
  template <int KIND> struct Functor {
    void operator()(Descriptor &result, const Descriptor &x, int dim,
        const Descriptor *mask, Terminator &terminator, const char *intrinsic,
        ACCUMULATOR &accumulator) const {
      PartialReduction<ACCUMULATOR, TypeCategory::Integer, KIND>(
          result, x, dim, mask, terminator, intrinsic, accumulator);
    }
  };
};

template <TypeCategory CAT, int KIND, bool IS_MAX,
    template <typename, bool, bool> class COMPARE, bool BACK>
static void DoPartialMaxOrMinLocDirection(const char *intrinsic,
    Descriptor &result, const Descriptor &x, int kind, int dim,
    const Descriptor *mask, Terminator &terminator) {
  using CppType = CppTypeFor<CAT, KIND>;
  using Accumulator = ExtremumLocAccumulator<COMPARE<CppType, IS_MAX, BACK>>;
  Accumulator accumulator{x};
  ApplyIntegerKind<PartialLocationHelper<Accumulator>::template Functor, void>(
      kind, terminator, result, x, dim, mask, terminator, intrinsic,
      accumulator);
}

template <TypeCategory CAT, int KIND, bool IS_MAX,
    template <typename, bool, bool> class COMPARE>
inline void DoPartialMaxOrMinLoc(const char *intrinsic, Descriptor &result,
    const Descriptor &x, int kind, int dim, const Descriptor *mask, bool back,
    Terminator &terminator) {
  if (back) {
    DoPartialMaxOrMinLocDirection<CAT, KIND, IS_MAX, COMPARE, true>(
        intrinsic, result, x, kind, dim, mask, terminator);
  } else {
    DoPartialMaxOrMinLocDirection<CAT, KIND, IS_MAX, COMPARE, false>(
        intrinsic, result, x, kind, dim, mask, terminator);
  }
}

template <TypeCategory CAT, bool IS_MAX,
    template <typename, bool, bool> class COMPARE>
struct DoPartialMaxOrMinLocHelper {
  template <int KIND> struct Functor {
    void operator()(const char *intrinsic, Descriptor &result,
        const Descriptor &x, int kind, int dim, const Descriptor *mask,
        bool back, Terminator &terminator) const {
      DoPartialMaxOrMinLoc<CAT, KIND, IS_MAX, COMPARE>(
          intrinsic, result, x, kind, dim, mask, back, terminator);
    }
  };
};

template <bool IS_MAX>
inline void TypedPartialMaxOrMinLoc(const char *intrinsic, Descriptor &result,
    const Descriptor &x, int kind, int dim, const char *source, int line,
    const Descriptor *mask, bool back) {
  Terminator terminator{source, line};
  CheckIntegerKind(terminator, kind, intrinsic);
  auto catKind{x.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, catKind.has_value());
  switch (catKind->first) {
  case TypeCategory::Integer:
    ApplyIntegerKind<DoPartialMaxOrMinLocHelper<TypeCategory::Integer, IS_MAX,
                         NumericCompare>::template Functor,
        void>(catKind->second, terminator, intrinsic, result, x, kind, dim,
        mask, back, terminator);
    break;
  case TypeCategory::Real:
    ApplyFloatingPointKind<DoPartialMaxOrMinLocHelper<TypeCategory::Real,
                               IS_MAX, NumericCompare>::template Functor,
        void>(catKind->second, terminator, intrinsic, result, x, kind, dim,
        mask, back, terminator);
    break;
  case TypeCategory::Character:
    ApplyCharacterKind<DoPartialMaxOrMinLocHelper<TypeCategory::Character,
                           IS_MAX, CharacterCompare>::template Functor,
        void>(catKind->second, terminator, intrinsic, result, x, kind, dim,
        mask, back, terminator);
    break;
  default:
    terminator.Crash(
        "%s: Bad data type code (%d) for array", intrinsic, x.type().raw());
  }
}

extern "C" {
void RTNAME(MaxlocDim)(Descriptor &result, const Descriptor &x, int kind,
    int dim, const char *source, int line, const Descriptor *mask, bool back) {
  TypedPartialMaxOrMinLoc<true>(
      "MAXLOC", result, x, kind, dim, source, line, mask, back);
}
void RTNAME(MinlocDim)(Descriptor &result, const Descriptor &x, int kind,
    int dim, const char *source, int line, const Descriptor *mask, bool back) {
  TypedPartialMaxOrMinLoc<false>(
      "MINLOC", result, x, kind, dim, source, line, mask, back);
}
} // extern "C"

// FINDLOC

template <TypeCategory CAT1, int KIND1, TypeCategory CAT2, int KIND2>
struct Equality {
  using Type1 = CppTypeFor<CAT1, KIND1>;
  using Type2 = CppTypeFor<CAT2, KIND2>;
  bool operator()(const Descriptor &array, const SubscriptValue at[],
      const Descriptor &target) const {
    return *array.Element<Type1>(at) == *target.OffsetElement<Type2>();
  }
};

template <int KIND1, int KIND2>
struct Equality<TypeCategory::Complex, KIND1, TypeCategory::Complex, KIND2> {
  using Type1 = CppTypeFor<TypeCategory::Complex, KIND1>;
  using Type2 = CppTypeFor<TypeCategory::Complex, KIND2>;
  bool operator()(const Descriptor &array, const SubscriptValue at[],
      const Descriptor &target) const {
    const Type1 &xz{*array.Element<Type1>(at)};
    const Type2 &tz{*target.OffsetElement<Type2>()};
    return xz.real() == tz.real() && xz.imag() == tz.imag();
  }
};

template <int KIND1, TypeCategory CAT2, int KIND2>
struct Equality<TypeCategory::Complex, KIND1, CAT2, KIND2> {
  using Type1 = CppTypeFor<TypeCategory::Complex, KIND1>;
  using Type2 = CppTypeFor<CAT2, KIND2>;
  bool operator()(const Descriptor &array, const SubscriptValue at[],
      const Descriptor &target) const {
    const Type1 &z{*array.Element<Type1>(at)};
    return z.imag() == 0 && z.real() == *target.OffsetElement<Type2>();
  }
};

template <TypeCategory CAT1, int KIND1, int KIND2>
struct Equality<CAT1, KIND1, TypeCategory::Complex, KIND2> {
  using Type1 = CppTypeFor<CAT1, KIND1>;
  using Type2 = CppTypeFor<TypeCategory::Complex, KIND2>;
  bool operator()(const Descriptor &array, const SubscriptValue at[],
      const Descriptor &target) const {
    const Type2 &z{*target.OffsetElement<Type2>()};
    return *array.Element<Type1>(at) == z.real() && z.imag() == 0;
  }
};

template <int KIND> struct CharacterEquality {
  using Type = CppTypeFor<TypeCategory::Character, KIND>;
  bool operator()(const Descriptor &array, const SubscriptValue at[],
      const Descriptor &target) const {
    return CharacterScalarCompare<Type>(array.Element<Type>(at),
               target.OffsetElement<Type>(),
               array.ElementBytes() / static_cast<unsigned>(KIND),
               target.ElementBytes() / static_cast<unsigned>(KIND)) == 0;
  }
};

struct LogicalEquivalence {
  bool operator()(const Descriptor &array, const SubscriptValue at[],
      const Descriptor &target) const {
    return IsLogicalElementTrue(array, at) ==
        IsLogicalElementTrue(target, at /*ignored*/);
  }
};

template <typename EQUALITY> class LocationAccumulator {
public:
  LocationAccumulator(
      const Descriptor &array, const Descriptor &target, bool back)
      : array_{array}, target_{target}, back_{back} {
    Reinitialize();
  }
  void Reinitialize() {
    // per standard: result indices are all zero if no data
    for (int j{0}; j < rank_; ++j) {
      location_[j] = 0;
    }
  }
  template <typename A> void GetResult(A *p, int zeroBasedDim = -1) {
    if (zeroBasedDim >= 0) {
      *p = location_[zeroBasedDim];
    } else {
      for (int j{0}; j < rank_; ++j) {
        p[j] = location_[j];
      }
    }
  }
  template <typename IGNORED> bool AccumulateAt(const SubscriptValue at[]) {
    if (equality_(array_, at, target_)) {
      for (int j{0}; j < rank_; ++j) {
        location_[j] = at[j];
      }
      return back_;
    } else {
      return true;
    }
  }

private:
  const Descriptor &array_;
  const Descriptor &target_;
  const bool back_{false};
  const int rank_{array_.rank()};
  SubscriptValue location_[maxRank];
  const EQUALITY equality_{};
};

template <TypeCategory XCAT, int XKIND, TypeCategory TARGET_CAT>
struct TotalNumericFindlocHelper {
  template <int TARGET_KIND> struct Functor {
    void operator()(Descriptor &result, const Descriptor &x,
        const Descriptor &target, int kind, int dim, const Descriptor *mask,
        bool back, Terminator &terminator) const {
      using Eq = Equality<XCAT, XKIND, TARGET_CAT, TARGET_KIND>;
      using Accumulator = LocationAccumulator<Eq>;
      Accumulator accumulator{x, target, back};
      DoTotalReduction<void>(x, dim, mask, accumulator, "FINDLOC", terminator);
      ApplyIntegerKind<LocationResultHelper<Accumulator>::template Functor,
          void>(kind, terminator, accumulator, result);
    }
  };
};

template <TypeCategory CAT,
    template <TypeCategory XCAT, int XKIND, TypeCategory TARGET_CAT>
    class HELPER>
struct NumericFindlocHelper {
  template <int KIND> struct Functor {
    void operator()(TypeCategory targetCat, int targetKind, Descriptor &result,
        const Descriptor &x, const Descriptor &target, int kind, int dim,
        const Descriptor *mask, bool back, Terminator &terminator) const {
      switch (targetCat) {
      case TypeCategory::Integer:
        ApplyIntegerKind<
            HELPER<CAT, KIND, TypeCategory::Integer>::template Functor, void>(
            targetKind, terminator, result, x, target, kind, dim, mask, back,
            terminator);
        break;
      case TypeCategory::Real:
        ApplyFloatingPointKind<
            HELPER<CAT, KIND, TypeCategory::Real>::template Functor, void>(
            targetKind, terminator, result, x, target, kind, dim, mask, back,
            terminator);
        break;
      case TypeCategory::Complex:
        ApplyFloatingPointKind<
            HELPER<CAT, KIND, TypeCategory::Complex>::template Functor, void>(
            targetKind, terminator, result, x, target, kind, dim, mask, back,
            terminator);
        break;
      default:
        terminator.Crash(
            "FINDLOC: bad target category %d for array category %d",
            static_cast<int>(targetCat), static_cast<int>(CAT));
      }
    }
  };
};

template <int KIND> struct CharacterFindlocHelper {
  void operator()(Descriptor &result, const Descriptor &x,
      const Descriptor &target, int kind, const Descriptor *mask, bool back,
      Terminator &terminator) {
    using Accumulator = LocationAccumulator<CharacterEquality<KIND>>;
    Accumulator accumulator{x, target, back};
    DoTotalReduction<void>(x, 0, mask, accumulator, "FINDLOC", terminator);
    ApplyIntegerKind<LocationResultHelper<Accumulator>::template Functor, void>(
        kind, terminator, accumulator, result);
  }
};

static void LogicalFindlocHelper(Descriptor &result, const Descriptor &x,
    const Descriptor &target, int kind, const Descriptor *mask, bool back,
    Terminator &terminator) {
  using Accumulator = LocationAccumulator<LogicalEquivalence>;
  Accumulator accumulator{x, target, back};
  DoTotalReduction<void>(x, 0, mask, accumulator, "FINDLOC", terminator);
  ApplyIntegerKind<LocationResultHelper<Accumulator>::template Functor, void>(
      kind, terminator, accumulator, result);
}

extern "C" {
void RTNAME(Findloc)(Descriptor &result, const Descriptor &x,
    const Descriptor &target, int kind, const char *source, int line,
    const Descriptor *mask, bool back) {
  int rank{x.rank()};
  SubscriptValue extent[1]{rank};
  result.Establish(TypeCategory::Integer, kind, nullptr, 1, extent,
      CFI_attribute_allocatable);
  result.GetDimension(0).SetBounds(1, extent[0]);
  Terminator terminator{source, line};
  if (int stat{result.Allocate()}) {
    terminator.Crash(
        "FINDLOC: could not allocate memory for result; STAT=%d", stat);
  }
  CheckIntegerKind(terminator, kind, "FINDLOC");
  auto xType{x.type().GetCategoryAndKind()};
  auto targetType{target.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, xType.has_value() && targetType.has_value());
  switch (xType->first) {
  case TypeCategory::Integer:
    ApplyIntegerKind<NumericFindlocHelper<TypeCategory::Integer,
                         TotalNumericFindlocHelper>::template Functor,
        void>(xType->second, terminator, targetType->first, targetType->second,
        result, x, target, kind, 0, mask, back, terminator);
    break;
  case TypeCategory::Real:
    ApplyFloatingPointKind<NumericFindlocHelper<TypeCategory::Real,
                               TotalNumericFindlocHelper>::template Functor,
        void>(xType->second, terminator, targetType->first, targetType->second,
        result, x, target, kind, 0, mask, back, terminator);
    break;
  case TypeCategory::Complex:
    ApplyFloatingPointKind<NumericFindlocHelper<TypeCategory::Complex,
                               TotalNumericFindlocHelper>::template Functor,
        void>(xType->second, terminator, targetType->first, targetType->second,
        result, x, target, kind, 0, mask, back, terminator);
    break;
  case TypeCategory::Character:
    RUNTIME_CHECK(terminator,
        targetType->first == TypeCategory::Character &&
            targetType->second == xType->second);
    ApplyCharacterKind<CharacterFindlocHelper, void>(xType->second, terminator,
        result, x, target, kind, mask, back, terminator);
    break;
  case TypeCategory::Logical:
    RUNTIME_CHECK(terminator, targetType->first == TypeCategory::Logical);
    LogicalFindlocHelper(result, x, target, kind, mask, back, terminator);
    break;
  default:
    terminator.Crash(
        "FINDLOC: Bad data type code (%d) for array", x.type().raw());
  }
}
} // extern "C"

// FINDLOC with DIM=

template <TypeCategory XCAT, int XKIND, TypeCategory TARGET_CAT>
struct PartialNumericFindlocHelper {
  template <int TARGET_KIND> struct Functor {
    void operator()(Descriptor &result, const Descriptor &x,
        const Descriptor &target, int kind, int dim, const Descriptor *mask,
        bool back, Terminator &terminator) const {
      using Eq = Equality<XCAT, XKIND, TARGET_CAT, TARGET_KIND>;
      using Accumulator = LocationAccumulator<Eq>;
      Accumulator accumulator{x, target, back};
      ApplyIntegerKind<PartialLocationHelper<Accumulator>::template Functor,
          void>(kind, terminator, result, x, dim, mask, terminator, "FINDLOC",
          accumulator);
    }
  };
};

template <int KIND> struct PartialCharacterFindlocHelper {
  void operator()(Descriptor &result, const Descriptor &x,
      const Descriptor &target, int kind, int dim, const Descriptor *mask,
      bool back, Terminator &terminator) {
    using Accumulator = LocationAccumulator<CharacterEquality<KIND>>;
    Accumulator accumulator{x, target, back};
    ApplyIntegerKind<PartialLocationHelper<Accumulator>::template Functor,
        void>(kind, terminator, result, x, dim, mask, terminator, "FINDLOC",
        accumulator);
  }
};

static void PartialLogicalFindlocHelper(Descriptor &result, const Descriptor &x,
    const Descriptor &target, int kind, int dim, const Descriptor *mask,
    bool back, Terminator &terminator) {
  using Accumulator = LocationAccumulator<LogicalEquivalence>;
  Accumulator accumulator{x, target, back};
  ApplyIntegerKind<PartialLocationHelper<Accumulator>::template Functor, void>(
      kind, terminator, result, x, dim, mask, terminator, "FINDLOC",
      accumulator);
}

extern "C" {
void RTNAME(FindlocDim)(Descriptor &result, const Descriptor &x,
    const Descriptor &target, int kind, int dim, const char *source, int line,
    const Descriptor *mask, bool back) {
  Terminator terminator{source, line};
  CheckIntegerKind(terminator, kind, "FINDLOC");
  auto xType{x.type().GetCategoryAndKind()};
  auto targetType{target.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, xType.has_value() && targetType.has_value());
  switch (xType->first) {
  case TypeCategory::Integer:
    ApplyIntegerKind<NumericFindlocHelper<TypeCategory::Integer,
                         PartialNumericFindlocHelper>::template Functor,
        void>(xType->second, terminator, targetType->first, targetType->second,
        result, x, target, kind, dim, mask, back, terminator);
    break;
  case TypeCategory::Real:
    ApplyFloatingPointKind<NumericFindlocHelper<TypeCategory::Real,
                               PartialNumericFindlocHelper>::template Functor,
        void>(xType->second, terminator, targetType->first, targetType->second,
        result, x, target, kind, dim, mask, back, terminator);
    break;
  case TypeCategory::Complex:
    ApplyFloatingPointKind<NumericFindlocHelper<TypeCategory::Complex,
                               PartialNumericFindlocHelper>::template Functor,
        void>(xType->second, terminator, targetType->first, targetType->second,
        result, x, target, kind, dim, mask, back, terminator);
    break;
  case TypeCategory::Character:
    RUNTIME_CHECK(terminator,
        targetType->first == TypeCategory::Character &&
            targetType->second == xType->second);
    ApplyCharacterKind<PartialCharacterFindlocHelper, void>(xType->second,
        terminator, result, x, target, kind, dim, mask, back, terminator);
    break;
  case TypeCategory::Logical:
    RUNTIME_CHECK(terminator, targetType->first == TypeCategory::Logical);
    PartialLogicalFindlocHelper(
        result, x, target, kind, dim, mask, back, terminator);
    break;
  default:
    terminator.Crash(
        "FINDLOC: Bad data type code (%d) for array", x.type().raw());
  }
}
} // extern "C"

// MAXVAL and MINVAL

template <TypeCategory CAT, int KIND, bool IS_MAXVAL> struct MaxOrMinIdentity {
  using Type = CppTypeFor<CAT, KIND>;
  static constexpr Type Value() {
    return IS_MAXVAL ? std::numeric_limits<Type>::lowest()
                     : std::numeric_limits<Type>::max();
  }
};

// std::numeric_limits<> may not know int128_t
template <bool IS_MAXVAL>
struct MaxOrMinIdentity<TypeCategory::Integer, 16, IS_MAXVAL> {
  using Type = CppTypeFor<TypeCategory::Integer, 16>;
  static constexpr Type Value() {
    return IS_MAXVAL ? Type{1} << 127 : ~Type{0} >> 1;
  }
};

template <TypeCategory CAT, int KIND, bool IS_MAXVAL>
class NumericExtremumAccumulator {
public:
  using Type = CppTypeFor<CAT, KIND>;
  explicit NumericExtremumAccumulator(const Descriptor &array)
      : array_{array} {}
  void Reinitialize() {
    extremum_ = MaxOrMinIdentity<CAT, KIND, IS_MAXVAL>::Value();
  }
  template <typename A> void GetResult(A *p, int /*zeroBasedDim*/ = -1) const {
    *p = extremum_;
  }
  bool Accumulate(Type x) {
    if constexpr (IS_MAXVAL) {
      if (x > extremum_) {
        extremum_ = x;
      }
    } else if (x < extremum_) {
      extremum_ = x;
    }
    return true;
  }
  template <typename A> bool AccumulateAt(const SubscriptValue at[]) {
    return Accumulate(*array_.Element<A>(at));
  }

private:
  const Descriptor &array_;
  Type extremum_{MaxOrMinIdentity<CAT, KIND, IS_MAXVAL>::Value()};
};

template <TypeCategory CAT, int KIND, bool IS_MAXVAL>
inline CppTypeFor<CAT, KIND> TotalNumericMaxOrMin(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask,
    const char *intrinsic) {
  return GetTotalReduction<CAT, KIND>(x, source, line, dim, mask,
      NumericExtremumAccumulator<CAT, KIND, IS_MAXVAL>{x}, intrinsic);
}

template <TypeCategory CAT, int KIND, bool IS_MAXVAL,
    template <TypeCategory, int, bool> class ACCUMULATOR>
static void DoMaxOrMin(Descriptor &result, const Descriptor &x, int dim,
    const Descriptor *mask, const char *intrinsic, Terminator &terminator) {
  using Type = CppTypeFor<CAT, KIND>;
  if (dim == 0 || x.rank() == 1) {
    // Total reduction
    result.Establish(x.type(), x.ElementBytes(), nullptr, 0, nullptr,
        CFI_attribute_allocatable);
    if (int stat{result.Allocate()}) {
      terminator.Crash(
          "%s: could not allocate memory for result; STAT=%d", intrinsic, stat);
    }
    ACCUMULATOR<CAT, KIND, IS_MAXVAL> accumulator{x};
    DoTotalReduction<Type>(x, dim, mask, accumulator, intrinsic, terminator);
    accumulator.GetResult(result.OffsetElement<Type>());
  } else {
    // Partial reduction
    using Accumulator = ACCUMULATOR<CAT, KIND, IS_MAXVAL>;
    Accumulator accumulator{x};
    PartialReduction<Accumulator, CAT, KIND>(
        result, x, dim, mask, terminator, intrinsic, accumulator);
  }
}

template <TypeCategory CAT, bool IS_MAXVAL> struct MaxOrMinHelper {
  template <int KIND> struct Functor {
    void operator()(Descriptor &result, const Descriptor &x, int dim,
        const Descriptor *mask, const char *intrinsic,
        Terminator &terminator) const {
      DoMaxOrMin<CAT, KIND, IS_MAXVAL, NumericExtremumAccumulator>(
          result, x, dim, mask, intrinsic, terminator);
    }
  };
};

template <bool IS_MAXVAL>
inline void NumericMaxOrMin(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line, const Descriptor *mask,
    const char *intrinsic) {
  Terminator terminator{source, line};
  auto type{x.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, type);
  switch (type->first) {
  case TypeCategory::Integer:
    ApplyIntegerKind<
        MaxOrMinHelper<TypeCategory::Integer, IS_MAXVAL>::template Functor,
        void>(
        type->second, terminator, result, x, dim, mask, intrinsic, terminator);
    break;
  case TypeCategory::Real:
    ApplyFloatingPointKind<
        MaxOrMinHelper<TypeCategory::Real, IS_MAXVAL>::template Functor, void>(
        type->second, terminator, result, x, dim, mask, intrinsic, terminator);
    break;
  default:
    terminator.Crash("%s: bad type code %d", intrinsic, x.type().raw());
  }
}

template <TypeCategory, int KIND, bool IS_MAXVAL>
class CharacterExtremumAccumulator {
public:
  using Type = CppTypeFor<TypeCategory::Character, KIND>;
  explicit CharacterExtremumAccumulator(const Descriptor &array)
      : array_{array}, charLen_{array_.ElementBytes() / KIND} {}
  void Reinitialize() { extremum_ = nullptr; }
  template <typename A> void GetResult(A *p, int /*zeroBasedDim*/ = -1) const {
    static_assert(std::is_same_v<A, Type>);
    if (extremum_) {
      std::memcpy(p, extremum_, charLen_);
    } else {
      // empty array: result is all zero-valued characters
      std::memset(p, 0, charLen_);
    }
  }
  bool Accumulate(const Type *x) {
    if (!extremum_) {
      extremum_ = x;
    } else {
      int cmp{CharacterScalarCompare(x, extremum_, charLen_, charLen_)};
      if (IS_MAXVAL == (cmp > 0)) {
        extremum_ = x;
      }
    }
    return true;
  }
  template <typename A> bool AccumulateAt(const SubscriptValue at[]) {
    return Accumulate(array_.Element<A>(at));
  }

private:
  const Descriptor &array_;
  std::size_t charLen_;
  const Type *extremum_{nullptr};
};

template <bool IS_MAXVAL> struct CharacterMaxOrMinHelper {
  template <int KIND> struct Functor {
    void operator()(Descriptor &result, const Descriptor &x, int dim,
        const Descriptor *mask, const char *intrinsic,
        Terminator &terminator) const {
      DoMaxOrMin<TypeCategory::Character, KIND, IS_MAXVAL,
          CharacterExtremumAccumulator>(
          result, x, dim, mask, intrinsic, terminator);
    }
  };
};

template <bool IS_MAXVAL>
inline void CharacterMaxOrMin(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line, const Descriptor *mask,
    const char *intrinsic) {
  Terminator terminator{source, line};
  auto type{x.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, type && type->first == TypeCategory::Character);
  ApplyCharacterKind<CharacterMaxOrMinHelper<IS_MAXVAL>::template Functor,
      void>(
      type->second, terminator, result, x, dim, mask, intrinsic, terminator);
}

extern "C" {
CppTypeFor<TypeCategory::Integer, 1> RTNAME(MaxvalInteger1)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 1, true>(
      x, source, line, dim, mask, "MAXVAL");
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(MaxvalInteger2)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 2, true>(
      x, source, line, dim, mask, "MAXVAL");
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(MaxvalInteger4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 4, true>(
      x, source, line, dim, mask, "MAXVAL");
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(MaxvalInteger8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 8, true>(
      x, source, line, dim, mask, "MAXVAL");
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(MaxvalInteger16)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 16, true>(
      x, source, line, dim, mask, "MAXVAL");
}
#endif

// TODO: REAL(2 & 3)
CppTypeFor<TypeCategory::Real, 4> RTNAME(MaxvalReal4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 4, true>(
      x, source, line, dim, mask, "MAXVAL");
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(MaxvalReal8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 8, true>(
      x, source, line, dim, mask, "MAXVAL");
}
#if LONG_DOUBLE == 80
CppTypeFor<TypeCategory::Real, 10> RTNAME(MaxvalReal10)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 10, true>(
      x, source, line, dim, mask, "MAXVAL");
}
#elif LONG_DOUBLE == 128
CppTypeFor<TypeCategory::Real, 16> RTNAME(MaxvalReal16)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 16, true>(
      x, source, line, dim, mask, "MAXVAL");
}
#endif

void RTNAME(MaxvalCharacter)(Descriptor &result, const Descriptor &x,
    const char *source, int line, const Descriptor *mask) {
  CharacterMaxOrMin<true>(result, x, 0, source, line, mask, "MAXVAL");
}

CppTypeFor<TypeCategory::Integer, 1> RTNAME(MinvalInteger1)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 1, false>(
      x, source, line, dim, mask, "MINVAL");
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(MinvalInteger2)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 2, false>(
      x, source, line, dim, mask, "MINVAL");
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(MinvalInteger4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 4, false>(
      x, source, line, dim, mask, "MINVAL");
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(MinvalInteger8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 8, false>(
      x, source, line, dim, mask, "MINVAL");
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(MinvalInteger16)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 16, false>(
      x, source, line, dim, mask, "MINVAL");
}
#endif

// TODO: REAL(2 & 3)
CppTypeFor<TypeCategory::Real, 4> RTNAME(MinvalReal4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 4, false>(
      x, source, line, dim, mask, "MINVAL");
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(MinvalReal8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 8, false>(
      x, source, line, dim, mask, "MINVAL");
}
#if LONG_DOUBLE == 80
CppTypeFor<TypeCategory::Real, 10> RTNAME(MinvalReal10)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 10, false>(
      x, source, line, dim, mask, "MINVAL");
}
#elif LONG_DOUBLE == 128
CppTypeFor<TypeCategory::Real, 16> RTNAME(MinvalReal16)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 16, false>(
      x, source, line, dim, mask, "MINVAL");
}
#endif

void RTNAME(MinvalCharacter)(Descriptor &result, const Descriptor &x,
    const char *source, int line, const Descriptor *mask) {
  CharacterMaxOrMin<false>(result, x, 0, source, line, mask, "MINVAL");
}

void RTNAME(MaxvalDim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line, const Descriptor *mask) {
  if (x.type().IsCharacter()) {
    CharacterMaxOrMin<true>(result, x, dim, source, line, mask, "MAXVAL");
  } else {
    NumericMaxOrMin<true>(result, x, dim, source, line, mask, "MAXVAL");
  }
}
void RTNAME(MinvalDim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line, const Descriptor *mask) {
  if (x.type().IsCharacter()) {
    CharacterMaxOrMin<false>(result, x, dim, source, line, mask, "MINVAL");
  } else {
    NumericMaxOrMin<false>(result, x, dim, source, line, mask, "MINVAL");
  }
}

} // extern "C"

// ALL, ANY, COUNT, & PARITY

enum class LogicalReduction { All, Any, Parity };

template <LogicalReduction REDUCTION> class LogicalAccumulator {
public:
  using Type = bool;
  explicit LogicalAccumulator(const Descriptor &array) : array_{array} {}
  void Reinitialize() { result_ = REDUCTION == LogicalReduction::All; }
  bool Result() const { return result_; }
  bool Accumulate(bool x) {
    if constexpr (REDUCTION == LogicalReduction::Parity) {
      result_ = result_ != x;
    } else if (x != (REDUCTION == LogicalReduction::All)) {
      result_ = x;
      return false;
    }
    return true;
  }
  template <typename IGNORED = void>
  bool AccumulateAt(const SubscriptValue at[]) {
    return Accumulate(IsLogicalElementTrue(array_, at));
  }

private:
  const Descriptor &array_;
  bool result_{REDUCTION == LogicalReduction::All};
};

template <typename ACCUMULATOR>
inline auto GetTotalLogicalReduction(const Descriptor &x, const char *source,
    int line, int dim, ACCUMULATOR &&accumulator, const char *intrinsic) ->
    typename ACCUMULATOR::Type {
  Terminator terminator{source, line};
  if (dim < 0 || dim > 1) {
    terminator.Crash("%s: bad DIM=%d", intrinsic, dim);
  }
  SubscriptValue xAt[maxRank];
  x.GetLowerBounds(xAt);
  for (auto elements{x.Elements()}; elements--; x.IncrementSubscripts(xAt)) {
    if (!accumulator.AccumulateAt(xAt)) {
      break; // cut short, result is known
    }
  }
  return accumulator.Result();
}

template <typename ACCUMULATOR>
inline auto ReduceLogicalDimToScalar(const Descriptor &x, int zeroBasedDim,
    SubscriptValue subscripts[]) -> typename ACCUMULATOR::Type {
  ACCUMULATOR accumulator{x};
  SubscriptValue xAt[maxRank];
  GetExpandedSubscripts(xAt, x, zeroBasedDim, subscripts);
  const auto &dim{x.GetDimension(zeroBasedDim)};
  SubscriptValue at{dim.LowerBound()};
  for (auto n{dim.Extent()}; n-- > 0; ++at) {
    xAt[zeroBasedDim] = at;
    if (!accumulator.AccumulateAt(xAt)) {
      break;
    }
  }
  return accumulator.Result();
}

template <LogicalReduction REDUCTION> struct LogicalReduceHelper {
  template <int KIND> struct Functor {
    void operator()(Descriptor &result, const Descriptor &x, int dim,
        Terminator &terminator, const char *intrinsic) const {
      // Standard requires result to have same LOGICAL kind as argument.
      CreatePartialReductionResult(
          result, x, dim, terminator, intrinsic, x.type());
      SubscriptValue at[maxRank];
      result.GetLowerBounds(at);
      INTERNAL_CHECK(at[0] == 1);
      using CppType = CppTypeFor<TypeCategory::Logical, KIND>;
      for (auto n{result.Elements()}; n-- > 0; result.IncrementSubscripts(at)) {
        *result.Element<CppType>(at) =
            ReduceLogicalDimToScalar<LogicalAccumulator<REDUCTION>>(
                x, dim - 1, at);
      }
    }
  };
};

template <LogicalReduction REDUCTION>
inline void DoReduceLogicalDimension(Descriptor &result, const Descriptor &x,
    int dim, Terminator &terminator, const char *intrinsic) {
  auto catKind{x.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, catKind && catKind->first == TypeCategory::Logical);
  ApplyLogicalKind<LogicalReduceHelper<REDUCTION>::template Functor, void>(
      catKind->second, terminator, result, x, dim, terminator, intrinsic);
}

// COUNT

class CountAccumulator {
public:
  using Type = std::int64_t;
  explicit CountAccumulator(const Descriptor &array) : array_{array} {}
  void Reinitialize() { result_ = 0; }
  Type Result() const { return result_; }
  template <typename IGNORED = void>
  bool AccumulateAt(const SubscriptValue at[]) {
    if (IsLogicalElementTrue(array_, at)) {
      ++result_;
    }
    return true;
  }

private:
  const Descriptor &array_;
  Type result_{0};
};

template <int KIND> struct CountDimension {
  void operator()(Descriptor &result, const Descriptor &x, int dim,
      Terminator &terminator) const {
    CreatePartialReductionResult(result, x, dim, terminator, "COUNT",
        TypeCode{TypeCategory::Integer, KIND});
    SubscriptValue at[maxRank];
    result.GetLowerBounds(at);
    INTERNAL_CHECK(at[0] == 1);
    using CppType = CppTypeFor<TypeCategory::Integer, KIND>;
    for (auto n{result.Elements()}; n-- > 0; result.IncrementSubscripts(at)) {
      *result.Element<CppType>(at) =
          ReduceLogicalDimToScalar<CountAccumulator>(x, dim - 1, at);
    }
  }
};

extern "C" {

bool RTNAME(All)(const Descriptor &x, const char *source, int line, int dim) {
  return GetTotalLogicalReduction(x, source, line, dim,
      LogicalAccumulator<LogicalReduction::All>{x}, "ALL");
}
void RTNAME(AllDim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line) {
  Terminator terminator{source, line};
  DoReduceLogicalDimension<LogicalReduction::All>(
      result, x, dim, terminator, "ALL");
}

bool RTNAME(Any)(const Descriptor &x, const char *source, int line, int dim) {
  return GetTotalLogicalReduction(x, source, line, dim,
      LogicalAccumulator<LogicalReduction::Any>{x}, "ANY");
}
void RTNAME(AnyDim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line) {
  Terminator terminator{source, line};
  DoReduceLogicalDimension<LogicalReduction::Any>(
      result, x, dim, terminator, "ANY");
}

std::int64_t RTNAME(Count)(
    const Descriptor &x, const char *source, int line, int dim) {
  return GetTotalLogicalReduction(
      x, source, line, dim, CountAccumulator{x}, "COUNT");
}

void RTNAME(CountDim)(Descriptor &result, const Descriptor &x, int dim,
    int kind, const char *source, int line) {
  Terminator terminator{source, line};
  ApplyIntegerKind<CountDimension, void>(
      kind, terminator, result, x, dim, terminator);
}

bool RTNAME(Parity)(
    const Descriptor &x, const char *source, int line, int dim) {
  return GetTotalLogicalReduction(x, source, line, dim,
      LogicalAccumulator<LogicalReduction::Parity>{x}, "PARITY");
}
void RTNAME(ParityDim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line) {
  Terminator terminator{source, line};
  DoReduceLogicalDimension<LogicalReduction::Parity>(
      result, x, dim, terminator, "PARITY");
}

} // extern "C"
} // namespace Fortran::runtime
