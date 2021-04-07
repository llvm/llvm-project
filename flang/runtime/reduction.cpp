//===-- runtime/reduction.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements ALL, ANY, COUNT, MAXLOC, MAXVAL, MINLOC, MINVAL, PRODUCT, and SUM
// for all required operand types and shapes and (for MAXLOC & MINLOC) kinds of
// results.
//
// * Real and complex SUM reductions attempt to reduce floating-point
//   cancellation on intermediate results by adding up partial sums
//   for positive and negative elements independently.
// * Partial reductions (i.e., those with DIM= arguments that are not
//   required to be 1 by the rank of the argument) return arrays that
//   are dynamically allocated in a caller-supplied descriptor.
// * Total reductions (i.e., no DIM= argument) with MAXLOC & MINLOC
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
// cases of MAXLOC & MINLOC).  These are the cases without DIM= or cases
// where the argument has rank 1 and DIM=, if present, must be 1.
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
    SubscriptValue subscripts[], TYPE *result) {
  ACCUMULATOR accumulator{x};
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
    SubscriptValue subscripts[], const Descriptor &mask, TYPE *result) {
  ACCUMULATOR accumulator{x};
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
    const Descriptor *mask, Terminator &terminator, const char *intrinsic) {
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
        ReduceDimMaskToScalar<CppType, ACCUMULATOR>(
            x, dim - 1, at, *mask, result.Element<CppType>(at));
      }
      return;
    } else if (!IsLogicalElementTrue(*mask, maskAt)) {
      // scalar MASK=.FALSE.
      ACCUMULATOR accumulator{x};
      for (auto n{result.Elements()}; n-- > 0; result.IncrementSubscripts(at)) {
        accumulator.GetResult(result.Element<CppType>(at));
      }
      return;
    }
  }
  // No MASK= or scalar MASK=.TRUE.
  for (auto n{result.Elements()}; n-- > 0; result.IncrementSubscripts(at)) {
    ReduceDimToScalar<CppType, ACCUMULATOR>(
        x, dim - 1, at, result.Element<CppType>(at));
  }
}

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
    switch (catKind->second) {
    case 1:
      PartialReduction<INTEGER_ACCUM<CppTypeFor<TypeCategory::Integer, 4>>,
          TypeCategory::Integer, 1>(
          result, x, dim, mask, terminator, intrinsic);
      return;
    case 2:
      PartialReduction<INTEGER_ACCUM<CppTypeFor<TypeCategory::Integer, 4>>,
          TypeCategory::Integer, 2>(
          result, x, dim, mask, terminator, intrinsic);
      return;
    case 4:
      PartialReduction<INTEGER_ACCUM<CppTypeFor<TypeCategory::Integer, 4>>,
          TypeCategory::Integer, 4>(
          result, x, dim, mask, terminator, intrinsic);
      return;
    case 8:
      PartialReduction<INTEGER_ACCUM<CppTypeFor<TypeCategory::Integer, 8>>,
          TypeCategory::Integer, 8>(
          result, x, dim, mask, terminator, intrinsic);
      return;
#ifdef __SIZEOF_INT128__
    case 16:
      PartialReduction<INTEGER_ACCUM<CppTypeFor<TypeCategory::Integer, 16>>,
          TypeCategory::Integer, 16>(
          result, x, dim, mask, terminator, intrinsic);
      return;
#endif
    }
    break;
  case TypeCategory::Real:
    switch (catKind->second) {
#if 0 // TODO
    case 2:
    case 3:
#endif
    case 4:
      PartialReduction<REAL_ACCUM<CppTypeFor<TypeCategory::Real, 8>>,
          TypeCategory::Real, 4>(result, x, dim, mask, terminator, intrinsic);
      return;
    case 8:
      PartialReduction<REAL_ACCUM<CppTypeFor<TypeCategory::Real, 8>>,
          TypeCategory::Real, 8>(result, x, dim, mask, terminator, intrinsic);
      return;
#if LONG_DOUBLE == 80
    case 10:
      PartialReduction<REAL_ACCUM<CppTypeFor<TypeCategory::Real, 10>>,
          TypeCategory::Real, 10>(result, x, dim, mask, terminator, intrinsic);
      return;
#elif LONG_DOUBLE == 128
    case 16:
      PartialReduction<REAL_ACCUM<CppTypeFor<TypeCategory::Real, 16>>,
          TypeCategory::Real, 16>(result, x, dim, mask, terminator, intrinsic);
      return;
#endif
    }
    break;
  case TypeCategory::Complex:
    switch (catKind->second) {
#if 0 // TODO
    case 2:
    case 3:
#endif
    case 4:
      PartialReduction<COMPLEX_ACCUM<CppTypeFor<TypeCategory::Real, 8>>,
          TypeCategory::Complex, 4>(
          result, x, dim, mask, terminator, intrinsic);
      return;
    case 8:
      PartialReduction<COMPLEX_ACCUM<CppTypeFor<TypeCategory::Real, 8>>,
          TypeCategory::Complex, 8>(
          result, x, dim, mask, terminator, intrinsic);
      return;
#if LONG_DOUBLE == 80
    case 10:
      PartialReduction<COMPLEX_ACCUM<CppTypeFor<TypeCategory::Real, 10>>,
          TypeCategory::Complex, 10>(
          result, x, dim, mask, terminator, intrinsic);
      return;
#elif LONG_DOUBLE == 128
    case 16:
      PartialReduction<COMPLEX_ACCUM<CppTypeFor<TypeCategory::Real, 16>>,
          TypeCategory::Complex, 16>(
          result, x, dim, mask, terminator, intrinsic);
      return;
#endif
    }
    break;
  default:
    break;
  }
  terminator.Crash("%s: invalid type code %d", intrinsic, x.type().raw());
}

// SUM()

template <typename INTERMEDIATE> class IntegerSumAccumulator {
public:
  explicit IntegerSumAccumulator(const Descriptor &array) : array_{array} {}
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

// MAXLOC and MINLOC

template <typename T, bool IS_MAX, bool BACK> struct NumericCompare {
  using Type = T;
  explicit NumericCompare(std::size_t /*elemLen; ignored*/) {}
  bool operator()(const T &value, const T &previous) {
    if (BACK && value == previous) {
      return true;
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
  bool operator()(const T &value, const T &previous) {
    int cmp{CharacterScalarCompare<T>(&value, &previous, chars_, chars_)};
    if (BACK && cmp == 0) {
      return true;
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
    // per standard: result indices are all zero if no data
    for (int j{0}; j < argRank_; ++j) {
      extremumLoc_[j] = 0;
    }
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

template <typename COMPARE, typename CPPTYPE>
static void DoMaxOrMinLocHelper(const char *intrinsic, Descriptor &result,
    const Descriptor &x, int kind, const Descriptor *mask,
    Terminator &terminator) {
  ExtremumLocAccumulator<COMPARE> accumulator{x};
  DoTotalReduction<CPPTYPE>(x, 0, mask, accumulator, intrinsic, terminator);
  switch (kind) {
  case 1:
    accumulator.GetResult(
        result.OffsetElement<CppTypeFor<TypeCategory::Integer, 1>>());
    break;
  case 2:
    accumulator.GetResult(
        result.OffsetElement<CppTypeFor<TypeCategory::Integer, 2>>());
    break;
  case 4:
    accumulator.GetResult(
        result.OffsetElement<CppTypeFor<TypeCategory::Integer, 4>>());
    break;
  case 8:
    accumulator.GetResult(
        result.OffsetElement<CppTypeFor<TypeCategory::Integer, 8>>());
    break;
#ifdef __SIZEOF_INT128__
  case 16:
    accumulator.GetResult(
        result.OffsetElement<CppTypeFor<TypeCategory::Integer, 16>>());
    break;
#endif
  default:
    terminator.Crash("%s: bad KIND=%d", intrinsic, kind);
  }
}

template <TypeCategory CAT, int KIND, bool IS_MAX,
    template <typename, bool, bool> class COMPARE>
inline void DoMaxOrMinLoc(const char *intrinsic, Descriptor &result,
    const Descriptor &x, int kind, const char *source, int line,
    const Descriptor *mask, bool back) {
  using CppType = CppTypeFor<CAT, KIND>;
  Terminator terminator{source, line};
  if (back) {
    DoMaxOrMinLocHelper<COMPARE<CppType, IS_MAX, true>, CppType>(
        intrinsic, result, x, kind, mask, terminator);
  } else {
    DoMaxOrMinLocHelper<COMPARE<CppType, IS_MAX, false>, CppType>(
        intrinsic, result, x, kind, mask, terminator);
  }
}

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
    switch (catKind->second) {
    case 1:
      DoMaxOrMinLoc<TypeCategory::Integer, 1, IS_MAX, NumericCompare>(
          intrinsic, result, x, kind, source, line, mask, back);
      return;
    case 2:
      DoMaxOrMinLoc<TypeCategory::Integer, 2, IS_MAX, NumericCompare>(
          intrinsic, result, x, kind, source, line, mask, back);
      return;
    case 4:
      DoMaxOrMinLoc<TypeCategory::Integer, 4, IS_MAX, NumericCompare>(
          intrinsic, result, x, kind, source, line, mask, back);
      return;
    case 8:
      DoMaxOrMinLoc<TypeCategory::Integer, 8, IS_MAX, NumericCompare>(
          intrinsic, result, x, kind, source, line, mask, back);
      return;
#ifdef __SIZEOF_INT128__
    case 16:
      DoMaxOrMinLoc<TypeCategory::Integer, 16, IS_MAX, NumericCompare>(
          intrinsic, result, x, kind, source, line, mask, back);
      return;
#endif
    }
    break;
  case TypeCategory::Real:
    switch (catKind->second) {
    // TODO: REAL(2 & 3)
    case 4:
      DoMaxOrMinLoc<TypeCategory::Real, 4, IS_MAX, NumericCompare>(
          intrinsic, result, x, kind, source, line, mask, back);
      return;
    case 8:
      DoMaxOrMinLoc<TypeCategory::Real, 8, IS_MAX, NumericCompare>(
          intrinsic, result, x, kind, source, line, mask, back);
      return;
#if LONG_DOUBLE == 80
    case 10:
      DoMaxOrMinLoc<TypeCategory::Real, 10, IS_MAX, NumericCompare>(
          intrinsic, result, x, kind, source, line, mask, back);
      return;
#elif LONG_DOUBLE == 128
    case 16:
      DoMaxOrMinLoc<TypeCategory::Real, 16, IS_MAX, NumericCompare>(
          intrinsic, result, x, kind, source, line, mask, back);
      return;
#endif
    }
    break;
  case TypeCategory::Character:
    switch (catKind->second) {
    case 1:
      DoMaxOrMinLoc<TypeCategory::Character, 1, IS_MAX, CharacterCompare>(
          intrinsic, result, x, kind, source, line, mask, back);
      return;
    case 2:
      DoMaxOrMinLoc<TypeCategory::Character, 2, IS_MAX, CharacterCompare>(
          intrinsic, result, x, kind, source, line, mask, back);
      return;
    case 4:
      DoMaxOrMinLoc<TypeCategory::Character, 4, IS_MAX, CharacterCompare>(
          intrinsic, result, x, kind, source, line, mask, back);
      return;
    }
    break;
  default:
    break;
  }
  terminator.Crash(
      "%s: Bad data type code (%d) for array", intrinsic, x.type().raw());
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

template <TypeCategory CAT, int KIND, bool IS_MAX,
    template <typename, bool, bool> class COMPARE, bool BACK>
static void DoPartialMaxOrMinLocDirection(const char *intrinsic,
    Descriptor &result, const Descriptor &x, int kind, int dim,
    const Descriptor *mask, Terminator &terminator) {
  using CppType = CppTypeFor<CAT, KIND>;
  switch (kind) {
  case 1:
    PartialReduction<ExtremumLocAccumulator<COMPARE<CppType, IS_MAX, BACK>>,
        TypeCategory::Integer, 1>(result, x, dim, mask, terminator, intrinsic);
    break;
  case 2:
    PartialReduction<ExtremumLocAccumulator<COMPARE<CppType, IS_MAX, BACK>>,
        TypeCategory::Integer, 2>(result, x, dim, mask, terminator, intrinsic);
    break;
  case 4:
    PartialReduction<ExtremumLocAccumulator<COMPARE<CppType, IS_MAX, BACK>>,
        TypeCategory::Integer, 4>(result, x, dim, mask, terminator, intrinsic);
    break;
  case 8:
    PartialReduction<ExtremumLocAccumulator<COMPARE<CppType, IS_MAX, BACK>>,
        TypeCategory::Integer, 8>(result, x, dim, mask, terminator, intrinsic);
    break;
#ifdef __SIZEOF_INT128__
  case 16:
    PartialReduction<ExtremumLocAccumulator<COMPARE<CppType, IS_MAX, BACK>>,
        TypeCategory::Integer, 16>(result, x, dim, mask, terminator, intrinsic);
    break;
#endif
  default:
    terminator.Crash("%s: bad KIND=%d", intrinsic, kind);
  }
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
    switch (catKind->second) {
    case 1:
      DoPartialMaxOrMinLoc<TypeCategory::Integer, 1, IS_MAX, NumericCompare>(
          intrinsic, result, x, kind, dim, mask, back, terminator);
      return;
    case 2:
      DoPartialMaxOrMinLoc<TypeCategory::Integer, 2, IS_MAX, NumericCompare>(
          intrinsic, result, x, kind, dim, mask, back, terminator);
      return;
    case 4:
      DoPartialMaxOrMinLoc<TypeCategory::Integer, 4, IS_MAX, NumericCompare>(
          intrinsic, result, x, kind, dim, mask, back, terminator);
      return;
    case 8:
      DoPartialMaxOrMinLoc<TypeCategory::Integer, 8, IS_MAX, NumericCompare>(
          intrinsic, result, x, kind, dim, mask, back, terminator);
      return;
#ifdef __SIZEOF_INT128__
    case 16:
      DoPartialMaxOrMinLoc<TypeCategory::Integer, 16, IS_MAX, NumericCompare>(
          intrinsic, result, x, kind, dim, mask, back, terminator);
      return;
#endif
    }
    break;
  case TypeCategory::Real:
    switch (catKind->second) {
    // TODO: REAL(2 & 3)
    case 4:
      DoPartialMaxOrMinLoc<TypeCategory::Real, 4, IS_MAX, NumericCompare>(
          intrinsic, result, x, kind, dim, mask, back, terminator);
      return;
    case 8:
      DoPartialMaxOrMinLoc<TypeCategory::Real, 8, IS_MAX, NumericCompare>(
          intrinsic, result, x, kind, dim, mask, back, terminator);
      return;
#if LONG_DOUBLE == 80
    case 10:
      DoPartialMaxOrMinLoc<TypeCategory::Real, 10, IS_MAX, NumericCompare>(
          intrinsic, result, x, kind, dim, mask, back, terminator);
      return;
#elif LONG_DOUBLE == 128
    case 16:
      DoPartialMaxOrMinLoc<TypeCategory::Real, 16, IS_MAX, NumericCompare>(
          intrinsic, result, x, kind, dim, mask, back, terminator);
      return;
#endif
    }
    break;
  case TypeCategory::Character:
    switch (catKind->second) {
    case 1:
      DoPartialMaxOrMinLoc<TypeCategory::Character, 1, IS_MAX,
          CharacterCompare>(
          intrinsic, result, x, kind, dim, mask, back, terminator);
      return;
    case 2:
      DoPartialMaxOrMinLoc<TypeCategory::Character, 2, IS_MAX,
          CharacterCompare>(
          intrinsic, result, x, kind, dim, mask, back, terminator);
      return;
    case 4:
      DoPartialMaxOrMinLoc<TypeCategory::Character, 4, IS_MAX,
          CharacterCompare>(
          intrinsic, result, x, kind, dim, mask, back, terminator);
      return;
    }
    break;
  default:
    break;
  }
  terminator.Crash(
      "%s: Bad data type code (%d) for array", intrinsic, x.type().raw());
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
  NumericExtremumAccumulator(const Descriptor &array) : array_{array} {}
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
    PartialReduction<ACCUMULATOR<CAT, KIND, IS_MAXVAL>, CAT, KIND>(
        result, x, dim, mask, terminator, intrinsic);
  }
}

template <bool IS_MAXVAL>
inline void NumericMaxOrMin(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line, const Descriptor *mask,
    const char *intrinsic) {
  Terminator terminator{source, line};
  auto type{x.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, type);
  switch (type->first) {
  case TypeCategory::Integer:
    switch (type->second) {
    case 1:
      DoMaxOrMin<TypeCategory::Integer, 1, IS_MAXVAL,
          NumericExtremumAccumulator>(
          result, x, dim, mask, intrinsic, terminator);
      return;
    case 2:
      DoMaxOrMin<TypeCategory::Integer, 2, IS_MAXVAL,
          NumericExtremumAccumulator>(
          result, x, dim, mask, intrinsic, terminator);
      return;
    case 4:
      DoMaxOrMin<TypeCategory::Integer, 4, IS_MAXVAL,
          NumericExtremumAccumulator>(
          result, x, dim, mask, intrinsic, terminator);
      return;
    case 8:
      DoMaxOrMin<TypeCategory::Integer, 8, IS_MAXVAL,
          NumericExtremumAccumulator>(
          result, x, dim, mask, intrinsic, terminator);
      return;
#ifdef __SIZEOF_INT128__
    case 16:
      DoMaxOrMin<TypeCategory::Integer, 16, IS_MAXVAL,
          NumericExtremumAccumulator>(
          result, x, dim, mask, intrinsic, terminator);
      return;
#endif
    }
    break;
  case TypeCategory::Real:
    switch (type->second) {
    // TODO: REAL(2 & 3)
    case 4:
      DoMaxOrMin<TypeCategory::Real, 4, IS_MAXVAL, NumericExtremumAccumulator>(
          result, x, dim, mask, intrinsic, terminator);
      return;
    case 8:
      DoMaxOrMin<TypeCategory::Real, 8, IS_MAXVAL, NumericExtremumAccumulator>(
          result, x, dim, mask, intrinsic, terminator);
      return;
#if LONG_DOUBLE == 80
    case 10:
      DoMaxOrMin<TypeCategory::Real, 10, IS_MAXVAL, NumericExtremumAccumulator>(
          result, x, dim, mask, intrinsic, terminator);
      return;
#elif LONG_DOUBLE == 128
    case 16:
      DoMaxOrMin<TypeCategory::Real, 16, IS_MAXVAL, NumericExtremumAccumulator>(
          result, x, dim, mask, intrinsic, terminator);
      return;
#endif
    }
    break;
  default:
    break;
  }
  terminator.Crash("%s: bad type code %d", intrinsic, x.type().raw());
}

template <TypeCategory, int KIND, bool IS_MAXVAL>
class CharacterExtremumAccumulator {
public:
  using Type = CppTypeFor<TypeCategory::Character, KIND>;
  CharacterExtremumAccumulator(const Descriptor &array)
      : array_{array}, charLen_{array_.ElementBytes() / KIND} {}
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

template <bool IS_MAXVAL>
inline void CharacterMaxOrMin(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line, const Descriptor *mask,
    const char *intrinsic) {
  Terminator terminator{source, line};
  auto type{x.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, type && type->first == TypeCategory::Character);
  switch (type->second) {
  case 1:
    DoMaxOrMin<TypeCategory::Character, 1, IS_MAXVAL,
        CharacterExtremumAccumulator>(
        result, x, dim, mask, intrinsic, terminator);
    break;
  case 2:
    DoMaxOrMin<TypeCategory::Character, 2, IS_MAXVAL,
        CharacterExtremumAccumulator>(
        result, x, dim, mask, intrinsic, terminator);
    break;
  case 4:
    DoMaxOrMin<TypeCategory::Character, 4, IS_MAXVAL,
        CharacterExtremumAccumulator>(
        result, x, dim, mask, intrinsic, terminator);
    break;
  default:
    terminator.Crash("%s: bad character kind %d", intrinsic, type->second);
  }
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

// ALL, ANY, & COUNT

template <bool IS_ALL> class LogicalAccumulator {
public:
  using Type = bool;
  explicit LogicalAccumulator(const Descriptor &array) : array_{array} {}
  bool Result() const { return result_; }
  bool Accumulate(bool x) {
    if (x == IS_ALL) {
      return true;
    } else {
      result_ = x;
      return false;
    }
  }
  template <typename IGNORED = void>
  bool AccumulateAt(const SubscriptValue at[]) {
    return Accumulate(IsLogicalElementTrue(array_, at));
  }

private:
  const Descriptor &array_;
  bool result_{IS_ALL};
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

template <bool IS_ALL, int KIND>
inline void ReduceLogicalDimension(Descriptor &result, const Descriptor &x,
    int dim, Terminator &terminator, const char *intrinsic) {
  // Standard requires result to have same LOGICAL kind as argument.
  CreatePartialReductionResult(result, x, dim, terminator, intrinsic, x.type());
  SubscriptValue at[maxRank];
  result.GetLowerBounds(at);
  INTERNAL_CHECK(at[0] == 1);
  using CppType = CppTypeFor<TypeCategory::Logical, KIND>;
  for (auto n{result.Elements()}; n-- > 0; result.IncrementSubscripts(at)) {
    *result.Element<CppType>(at) =
        ReduceLogicalDimToScalar<LogicalAccumulator<IS_ALL>>(x, dim - 1, at);
  }
}

template <bool IS_ALL>
inline void DoReduceLogicalDimension(Descriptor &result, const Descriptor &x,
    int dim, Terminator &terminator, const char *intrinsic) {
  auto catKind{x.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, catKind && catKind->first == TypeCategory::Logical);
  switch (catKind->second) {
  case 1:
    ReduceLogicalDimension<IS_ALL, 1>(result, x, dim, terminator, intrinsic);
    break;
  case 2:
    ReduceLogicalDimension<IS_ALL, 2>(result, x, dim, terminator, intrinsic);
    break;
  case 4:
    ReduceLogicalDimension<IS_ALL, 4>(result, x, dim, terminator, intrinsic);
    break;
  case 8:
    ReduceLogicalDimension<IS_ALL, 8>(result, x, dim, terminator, intrinsic);
    break;
  default:
    terminator.Crash(
        "%s: bad argument type LOGICAL(%d)", intrinsic, catKind->second);
  }
}

// COUNT

class CountAccumulator {
public:
  using Type = std::int64_t;
  explicit CountAccumulator(const Descriptor &array) : array_{array} {}
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

template <int KIND>
inline void CountDimension(
    Descriptor &result, const Descriptor &x, int dim, Terminator &terminator) {
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

extern "C" {

bool RTNAME(All)(const Descriptor &x, const char *source, int line, int dim) {
  return GetTotalLogicalReduction(
      x, source, line, dim, LogicalAccumulator<true>{x}, "ALL");
}
void RTNAME(AllDim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line) {
  Terminator terminator{source, line};
  DoReduceLogicalDimension<true>(result, x, dim, terminator, "ALL");
}

bool RTNAME(Any)(const Descriptor &x, const char *source, int line, int dim) {
  return GetTotalLogicalReduction(
      x, source, line, dim, LogicalAccumulator<false>{x}, "ANY");
}
void RTNAME(AnyDim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line) {
  Terminator terminator{source, line};
  DoReduceLogicalDimension<false>(result, x, dim, terminator, "ANY");
}

std::int64_t RTNAME(Count)(
    const Descriptor &x, const char *source, int line, int dim) {
  return GetTotalLogicalReduction(
      x, source, line, dim, CountAccumulator{x}, "COUNT");
}
void RTNAME(CountDim)(Descriptor &result, const Descriptor &x, int dim,
    int kind, const char *source, int line) {
  Terminator terminator{source, line};
  switch (kind) {
  case 1:
    CountDimension<1>(result, x, dim, terminator);
    break;
  case 2:
    CountDimension<2>(result, x, dim, terminator);
    break;
  case 4:
    CountDimension<4>(result, x, dim, terminator);
    break;
  case 8:
    CountDimension<8>(result, x, dim, terminator);
    break;
#ifdef __SIZEOF_INT128__
  case 16:
    CountDimension<16>(result, x, dim, terminator);
    break;
#endif
  default:
    terminator.Crash("COUNT: bad KIND=%d", kind);
  }
}

} // extern "C"
} // namespace Fortran::runtime
