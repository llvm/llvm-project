//===-- runtime/reduction-templates.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Generic function templates used by various reduction transformation
// intrinsic functions (SUM, PRODUCT, &c.)
//
// * Partial reductions (i.e., those with DIM= arguments that are not
//   required to be 1 by the rank of the argument) return arrays that
//   are dynamically allocated in a caller-supplied descriptor.
// * Total reductions (i.e., no DIM= argument) with FINDLOC, MAXLOC, & MINLOC
//   return integer vectors of some kind, not scalars; a caller-supplied
//   descriptor is used
// * Character-valued reductions (MAXVAL & MINVAL) return arbitrary
//   length results, dynamically allocated in a caller-supplied descriptor

#ifndef FORTRAN_RUNTIME_REDUCTION_TEMPLATES_H_
#define FORTRAN_RUNTIME_REDUCTION_TEMPLATES_H_

#include "numeric-templates.h"
#include "terminator.h"
#include "tools.h"
#include "flang/Runtime/cpp-type.h"
#include "flang/Runtime/descriptor.h"
#include <algorithm>

namespace Fortran::runtime {

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
inline RT_API_ATTRS void DoTotalReduction(const Descriptor &x, int dim,
    const Descriptor *mask, ACCUMULATOR &accumulator, const char *intrinsic,
    Terminator &terminator) {
  if (dim < 0 || dim > 1) {
    terminator.Crash("%s: bad DIM=%d for ARRAY argument with rank %d",
        intrinsic, dim, x.rank());
  }
  SubscriptValue xAt[maxRank];
  x.GetLowerBounds(xAt);
  if (mask) {
    CheckConformability(x, *mask, terminator, intrinsic, "ARRAY", "MASK");
    if (mask->rank() > 0) {
      SubscriptValue maskAt[maxRank];
      mask->GetLowerBounds(maskAt);
      for (auto elements{x.Elements()}; elements--;
           x.IncrementSubscripts(xAt), mask->IncrementSubscripts(maskAt)) {
        if (IsLogicalElementTrue(*mask, maskAt)) {
          if (!accumulator.template AccumulateAt<TYPE>(xAt)) {
            break;
          }
        }
      }
      return;
    } else if (!IsLogicalScalarTrue(*mask)) {
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
inline RT_API_ATTRS CppTypeFor<CAT, KIND> GetTotalReduction(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask,
    ACCUMULATOR &&accumulator, const char *intrinsic) {
  Terminator terminator{source, line};
  RUNTIME_CHECK(terminator, TypeCode(CAT, KIND) == x.type());
  using CppType = CppTypeFor<CAT, KIND>;
  DoTotalReduction<CppType>(x, dim, mask, accumulator, intrinsic, terminator);
  if constexpr (std::is_void_v<CppType>) {
    // Result is returned from accumulator, as in REDUCE() for derived type
#ifdef _MSC_VER // work around MSVC spurious error
    accumulator.GetResult();
#else
    accumulator.template GetResult<CppType>();
#endif
  } else {
    CppType result;
#ifdef _MSC_VER // work around MSVC spurious error
    accumulator.GetResult(&result);
#else
    accumulator.template GetResult<CppType>(&result);
#endif
    return result;
  }
}

// For reductions on a dimension, e.g. SUM(array,DIM=2) where the shape
// of the array is [2,3,5], the shape of the result is [2,5] and
// result(j,k) = SUM(array(j,:,k)), possibly modified if the array has
// lower bounds other than one.  This utility subroutine creates an
// array of subscripts [j,_,k] for result subscripts [j,k] so that the
// elements of array(j,:,k) can be reduced.
inline RT_API_ATTRS void GetExpandedSubscripts(SubscriptValue at[],
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
inline RT_API_ATTRS void ReduceDimToScalar(const Descriptor &x,
    int zeroBasedDim, SubscriptValue subscripts[], TYPE *result,
    ACCUMULATOR &accumulator) {
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
  accumulator.template GetResult<TYPE>(result, zeroBasedDim);
#endif
}

template <typename TYPE, typename ACCUMULATOR>
inline RT_API_ATTRS void ReduceDimMaskToScalar(const Descriptor &x,
    int zeroBasedDim, SubscriptValue subscripts[], const Descriptor &mask,
    TYPE *result, ACCUMULATOR &accumulator) {
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
  accumulator.template GetResult<TYPE>(result, zeroBasedDim);
#endif
}

// Partial reductions with DIM=

template <typename ACCUMULATOR, TypeCategory CAT, int KIND>
inline RT_API_ATTRS void PartialReduction(Descriptor &result,
    const Descriptor &x, std::size_t resultElementSize, int dim,
    const Descriptor *mask, Terminator &terminator, const char *intrinsic,
    ACCUMULATOR &accumulator) {
  CreatePartialReductionResult(result, x, resultElementSize, dim, terminator,
      intrinsic, TypeCode{CAT, KIND});
  SubscriptValue at[maxRank];
  result.GetLowerBounds(at);
  INTERNAL_CHECK(result.rank() == 0 || at[0] == 1);
  using CppType = CppTypeFor<CAT, KIND>;
  if (mask) {
    CheckConformability(x, *mask, terminator, intrinsic, "ARRAY", "MASK");
    if (mask->rank() > 0) {
      for (auto n{result.Elements()}; n-- > 0; result.IncrementSubscripts(at)) {
        accumulator.Reinitialize();
        ReduceDimMaskToScalar<CppType, ACCUMULATOR>(
            x, dim - 1, at, *mask, result.Element<CppType>(at), accumulator);
      }
      return;
    } else if (!IsLogicalScalarTrue(*mask)) {
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
    RT_API_ATTRS void operator()(Descriptor &result, const Descriptor &x,
        int dim, const Descriptor *mask, Terminator &terminator,
        const char *intrinsic) const {
      using Accumulator =
          ACCUM<CppTypeFor<TypeCategory::Integer, Intermediate>>;
      Accumulator accumulator{x};
      // Element size of the destination descriptor is the same
      // as the element size of the source.
      PartialReduction<Accumulator, TypeCategory::Integer, KIND>(result, x,
          x.ElementBytes(), dim, mask, terminator, intrinsic, accumulator);
    }
  };
};

template <template <typename> class INTEGER_ACCUM>
inline RT_API_ATTRS void PartialIntegerReduction(Descriptor &result,
    const Descriptor &x, int dim, int kind, const Descriptor *mask,
    const char *intrinsic, Terminator &terminator) {
  ApplyIntegerKind<
      PartialIntegerReductionHelper<INTEGER_ACCUM>::template Functor, void>(
      kind, terminator, result, x, dim, mask, terminator, intrinsic);
}

template <TypeCategory CAT, template <typename> class ACCUM, int MIN_KIND>
struct PartialFloatingReductionHelper {
  template <int KIND> struct Functor {
    static constexpr int Intermediate{std::max(KIND, MIN_KIND)};
    RT_API_ATTRS void operator()(Descriptor &result, const Descriptor &x,
        int dim, const Descriptor *mask, Terminator &terminator,
        const char *intrinsic) const {
      using Accumulator = ACCUM<CppTypeFor<TypeCategory::Real, Intermediate>>;
      Accumulator accumulator{x};
      // Element size of the destination descriptor is the same
      // as the element size of the source.
      PartialReduction<Accumulator, CAT, KIND>(result, x, x.ElementBytes(), dim,
          mask, terminator, intrinsic, accumulator);
    }
  };
};

template <template <typename> class INTEGER_ACCUM,
    template <typename> class REAL_ACCUM,
    template <typename> class COMPLEX_ACCUM, int MIN_REAL_KIND>
inline RT_API_ATTRS void TypedPartialNumericReduction(Descriptor &result,
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
                               REAL_ACCUM, MIN_REAL_KIND>::template Functor,
        void>(catKind->second, terminator, result, x, dim, mask, terminator,
        intrinsic);
    break;
  case TypeCategory::Complex:
    ApplyFloatingPointKind<PartialFloatingReductionHelper<TypeCategory::Complex,
                               COMPLEX_ACCUM, MIN_REAL_KIND>::template Functor,
        void>(catKind->second, terminator, result, x, dim, mask, terminator,
        intrinsic);
    break;
  default:
    terminator.Crash("%s: bad type code %d", intrinsic, x.type().raw());
  }
}

template <typename ACCUMULATOR> struct LocationResultHelper {
  template <int KIND> struct Functor {
    RT_API_ATTRS void operator()(
        ACCUMULATOR &accumulator, const Descriptor &result) const {
      accumulator.GetResult(
          result.OffsetElement<CppTypeFor<TypeCategory::Integer, KIND>>());
    }
  };
};

template <typename ACCUMULATOR> struct PartialLocationHelper {
  template <int KIND> struct Functor {
    RT_API_ATTRS void operator()(Descriptor &result, const Descriptor &x,
        int dim, const Descriptor *mask, Terminator &terminator,
        const char *intrinsic, ACCUMULATOR &accumulator) const {
      // Element size of the destination descriptor is the size
      // of {TypeCategory::Integer, KIND}.
      PartialReduction<ACCUMULATOR, TypeCategory::Integer, KIND>(result, x,
          Descriptor::BytesFor(TypeCategory::Integer, KIND), dim, mask,
          terminator, intrinsic, accumulator);
    }
  };
};

// NORM2 templates

RT_VAR_GROUP_BEGIN

// Use at least double precision for accumulators.
// Don't use __float128, it doesn't work with abs() or sqrt() yet.
static constexpr RT_CONST_VAR_ATTRS int Norm2LargestLDKind {
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
  16
#elif LDBL_MANT_DIG == 64
  10
#else
  8
#endif
};

RT_VAR_GROUP_END

template <TypeCategory CAT, int KIND, typename ACCUMULATOR>
inline RT_API_ATTRS void DoMaxMinNorm2(Descriptor &result, const Descriptor &x,
    int dim, const Descriptor *mask, const char *intrinsic,
    Terminator &terminator) {
  using Type = CppTypeFor<CAT, KIND>;
  ACCUMULATOR accumulator{x};
  if (dim == 0 || x.rank() == 1) {
    // Total reduction

    // Element size of the destination descriptor is the same
    // as the element size of the source.
    result.Establish(x.type(), x.ElementBytes(), nullptr, 0, nullptr,
        CFI_attribute_allocatable);
    if (int stat{result.Allocate()}) {
      terminator.Crash(
          "%s: could not allocate memory for result; STAT=%d", intrinsic, stat);
    }
    DoTotalReduction<Type>(x, dim, mask, accumulator, intrinsic, terminator);
    accumulator.GetResult(result.OffsetElement<Type>());
  } else {
    // Partial reduction

    // Element size of the destination descriptor is the same
    // as the element size of the source.
    PartialReduction<ACCUMULATOR, CAT, KIND>(result, x, x.ElementBytes(), dim,
        mask, terminator, intrinsic, accumulator);
  }
}

// The data type used by Norm2Accumulator.
template <int KIND>
using Norm2AccumType =
    CppTypeFor<TypeCategory::Real, std::clamp(KIND, 8, Norm2LargestLDKind)>;

template <int KIND> class Norm2Accumulator {
public:
  using Type = CppTypeFor<TypeCategory::Real, KIND>;
  using AccumType = Norm2AccumType<KIND>;
  explicit RT_API_ATTRS Norm2Accumulator(const Descriptor &array)
      : array_{array} {}
  RT_API_ATTRS void Reinitialize() { max_ = sum_ = 0; }
  template <typename A>
  RT_API_ATTRS void GetResult(A *p, int /*zeroBasedDim*/ = -1) const {
    // m * sqrt(1 + sum((others(:)/m)**2))
    *p = static_cast<Type>(max_ * SQRTTy<AccumType>::compute(1 + sum_));
  }
  RT_API_ATTRS bool Accumulate(Type x) {
    auto absX{ABSTy<AccumType>::compute(static_cast<AccumType>(x))};
    if (!max_) {
      max_ = absX;
    } else if (absX > max_) {
      auto t{max_ / absX}; // < 1.0
      auto tsq{t * t};
      sum_ *= tsq; // scale sum to reflect change to the max
      sum_ += tsq; // include a term for the previous max
      max_ = absX;
    } else { // absX <= max_
      auto t{absX / max_};
      sum_ += t * t;
    }
    return true;
  }
  template <typename A>
  RT_API_ATTRS bool AccumulateAt(const SubscriptValue at[]) {
    return Accumulate(*array_.Element<A>(at));
  }

private:
  const Descriptor &array_;
  AccumType max_{0}; // value (m) with largest magnitude
  AccumType sum_{0}; // sum((others(:)/m)**2)
};

template <int KIND> struct Norm2Helper {
  RT_API_ATTRS void operator()(Descriptor &result, const Descriptor &x, int dim,
      const Descriptor *mask, Terminator &terminator) const {
    DoMaxMinNorm2<TypeCategory::Real, KIND, Norm2Accumulator<KIND>>(
        result, x, dim, mask, "NORM2", terminator);
  }
};

} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_REDUCTION_TEMPLATES_H_
