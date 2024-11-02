//===-- runtime/reduction-templates.h -------------------------------------===//
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

#include "terminator.h"
#include "tools.h"
#include "flang/Runtime/cpp-type.h"
#include "flang/Runtime/descriptor.h"

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
inline void DoTotalReduction(const Descriptor &x, int dim,
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
    SubscriptValue maskAt[maxRank];
    mask->GetLowerBounds(maskAt);
    if (mask->rank() > 0) {
      for (auto elements{x.Elements()}; elements--;
           x.IncrementSubscripts(xAt), mask->IncrementSubscripts(maskAt)) {
        if (IsLogicalElementTrue(*mask, maskAt)) {
          if (!accumulator.template AccumulateAt<TYPE>(xAt))
            break;
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
// elements of array(j,:,k) can be reduced.
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
    const Descriptor &x, std::size_t resultElementSize, int dim,
    Terminator &terminator, const char *intrinsic, TypeCode typeCode) {
  int xRank{x.rank()};
  if (dim < 1 || dim > xRank) {
    terminator.Crash(
        "%s: bad DIM=%d for ARRAY with rank %d", intrinsic, dim, xRank);
  }
  int zeroBasedDim{dim - 1};
  SubscriptValue resultExtent[maxRank];
  for (int j{0}; j < zeroBasedDim; ++j) {
    resultExtent[j] = x.GetDimension(j).Extent();
  }
  for (int j{zeroBasedDim + 1}; j < xRank; ++j) {
    resultExtent[j - 1] = x.GetDimension(j).Extent();
  }
  result.Establish(typeCode, resultElementSize, nullptr, xRank - 1,
      resultExtent, CFI_attribute_allocatable);
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
inline void PartialReduction(Descriptor &result, const Descriptor &x,
    std::size_t resultElementSize, int dim, const Descriptor *mask,
    Terminator &terminator, const char *intrinsic, ACCUMULATOR &accumulator) {
  CreatePartialReductionResult(result, x, resultElementSize, dim, terminator,
      intrinsic, TypeCode{CAT, KIND});
  SubscriptValue at[maxRank];
  result.GetLowerBounds(at);
  INTERNAL_CHECK(result.rank() == 0 || at[0] == 1);
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
      // Element size of the destination descriptor is the same
      // as the element size of the source.
      PartialReduction<Accumulator, TypeCategory::Integer, KIND>(result, x,
          x.ElementBytes(), dim, mask, terminator, intrinsic, accumulator);
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
      // Element size of the destination descriptor is the same
      // as the element size of the source.
      PartialReduction<Accumulator, CAT, KIND>(result, x, x.ElementBytes(), dim,
          mask, terminator, intrinsic, accumulator);
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
    terminator.Crash("%s: bad type code %d", intrinsic, x.type().raw());
  }
}

template <typename ACCUMULATOR> struct LocationResultHelper {
  template <int KIND> struct Functor {
    void operator()(ACCUMULATOR &accumulator, const Descriptor &result) const {
      accumulator.GetResult(
          result.OffsetElement<CppTypeFor<TypeCategory::Integer, KIND>>());
    }
  };
};

template <typename ACCUMULATOR> struct PartialLocationHelper {
  template <int KIND> struct Functor {
    void operator()(Descriptor &result, const Descriptor &x, int dim,
        const Descriptor *mask, Terminator &terminator, const char *intrinsic,
        ACCUMULATOR &accumulator) const {
      // Element size of the destination descriptor is the size
      // of {TypeCategory::Integer, KIND}.
      PartialReduction<ACCUMULATOR, TypeCategory::Integer, KIND>(result, x,
          Descriptor::BytesFor(TypeCategory::Integer, KIND), dim, mask,
          terminator, intrinsic, accumulator);
    }
  };
};

} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_REDUCTION_TEMPLATES_H_
