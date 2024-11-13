//===-- runtime/transformational.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements the transformational intrinsic functions of Fortran 2018 that
// rearrange or duplicate data without (much) regard to type.  These are
// CSHIFT, EOSHIFT, PACK, RESHAPE, SPREAD, TRANSPOSE, and UNPACK.
//
// Many of these are defined in the 2018 standard with text that makes sense
// only if argument arrays have lower bounds of one.  Rather than interpret
// these cases as implying a hidden constraint, these implementations
// work with arbitrary lower bounds.  This may be technically an extension
// of the standard but it more likely to conform with its intent.

#include "flang/Runtime/transformational.h"
#include "copy.h"
#include "terminator.h"
#include "tools.h"
#include "flang/Common/float128.h"
#include "flang/Runtime/descriptor.h"

namespace Fortran::runtime {

// Utility for CSHIFT & EOSHIFT rank > 1 cases that determines the shift count
// for each of the vector sections of the result.
class ShiftControl {
public:
  RT_API_ATTRS ShiftControl(const Descriptor &s, Terminator &t, int dim)
      : shift_{s}, terminator_{t}, shiftRank_{s.rank()}, dim_{dim} {}
  RT_API_ATTRS void Init(const Descriptor &source, const char *which) {
    int rank{source.rank()};
    RUNTIME_CHECK(terminator_, shiftRank_ == 0 || shiftRank_ == rank - 1);
    auto catAndKind{shift_.type().GetCategoryAndKind()};
    RUNTIME_CHECK(
        terminator_, catAndKind && catAndKind->first == TypeCategory::Integer);
    shiftElemLen_ = catAndKind->second;
    if (shiftRank_ > 0) {
      int k{0};
      for (int j{0}; j < rank; ++j) {
        if (j + 1 != dim_) {
          const Dimension &shiftDim{shift_.GetDimension(k)};
          lb_[k++] = shiftDim.LowerBound();
          if (shiftDim.Extent() != source.GetDimension(j).Extent()) {
            terminator_.Crash("%s: on dimension %d, SHIFT= has extent %jd but "
                              "SOURCE= has extent %jd",
                which, k, static_cast<std::intmax_t>(shiftDim.Extent()),
                static_cast<std::intmax_t>(source.GetDimension(j).Extent()));
          }
        }
      }
    } else if (auto count{GetInt64Safe(
                   shift_.OffsetElement<char>(), shiftElemLen_, terminator_)}) {
      shiftCount_ = *count;
    } else {
      terminator_.Crash("%s: SHIFT= value exceeds 64 bits", which);
    }
  }
  RT_API_ATTRS SubscriptValue GetShift(const SubscriptValue resultAt[]) const {
    if (shiftRank_ > 0) {
      SubscriptValue shiftAt[maxRank];
      int k{0};
      for (int j{0}; j < shiftRank_ + 1; ++j) {
        if (j + 1 != dim_) {
          shiftAt[k] = lb_[k] + resultAt[j] - 1;
          ++k;
        }
      }
      auto count{GetInt64Safe(
          shift_.Element<char>(shiftAt), shiftElemLen_, terminator_)};
      RUNTIME_CHECK(terminator_, count.has_value());
      return *count;
    } else {
      return shiftCount_; // invariant count extracted in Init()
    }
  }

private:
  const Descriptor &shift_;
  Terminator &terminator_;
  int shiftRank_;
  int dim_;
  SubscriptValue lb_[maxRank];
  std::size_t shiftElemLen_;
  SubscriptValue shiftCount_{};
};

// Fill an EOSHIFT result with default boundary values
static RT_API_ATTRS void DefaultInitialize(
    const Descriptor &result, Terminator &terminator) {
  auto catAndKind{result.type().GetCategoryAndKind()};
  RUNTIME_CHECK(
      terminator, catAndKind && catAndKind->first != TypeCategory::Derived);
  std::size_t elementLen{result.ElementBytes()};
  std::size_t bytes{result.Elements() * elementLen};
  if (catAndKind->first == TypeCategory::Character) {
    switch (int kind{catAndKind->second}) {
    case 1:
      Fortran::runtime::fill_n(result.OffsetElement<char>(), bytes, ' ');
      break;
    case 2:
      Fortran::runtime::fill_n(result.OffsetElement<char16_t>(), bytes / 2,
          static_cast<char16_t>(' '));
      break;
    case 4:
      Fortran::runtime::fill_n(result.OffsetElement<char32_t>(), bytes / 4,
          static_cast<char32_t>(' '));
      break;
    default:
      terminator.Crash(
          "not yet implemented: CHARACTER(KIND=%d) in EOSHIFT intrinsic", kind);
    }
  } else {
    std::memset(result.raw().base_addr, 0, bytes);
  }
}

static inline RT_API_ATTRS std::size_t AllocateResult(Descriptor &result,
    const Descriptor &source, int rank, const SubscriptValue extent[],
    Terminator &terminator, const char *function) {
  std::size_t elementLen{source.ElementBytes()};
  const DescriptorAddendum *sourceAddendum{source.Addendum()};
  result.Establish(source.type(), elementLen, nullptr, rank, extent,
      CFI_attribute_allocatable, sourceAddendum != nullptr);
  if (sourceAddendum) {
    *result.Addendum() = *sourceAddendum;
  }
  for (int j{0}; j < rank; ++j) {
    result.GetDimension(j).SetBounds(1, extent[j]);
  }
  if (int stat{result.Allocate()}) {
    terminator.Crash(
        "%s: Could not allocate memory for result (stat=%d)", function, stat);
  }
  return elementLen;
}

template <TypeCategory CAT, int KIND>
static inline RT_API_ATTRS std::size_t AllocateBesselResult(Descriptor &result,
    int32_t n1, int32_t n2, Terminator &terminator, const char *function) {
  int rank{1};
  SubscriptValue extent[maxRank];
  for (int j{0}; j < maxRank; j++) {
    extent[j] = 0;
  }
  if (n1 <= n2) {
    extent[0] = n2 - n1 + 1;
  }

  std::size_t elementLen{Descriptor::BytesFor(CAT, KIND)};
  result.Establish(TypeCode{CAT, KIND}, elementLen, nullptr, rank, extent,
      CFI_attribute_allocatable, false);
  for (int j{0}; j < rank; ++j) {
    result.GetDimension(j).SetBounds(1, extent[j]);
  }
  if (int stat{result.Allocate()}) {
    terminator.Crash(
        "%s: Could not allocate memory for result (stat=%d)", function, stat);
  }
  return elementLen;
}

template <TypeCategory CAT, int KIND>
static inline RT_API_ATTRS void DoBesselJn(Descriptor &result, int32_t n1,
    int32_t n2, CppTypeFor<CAT, KIND> x, CppTypeFor<CAT, KIND> bn2,
    CppTypeFor<CAT, KIND> bn2_1, const char *sourceFile, int line) {
  Terminator terminator{sourceFile, line};
  AllocateBesselResult<CAT, KIND>(result, n1, n2, terminator, "BESSEL_JN");

  // The standard requires that n1 and n2 be non-negative. However, some other
  // compilers generate results even when n1 and/or n2 are negative. For now,
  // we also do not enforce the non-negativity constraint.
  if (n2 < n1) {
    return;
  }

  SubscriptValue at[maxRank];
  for (int j{0}; j < maxRank; ++j) {
    at[j] = 0;
  }

  // if n2 >= n1, there will be at least one element in the result.
  at[0] = n2 - n1 + 1;
  *result.Element<CppTypeFor<CAT, KIND>>(at) = bn2;

  if (n2 == n1) {
    return;
  }

  at[0] = n2 - n1;
  *result.Element<CppTypeFor<CAT, KIND>>(at) = bn2_1;

  // Bessel functions of the first kind are stable for a backward recursion
  // (see https://dlmf.nist.gov/10.74.iv and https://dlmf.nist.gov/10.6.E1).
  //
  //     J(n-1, x) = (2.0 / x) * n * J(n, x) - J(n+1, x)
  //
  // which is equivalent to
  //
  //     J(n, x) = (2.0 / x) * (n + 1) * J(n+1, x) - J(n+2, x)
  //
  CppTypeFor<CAT, KIND> bn_2 = bn2;
  CppTypeFor<CAT, KIND> bn_1 = bn2_1;
  CppTypeFor<CAT, KIND> twoOverX = 2.0 / x;
  for (int n{n2 - 2}; n >= n1; --n) {
    auto bn = twoOverX * (n + 1) * bn_1 - bn_2;

    at[0] = n - n1 + 1;
    *result.Element<CppTypeFor<CAT, KIND>>(at) = bn;

    bn_2 = bn_1;
    bn_1 = bn;
  }
}

template <TypeCategory CAT, int KIND>
static inline RT_API_ATTRS void DoBesselJnX0(Descriptor &result, int32_t n1,
    int32_t n2, const char *sourceFile, int line) {
  Terminator terminator{sourceFile, line};
  AllocateBesselResult<CAT, KIND>(result, n1, n2, terminator, "BESSEL_JN");

  // The standard requires that n1 and n2 be non-negative. However, some other
  // compilers generate results even when n1 and/or n2 are negative. For now,
  // we also do not enforce the non-negativity constraint.
  if (n2 < n1) {
    return;
  }

  SubscriptValue at[maxRank];
  for (int j{0}; j < maxRank; ++j) {
    at[j] = 0;
  }

  // J(0, 0.0) = 1.0, when n == 0.
  // J(n, 0.0) = 0.0, when n > 0.
  at[0] = 1;
  *result.Element<CppTypeFor<CAT, KIND>>(at) = (n1 == 0) ? 1.0 : 0.0;
  for (int j{2}; j <= n2 - n1 + 1; ++j) {
    at[0] = j;
    *result.Element<CppTypeFor<CAT, KIND>>(at) = 0.0;
  }
}

template <TypeCategory CAT, int KIND>
static inline RT_API_ATTRS void DoBesselYn(Descriptor &result, int32_t n1,
    int32_t n2, CppTypeFor<CAT, KIND> x, CppTypeFor<CAT, KIND> bn1,
    CppTypeFor<CAT, KIND> bn1_1, const char *sourceFile, int line) {
  Terminator terminator{sourceFile, line};
  AllocateBesselResult<CAT, KIND>(result, n1, n2, terminator, "BESSEL_YN");

  // The standard requires that n1 and n2 be non-negative. However, some other
  // compilers generate results even when n1 and/or n2 are negative. For now,
  // we also do not enforce the non-negativity constraint.
  if (n2 < n1) {
    return;
  }

  SubscriptValue at[maxRank];
  for (int j{0}; j < maxRank; ++j) {
    at[j] = 0;
  }

  // if n2 >= n1, there will be at least one element in the result.
  at[0] = 1;
  *result.Element<CppTypeFor<CAT, KIND>>(at) = bn1;

  if (n2 == n1) {
    return;
  }

  at[0] = 2;
  *result.Element<CppTypeFor<CAT, KIND>>(at) = bn1_1;

  // Bessel functions of the second kind are stable for a forward recursion
  // (see https://dlmf.nist.gov/10.74.iv and https://dlmf.nist.gov/10.6.E1).
  //
  //     Y(n+1, x) = (2.0 / x) * n * Y(n, x) - Y(n-1, x)
  //
  // which is equivalent to
  //
  //     Y(n, x) = (2.0 / x) * (n - 1) * Y(n-1, x) - Y(n-2, x)
  //
  CppTypeFor<CAT, KIND> bn_2 = bn1;
  CppTypeFor<CAT, KIND> bn_1 = bn1_1;
  CppTypeFor<CAT, KIND> twoOverX = 2.0 / x;
  for (int n{n1 + 2}; n <= n2; ++n) {
    auto bn = twoOverX * (n - 1) * bn_1 - bn_2;

    at[0] = n - n1 + 1;
    *result.Element<CppTypeFor<CAT, KIND>>(at) = bn;

    bn_2 = bn_1;
    bn_1 = bn;
  }
}

template <TypeCategory CAT, int KIND>
static inline RT_API_ATTRS void DoBesselYnX0(Descriptor &result, int32_t n1,
    int32_t n2, const char *sourceFile, int line) {
  Terminator terminator{sourceFile, line};
  AllocateBesselResult<CAT, KIND>(result, n1, n2, terminator, "BESSEL_YN");

  // The standard requires that n1 and n2 be non-negative. However, some other
  // compilers generate results even when n1 and/or n2 are negative. For now,
  // we also do not enforce the non-negativity constraint.
  if (n2 < n1) {
    return;
  }

  SubscriptValue at[maxRank];
  for (int j{0}; j < maxRank; ++j) {
    at[j] = 0;
  }

  // Y(n, 0.0) = -Inf, when n >= 0
  for (int j{1}; j <= n2 - n1 + 1; ++j) {
    at[0] = j;
    *result.Element<CppTypeFor<CAT, KIND>>(at) =
        -std::numeric_limits<CppTypeFor<CAT, KIND>>::infinity();
  }
}

extern "C" {
RT_EXT_API_GROUP_BEGIN

// BESSEL_JN
// TODO: REAL(2 & 3)
void RTDEF(BesselJn_4)(Descriptor &result, int32_t n1, int32_t n2,
    CppTypeFor<TypeCategory::Real, 4> x, CppTypeFor<TypeCategory::Real, 4> bn2,
    CppTypeFor<TypeCategory::Real, 4> bn2_1, const char *sourceFile, int line) {
  DoBesselJn<TypeCategory::Real, 4>(
      result, n1, n2, x, bn2, bn2_1, sourceFile, line);
}

void RTDEF(BesselJn_8)(Descriptor &result, int32_t n1, int32_t n2,
    CppTypeFor<TypeCategory::Real, 8> x, CppTypeFor<TypeCategory::Real, 8> bn2,
    CppTypeFor<TypeCategory::Real, 8> bn2_1, const char *sourceFile, int line) {
  DoBesselJn<TypeCategory::Real, 8>(
      result, n1, n2, x, bn2, bn2_1, sourceFile, line);
}

#if HAS_FLOAT80
void RTDEF(BesselJn_10)(Descriptor &result, int32_t n1, int32_t n2,
    CppTypeFor<TypeCategory::Real, 10> x,
    CppTypeFor<TypeCategory::Real, 10> bn2,
    CppTypeFor<TypeCategory::Real, 10> bn2_1, const char *sourceFile,
    int line) {
  DoBesselJn<TypeCategory::Real, 10>(
      result, n1, n2, x, bn2, bn2_1, sourceFile, line);
}
#endif

#if HAS_LDBL128 || HAS_FLOAT128
void RTDEF(BesselJn_16)(Descriptor &result, int32_t n1, int32_t n2,
    CppTypeFor<TypeCategory::Real, 16> x,
    CppTypeFor<TypeCategory::Real, 16> bn2,
    CppTypeFor<TypeCategory::Real, 16> bn2_1, const char *sourceFile,
    int line) {
  DoBesselJn<TypeCategory::Real, 16>(
      result, n1, n2, x, bn2, bn2_1, sourceFile, line);
}
#endif

// TODO: REAL(2 & 3)
void RTDEF(BesselJnX0_4)(Descriptor &result, int32_t n1, int32_t n2,
    const char *sourceFile, int line) {
  DoBesselJnX0<TypeCategory::Real, 4>(result, n1, n2, sourceFile, line);
}

void RTDEF(BesselJnX0_8)(Descriptor &result, int32_t n1, int32_t n2,
    const char *sourceFile, int line) {
  DoBesselJnX0<TypeCategory::Real, 8>(result, n1, n2, sourceFile, line);
}

#if HAS_FLOAT80
void RTDEF(BesselJnX0_10)(Descriptor &result, int32_t n1, int32_t n2,
    const char *sourceFile, int line) {
  DoBesselJnX0<TypeCategory::Real, 10>(result, n1, n2, sourceFile, line);
}
#endif

#if HAS_LDBL128 || HAS_FLOAT128
void RTDEF(BesselJnX0_16)(Descriptor &result, int32_t n1, int32_t n2,
    const char *sourceFile, int line) {
  DoBesselJnX0<TypeCategory::Real, 16>(result, n1, n2, sourceFile, line);
}
#endif

// BESSEL_YN
// TODO: REAL(2 & 3)
void RTDEF(BesselYn_4)(Descriptor &result, int32_t n1, int32_t n2,
    CppTypeFor<TypeCategory::Real, 4> x, CppTypeFor<TypeCategory::Real, 4> bn1,
    CppTypeFor<TypeCategory::Real, 4> bn1_1, const char *sourceFile, int line) {
  DoBesselYn<TypeCategory::Real, 4>(
      result, n1, n2, x, bn1, bn1_1, sourceFile, line);
}

void RTDEF(BesselYn_8)(Descriptor &result, int32_t n1, int32_t n2,
    CppTypeFor<TypeCategory::Real, 8> x, CppTypeFor<TypeCategory::Real, 8> bn1,
    CppTypeFor<TypeCategory::Real, 8> bn1_1, const char *sourceFile, int line) {
  DoBesselYn<TypeCategory::Real, 8>(
      result, n1, n2, x, bn1, bn1_1, sourceFile, line);
}

#if HAS_FLOAT80
void RTDEF(BesselYn_10)(Descriptor &result, int32_t n1, int32_t n2,
    CppTypeFor<TypeCategory::Real, 10> x,
    CppTypeFor<TypeCategory::Real, 10> bn1,
    CppTypeFor<TypeCategory::Real, 10> bn1_1, const char *sourceFile,
    int line) {
  DoBesselYn<TypeCategory::Real, 10>(
      result, n1, n2, x, bn1, bn1_1, sourceFile, line);
}
#endif

#if HAS_LDBL128 || HAS_FLOAT128
void RTDEF(BesselYn_16)(Descriptor &result, int32_t n1, int32_t n2,
    CppTypeFor<TypeCategory::Real, 16> x,
    CppTypeFor<TypeCategory::Real, 16> bn1,
    CppTypeFor<TypeCategory::Real, 16> bn1_1, const char *sourceFile,
    int line) {
  DoBesselYn<TypeCategory::Real, 16>(
      result, n1, n2, x, bn1, bn1_1, sourceFile, line);
}
#endif

// TODO: REAL(2 & 3)
void RTDEF(BesselYnX0_4)(Descriptor &result, int32_t n1, int32_t n2,
    const char *sourceFile, int line) {
  DoBesselYnX0<TypeCategory::Real, 4>(result, n1, n2, sourceFile, line);
}

void RTDEF(BesselYnX0_8)(Descriptor &result, int32_t n1, int32_t n2,
    const char *sourceFile, int line) {
  DoBesselYnX0<TypeCategory::Real, 8>(result, n1, n2, sourceFile, line);
}

#if HAS_FLOAT80
void RTDEF(BesselYnX0_10)(Descriptor &result, int32_t n1, int32_t n2,
    const char *sourceFile, int line) {
  DoBesselYnX0<TypeCategory::Real, 10>(result, n1, n2, sourceFile, line);
}
#endif

#if HAS_LDBL128 || HAS_FLOAT128
void RTDEF(BesselYnX0_16)(Descriptor &result, int32_t n1, int32_t n2,
    const char *sourceFile, int line) {
  DoBesselYnX0<TypeCategory::Real, 16>(result, n1, n2, sourceFile, line);
}
#endif

// CSHIFT where rank of ARRAY argument > 1
void RTDEF(Cshift)(Descriptor &result, const Descriptor &source,
    const Descriptor &shift, int dim, const char *sourceFile, int line) {
  Terminator terminator{sourceFile, line};
  int rank{source.rank()};
  RUNTIME_CHECK(terminator, rank > 1);
  if (dim < 1 || dim > rank) {
    terminator.Crash(
        "CSHIFT: DIM=%d must be >= 1 and <= SOURCE= rank %d", dim, rank);
  }
  ShiftControl shiftControl{shift, terminator, dim};
  shiftControl.Init(source, "CSHIFT");
  SubscriptValue extent[maxRank];
  source.GetShape(extent);
  AllocateResult(result, source, rank, extent, terminator, "CSHIFT");
  SubscriptValue resultAt[maxRank];
  for (int j{0}; j < rank; ++j) {
    resultAt[j] = 1;
  }
  SubscriptValue sourceLB[maxRank];
  source.GetLowerBounds(sourceLB);
  SubscriptValue dimExtent{extent[dim - 1]};
  SubscriptValue dimLB{sourceLB[dim - 1]};
  SubscriptValue &resDim{resultAt[dim - 1]};
  for (std::size_t n{result.Elements()}; n > 0; n -= dimExtent) {
    SubscriptValue shiftCount{shiftControl.GetShift(resultAt)};
    SubscriptValue sourceAt[maxRank];
    for (int j{0}; j < rank; ++j) {
      sourceAt[j] = sourceLB[j] + resultAt[j] - 1;
    }
    SubscriptValue &sourceDim{sourceAt[dim - 1]};
    sourceDim = dimLB + shiftCount % dimExtent;
    if (sourceDim < dimLB) {
      sourceDim += dimExtent;
    }
    for (resDim = 1; resDim <= dimExtent; ++resDim) {
      CopyElement(result, resultAt, source, sourceAt, terminator);
      if (++sourceDim == dimLB + dimExtent) {
        sourceDim = dimLB;
      }
    }
    result.IncrementSubscripts(resultAt);
  }
}

// CSHIFT where rank of ARRAY argument == 1
void RTDEF(CshiftVector)(Descriptor &result, const Descriptor &source,
    std::int64_t shift, const char *sourceFile, int line) {
  Terminator terminator{sourceFile, line};
  RUNTIME_CHECK(terminator, source.rank() == 1);
  const Dimension &sourceDim{source.GetDimension(0)};
  SubscriptValue extent{sourceDim.Extent()};
  AllocateResult(result, source, 1, &extent, terminator, "CSHIFT");
  SubscriptValue lb{sourceDim.LowerBound()};
  for (SubscriptValue j{0}; j < extent; ++j) {
    SubscriptValue resultAt{1 + j};
    SubscriptValue sourceAt{
        lb + static_cast<SubscriptValue>(j + shift) % extent};
    if (sourceAt < lb) {
      sourceAt += extent;
    }
    CopyElement(result, &resultAt, source, &sourceAt, terminator);
  }
}

// EOSHIFT of rank > 1
void RTDEF(Eoshift)(Descriptor &result, const Descriptor &source,
    const Descriptor &shift, const Descriptor *boundary, int dim,
    const char *sourceFile, int line) {
  Terminator terminator{sourceFile, line};
  SubscriptValue extent[maxRank];
  int rank{source.GetShape(extent)};
  RUNTIME_CHECK(terminator, rank > 1);
  if (dim < 1 || dim > rank) {
    terminator.Crash(
        "EOSHIFT: DIM=%d must be >= 1 and <= SOURCE= rank %d", dim, rank);
  }
  std::size_t elementLen{
      AllocateResult(result, source, rank, extent, terminator, "EOSHIFT")};
  int boundaryRank{-1};
  if (boundary) {
    boundaryRank = boundary->rank();
    RUNTIME_CHECK(terminator, boundaryRank == 0 || boundaryRank == rank - 1);
    RUNTIME_CHECK(terminator, boundary->type() == source.type());
    if (boundary->ElementBytes() != elementLen) {
      terminator.Crash("EOSHIFT: BOUNDARY= has element byte length %zd, but "
                       "SOURCE= has length %zd",
          boundary->ElementBytes(), elementLen);
    }
    if (boundaryRank > 0) {
      int k{0};
      for (int j{0}; j < rank; ++j) {
        if (j != dim - 1) {
          if (boundary->GetDimension(k).Extent() != extent[j]) {
            terminator.Crash("EOSHIFT: BOUNDARY= has extent %jd on dimension "
                             "%d but must conform with extent %jd of SOURCE=",
                static_cast<std::intmax_t>(boundary->GetDimension(k).Extent()),
                k + 1, static_cast<std::intmax_t>(extent[j]));
          }
          ++k;
        }
      }
    }
  }
  ShiftControl shiftControl{shift, terminator, dim};
  shiftControl.Init(source, "EOSHIFT");
  SubscriptValue resultAt[maxRank];
  for (int j{0}; j < rank; ++j) {
    resultAt[j] = 1;
  }
  if (!boundary) {
    DefaultInitialize(result, terminator);
  }
  SubscriptValue sourceLB[maxRank];
  source.GetLowerBounds(sourceLB);
  SubscriptValue boundaryAt[maxRank];
  if (boundaryRank > 0) {
    boundary->GetLowerBounds(boundaryAt);
  }
  SubscriptValue dimExtent{extent[dim - 1]};
  SubscriptValue dimLB{sourceLB[dim - 1]};
  SubscriptValue &resDim{resultAt[dim - 1]};
  for (std::size_t n{result.Elements()}; n > 0; n -= dimExtent) {
    SubscriptValue shiftCount{shiftControl.GetShift(resultAt)};
    SubscriptValue sourceAt[maxRank];
    for (int j{0}; j < rank; ++j) {
      sourceAt[j] = sourceLB[j] + resultAt[j] - 1;
    }
    SubscriptValue &sourceDim{sourceAt[dim - 1]};
    sourceDim = dimLB + shiftCount;
    for (resDim = 1; resDim <= dimExtent; ++resDim) {
      if (sourceDim >= dimLB && sourceDim < dimLB + dimExtent) {
        CopyElement(result, resultAt, source, sourceAt, terminator);
      } else if (boundary) {
        CopyElement(result, resultAt, *boundary, boundaryAt, terminator);
      }
      ++sourceDim;
    }
    result.IncrementSubscripts(resultAt);
    if (boundaryRank > 0) {
      boundary->IncrementSubscripts(boundaryAt);
    }
  }
}

// EOSHIFT of vector
void RTDEF(EoshiftVector)(Descriptor &result, const Descriptor &source,
    std::int64_t shift, const Descriptor *boundary, const char *sourceFile,
    int line) {
  Terminator terminator{sourceFile, line};
  RUNTIME_CHECK(terminator, source.rank() == 1);
  SubscriptValue extent{source.GetDimension(0).Extent()};
  std::size_t elementLen{
      AllocateResult(result, source, 1, &extent, terminator, "EOSHIFT")};
  if (boundary) {
    RUNTIME_CHECK(terminator, boundary->rank() == 0);
    RUNTIME_CHECK(terminator, boundary->type() == source.type());
    if (boundary->ElementBytes() != elementLen) {
      terminator.Crash("EOSHIFT: BOUNDARY= has element byte length %zd but "
                       "SOURCE= has length %zd",
          boundary->ElementBytes(), elementLen);
    }
  }
  if (!boundary) {
    DefaultInitialize(result, terminator);
  }
  SubscriptValue lb{source.GetDimension(0).LowerBound()};
  for (SubscriptValue j{1}; j <= extent; ++j) {
    SubscriptValue sourceAt{lb + j - 1 + static_cast<SubscriptValue>(shift)};
    if (sourceAt >= lb && sourceAt < lb + extent) {
      CopyElement(result, &j, source, &sourceAt, terminator);
    } else if (boundary) {
      CopyElement(result, &j, *boundary, 0, terminator);
    }
  }
}

// PACK
void RTDEF(Pack)(Descriptor &result, const Descriptor &source,
    const Descriptor &mask, const Descriptor *vector, const char *sourceFile,
    int line) {
  Terminator terminator{sourceFile, line};
  CheckConformability(source, mask, terminator, "PACK", "ARRAY=", "MASK=");
  auto maskType{mask.type().GetCategoryAndKind()};
  RUNTIME_CHECK(
      terminator, maskType && maskType->first == TypeCategory::Logical);
  SubscriptValue trues{0};
  if (mask.rank() == 0) {
    if (IsLogicalElementTrue(mask, nullptr)) {
      trues = source.Elements();
    }
  } else {
    SubscriptValue maskAt[maxRank];
    mask.GetLowerBounds(maskAt);
    for (std::size_t n{mask.Elements()}; n > 0; --n) {
      if (IsLogicalElementTrue(mask, maskAt)) {
        ++trues;
      }
      mask.IncrementSubscripts(maskAt);
    }
  }
  SubscriptValue extent{trues};
  if (vector) {
    RUNTIME_CHECK(terminator, vector->rank() == 1);
    RUNTIME_CHECK(terminator, source.type() == vector->type());
    if (source.ElementBytes() != vector->ElementBytes()) {
      terminator.Crash("PACK: SOURCE= has element byte length %zd, but VECTOR= "
                       "has length %zd",
          source.ElementBytes(), vector->ElementBytes());
    }
    extent = vector->GetDimension(0).Extent();
    if (extent < trues) {
      terminator.Crash("PACK: VECTOR= has extent %jd but there are %jd MASK= "
                       "elements that are .TRUE.",
          static_cast<std::intmax_t>(extent),
          static_cast<std::intmax_t>(trues));
    }
  }
  AllocateResult(result, source, 1, &extent, terminator, "PACK");
  SubscriptValue sourceAt[maxRank], resultAt{1};
  source.GetLowerBounds(sourceAt);
  if (mask.rank() == 0) {
    if (IsLogicalElementTrue(mask, nullptr)) {
      for (SubscriptValue n{trues}; n > 0; --n) {
        CopyElement(result, &resultAt, source, sourceAt, terminator);
        ++resultAt;
        source.IncrementSubscripts(sourceAt);
      }
    }
  } else {
    SubscriptValue maskAt[maxRank];
    mask.GetLowerBounds(maskAt);
    for (std::size_t n{source.Elements()}; n > 0; --n) {
      if (IsLogicalElementTrue(mask, maskAt)) {
        CopyElement(result, &resultAt, source, sourceAt, terminator);
        ++resultAt;
      }
      source.IncrementSubscripts(sourceAt);
      mask.IncrementSubscripts(maskAt);
    }
  }
  if (vector) {
    SubscriptValue vectorAt{
        vector->GetDimension(0).LowerBound() + resultAt - 1};
    for (; resultAt <= extent; ++resultAt, ++vectorAt) {
      CopyElement(result, &resultAt, *vector, &vectorAt, terminator);
    }
  }
}

// RESHAPE
// F2018 16.9.163
void RTDEF(Reshape)(Descriptor &result, const Descriptor &source,
    const Descriptor &shape, const Descriptor *pad, const Descriptor *order,
    const char *sourceFile, int line) {
  // Compute and check the rank of the result.
  Terminator terminator{sourceFile, line};
  RUNTIME_CHECK(terminator, shape.rank() == 1);
  RUNTIME_CHECK(terminator, shape.type().IsInteger());
  SubscriptValue resultRank{shape.GetDimension(0).Extent()};
  if (resultRank < 0 || resultRank > static_cast<SubscriptValue>(maxRank)) {
    terminator.Crash(
        "RESHAPE: SHAPE= vector length %jd implies a bad result rank",
        static_cast<std::intmax_t>(resultRank));
  }

  // Extract and check the shape of the result; compute its element count.
  SubscriptValue resultExtent[maxRank];
  std::size_t shapeElementBytes{shape.ElementBytes()};
  std::size_t resultElements{1};
  SubscriptValue shapeSubscript{shape.GetDimension(0).LowerBound()};
  for (int j{0}; j < resultRank; ++j, ++shapeSubscript) {
    auto extent{GetInt64Safe(
        shape.Element<char>(&shapeSubscript), shapeElementBytes, terminator)};
    if (!extent) {
      terminator.Crash("RESHAPE: value of SHAPE(%d) exceeds 64 bits", j + 1);
    } else if (*extent < 0) {
      terminator.Crash("RESHAPE: bad value for SHAPE(%d)=%jd", j + 1,
          static_cast<std::intmax_t>(*extent));
    }
    resultExtent[j] = *extent;
    resultElements *= resultExtent[j];
  }

  // Check that there are sufficient elements in the SOURCE=, or that
  // the optional PAD= argument is present and nonempty.
  std::size_t elementBytes{source.ElementBytes()};
  std::size_t sourceElements{source.Elements()};
  std::size_t padElements{pad ? pad->Elements() : 0};
  if (resultElements > sourceElements) {
    if (padElements <= 0) {
      terminator.Crash(
          "RESHAPE: not enough elements, need %zd but only have %zd",
          resultElements, sourceElements);
    }
    if (pad->ElementBytes() != elementBytes) {
      terminator.Crash("RESHAPE: PAD= has element byte length %zd but SOURCE= "
                       "has length %zd",
          pad->ElementBytes(), elementBytes);
    }
  }

  // Extract and check the optional ORDER= argument, which must be a
  // permutation of [1..resultRank].
  int dimOrder[maxRank];
  if (order) {
    RUNTIME_CHECK(terminator, order->rank() == 1);
    RUNTIME_CHECK(terminator, order->type().IsInteger());
    if (order->GetDimension(0).Extent() != resultRank) {
      terminator.Crash("RESHAPE: the extent of ORDER (%jd) must match the rank"
                       " of the SHAPE (%d)",
          static_cast<std::intmax_t>(order->GetDimension(0).Extent()),
          resultRank);
    }
    std::uint64_t values{0};
    SubscriptValue orderSubscript{order->GetDimension(0).LowerBound()};
    std::size_t orderElementBytes{order->ElementBytes()};
    for (SubscriptValue j{0}; j < resultRank; ++j, ++orderSubscript) {
      auto k{GetInt64Safe(order->Element<char>(&orderSubscript),
          orderElementBytes, terminator)};
      if (!k) {
        terminator.Crash("RESHAPE: ORDER element value exceeds 64 bits");
      } else if (*k < 1 || *k > resultRank || ((values >> *k) & 1)) {
        terminator.Crash("RESHAPE: bad value for ORDER element (%jd)",
            static_cast<std::intmax_t>(*k));
      }
      values |= std::uint64_t{1} << *k;
      dimOrder[j] = *k - 1;
    }
  } else {
    for (int j{0}; j < resultRank; ++j) {
      dimOrder[j] = j;
    }
  }

  // Allocate result descriptor
  AllocateResult(
      result, source, resultRank, resultExtent, terminator, "RESHAPE");

  // Populate the result's elements.
  SubscriptValue resultSubscript[maxRank];
  result.GetLowerBounds(resultSubscript);
  SubscriptValue sourceSubscript[maxRank];
  source.GetLowerBounds(sourceSubscript);
  std::size_t resultElement{0};
  std::size_t elementsFromSource{std::min(resultElements, sourceElements)};
  for (; resultElement < elementsFromSource; ++resultElement) {
    CopyElement(result, resultSubscript, source, sourceSubscript, terminator);
    source.IncrementSubscripts(sourceSubscript);
    result.IncrementSubscripts(resultSubscript, dimOrder);
  }
  if (resultElement < resultElements) {
    // Remaining elements come from the optional PAD= argument.
    SubscriptValue padSubscript[maxRank];
    pad->GetLowerBounds(padSubscript);
    for (; resultElement < resultElements; ++resultElement) {
      CopyElement(result, resultSubscript, *pad, padSubscript, terminator);
      pad->IncrementSubscripts(padSubscript);
      result.IncrementSubscripts(resultSubscript, dimOrder);
    }
  }
}

// SPREAD
void RTDEF(Spread)(Descriptor &result, const Descriptor &source, int dim,
    std::int64_t ncopies, const char *sourceFile, int line) {
  Terminator terminator{sourceFile, line};
  int rank{source.rank() + 1};
  RUNTIME_CHECK(terminator, rank <= maxRank);
  if (dim < 1 || dim > rank) {
    terminator.Crash("SPREAD: DIM=%d argument for rank-%d source array "
                     "must be greater than 1 and less than or equal to %d",
        dim, rank - 1, rank);
  }
  ncopies = std::max<std::int64_t>(ncopies, 0);
  SubscriptValue extent[maxRank];
  int k{0};
  for (int j{0}; j < rank; ++j) {
    extent[j] = j == dim - 1 ? ncopies : source.GetDimension(k++).Extent();
  }
  AllocateResult(result, source, rank, extent, terminator, "SPREAD");
  SubscriptValue resultAt[maxRank];
  for (int j{0}; j < rank; ++j) {
    resultAt[j] = 1;
  }
  SubscriptValue &resultDim{resultAt[dim - 1]};
  SubscriptValue sourceAt[maxRank];
  source.GetLowerBounds(sourceAt);
  for (std::size_t n{result.Elements()}; n > 0; n -= ncopies) {
    for (resultDim = 1; resultDim <= ncopies; ++resultDim) {
      CopyElement(result, resultAt, source, sourceAt, terminator);
    }
    result.IncrementSubscripts(resultAt);
    source.IncrementSubscripts(sourceAt);
  }
}

// TRANSPOSE
void RTDEF(Transpose)(Descriptor &result, const Descriptor &matrix,
    const char *sourceFile, int line) {
  Terminator terminator{sourceFile, line};
  RUNTIME_CHECK(terminator, matrix.rank() == 2);
  SubscriptValue extent[2]{
      matrix.GetDimension(1).Extent(), matrix.GetDimension(0).Extent()};
  AllocateResult(result, matrix, 2, extent, terminator, "TRANSPOSE");
  SubscriptValue resultAt[2]{1, 1};
  SubscriptValue matrixLB[2];
  matrix.GetLowerBounds(matrixLB);
  for (std::size_t n{result.Elements()}; n-- > 0;
       result.IncrementSubscripts(resultAt)) {
    SubscriptValue matrixAt[2]{
        matrixLB[0] + resultAt[1] - 1, matrixLB[1] + resultAt[0] - 1};
    CopyElement(result, resultAt, matrix, matrixAt, terminator);
  }
}

// UNPACK
void RTDEF(Unpack)(Descriptor &result, const Descriptor &vector,
    const Descriptor &mask, const Descriptor &field, const char *sourceFile,
    int line) {
  Terminator terminator{sourceFile, line};
  RUNTIME_CHECK(terminator, vector.rank() == 1);
  int rank{mask.rank()};
  RUNTIME_CHECK(terminator, rank > 0);
  SubscriptValue extent[maxRank];
  mask.GetShape(extent);
  CheckConformability(mask, field, terminator, "UNPACK", "MASK=", "FIELD=");
  std::size_t elementLen{
      AllocateResult(result, field, rank, extent, terminator, "UNPACK")};
  RUNTIME_CHECK(terminator, vector.type() == field.type());
  if (vector.ElementBytes() != elementLen) {
    terminator.Crash(
        "UNPACK: VECTOR= has element byte length %zd but FIELD= has length %zd",
        vector.ElementBytes(), elementLen);
  }
  SubscriptValue resultAt[maxRank], maskAt[maxRank], fieldAt[maxRank],
      vectorAt{vector.GetDimension(0).LowerBound()};
  for (int j{0}; j < rank; ++j) {
    resultAt[j] = 1;
  }
  mask.GetLowerBounds(maskAt);
  field.GetLowerBounds(fieldAt);
  SubscriptValue vectorElements{vector.GetDimension(0).Extent()};
  SubscriptValue vectorLeft{vectorElements};
  for (std::size_t n{result.Elements()}; n-- > 0;) {
    if (IsLogicalElementTrue(mask, maskAt)) {
      if (vectorLeft-- == 0) {
        terminator.Crash(
            "UNPACK: VECTOR= argument has fewer elements (%d) than "
            "MASK= has .TRUE. entries",
            vectorElements);
      }
      CopyElement(result, resultAt, vector, &vectorAt, terminator);
      ++vectorAt;
    } else {
      CopyElement(result, resultAt, field, fieldAt, terminator);
    }
    result.IncrementSubscripts(resultAt);
    mask.IncrementSubscripts(maskAt);
    field.IncrementSubscripts(fieldAt);
  }
}

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime
