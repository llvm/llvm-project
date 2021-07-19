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

#include "transformational.h"
#include "copy.h"
#include "terminator.h"
#include "tools.h"
#include <algorithm>

namespace Fortran::runtime {

// Utility for CSHIFT & EOSHIFT rank > 1 cases that determines the shift count
// for each of the vector sections of the result.
class ShiftControl {
public:
  ShiftControl(const Descriptor &s, Terminator &t, int dim)
      : shift_{s}, terminator_{t}, shiftRank_{s.rank()}, dim_{dim} {}
  void Init(const Descriptor &source) {
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
          RUNTIME_CHECK(terminator_,
              shiftDim.Extent() == source.GetDimension(j).Extent());
        }
      }
    } else {
      shiftCount_ =
          GetInt64(shift_.OffsetElement<char>(), shiftElemLen_, terminator_);
    }
  }
  SubscriptValue GetShift(const SubscriptValue resultAt[]) const {
    if (shiftRank_ > 0) {
      SubscriptValue shiftAt[maxRank];
      int k{0};
      for (int j{0}; j < shiftRank_ + 1; ++j) {
        if (j + 1 != dim_) {
          shiftAt[k] = lb_[k] + resultAt[j] - 1;
          ++k;
        }
      }
      return GetInt64(
          shift_.Element<char>(shiftAt), shiftElemLen_, terminator_);
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
static void DefaultInitialize(
    const Descriptor &result, Terminator &terminator) {
  auto catAndKind{result.type().GetCategoryAndKind()};
  RUNTIME_CHECK(
      terminator, catAndKind && catAndKind->first != TypeCategory::Derived);
  std::size_t elementLen{result.ElementBytes()};
  std::size_t bytes{result.Elements() * elementLen};
  if (catAndKind->first == TypeCategory::Character) {
    switch (int kind{catAndKind->second}) {
    case 1:
      std::fill_n(result.OffsetElement<char>(), bytes, ' ');
      break;
    case 2:
      std::fill_n(result.OffsetElement<char16_t>(), bytes / 2,
          static_cast<char16_t>(' '));
      break;
    case 4:
      std::fill_n(result.OffsetElement<char32_t>(), bytes / 4,
          static_cast<char32_t>(' '));
      break;
    default:
      terminator.Crash("EOSHIFT: bad CHARACTER kind %d", kind);
    }
  } else {
    std::memset(result.raw().base_addr, 0, bytes);
  }
}

static inline std::size_t AllocateResult(Descriptor &result,
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

extern "C" {

// CSHIFT where rank of ARRAY argument > 1
void RTNAME(Cshift)(Descriptor &result, const Descriptor &source,
    const Descriptor &shift, int dim, const char *sourceFile, int line) {
  Terminator terminator{sourceFile, line};
  int rank{source.rank()};
  RUNTIME_CHECK(terminator, rank > 1);
  RUNTIME_CHECK(terminator, dim >= 1 && dim <= rank);
  ShiftControl shiftControl{shift, terminator, dim};
  shiftControl.Init(source);
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
    if (shiftCount < 0) {
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
void RTNAME(CshiftVector)(Descriptor &result, const Descriptor &source,
    std::int64_t shift, const char *sourceFile, int line) {
  Terminator terminator{sourceFile, line};
  RUNTIME_CHECK(terminator, source.rank() == 1);
  const Dimension &sourceDim{source.GetDimension(0)};
  SubscriptValue extent{sourceDim.Extent()};
  AllocateResult(result, source, 1, &extent, terminator, "CSHIFT");
  SubscriptValue lb{sourceDim.LowerBound()};
  for (SubscriptValue j{0}; j < extent; ++j) {
    SubscriptValue resultAt{1 + j};
    SubscriptValue sourceAt{lb + (j + shift) % extent};
    if (sourceAt < 0) {
      sourceAt += extent;
    }
    CopyElement(result, &resultAt, source, &sourceAt, terminator);
  }
}

// EOSHIFT of rank > 1
void RTNAME(Eoshift)(Descriptor &result, const Descriptor &source,
    const Descriptor &shift, const Descriptor *boundary, int dim,
    const char *sourceFile, int line) {
  Terminator terminator{sourceFile, line};
  SubscriptValue extent[maxRank];
  int rank{source.GetShape(extent)};
  RUNTIME_CHECK(terminator, rank > 1);
  RUNTIME_CHECK(terminator, dim >= 1 && dim <= rank);
  std::size_t elementLen{
      AllocateResult(result, source, rank, extent, terminator, "EOSHIFT")};
  int boundaryRank{-1};
  if (boundary) {
    boundaryRank = boundary->rank();
    RUNTIME_CHECK(terminator, boundaryRank == 0 || boundaryRank == rank - 1);
    RUNTIME_CHECK(terminator,
        boundary->type() == source.type() &&
            boundary->ElementBytes() == elementLen);
    if (boundaryRank > 0) {
      int k{0};
      for (int j{0}; j < rank; ++j) {
        if (j != dim - 1) {
          RUNTIME_CHECK(
              terminator, boundary->GetDimension(k).Extent() == extent[j]);
          ++k;
        }
      }
    }
  }
  ShiftControl shiftControl{shift, terminator, dim};
  shiftControl.Init(source);
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
void RTNAME(EoshiftVector)(Descriptor &result, const Descriptor &source,
    std::int64_t shift, const Descriptor *boundary, const char *sourceFile,
    int line) {
  Terminator terminator{sourceFile, line};
  RUNTIME_CHECK(terminator, source.rank() == 1);
  SubscriptValue extent{source.GetDimension(0).Extent()};
  std::size_t elementLen{
      AllocateResult(result, source, 1, &extent, terminator, "EOSHIFT")};
  if (boundary) {
    RUNTIME_CHECK(terminator, boundary->rank() == 0);
    RUNTIME_CHECK(terminator,
        boundary->type() == source.type() &&
            boundary->ElementBytes() == elementLen);
  }
  if (!boundary) {
    DefaultInitialize(result, terminator);
  }
  SubscriptValue lb{source.GetDimension(0).LowerBound()};
  for (SubscriptValue j{1}; j <= extent; ++j) {
    SubscriptValue sourceAt{lb + j - 1 + shift};
    if (sourceAt >= lb && sourceAt < lb + extent) {
      CopyElement(result, &j, source, &sourceAt, terminator);
    }
  }
}

// PACK
void RTNAME(Pack)(Descriptor &result, const Descriptor &source,
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
    RUNTIME_CHECK(terminator,
        source.type() == vector->type() &&
            source.ElementBytes() == vector->ElementBytes());
    extent = vector->GetDimension(0).Extent();
    RUNTIME_CHECK(terminator, extent >= trues);
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
void RTNAME(Reshape)(Descriptor &result, const Descriptor &source,
    const Descriptor &shape, const Descriptor *pad, const Descriptor *order,
    const char *sourceFile, int line) {
  // Compute and check the rank of the result.
  Terminator terminator{sourceFile, line};
  RUNTIME_CHECK(terminator, shape.rank() == 1);
  RUNTIME_CHECK(terminator, shape.type().IsInteger());
  SubscriptValue resultRank{shape.GetDimension(0).Extent()};
  RUNTIME_CHECK(terminator,
      resultRank >= 0 && resultRank <= static_cast<SubscriptValue>(maxRank));

  // Extract and check the shape of the result; compute its element count.
  SubscriptValue resultExtent[maxRank];
  std::size_t shapeElementBytes{shape.ElementBytes()};
  std::size_t resultElements{1};
  SubscriptValue shapeSubscript{shape.GetDimension(0).LowerBound()};
  for (SubscriptValue j{0}; j < resultRank; ++j, ++shapeSubscript) {
    resultExtent[j] = GetInt64(
        shape.Element<char>(&shapeSubscript), shapeElementBytes, terminator);
    RUNTIME_CHECK(terminator, resultExtent[j] >= 0);
    resultElements *= resultExtent[j];
  }

  // Check that there are sufficient elements in the SOURCE=, or that
  // the optional PAD= argument is present and nonempty.
  std::size_t elementBytes{source.ElementBytes()};
  std::size_t sourceElements{source.Elements()};
  std::size_t padElements{pad ? pad->Elements() : 0};
  if (resultElements < sourceElements) {
    RUNTIME_CHECK(terminator, padElements > 0);
    RUNTIME_CHECK(terminator, pad->ElementBytes() == elementBytes);
  }

  // Extract and check the optional ORDER= argument, which must be a
  // permutation of [1..resultRank].
  int dimOrder[maxRank];
  if (order) {
    RUNTIME_CHECK(terminator, order->rank() == 1);
    RUNTIME_CHECK(terminator, order->type().IsInteger());
    RUNTIME_CHECK(terminator, order->GetDimension(0).Extent() == resultRank);
    std::uint64_t values{0};
    SubscriptValue orderSubscript{order->GetDimension(0).LowerBound()};
    std::size_t orderElementBytes{order->ElementBytes()};
    for (SubscriptValue j{0}; j < resultRank; ++j, ++orderSubscript) {
      auto k{GetInt64(order->Element<char>(&orderSubscript), orderElementBytes,
          terminator)};
      RUNTIME_CHECK(
          terminator, k >= 1 && k <= resultRank && !((values >> k) & 1));
      values |= std::uint64_t{1} << k;
      dimOrder[k - 1] = j;
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
void RTNAME(Spread)(Descriptor &result, const Descriptor &source, int dim,
    std::int64_t ncopies, const char *sourceFile, int line) {
  Terminator terminator{sourceFile, line};
  int rank{source.rank() + 1};
  RUNTIME_CHECK(terminator, rank <= maxRank);
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
void RTNAME(Transpose)(Descriptor &result, const Descriptor &matrix,
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
void RTNAME(Unpack)(Descriptor &result, const Descriptor &vector,
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
  RUNTIME_CHECK(terminator,
      vector.type() == field.type() && vector.ElementBytes() == elementLen);
  SubscriptValue resultAt[maxRank], maskAt[maxRank], fieldAt[maxRank],
      vectorAt{vector.GetDimension(0).LowerBound()};
  for (int j{0}; j < rank; ++j) {
    resultAt[j] = 1;
  }
  mask.GetLowerBounds(maskAt);
  field.GetLowerBounds(fieldAt);
  SubscriptValue vectorLeft{vector.GetDimension(0).Extent()};
  for (std::size_t n{result.Elements()}; n-- > 0;) {
    if (IsLogicalElementTrue(mask, maskAt)) {
      if (vectorLeft-- == 0) {
        terminator.Crash("UNPACK: VECTOR= argument has fewer elements than "
                         "MASK= has .TRUE. entries");
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

} // extern "C"
} // namespace Fortran::runtime
