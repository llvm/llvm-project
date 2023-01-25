//===-- lib/Semantics/target.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/target.h"
#include "flang/Common/template.h"
#include "flang/Evaluate/common.h"
#include "flang/Evaluate/type.h"

namespace Fortran::evaluate {

Rounding TargetCharacteristics::defaultRounding;

TargetCharacteristics::TargetCharacteristics() {
  auto enableCategoryKinds{[this](TypeCategory category) {
    for (int kind{0}; kind < maxKind; ++kind) {
      if (CanSupportType(category, kind)) {
        auto byteSize{static_cast<std::size_t>(kind)};
        if (category == TypeCategory::Real ||
            category == TypeCategory::Complex) {
          if (kind == 3) {
            // non-IEEE 16-bit format (truncated 32-bit)
            byteSize = 2;
          } else if (kind == 10) {
            // x87 floating-point
            // Follow gcc precedent for "long double"
            byteSize = 16;
          }
        }
        std::size_t align{byteSize};
        if (category == TypeCategory::Complex) {
          byteSize = 2 * byteSize;
        }
        EnableType(category, kind, byteSize, align);
      }
    }
  }};
  enableCategoryKinds(TypeCategory::Integer);
  enableCategoryKinds(TypeCategory::Real);
  enableCategoryKinds(TypeCategory::Complex);
  enableCategoryKinds(TypeCategory::Character);
  enableCategoryKinds(TypeCategory::Logical);

  isBigEndian_ = !isHostLittleEndian;

  areSubnormalsFlushedToZero_ = false;
}

bool TargetCharacteristics::CanSupportType(
    TypeCategory category, std::int64_t kind) {
  return IsValidKindOfIntrinsicType(category, kind);
}

bool TargetCharacteristics::EnableType(common::TypeCategory category,
    std::int64_t kind, std::size_t byteSize, std::size_t align) {
  if (CanSupportType(category, kind)) {
    byteSize_[static_cast<int>(category)][kind] = byteSize;
    align_[static_cast<int>(category)][kind] = align;
    maxByteSize_ = std::max(maxByteSize_, byteSize);
    maxAlignment_ = std::max(maxAlignment_, align);
    return true;
  } else {
    return false;
  }
}

void TargetCharacteristics::DisableType(
    common::TypeCategory category, std::int64_t kind) {
  if (kind >= 0 && kind < maxKind) {
    align_[static_cast<int>(category)][kind] = 0;
  }
}

std::size_t TargetCharacteristics::GetByteSize(
    common::TypeCategory category, std::int64_t kind) const {
  if (kind >= 0 && kind < maxKind) {
    return byteSize_[static_cast<int>(category)][kind];
  } else {
    return 0;
  }
}

std::size_t TargetCharacteristics::GetAlignment(
    common::TypeCategory category, std::int64_t kind) const {
  if (kind >= 0 && kind < maxKind) {
    return align_[static_cast<int>(category)][kind];
  } else {
    return 0;
  }
}

bool TargetCharacteristics::IsTypeEnabled(
    common::TypeCategory category, std::int64_t kind) const {
  return GetAlignment(category, kind) > 0;
}

void TargetCharacteristics::set_isBigEndian(bool isBig) {
  isBigEndian_ = isBig;
}

void TargetCharacteristics::set_areSubnormalsFlushedToZero(bool yes) {
  areSubnormalsFlushedToZero_ = yes;
}

void TargetCharacteristics::set_roundingMode(Rounding rounding) {
  roundingMode_ = rounding;
}

// SELECTED_INT_KIND() -- F'2018 16.9.169
class SelectedIntKindVisitor {
public:
  SelectedIntKindVisitor(
      const TargetCharacteristics &targetCharacteristics, std::int64_t p)
      : targetCharacteristics_{targetCharacteristics}, precision_{p} {}
  using Result = std::optional<int>;
  using Types = IntegerTypes;
  template <typename T> Result Test() const {
    if (Scalar<T>::RANGE >= precision_ &&
        targetCharacteristics_.IsTypeEnabled(T::category, T::kind)) {
      return T::kind;
    } else {
      return std::nullopt;
    }
  }

private:
  const TargetCharacteristics &targetCharacteristics_;
  std::int64_t precision_;
};

int TargetCharacteristics::SelectedIntKind(std::int64_t precision) const {
  if (auto kind{
          common::SearchTypes(SelectedIntKindVisitor{*this, precision})}) {
    return *kind;
  } else {
    return -1;
  }
}

// SELECTED_REAL_KIND() -- F'2018 16.9.170
class SelectedRealKindVisitor {
public:
  SelectedRealKindVisitor(const TargetCharacteristics &targetCharacteristics,
      std::int64_t p, std::int64_t r)
      : targetCharacteristics_{targetCharacteristics}, precision_{p}, range_{
                                                                          r} {}
  using Result = std::optional<int>;
  using Types = RealTypes;
  template <typename T> Result Test() const {
    if (Scalar<T>::PRECISION >= precision_ && Scalar<T>::RANGE >= range_ &&
        targetCharacteristics_.IsTypeEnabled(T::category, T::kind)) {
      return {T::kind};
    } else {
      return std::nullopt;
    }
  }

private:
  const TargetCharacteristics &targetCharacteristics_;
  std::int64_t precision_, range_;
};

int TargetCharacteristics::SelectedRealKind(
    std::int64_t precision, std::int64_t range, std::int64_t radix) const {
  if (radix != 2) {
    return -5;
  }
  if (auto kind{common::SearchTypes(
          SelectedRealKindVisitor{*this, precision, range})}) {
    return *kind;
  }
  // No kind has both sufficient precision and sufficient range.
  // The negative return value encodes whether any kinds exist that
  // could satisfy either constraint independently.
  bool pOK{common::SearchTypes(SelectedRealKindVisitor{*this, precision, 0})};
  bool rOK{common::SearchTypes(SelectedRealKindVisitor{*this, 0, range})};
  if (pOK) {
    if (rOK) {
      return -4;
    } else {
      return -2;
    }
  } else {
    if (rOK) {
      return -1;
    } else {
      return -3;
    }
  }
}

} // namespace Fortran::evaluate
