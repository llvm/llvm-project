//===-- include/flang/Evaluate/target.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Represents the minimal amount of target architecture information required by
// semantics.

#ifndef FORTRAN_EVALUATE_TARGET_H_
#define FORTRAN_EVALUATE_TARGET_H_

#include "flang/Common/Fortran.h"
#include "flang/Common/enum-class.h"
#include "flang/Common/enum-set.h"
#include "flang/Common/target-rounding.h"
#include "flang/Evaluate/common.h"
#include <cstdint>

namespace Fortran::evaluate {

using common::Rounding;

ENUM_CLASS(IeeeFeature, Denormal, Divide, Flags, Halting, Inf, Io, NaN,
    Rounding, Sqrt, Standard, Subnormal, UnderflowControl)

using IeeeFeatures = common::EnumSet<IeeeFeature, 16>;

class TargetCharacteristics {
public:
  TargetCharacteristics();
  TargetCharacteristics &operator=(const TargetCharacteristics &) = default;

  bool isBigEndian() const { return isBigEndian_; }
  void set_isBigEndian(bool isBig = true);

  bool haltingSupportIsUnknownAtCompileTime() const {
    return haltingSupportIsUnknownAtCompileTime_;
  }
  void set_haltingSupportIsUnknownAtCompileTime(bool yes = true) {
    haltingSupportIsUnknownAtCompileTime_ = yes;
  }

  bool areSubnormalsFlushedToZero() const {
    return areSubnormalsFlushedToZero_;
  }
  void set_areSubnormalsFlushedToZero(bool yes = true);

  // Check if a given real kind has flushing control.
  bool hasSubnormalFlushingControl(int kind) const;
  // Check if any or all real kinds have flushing control.
  bool hasSubnormalFlushingControl(bool any = false) const;
  void set_hasSubnormalFlushingControl(int kind, bool yes = true);

  Rounding roundingMode() const { return roundingMode_; }
  void set_roundingMode(Rounding);

  void set_ieeeFeature(IeeeFeature ieeeFeature, bool yes = true) {
    if (yes) {
      ieeeFeatures_.set(ieeeFeature);
    } else {
      ieeeFeatures_.reset(ieeeFeature);
    }
  }

  std::size_t procedurePointerByteSize() const {
    return procedurePointerByteSize_;
  }
  std::size_t procedurePointerAlignment() const {
    return procedurePointerAlignment_;
  }
  std::size_t descriptorAlignment() const { return descriptorAlignment_; }
  std::size_t maxByteSize() const { return maxByteSize_; }
  std::size_t maxAlignment() const { return maxAlignment_; }

  static bool CanSupportType(common::TypeCategory, std::int64_t kind);
  bool EnableType(common::TypeCategory category, std::int64_t kind,
      std::size_t byteSize, std::size_t align);
  void DisableType(common::TypeCategory category, std::int64_t kind);

  std::size_t GetByteSize(
      common::TypeCategory category, std::int64_t kind) const;
  std::size_t GetAlignment(
      common::TypeCategory category, std::int64_t kind) const;
  bool IsTypeEnabled(common::TypeCategory category, std::int64_t kind) const;

  int SelectedIntKind(std::int64_t precision = 0) const;
  int SelectedLogicalKind(std::int64_t bits = 1) const;
  int SelectedRealKind(std::int64_t precision = 0, std::int64_t range = 0,
      std::int64_t radix = 2) const;

  static Rounding defaultRounding;

  const std::string &compilerOptionsString() const {
    return compilerOptionsString_;
  };
  TargetCharacteristics &set_compilerOptionsString(std::string x) {
    compilerOptionsString_ = x;
    return *this;
  }

  const std::string &compilerVersionString() const {
    return compilerVersionString_;
  };
  TargetCharacteristics &set_compilerVersionString(std::string x) {
    compilerVersionString_ = x;
    return *this;
  }

  bool isPPC() const { return isPPC_; }
  void set_isPPC(bool isPPC = false);

  bool isSPARC() const { return isSPARC_; }
  void set_isSPARC(bool isSPARC = false);

  bool isOSWindows() const { return isOSWindows_; }
  void set_isOSWindows(bool isOSWindows = false) {
    isOSWindows_ = isOSWindows;
  };

  IeeeFeatures &ieeeFeatures() { return ieeeFeatures_; }
  const IeeeFeatures &ieeeFeatures() const { return ieeeFeatures_; }

private:
  static constexpr int maxKind{16};
  std::uint8_t byteSize_[common::TypeCategory_enumSize][maxKind + 1]{};
  std::uint8_t align_[common::TypeCategory_enumSize][maxKind + 1]{};
  bool isBigEndian_{false};
  bool isPPC_{false};
  bool isSPARC_{false};
  bool isOSWindows_{false};
  bool haltingSupportIsUnknownAtCompileTime_{false};
  bool areSubnormalsFlushedToZero_{false};
  bool hasSubnormalFlushingControl_[maxKind + 1]{};
  Rounding roundingMode_{defaultRounding};
  std::size_t procedurePointerByteSize_{8};
  std::size_t procedurePointerAlignment_{8};
  std::size_t descriptorAlignment_{8};
  std::size_t maxByteSize_{8 /*at least*/};
  std::size_t maxAlignment_{8 /*at least*/};
  std::string compilerOptionsString_;
  std::string compilerVersionString_;
  IeeeFeatures ieeeFeatures_{IeeeFeature::Denormal, IeeeFeature::Divide,
      IeeeFeature::Flags, IeeeFeature::Inf, IeeeFeature::Io, IeeeFeature::NaN,
      IeeeFeature::Rounding, IeeeFeature::Sqrt, IeeeFeature::Standard,
      IeeeFeature::Subnormal, IeeeFeature::UnderflowControl};
};

} // namespace Fortran::evaluate
#endif // FORTRAN_EVALUATE_TARGET_H_
