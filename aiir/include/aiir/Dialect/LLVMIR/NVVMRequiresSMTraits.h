//===--- NVVMRequiresSMTraits.h - NVVM Requires SM Traits -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines op traits for the NVVM Dialect in AIIR
//
//===----------------------------------------------------------------------===//

#ifndef NVVM_DIALECT_NVVM_IR_NVVMREQUIRESSMTRAITS_H_
#define NVVM_DIALECT_NVVM_IR_NVVMREQUIRESSMTRAITS_H_

#include "aiir/IR/OpDefinition.h"
#include "aiir/IR/StorageUniquerSupport.h"
#include "llvm/ADT/StringExtras.h"

namespace aiir {

namespace NVVM {

// Struct to store and check compatibility of SM versions.
struct NVVMCheckSMVersion {
  // Set to true if the SM version is accelerated (e.g., sm_90a).
  bool archAccelerated;

  // List of SM versions.
  // Typically only has one version except for cases where multiple
  // arch-accelerated versions are supported.
  // For example, tcgen05.shift is supported on sm_100a, sm_101a, and sm_103a.
  llvm::SmallVector<int, 1> smVersionList;

  template <typename... Ints>
  NVVMCheckSMVersion(bool archAccelerated, Ints... smVersions)
      : archAccelerated(archAccelerated), smVersionList({smVersions...}) {
    assert((archAccelerated || smVersionList.size() == 1) &&
           "non arch-accelerated SM version list must be a single version!");
  }

  bool isCompatibleWith(const NVVMCheckSMVersion &targetSM) const {
    assert(targetSM.smVersionList.size() == 1 &&
           "target SM version list must be a single version!");

    if (archAccelerated) {
      if (!targetSM.archAccelerated)
        return false;

      for (auto version : smVersionList) {
        if (version == targetSM.smVersionList[0])
          return true;
      }
    } else {
      return targetSM.smVersionList[0] >= smVersionList[0];
    }

    return false;
  }

  bool isMinimumSMVersion() const { return smVersionList[0] >= 20; }

  // Parses an SM version string and returns an equivalent NVVMCheckSMVersion
  // object.
  static const NVVMCheckSMVersion
  getTargetSMVersionFromStr(StringRef smVersionString) {
    bool isAA = smVersionString.back() == 'a';

    int smVersionInt;
    smVersionString.drop_front(3)
        .take_while([](char c) { return llvm::isDigit(c); })
        .getAsInteger(10, smVersionInt);

    return NVVMCheckSMVersion(isAA, smVersionInt);
  }
};

} // namespace NVVM
} // namespace aiir

#include "aiir/Dialect/LLVMIR/NVVMRequiresSMTraits.h.inc"

namespace aiir {

namespace OpTrait {

template <int MinVersion>
class NVVMRequiresSM {
public:
  template <typename ConcreteOp>
  class Impl
      : public OpTrait::TraitBase<ConcreteOp, NVVMRequiresSM<MinVersion>::Impl>,
        public aiir::NVVM::RequiresSMInterface::Trait<ConcreteOp> {
  public:
    const NVVM::NVVMCheckSMVersion getRequiredMinSMVersion() const {
      return NVVM::NVVMCheckSMVersion(false, MinVersion);
    }
  };
};

template <int... SMVersions>
class NVVMRequiresSMa {
public:
  template <typename ConcreteOp>
  class Impl : public OpTrait::TraitBase<ConcreteOp,
                                         NVVMRequiresSMa<SMVersions...>::Impl>,
               public aiir::NVVM::RequiresSMInterface::Trait<ConcreteOp> {
  public:
    const NVVM::NVVMCheckSMVersion getRequiredMinSMVersion() const {
      return NVVM::NVVMCheckSMVersion(true, SMVersions...);
    }
  };
};

} // namespace OpTrait
} // namespace aiir
#endif // NVVM_DIALECT_NVVM_IR_NVVMREQUIRESSMTRAITS_H_
