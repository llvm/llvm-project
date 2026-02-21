//===- X86.cpp - TargetInfo for X86 -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABIInfo.h"
#include "LowerModule.h"
#include "LowerTypes.h"
#include "TargetInfo.h"
#include "clang/CIR/MissingFeatures.h"
#include <memory>

namespace cir {

//===----------------------------------------------------------------------===//
// X86 ABI Implementation
//===----------------------------------------------------------------------===//

class X86_64ABIInfo : public ABIInfo {

public:
  X86_64ABIInfo(LowerTypes &lt, X86AVXABILevel avxLevel) : ABIInfo(lt) {}
};

class X86_64TargetLoweringInfo : public TargetLoweringInfo {
public:
  X86_64TargetLoweringInfo(LowerTypes &lt, X86AVXABILevel avxLevel)
      : TargetLoweringInfo(std::make_unique<X86_64ABIInfo>(lt, avxLevel)) {
    assert(!cir::MissingFeatures::swift());
  }
};

std::unique_ptr<TargetLoweringInfo>
createX86_64TargetLoweringInfo(LowerModule &lm, X86AVXABILevel avxLevel) {
  return std::make_unique<X86_64TargetLoweringInfo>(lm.getTypes(), avxLevel);
}

} // namespace cir
