//===- NVPTX.cpp - TargetInfo for NVPTX -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LowerTypes.h"
#include "TargetInfo.h"
#include "TargetLoweringInfo.h"

namespace cir {

//===----------------------------------------------------------------------===//
// NVPTX ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

class NVPTXABIInfo : public ABIInfo {
public:
  NVPTXABIInfo(LowerTypes &lt) : ABIInfo(lt) {}
};

class NVPTXTargetLoweringInfo : public TargetLoweringInfo {
public:
  NVPTXTargetLoweringInfo(LowerTypes &lt)
      : TargetLoweringInfo(std::make_unique<NVPTXABIInfo>(lt)) {}
};

} // namespace

std::unique_ptr<TargetLoweringInfo>
createNVPTXTargetLoweringInfo(LowerModule &lm) {
  return std::make_unique<NVPTXTargetLoweringInfo>(lm.getTypes());
}

} // namespace cir