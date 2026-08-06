//===- bolt/Core/BranchLivenessInfo.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/BranchLivenessInfo.h"
#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "llvm/MC/MCInst.h"
#include <cassert>
#include <utility>

namespace llvm {
namespace bolt {

BranchLivenessInfo::BranchLivenessInfo(BinaryFunction &BF)
    : BF(&BF),
      AnnotationIndex(
          BF.getBinaryContext().MIB->getOrCreateAnnotationIndex("DeadFlags")) {}

BranchLivenessInfo::~BranchLivenessInfo() {
  if (!BF)
    return;

  MCPlusBuilder &MIB = *BF->getBinaryContext().MIB;
  for (BinaryBasicBlock &BB : *BF)
    for (MCInst &Inst : BB)
      MIB.removeAnnotation(Inst, AnnotationIndex);
}

BranchLivenessInfo::BranchLivenessInfo(BranchLivenessInfo &&Other) noexcept
    : BF(nullptr), AnnotationIndex(0) {
  swap(Other);
}

BranchLivenessInfo &
BranchLivenessInfo::operator=(BranchLivenessInfo &&Other) noexcept {
  BranchLivenessInfo Tmp(std::move(Other));
  swap(Tmp);
  return *this;
}

void BranchLivenessInfo::swap(BranchLivenessInfo &Other) noexcept {
  std::swap(BF, Other.BF);
  std::swap(AnnotationIndex, Other.AnnotationIndex);
}

bool BranchLivenessInfo::mustPreserveFlags(const MCInst &Inst) const {
  if (!BF)
    return true;

  return !BF->getBinaryContext().MIB->hasAnnotation(Inst, AnnotationIndex);
}

void BranchLivenessInfo::removeAnnotation(MCInst &Inst) const {
  assert(BF && "branch liveness info is not initialized");

  BF->getBinaryContext().MIB->removeAnnotation(Inst, AnnotationIndex);
}

void BranchLivenessInfo::setFlagsDead(MCInst &Inst) {
  assert(BF && "branch liveness info is not initialized");

  MCPlusBuilder &MIB = *BF->getBinaryContext().MIB;
  if (!MIB.hasAnnotation(Inst, AnnotationIndex))
    MIB.addAnnotation(Inst, AnnotationIndex, true);
}

} // namespace bolt
} // namespace llvm
