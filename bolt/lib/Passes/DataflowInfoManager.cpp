//===- bolt/Passes/DataflowInfoManager.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the DataflowInfoManager class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/DataflowInfoManager.h"

namespace llvm {
namespace bolt {

ReachingDefOrUse</*Def=*/true> &DataflowInfoManager::getReachingDefs() {
  if (RD)
    return *RD;
  assert(RA && "RegAnalysis required");
  RD.reset(new ReachingDefOrUse<true>(*RA, BF, None, AllocatorId));
  RD->run();
  return *RD;
}

void DataflowInfoManager::invalidateReachingDefs() { RD.reset(nullptr); }

ReachingDefOrUse</*Def=*/false> &DataflowInfoManager::getReachingUses() {
  if (RU)
    return *RU;
  assert(RA && "RegAnalysis required");
  RU.reset(new ReachingDefOrUse<false>(*RA, BF, None, AllocatorId));
  RU->run();
  return *RU;
}

void DataflowInfoManager::invalidateReachingUses() { RU.reset(nullptr); }

LivenessAnalysis &DataflowInfoManager::getLivenessAnalysis() {
  if (LA)
    return *LA;
  assert(RA && "RegAnalysis required");
  LA.reset(new LivenessAnalysis(*RA, BF, AllocatorId));
  LA->run();
  return *LA;
}

void DataflowInfoManager::invalidateLivenessAnalysis() { LA.reset(nullptr); }

StackReachingUses &DataflowInfoManager::getStackReachingUses() {
  if (SRU)
    return *SRU;
  assert(FA && "FrameAnalysis required");
  SRU.reset(new StackReachingUses(*FA, BF, AllocatorId));
  SRU->run();
  return *SRU;
}

void DataflowInfoManager::invalidateStackReachingUses() { SRU.reset(nullptr); }

DominatorAnalysis<false> &DataflowInfoManager::getDominatorAnalysis() {
  if (DA)
    return *DA;
  DA.reset(new DominatorAnalysis<false>(BF, AllocatorId));
  DA->run();
  return *DA;
}

void DataflowInfoManager::invalidateDominatorAnalysis() { DA.reset(nullptr); }

DominatorAnalysis<true> &DataflowInfoManager::getPostDominatorAnalysis() {
  if (PDA)
    return *PDA;
  PDA.reset(new DominatorAnalysis<true>(BF, AllocatorId));
  PDA->run();
  return *PDA;
}

void DataflowInfoManager::invalidatePostDominatorAnalysis() {
  PDA.reset(nullptr);
}

StackPointerTracking &DataflowInfoManager::getStackPointerTracking() {
  if (SPT)
    return *SPT;
  SPT.reset(new StackPointerTracking(BF, AllocatorId));
  SPT->run();
  return *SPT;
}

void DataflowInfoManager::invalidateStackPointerTracking() {
  invalidateStackAllocationAnalysis();
  SPT.reset(nullptr);
}

ReachingInsns<false> &DataflowInfoManager::getReachingInsns() {
  if (RI)
    return *RI;
  RI.reset(new ReachingInsns<false>(BF, AllocatorId));
  RI->run();
  return *RI;
}

void DataflowInfoManager::invalidateReachingInsns() { RI.reset(nullptr); }

ReachingInsns<true> &DataflowInfoManager::getReachingInsnsBackwards() {
  if (RIB)
    return *RIB;
  RIB.reset(new ReachingInsns<true>(BF, AllocatorId));
  RIB->run();
  return *RIB;
}

void DataflowInfoManager::invalidateReachingInsnsBackwards() {
  RIB.reset(nullptr);
}

StackAllocationAnalysis &DataflowInfoManager::getStackAllocationAnalysis() {
  if (SAA)
    return *SAA;
  SAA.reset(
      new StackAllocationAnalysis(BF, getStackPointerTracking(), AllocatorId));
  SAA->run();
  return *SAA;
}

void DataflowInfoManager::invalidateStackAllocationAnalysis() {
  SAA.reset(nullptr);
}

std::unordered_map<const MCInst *, BinaryBasicBlock *> &
DataflowInfoManager::getInsnToBBMap() {
  if (InsnToBB)
    return *InsnToBB;
  InsnToBB.reset(new std::unordered_map<const MCInst *, BinaryBasicBlock *>());
  for (BinaryBasicBlock &BB : BF) {
    for (MCInst &Inst : BB)
      (*InsnToBB)[&Inst] = &BB;
  }
  return *InsnToBB;
}

void DataflowInfoManager::invalidateInsnToBBMap() { InsnToBB.reset(nullptr); }

void DataflowInfoManager::invalidateAll() {
  invalidateReachingDefs();
  invalidateReachingUses();
  invalidateLivenessAnalysis();
  invalidateStackReachingUses();
  invalidateDominatorAnalysis();
  invalidatePostDominatorAnalysis();
  invalidateStackPointerTracking();
  invalidateReachingInsns();
  invalidateReachingInsnsBackwards();
  invalidateStackAllocationAnalysis();
  invalidateInsnToBBMap();
}

} // end namespace bolt
} // end namespace llvm
