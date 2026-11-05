//===- bolt/Core/BranchLivenessInfo.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_CORE_BRANCHLIVENESSINFO_H
#define BOLT_CORE_BRANCHLIVENESSINFO_H

namespace llvm {
class MCInst;

namespace bolt {
class BinaryFunction;

class BranchLivenessInfo {
  BinaryFunction *BF;
  unsigned AnnotationIndex;

  void swap(BranchLivenessInfo &Other) noexcept;

public:
  explicit BranchLivenessInfo(BinaryFunction &BF);
  ~BranchLivenessInfo();

  // Copies would create multiple owners for removing the same annotations.
  BranchLivenessInfo(const BranchLivenessInfo &) = delete;
  BranchLivenessInfo &operator=(const BranchLivenessInfo &) = delete;

  BranchLivenessInfo(BranchLivenessInfo &&Other) noexcept;
  BranchLivenessInfo &operator=(BranchLivenessInfo &&Other) noexcept;

  bool mustPreserveFlags(const MCInst &Inst) const;
  void removeAnnotation(MCInst &Inst) const;
  void setFlagsDead(MCInst &Inst);
};

} // namespace bolt
} // namespace llvm

#endif
