//===---- TargetLoweringInfo.h - Encapsulate target details -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics the TargetCodeGenInfo class from the file
// clang/lib/CodeGen/TargetInfo.h.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_TARGETLOWERINGINFO_H
#define LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_TARGETLOWERINGINFO_H

#include "ABIInfo.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"

namespace cir {

class TargetLoweringInfo {
private:
  std::unique_ptr<ABIInfo> info;

public:
  TargetLoweringInfo(std::unique_ptr<ABIInfo> info);
  virtual ~TargetLoweringInfo();

  const ABIInfo &getABIInfo() const { return *info; }

  virtual cir::SyncScopeKind
  convertSyncScope(cir::SyncScopeKind syncScope) const;
};

} // namespace cir

#endif
