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

#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include <string>

namespace cir {

class TargetLoweringInfo {
public:
  virtual ~TargetLoweringInfo();

  virtual std::string getLLVMSyncScope(cir::SyncScopeKind syncScope) const;
};

} // namespace cir

#endif
