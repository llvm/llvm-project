//===---- TargetLoweringInfo.cpp - Encapsulate target details ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics the TargetCodeGenInfo class from the file
// clang/lib/CodeGen/TargetInfo.cpp.
//
//===----------------------------------------------------------------------===//

#include "TargetLoweringInfo.h"

namespace cir {

TargetLoweringInfo::~TargetLoweringInfo() = default;

std::string
TargetLoweringInfo::getLLVMSyncScope(cir::SyncScopeKind syncScope) const {
  return ""; // default sync scope
}

} // namespace cir
