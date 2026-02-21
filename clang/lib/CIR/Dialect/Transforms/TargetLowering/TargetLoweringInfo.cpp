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

TargetLoweringInfo::TargetLoweringInfo(std::unique_ptr<ABIInfo> info)
    : info(std::move(info)) {}

TargetLoweringInfo::~TargetLoweringInfo() = default;

cir::SyncScopeKind
TargetLoweringInfo::convertSyncScope(cir::SyncScopeKind syncScope) const {
  // By default, targets don't deal with sync scopes other than system scope.
  return cir::SyncScopeKind::System;
}

} // namespace cir
