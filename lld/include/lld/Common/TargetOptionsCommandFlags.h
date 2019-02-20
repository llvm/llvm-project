//===-- TargetOptionsCommandFlags.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper to create TargetOptions from command line flags.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Optional.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetOptions.h"

namespace lld {
llvm::TargetOptions InitTargetOptionsFromCodeGenFlags();
llvm::Optional<llvm::CodeModel::Model> GetCodeModelFromCMModel();
std::string GetCPUStr();
std::vector<std::string> GetMAttrs();
}
