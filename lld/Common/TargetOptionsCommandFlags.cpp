//===-- TargetOptionsCommandFlags.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lld/Common/TargetOptionsCommandFlags.h"

#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/Target/TargetOptions.h"

static llvm::codegen::RegisterCodeGenFlags CGF;

llvm::TargetOptions lld::initTargetOptionsFromCodeGenFlags() {
  return llvm::codegen::InitTargetOptionsFromCodeGenFlags(llvm::Triple());
}

llvm::Optional<llvm::Reloc::Model> lld::getRelocModelFromCMModel() {
  return llvm::codegen::getExplicitRelocModel();
}

llvm::Optional<llvm::CodeModel::Model> lld::getCodeModelFromCMModel() {
  return llvm::codegen::getExplicitCodeModel();
}

std::string lld::getCPUStr() { return llvm::codegen::getCPUStr(); }

std::vector<std::string> lld::getMAttrs() { return llvm::codegen::getMAttrs(); }
