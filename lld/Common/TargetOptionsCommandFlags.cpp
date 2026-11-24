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
#include "llvm/TargetParser/Triple.h"
#include <optional>

llvm::TargetOptions
lld::getTargetOptions(const llvm::cl::LibraryOptionsContext &libraryOptions) {
  return llvm::codegen::getTargetOptions(libraryOptions, llvm::Triple());
}

llvm::ArrayRef<const llvm::cl::LibraryOptions *> lld::getToolOptions() {
  return llvm::codegen::getToolOptions();
}

std::optional<llvm::Reloc::Model> lld::getRelocModelFromCMModel(
    const llvm::cl::LibraryOptionsContext &libraryOptions) {
  return llvm::codegen::getExplicitRelocModel(libraryOptions);
}

std::optional<llvm::CodeModel::Model> lld::getCodeModelFromCMModel(
    const llvm::cl::LibraryOptionsContext &libraryOptions) {
  return llvm::codegen::getExplicitCodeModel(libraryOptions);
}

std::string
lld::getCPUStr(const llvm::cl::LibraryOptionsContext &libraryOptions) {
  return llvm::codegen::getCPUStr(libraryOptions);
}

std::vector<std::string>
lld::getMAttrs(const llvm::cl::LibraryOptionsContext &libraryOptions) {
  return llvm::codegen::getMAttrs(libraryOptions);
}
