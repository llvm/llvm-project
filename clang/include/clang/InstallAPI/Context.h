//===- InstallAPI/Context.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INSTALLAPI_CONTEXT_H
#define LLVM_CLANG_INSTALLAPI_CONTEXT_H

#include "llvm/TextAPI/InterfaceFile.h"
#include "llvm/TextAPI/RecordVisitor.h"
#include "llvm/TextAPI/RecordsSlice.h"

namespace clang {
namespace installapi {

/// Struct used for generating validating InstallAPI.
/// The attributes captured represent all necessary information
/// to generate TextAPI output.
struct InstallAPIContext {

  /// Library attributes that are typically passed as linker inputs.
  llvm::MachO::RecordsSlice::BinaryAttrs BA;

  /// Active target triple to parse.
  llvm::Triple TargetTriple{};

  /// File Path of output location.
  llvm::StringRef OutputLoc{};

  /// What encoding to write output as.
  llvm::MachO::FileType FT = llvm::MachO::FileType::TBD_V5;
};

} // namespace installapi
} // namespace clang

#endif // LLVM_CLANG_INSTALLAPI_CONTEXT_H
