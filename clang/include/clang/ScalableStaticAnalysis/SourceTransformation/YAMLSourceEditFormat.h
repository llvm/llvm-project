//===- YAMLSourceEditFormat.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Built-in YAML source-edit writer. The on-disk layout is the existing
// `clang::tooling::TranslationUnitReplacements` YAML schema, byte-for-byte
// consumable by `clang-apply-replacements`.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_YAMLSOURCEEDITFORMAT_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_YAMLSOURCEEDITFORMAT_H

#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace clang::ssaf {

/// Writes \p Doc to \p Path as a YAML document compatible with
/// `clang-apply-replacements`.
llvm::Error
writeYAMLSourceEdits(const clang::tooling::TranslationUnitReplacements &Doc,
                     llvm::StringRef Path);

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_YAMLSOURCEEDITFORMAT_H
