//===--- NoLintFixes.h ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_NOLINTFIXES_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_NOLINTFIXES_H

#include "../clang-tidy/ClangTidyDiagnosticConsumer.h"
#include "../clang-tidy/ClangTidyModule.h"
#include "Diagnostics.h"
#include "FeatureModule.h"
#include "clang/Basic/Diagnostic.h"
#include <cassert>
#include <vector>

namespace clang {
namespace clangd {

/// Suggesting to insert "\\ NOLINTNEXTLINE(...)" to suppress clang-tidy
/// diagnostics.
std::vector<Fix>
clangTidyNoLintFixes(const clang::tidy::ClangTidyContext &CTContext,
                     const clang::Diagnostic &Info, const Diag &Diag);

/// Check if a fix created by clangTidyNoLintFixes().
bool isClangTidyNoLintFixes(const Fix &F);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_NOLINTFIXES_H
