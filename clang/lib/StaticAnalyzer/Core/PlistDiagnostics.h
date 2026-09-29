//==- PlistDiagnostics.h - Plist Diagnostics for Paths -------------*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_STATICANALYZER_CORE_PLISTDIAGNOSTICS_H
#define LLVM_CLANG_LIB_STATICANALYZER_CORE_PLISTDIAGNOSTICS_H

#include "clang/CrossTU/CrossTranslationUnit.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/StaticAnalyzer/Core/PathDiagnosticConsumers.h"
#include <string>

namespace clang::ento {

void createPlistDiagnosticConsumerImpl(
    PathDiagnosticConsumerOptions DiagOpts, PathDiagnosticConsumers &C,
    const std::string &Output, const Preprocessor &PP,
    const cross_tu::CrossTranslationUnitContext &CTU,
    const MacroExpansionContext &MacroExpansions, bool SupportsMultipleFiles);

} // namespace clang::ento

#endif
