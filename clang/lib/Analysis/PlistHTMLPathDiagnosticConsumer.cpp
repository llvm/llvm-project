//===--- PlistHTMLDiagnostics.cpp - The Plist-HTML Diagnostic Consumer. ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This diagnostic consumer produces both the HTML output and the Plist output.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathDiagnostic.h"
#include "clang/Analysis/PathDiagnosticConsumers.h"
#include "clang/Basic/SourceManager.h"
#include "clang/CrossTU/CrossTranslationUnit.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang;
using namespace ento;

void ento::createPlistHTMLDiagnosticConsumer(
    PathDiagnosticConsumerOptions DiagOpts, PathDiagnosticConsumers &C,
    const std::string &Prefix, const Preprocessor &PP,
    const cross_tu::CrossTranslationUnitContext &CTU) {
  createHTMLDiagnosticConsumer(
      DiagOpts, C, std::string(llvm::sys::path::parent_path(Prefix)), PP, CTU);
  createPlistMultiFileDiagnosticConsumer(DiagOpts, C, Prefix, PP, CTU);
  createTextMinimalPathDiagnosticConsumer(std::move(DiagOpts), C, Prefix, PP,
                                          CTU);
}
