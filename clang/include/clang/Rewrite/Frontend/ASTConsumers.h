//===--- ASTConsumers.h - ASTConsumer implementations -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// AST Consumers.
//
//===----------------------------------------------------------------------===//

#ifndef REWRITE_ASTCONSUMERS_H
#define REWRITE_ASTCONSUMERS_H

#include "clang/Basic/LLVM.h"
#include <string>

namespace clang {

class ASTConsumer;
class DiagnosticsEngine;
class LangOptions;
class Preprocessor;

// ObjC rewriter: attempts to rewrite ObjC constructs into pure C code.
// This is considered experimental, and only works with Apple's ObjC runtime.
ASTConsumer *CreateObjCRewriter(const std::string &InFile,
                                raw_ostream *OS,
                                DiagnosticsEngine &Diags,
                                const LangOptions &LOpts,
                                bool SilenceRewriteMacroWarning);
ASTConsumer *CreateModernObjCRewriter(const std::string &InFile,
                                raw_ostream *OS,
                                DiagnosticsEngine &Diags,
                                const LangOptions &LOpts,
                                bool SilenceRewriteMacroWarning);

/// CreateHTMLPrinter - Create an AST consumer which rewrites source code to
/// HTML with syntax highlighting suitable for viewing in a web-browser.
ASTConsumer *CreateHTMLPrinter(raw_ostream *OS, Preprocessor &PP,
                               bool SyntaxHighlight = true,
                               bool HighlightMacros = true);

} // end clang namespace

#endif
