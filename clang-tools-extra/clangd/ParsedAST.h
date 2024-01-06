//===--- ParsedAST.h - Building translation units ----------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes building a file as if it were open in clangd, and defines
// the ParsedAST structure that holds the results.
//
// This is similar to a clang -fsyntax-only run that produces a clang AST, but
// we have several customizations:
//  - preamble handling
//  - capturing diagnostics for later access
//  - running clang-tidy checks
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_PARSEDAST_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_PARSEDAST_H

#include "CollectMacros.h"
#include "Compiler.h"
#include "Diagnostics.h"
#include "Headers.h"
#include "Preamble.h"
#include "clang-include-cleaner/Record.h"
#include "support/Path.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace clang {
class Sema;
namespace clangd {
class HeuristicResolver;

/// Stores and provides access to parsed AST.
class ParsedAST {
public:
  /// Attempts to run Clang and store the parsed AST.
  /// If \p Preamble is non-null it is reused during parsing.
  /// This function does not check if preamble is valid to reuse.
  static std::optional<ParsedAST>
  build(llvm::StringRef Filename, const ParseInputs &Inputs,
        std::unique_ptr<clang::CompilerInvocation> CI,
        llvm::ArrayRef<Diag> CompilerInvocationDiags,
        std::shared_ptr<const PreambleData> Preamble);

  ParsedAST(ParsedAST &&Other);
  ParsedAST &operator=(ParsedAST &&Other);

  ~ParsedAST();

  /// Note that the returned ast will not contain decls from the preamble that
  /// were not deserialized during parsing. Clients should expect only decls
  /// from the main file to be in the AST.
  ASTContext &getASTContext();
  const ASTContext &getASTContext() const;

  Sema &getSema();

  Preprocessor &getPreprocessor();
  std::shared_ptr<Preprocessor> getPreprocessorPtr();
  const Preprocessor &getPreprocessor() const;

  SourceManager &getSourceManager() {
    return getASTContext().getSourceManager();
  }
  const SourceManager &getSourceManager() const {
    return getASTContext().getSourceManager();
  }

  const LangOptions &getLangOpts() const {
    return getASTContext().getLangOpts();
  }

  /// This function returns top-level decls present in the main file of the AST.
  /// The result does not include the decls that come from the preamble.
  /// (These should be const, but RecursiveASTVisitor requires Decl*).
  ArrayRef<Decl *> getLocalTopLevelDecls();
  ArrayRef<const Decl *> getLocalTopLevelDecls() const;

  llvm::ArrayRef<Diag> getDiagnostics() const;

  /// Returns the estimated size of the AST and the accessory structures, in
  /// bytes. Does not include the size of the preamble.
  std::size_t getUsedBytes() const;
  const IncludeStructure &getIncludeStructure() const;

  /// Gets all macro references (definition, expansions) present in the main
  /// file, including those in the preamble region.
  const MainFileMacros &getMacros() const;
  /// Gets all pragma marks in the main file.
  const std::vector<PragmaMark> &getMarks() const;
  /// Tokens recorded while parsing the main file.
  /// (!) does not have tokens from the preamble.
  const syntax::TokenBuffer &getTokens() const { return Tokens; }
  /// Returns the PramaIncludes for preamble + main file includes.
  const include_cleaner::PragmaIncludes &getPragmaIncludes() const;

  /// Returns the version of the ParseInputs this AST was built from.
  llvm::StringRef version() const { return Version; }

  /// Returns the path passed by the caller when building this AST.
  PathRef tuPath() const { return TUPath; }

  /// Returns the version of the ParseInputs used to build Preamble part of this
  /// AST. Might be std::nullopt if no Preamble is used.
  std::optional<llvm::StringRef> preambleVersion() const;

  const HeuristicResolver *getHeuristicResolver() const {
    return Resolver.get();
  }

private:
  ParsedAST(PathRef TUPath, llvm::StringRef Version,
            std::shared_ptr<const PreambleData> Preamble,
            std::unique_ptr<CompilerInstance> Clang,
            std::unique_ptr<FrontendAction> Action, syntax::TokenBuffer Tokens,
            MainFileMacros Macros, std::vector<PragmaMark> Marks,
            std::vector<Decl *> LocalTopLevelDecls, std::vector<Diag> Diags,
            IncludeStructure Includes, include_cleaner::PragmaIncludes PI);
  Path TUPath;
  std::string Version;
  // In-memory preambles must outlive the AST, it is important that this member
  // goes before Clang and Action.
  std::shared_ptr<const PreambleData> Preamble;
  // We store an "incomplete" FrontendAction (i.e. no EndSourceFile was called
  // on it) and CompilerInstance used to run it. That way we don't have to do
  // complex memory management of all Clang structures on our own. (They are
  // stored in CompilerInstance and cleaned up by
  // FrontendAction.EndSourceFile).
  std::unique_ptr<CompilerInstance> Clang;
  std::unique_ptr<FrontendAction> Action;
  /// Tokens recorded after the preamble finished.
  ///   - Includes all spelled tokens for the main file.
  ///   - Includes expanded tokens produced **after** preamble.
  ///   - Does not have spelled or expanded tokens for files from preamble.
  syntax::TokenBuffer Tokens;

  /// All macro definitions and expansions in the main file.
  MainFileMacros Macros;
  // Pragma marks in the main file.
  std::vector<PragmaMark> Marks;
  // Diags emitted while parsing this AST (including preamble and compiler
  // invocation).
  std::vector<Diag> Diags;
  // Top-level decls inside the current file. Not that this does not include
  // top-level decls from the preamble.
  std::vector<Decl *> LocalTopLevelDecls;
  IncludeStructure Includes;
  include_cleaner::PragmaIncludes PI;
  std::unique_ptr<HeuristicResolver> Resolver;
};

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_PARSEDAST_H
