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
//  - running clang-tidy checks checks
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_PARSEDAST_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_PARSEDAST_H

#include "CollectMacros.h"
#include "Compiler.h"
#include "Diagnostics.h"
#include "Headers.h"
#include "Path.h"
#include "Preamble.h"
#include "index/CanonicalIncludes.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/PrecompiledPreamble.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/ArrayRef.h"
#include <memory>
#include <string>
#include <vector>

namespace clang {
namespace clangd {
class SymbolIndex;

/// Stores and provides access to parsed AST.
class ParsedAST {
public:
  /// Attempts to run Clang and store parsed AST. If \p Preamble is non-null
  /// it is reused during parsing.
  static llvm::Optional<ParsedAST>
  build(std::unique_ptr<clang::CompilerInvocation> CI,
        llvm::ArrayRef<Diag> CompilerInvocationDiags,
        std::shared_ptr<const PreambleData> Preamble,
        std::unique_ptr<llvm::MemoryBuffer> Buffer,
        llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS,
        const SymbolIndex *Index, const ParseOptions &Opts);

  ParsedAST(ParsedAST &&Other);
  ParsedAST &operator=(ParsedAST &&Other);

  ~ParsedAST();

  /// Note that the returned ast will not contain decls from the preamble that
  /// were not deserialized during parsing. Clients should expect only decls
  /// from the main file to be in the AST.
  ASTContext &getASTContext();
  const ASTContext &getASTContext() const;

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

  const std::vector<Diag> &getDiagnostics() const;

  /// Returns the esitmated size of the AST and the accessory structures, in
  /// bytes. Does not include the size of the preamble.
  std::size_t getUsedBytes() const;
  const IncludeStructure &getIncludeStructure() const;
  const CanonicalIncludes &getCanonicalIncludes() const;

  /// Gets all macro references (definition, expansions) present in the main
  /// file, including those in the preamble region.
  const MainFileMacros &getMacros() const;
  /// Tokens recorded while parsing the main file.
  /// (!) does not have tokens from the preamble.
  const syntax::TokenBuffer &getTokens() const { return Tokens; }

private:
  ParsedAST(std::shared_ptr<const PreambleData> Preamble,
            std::unique_ptr<CompilerInstance> Clang,
            std::unique_ptr<FrontendAction> Action, syntax::TokenBuffer Tokens,
            MainFileMacros Macros, std::vector<Decl *> LocalTopLevelDecls,
            std::vector<Diag> Diags, IncludeStructure Includes,
            CanonicalIncludes CanonIncludes);

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
  ///   - Includes expanded tokens produced **after** preabmle.
  ///   - Does not have spelled or expanded tokens for files from preamble.
  syntax::TokenBuffer Tokens;

  /// All macro definitions and expansions in the main file.
  MainFileMacros Macros;
  // Data, stored after parsing.
  std::vector<Diag> Diags;
  // Top-level decls inside the current file. Not that this does not include
  // top-level decls from the preamble.
  std::vector<Decl *> LocalTopLevelDecls;
  IncludeStructure Includes;
  CanonicalIncludes CanonIncludes;
};

/// Build an AST from provided user inputs. This function does not check if
/// preamble can be reused, as this function expects that \p Preamble is the
/// result of calling buildPreamble.
llvm::Optional<ParsedAST>
buildAST(PathRef FileName, std::unique_ptr<CompilerInvocation> Invocation,
         llvm::ArrayRef<Diag> CompilerInvocationDiags,
         const ParseInputs &Inputs,
         std::shared_ptr<const PreambleData> Preamble);

/// For testing/debugging purposes. Note that this method deserializes all
/// unserialized Decls, so use with care.
void dumpAST(ParsedAST &AST, llvm::raw_ostream &OS);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_PARSEDAST_H
