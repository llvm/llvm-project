//===--- TestTU.cpp - Scratch source files for testing ------------*-
//C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
#include "TestTU.h"
#include "TestFS.h"
#include "index/FileIndex.h"
#include "index/MemIndex.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/PCHContainerOperations.h"
#include "clang/Frontend/Utils.h"

namespace clang {
namespace clangd {
using namespace llvm;

ParsedAST TestTU::build() const {
  std::string FullFilename = testPath(Filename),
              FullHeaderName = testPath(HeaderFilename);
  std::vector<const char *> Cmd = {"clang", FullFilename.c_str()};
  // FIXME: this shouldn't need to be conditional, but it breaks a
  // GoToDefinition test for some reason (getMacroArgExpandedLocation fails).
  if (!HeaderCode.empty()) {
    Cmd.push_back("-include");
    Cmd.push_back(FullHeaderName.c_str());
  }
  auto AST = ParsedAST::Build(
      createInvocationFromCommandLine(Cmd), nullptr,
      MemoryBuffer::getMemBufferCopy(Code),
      std::make_shared<PCHContainerOperations>(),
      buildTestFS({{FullFilename, Code}, {FullHeaderName, HeaderCode}}));
  if (!AST.hasValue()) {
    ADD_FAILURE() << "Failed to build code:\n" << Code;
    llvm_unreachable("Failed to build TestTU!");
  }
  return std::move(*AST);
}

SymbolSlab TestTU::headerSymbols() const {
  auto AST = build();
  return indexAST(AST.getASTContext(), AST.getPreprocessorPtr());
}

std::unique_ptr<SymbolIndex> TestTU::index() const {
  return MemIndex::build(headerSymbols());
}

// Look up a symbol by qualified name, which must be unique.
const Symbol &findSymbol(const SymbolSlab &Slab, llvm::StringRef QName) {
  const Symbol *Result = nullptr;
  for (const Symbol &S : Slab) {
    if (QName != (S.Scope + S.Name).str())
      continue;
    if (Result) {
      ADD_FAILURE() << "Multiple symbols named " << QName << ":\n"
                    << *Result << "\n---\n"
                    << S;
      assert(false && "QName is not unique");
    }
    Result = &S;
  }
  if (!Result) {
    ADD_FAILURE() << "No symbol named " << QName << " in "
                  << ::testing::PrintToString(Slab);
    assert(false && "No symbol with QName");
  }
  return *Result;
}

const NamedDecl &findDecl(ParsedAST &AST, llvm::StringRef QName) {
  const NamedDecl *Result = nullptr;
  for (const Decl *D : AST.getTopLevelDecls()) {
    auto *ND = dyn_cast<NamedDecl>(D);
    if (!ND || ND->getNameAsString() != QName)
      continue;
    if (Result) {
      ADD_FAILURE() << "Multiple Decls named " << QName;
      assert(false && "QName is not unique");
    }
    Result = ND;
  }
  if (!Result) {
    ADD_FAILURE() << "No Decl named " << QName << " in AST";
    assert(false && "No Decl with QName");
  }
  return *Result;
}

} // namespace clangd
} // namespace clang
