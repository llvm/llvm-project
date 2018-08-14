//===--- TestTU.cpp - Scratch source files for testing --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TestTU.h"
#include "TestFS.h"
#include "index/FileIndex.h"
#include "index/MemIndex.h"
#include "clang/AST/RecursiveASTVisitor.h"
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
  Cmd.insert(Cmd.end(), ExtraArgs.begin(), ExtraArgs.end());
  auto AST = ParsedAST::build(
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
  llvm::SmallVector<llvm::StringRef, 4> Components;
  QName.split(Components, "::");

  auto &Ctx = AST.getASTContext();
  auto LookupDecl = [&Ctx](const DeclContext &Scope,
                           llvm::StringRef Name) -> const NamedDecl & {
    auto LookupRes = Scope.lookup(DeclarationName(&Ctx.Idents.get(Name)));
    assert(!LookupRes.empty() && "Lookup failed");
    assert(LookupRes.size() == 1 && "Lookup returned multiple results");
    return *LookupRes.front();
  };

  const DeclContext *Scope = Ctx.getTranslationUnitDecl();
  for (auto NameIt = Components.begin(), End = Components.end() - 1;
       NameIt != End; ++NameIt) {
    Scope = &cast<DeclContext>(LookupDecl(*Scope, *NameIt));
  }
  return LookupDecl(*Scope, Components.back());
}

const NamedDecl &findAnyDecl(ParsedAST &AST,
                             std::function<bool(const NamedDecl &)> Callback) {
  struct Visitor : RecursiveASTVisitor<Visitor> {
    decltype(Callback) CB;
    llvm::SmallVector<const NamedDecl *, 1> Decls;
    bool VisitNamedDecl(const NamedDecl *ND) {
      if (CB(*ND))
        Decls.push_back(ND);
      return true;
    }
  } Visitor;
  Visitor.CB = Callback;
  for (Decl *D : AST.getLocalTopLevelDecls())
    Visitor.TraverseDecl(D);
  if (Visitor.Decls.size() != 1) {
    ADD_FAILURE() << Visitor.Decls.size() << " symbols matched.";
    assert(Visitor.Decls.size() == 1);
  }
  return *Visitor.Decls.front();
}

const NamedDecl &findAnyDecl(ParsedAST &AST, llvm::StringRef Name) {
  return findAnyDecl(AST, [Name](const NamedDecl &ND) {
    if (auto *ID = ND.getIdentifier())
      if (ID->getName() == Name)
        return true;
    return false;
  });
}

} // namespace clangd
} // namespace clang
