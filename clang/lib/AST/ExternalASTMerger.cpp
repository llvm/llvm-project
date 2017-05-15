//===- ExternalASTMerger.cpp - Merging External AST Interface ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the ExternalASTMerger, which vends a combination of
//  ASTs from several different ASTContext/FileManager pairs
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ExternalASTMerger.h"

using namespace clang;

namespace {

template <typename T> struct Source {
  T t;
  Source(T t) : t(t) {}
  operator T() { return t; }
  template <typename U = T> U &get() { return t; }
  template <typename U = T> const U &get() const { return t; }
  template <typename U> operator Source<U>() { return Source<U>(t); }
};

typedef std::pair<Source<NamedDecl *>, ASTImporter *> Candidate;

class LazyASTImporter : public ASTImporter {
public:
  LazyASTImporter(ASTContext &ToContext, FileManager &ToFileManager,
                  ASTContext &FromContext, FileManager &FromFileManager)
      : ASTImporter(ToContext, ToFileManager, FromContext, FromFileManager,
                    /*MinimalImport=*/true) {}
  Decl *Imported(Decl *From, Decl *To) override {
    if (auto ToTag = dyn_cast<TagDecl>(To)) {
      ToTag->setHasExternalLexicalStorage();
    } else if (auto ToNamespace = dyn_cast<NamespaceDecl>(To)) {
      ToNamespace->setHasExternalVisibleStorage();
    }
    return ASTImporter::Imported(From, To);
  }
};

Source<const DeclContext *>
LookupSameContext(Source<TranslationUnitDecl *> SourceTU, const DeclContext *DC,
                  ASTImporter &ReverseImporter) {
  if (DC->isTranslationUnit()) {
    return SourceTU;
  }
  Source<const DeclContext *> SourceParentDC =
      LookupSameContext(SourceTU, DC->getParent(), ReverseImporter);
  if (!SourceParentDC) {
    // If we couldn't find the parent DC in this TranslationUnit, give up.
    return nullptr;
  }
  auto ND = cast<NamedDecl>(DC);
  DeclarationName Name = ND->getDeclName();
  Source<DeclarationName> SourceName = ReverseImporter.Import(Name);
  DeclContext::lookup_result SearchResult =
      SourceParentDC.get()->lookup(SourceName.get());
  size_t SearchResultSize = SearchResult.size();
  // Handle multiple candidates once we have a test for it.
  // This may turn up when we import template specializations correctly.
  assert(SearchResultSize < 2);
  if (SearchResultSize == 0) {
    // couldn't find the name, so we have to give up
    return nullptr;
  } else {
    NamedDecl *SearchResultDecl = SearchResult[0];
    return dyn_cast<DeclContext>(SearchResultDecl);
  }
}

bool IsForwardDeclaration(Decl *D) {
  assert(!isa<ObjCInterfaceDecl>(D)); // TODO handle this case
  if (auto TD = dyn_cast<TagDecl>(D)) {
    return !TD->isThisDeclarationADefinition();
  } else if (auto FD = dyn_cast<FunctionDecl>(D)) {
    return !FD->isThisDeclarationADefinition();
  } else {
    return false;
  }
}

template <typename CallbackType>
void ForEachMatchingDC(
    const DeclContext *DC,
    llvm::ArrayRef<ExternalASTMerger::ImporterPair> Importers,
    CallbackType Callback) {
  for (const ExternalASTMerger::ImporterPair &IP : Importers) {
    Source<TranslationUnitDecl *> SourceTU =
        IP.Forward->getFromContext().getTranslationUnitDecl();
    if (auto SourceDC = LookupSameContext(SourceTU, DC, *IP.Reverse))
      Callback(IP, SourceDC);
  }
}

bool HasDeclOfSameType(llvm::ArrayRef<Candidate> Decls, const Candidate &C) {
  return llvm::any_of(Decls, [&](const Candidate &D) {
    return C.first.get()->getKind() == D.first.get()->getKind();
  });
}
} // end namespace

ExternalASTMerger::ExternalASTMerger(const ImporterEndpoint &Target,
                                     llvm::ArrayRef<ImporterEndpoint> Sources) {
  for (const ImporterEndpoint &S : Sources) {
    Importers.push_back(
        {llvm::make_unique<LazyASTImporter>(Target.AST, Target.FM, S.AST, S.FM),
         llvm::make_unique<ASTImporter>(S.AST, S.FM, Target.AST, Target.FM,
                                        /*MinimalImport=*/true)});
  }
}

bool ExternalASTMerger::FindExternalVisibleDeclsByName(const DeclContext *DC,
                                                       DeclarationName Name) {
  llvm::SmallVector<NamedDecl *, 1> Decls;
  llvm::SmallVector<Candidate, 4> CompleteDecls;
  llvm::SmallVector<Candidate, 4> ForwardDecls;

  auto FilterFoundDecl = [&CompleteDecls, &ForwardDecls](const Candidate &C) {
    if (IsForwardDeclaration(C.first.get())) {
      if (!HasDeclOfSameType(ForwardDecls, C)) {
        ForwardDecls.push_back(C);
      }
    } else {
      CompleteDecls.push_back(C);
    }
  };

  ForEachMatchingDC(
      DC, Importers,
      [&](const ImporterPair &IP, Source<const DeclContext *> SourceDC) {
        DeclarationName FromName = IP.Reverse->Import(Name);
        DeclContextLookupResult Result = SourceDC.get()->lookup(FromName);
        for (NamedDecl *FromD : Result) {
          FilterFoundDecl(std::make_pair(FromD, IP.Forward.get()));
        }
      });

  llvm::ArrayRef<Candidate> DeclsToReport =
      CompleteDecls.empty() ? ForwardDecls : CompleteDecls;

  if (DeclsToReport.empty()) {
    return false;
  }

  Decls.reserve(DeclsToReport.size());
  for (const Candidate &C : DeclsToReport) {
    NamedDecl *d = cast<NamedDecl>(C.second->Import(C.first.get()));
    assert(d);
    Decls.push_back(d);
  }
  SetExternalVisibleDeclsForName(DC, Name, Decls);
  return true;
}

void ExternalASTMerger::FindExternalLexicalDecls(
    const DeclContext *DC, llvm::function_ref<bool(Decl::Kind)> IsKindWeWant,
    SmallVectorImpl<Decl *> &Result) {
  ForEachMatchingDC(
      DC, Importers,
      [&](const ImporterPair &IP, Source<const DeclContext *> SourceDC) {
        for (const Decl *SourceDecl : SourceDC.get()->decls()) {
          if (IsKindWeWant(SourceDecl->getKind())) {
            Decl *ImportedDecl =
                IP.Forward->Import(const_cast<Decl *>(SourceDecl));
            assert(ImportedDecl->getDeclContext() == DC);
            (void)ImportedDecl;
          }
        }
      });
}

void ExternalASTMerger::CompleteType(TagDecl *Tag) {
  SmallVector<Decl *, 0> Result;
  FindExternalLexicalDecls(Tag, [](Decl::Kind) { return true; }, Result);
  Tag->setHasExternalLexicalStorage(false);
}
