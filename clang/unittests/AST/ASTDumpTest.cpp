//===- unittests/AST/ASTDumpTest.cpp --- Declaration tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests Decl::dump().
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "gtest/gtest.h"

using namespace clang;

namespace clang {
namespace ast {

namespace {
/// An ExternalASTSource that asserts if it is queried for information about
/// any declaration.
class TrappingExternalASTSource : public ExternalASTSource {
  ~TrappingExternalASTSource() override = default;
  bool FindExternalVisibleDeclsByName(const DeclContext *,
                                      DeclarationName) override {
    assert(false && "Unexpected call to FindExternalVisibleDeclsByName");
    return true;
  }

  void FindExternalLexicalDecls(const DeclContext *,
                                llvm::function_ref<bool(Decl::Kind)>,
                                SmallVectorImpl<Decl *> &) override {
    assert(false && "Unexpected call to FindExternalLexicalDecls");
  }

  void completeVisibleDeclsMap(const DeclContext *) override {
    assert(false && "Unexpected call to completeVisibleDeclsMap");
  }

  void CompleteRedeclChain(const Decl *) override {
    assert(false && "Unexpected call to CompleteRedeclChain");
  }

  void CompleteType(TagDecl *) override {
    assert(false && "Unexpected call to CompleteType(Tag Decl*)");
  }

  void CompleteType(ObjCInterfaceDecl *) override {
    assert(false && "Unexpected call to CompleteType(ObjCInterfaceDecl *)");
  }
};

/// Tests that Decl::dump doesn't load additional declarations from the
/// ExternalASTSource.
class ExternalASTSourceDumpTest : public ::testing::Test {
protected:
  ExternalASTSourceDumpTest()
      : FileMgr(FileMgrOpts), DiagID(new DiagnosticIDs()),
        Diags(DiagID, new DiagnosticOptions, new IgnoringDiagConsumer()),
        SourceMgr(Diags, FileMgr), Idents(LangOpts, nullptr),
        Ctxt(LangOpts, SourceMgr, Idents, Sels, Builtins) {
    Ctxt.setExternalSource(new TrappingExternalASTSource());
  }

  FileSystemOptions FileMgrOpts;
  FileManager FileMgr;
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID;
  DiagnosticsEngine Diags;
  SourceManager SourceMgr;
  LangOptions LangOpts;
  IdentifierTable Idents;
  SelectorTable Sels;
  Builtin::Context Builtins;
  ASTContext Ctxt;
};
} // unnamed namespace

/// Set all flags that activate queries to the ExternalASTSource.
static void setExternalStorageFlags(DeclContext *DC) {
  DC->setHasExternalLexicalStorage();
  DC->setHasExternalVisibleStorage();
  DC->setMustBuildLookupTable();
}

/// Dumps the given Decl.
static void dumpDecl(Decl *D) {
  // Try dumping the decl which shouldn't trigger any calls to the
  // ExternalASTSource.

  std::string Out;
  llvm::raw_string_ostream OS(Out);
  D->dump(OS);
}

TEST_F(ExternalASTSourceDumpTest, DumpObjCInterfaceDecl) {
  // Define an Objective-C interface.
  ObjCInterfaceDecl *I = ObjCInterfaceDecl::Create(
      Ctxt, Ctxt.getTranslationUnitDecl(), SourceLocation(),
      &Ctxt.Idents.get("c"), nullptr, nullptr);
  Ctxt.getTranslationUnitDecl()->addDecl(I);

  setExternalStorageFlags(I);
  dumpDecl(I);
}

TEST_F(ExternalASTSourceDumpTest, DumpRecordDecl) {
  // Define a struct.
  RecordDecl *R = RecordDecl::Create(
      Ctxt, TagDecl::TagKind::TTK_Class, Ctxt.getTranslationUnitDecl(),
      SourceLocation(), SourceLocation(), &Ctxt.Idents.get("c"));
  R->startDefinition();
  R->completeDefinition();
  Ctxt.getTranslationUnitDecl()->addDecl(R);

  setExternalStorageFlags(R);
  dumpDecl(R);
}

TEST_F(ExternalASTSourceDumpTest, DumpCXXRecordDecl) {
  // Define a class.
  CXXRecordDecl *R = CXXRecordDecl::Create(
      Ctxt, TagDecl::TagKind::TTK_Class, Ctxt.getTranslationUnitDecl(),
      SourceLocation(), SourceLocation(), &Ctxt.Idents.get("c"));
  R->startDefinition();
  R->completeDefinition();
  Ctxt.getTranslationUnitDecl()->addDecl(R);

  setExternalStorageFlags(R);
  dumpDecl(R);
}

} // end namespace ast
} // end namespace clang
