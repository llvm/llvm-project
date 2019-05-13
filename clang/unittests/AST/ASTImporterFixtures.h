//===- unittest/AST/ASTImporterFixtures.h - AST unit test support ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Fixture classes for testing the ASTImporter.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_AST_IMPORTER_FIXTURES_H
#define LLVM_CLANG_UNITTESTS_AST_IMPORTER_FIXTURES_H

#include "gmock/gmock.h"

#include "clang/AST/ASTImporter.h"
#include "clang/AST/ASTImporterLookupTable.h"
#include "clang/Frontend/ASTUnit.h"

#include "DeclMatcher.h"
#include "Language.h"

namespace clang {

class ASTImporter;
class ASTImporterLookupTable;
class ASTUnit;

namespace ast_matchers {

const StringRef DeclToImportID = "declToImport";
const StringRef DeclToVerifyID = "declToVerify";

// Creates a virtual file and assigns that to the context of given AST. If the
// file already exists then the file will not be created again as a duplicate.
void createVirtualFileIfNeeded(ASTUnit *ToAST, StringRef FileName,
                               std::unique_ptr<llvm::MemoryBuffer> &&Buffer);

void createVirtualFileIfNeeded(ASTUnit *ToAST, StringRef FileName,
                               StringRef Code);

// Common base for the different families of ASTImporter tests that are
// parameterized on the compiler options which may result a different AST. E.g.
// -fms-compatibility or -fdelayed-template-parsing.
class CompilerOptionSpecificTest : public ::testing::Test {
protected:
  // Return the extra arguments appended to runtime options at compilation.
  virtual ArgVector getExtraArgs() const { return ArgVector(); }

  // Returns the argument vector used for a specific language option, this set
  // can be tweaked by the test parameters.
  ArgVector getArgVectorForLanguage(Language Lang) const {
    ArgVector Args = getBasicRunOptionsForLanguage(Lang);
    ArgVector ExtraArgs = getExtraArgs();
    for (const auto &Arg : ExtraArgs) {
      Args.push_back(Arg);
    }
    return Args;
  }
};

const auto DefaultTestValuesForRunOptions = ::testing::Values(
    ArgVector(), ArgVector{"-fdelayed-template-parsing"},
    ArgVector{"-fms-compatibility"},
    ArgVector{"-fdelayed-template-parsing", "-fms-compatibility"});

// This class provides generic methods to write tests which can check internal
// attributes of AST nodes like getPreviousDecl(), isVirtual(), etc. Also,
// this fixture makes it possible to import from several "From" contexts.
class ASTImporterTestBase : public CompilerOptionSpecificTest {

  const char *const InputFileName = "input.cc";
  const char *const OutputFileName = "output.cc";

public:
  /// Allocates an ASTImporter (or one of its subclasses).
  typedef std::function<ASTImporter *(ASTContext &, FileManager &, ASTContext &,
                                      FileManager &, bool,
                                      ASTImporterLookupTable *)>
      ImporterConstructor;

  // The lambda that constructs the ASTImporter we use in this test.
  ImporterConstructor Creator;

private:
  // Buffer for the To context, must live in the test scope.
  std::string ToCode;

  // Represents a "From" translation unit and holds an importer object which we
  // use to import from this translation unit.
  struct TU {
    // Buffer for the context, must live in the test scope.
    std::string Code;
    std::string FileName;
    std::unique_ptr<ASTUnit> Unit;
    TranslationUnitDecl *TUDecl = nullptr;
    std::unique_ptr<ASTImporter> Importer;
    ImporterConstructor Creator;

    TU(StringRef Code, StringRef FileName, ArgVector Args,
       ImporterConstructor C = ImporterConstructor());
    ~TU();

    void lazyInitImporter(ASTImporterLookupTable &LookupTable, ASTUnit *ToAST);
    Decl *import(ASTImporterLookupTable &LookupTable, ASTUnit *ToAST,
                 Decl *FromDecl);
    QualType import(ASTImporterLookupTable &LookupTable, ASTUnit *ToAST,
                    QualType FromType);
  };

  // We may have several From contexts and related translation units. In each
  // AST, the buffers for the source are handled via references and are set
  // during the creation of the AST. These references must point to a valid
  // buffer until the AST is alive. Thus, we must use a list in order to avoid
  // moving of the stored objects because that would mean breaking the
  // references in the AST. By using a vector a move could happen when the
  // vector is expanding, with the list we won't have these issues.
  std::list<TU> FromTUs;

  // Initialize the lookup table if not initialized already.
  void lazyInitLookupTable(TranslationUnitDecl *ToTU);

  void lazyInitToAST(Language ToLang, StringRef ToSrcCode, StringRef FileName);
  TU *findFromTU(Decl *From);

protected:
  std::unique_ptr<ASTImporterLookupTable> LookupTablePtr;

public:
  // We may have several From context but only one To context.
  std::unique_ptr<ASTUnit> ToAST;

  // Creates an AST both for the From and To source code and imports the Decl
  // of the identifier into the To context.
  // Must not be called more than once within the same test.
  std::tuple<Decl *, Decl *>
  getImportedDecl(StringRef FromSrcCode, Language FromLang, StringRef ToSrcCode,
                  Language ToLang, StringRef Identifier = DeclToImportID);

  // Creates a TU decl for the given source code which can be used as a From
  // context.  May be called several times in a given test (with different file
  // name).
  TranslationUnitDecl *getTuDecl(StringRef SrcCode, Language Lang,
                                 StringRef FileName = "input.cc");

  // Creates the To context with the given source code and returns the TU decl.
  TranslationUnitDecl *getToTuDecl(StringRef ToSrcCode, Language ToLang);

  // Import the given Decl into the ToCtx.
  // May be called several times in a given test.
  // The different instances of the param From may have different ASTContext.
  Decl *Import(Decl *From, Language ToLang);

  template <class DeclT> DeclT *Import(DeclT *From, Language Lang) {
    return cast_or_null<DeclT>(Import(cast<Decl>(From), Lang));
  }

  QualType ImportType(QualType FromType, Decl *TUDecl, Language ToLang);

  ~ASTImporterTestBase();
};

class ASTImporterOptionSpecificTestBase
    : public ASTImporterTestBase,
      public ::testing::WithParamInterface<ArgVector> {
protected:
  ArgVector getExtraArgs() const override { return GetParam(); }
};

} // end namespace ast_matchers
} // end namespace clang

#endif
