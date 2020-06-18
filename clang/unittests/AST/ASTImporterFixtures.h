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
#include "clang/AST/ASTImporterSharedState.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Testing/CommandLineArgs.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

#include "DeclMatcher.h"

#include <sstream>

namespace clang {

class ASTImporter;
class ASTImporterSharedState;
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
  virtual std::vector<std::string> getExtraArgs() const { return {}; }

  // Returns the argument vector used for a specific language option, this set
  // can be tweaked by the test parameters.
  std::vector<std::string>
  getCommandLineArgsForLanguage(TestLanguage Lang) const {
    std::vector<std::string> Args = getCommandLineArgsForTesting(Lang);
    std::vector<std::string> ExtraArgs = getExtraArgs();
    for (const auto &Arg : ExtraArgs) {
      Args.push_back(Arg);
    }
    return Args;
  }
};

const auto DefaultTestArrayForRunOptions =
    std::array<std::vector<std::string>, 4>{
        {std::vector<std::string>(),
         std::vector<std::string>{"-fdelayed-template-parsing"},
         std::vector<std::string>{"-fms-compatibility"},
         std::vector<std::string>{"-fdelayed-template-parsing",
                                  "-fms-compatibility"}}};

const auto DefaultTestValuesForRunOptions =
    ::testing::ValuesIn(DefaultTestArrayForRunOptions);

// This class provides generic methods to write tests which can check internal
// attributes of AST nodes like getPreviousDecl(), isVirtual(), etc. Also,
// this fixture makes it possible to import from several "From" contexts.
class ASTImporterTestBase : public CompilerOptionSpecificTest {

  const char *const InputFileName = "input.cc";
  const char *const OutputFileName = "output.cc";

public:
  /// Allocates an ASTImporter (or one of its subclasses).
  typedef std::function<ASTImporter *(
      ASTContext &, FileManager &, ASTContext &, FileManager &, bool,
      const std::shared_ptr<ASTImporterSharedState> &SharedState)>
      ImporterConstructor;

  // ODR handling type for the AST importer.
  ASTImporter::ODRHandlingType ODRHandling;

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
    ASTImporter::ODRHandlingType ODRHandling;

    TU(StringRef Code, StringRef FileName, std::vector<std::string> Args,
       ImporterConstructor C = ImporterConstructor(),
       ASTImporter::ODRHandlingType ODRHandling =
           ASTImporter::ODRHandlingType::Conservative);
    ~TU();

    void
    lazyInitImporter(const std::shared_ptr<ASTImporterSharedState> &SharedState,
                     ASTUnit *ToAST);
    Decl *import(const std::shared_ptr<ASTImporterSharedState> &SharedState,
                 ASTUnit *ToAST, Decl *FromDecl);
    llvm::Expected<Decl *>
    importOrError(const std::shared_ptr<ASTImporterSharedState> &SharedState,
                  ASTUnit *ToAST, Decl *FromDecl);
    QualType import(const std::shared_ptr<ASTImporterSharedState> &SharedState,
                    ASTUnit *ToAST, QualType FromType);
  };

  // We may have several From contexts and related translation units. In each
  // AST, the buffers for the source are handled via references and are set
  // during the creation of the AST. These references must point to a valid
  // buffer until the AST is alive. Thus, we must use a list in order to avoid
  // moving of the stored objects because that would mean breaking the
  // references in the AST. By using a vector a move could happen when the
  // vector is expanding, with the list we won't have these issues.
  std::list<TU> FromTUs;

  // Initialize the shared state if not initialized already.
  void lazyInitSharedState(TranslationUnitDecl *ToTU);

  void lazyInitToAST(TestLanguage ToLang, StringRef ToSrcCode,
                     StringRef FileName);

protected:
  std::shared_ptr<ASTImporterSharedState> SharedStatePtr;

public:
  // We may have several From context but only one To context.
  std::unique_ptr<ASTUnit> ToAST;

  // Returns with the TU associated with the given Decl.
  TU *findFromTU(Decl *From);

  // Creates an AST both for the From and To source code and imports the Decl
  // of the identifier into the To context.
  // Must not be called more than once within the same test.
  std::tuple<Decl *, Decl *>
  getImportedDecl(StringRef FromSrcCode, TestLanguage FromLang,
                  StringRef ToSrcCode, TestLanguage ToLang,
                  StringRef Identifier = DeclToImportID);

  // Creates a TU decl for the given source code which can be used as a From
  // context.  May be called several times in a given test (with different file
  // name).
  TranslationUnitDecl *getTuDecl(StringRef SrcCode, TestLanguage Lang,
                                 StringRef FileName = "input.cc");

  // Creates the To context with the given source code and returns the TU decl.
  TranslationUnitDecl *getToTuDecl(StringRef ToSrcCode, TestLanguage ToLang);

  // Import the given Decl into the ToCtx.
  // May be called several times in a given test.
  // The different instances of the param From may have different ASTContext.
  Decl *Import(Decl *From, TestLanguage ToLang);

  template <class DeclT> DeclT *Import(DeclT *From, TestLanguage Lang) {
    return cast_or_null<DeclT>(Import(cast<Decl>(From), Lang));
  }

  // Import the given Decl into the ToCtx.
  // Same as Import but returns the result of the import which can be an error.
  llvm::Expected<Decl *> importOrError(Decl *From, TestLanguage ToLang);

  QualType ImportType(QualType FromType, Decl *TUDecl, TestLanguage ToLang);

  ASTImporterTestBase()
      : ODRHandling(ASTImporter::ODRHandlingType::Conservative) {}
  ~ASTImporterTestBase();
};

class ASTImporterOptionSpecificTestBase
    : public ASTImporterTestBase,
      public ::testing::WithParamInterface<std::vector<std::string>> {
protected:
  std::vector<std::string> getExtraArgs() const override { return GetParam(); }
};

template <class T>
::testing::AssertionResult isSuccess(llvm::Expected<T> &ValOrErr) {
  if (ValOrErr)
    return ::testing::AssertionSuccess() << "Expected<> contains no error.";
  else
    return ::testing::AssertionFailure()
           << "Expected<> contains error: " << toString(ValOrErr.takeError());
}

template <class T>
::testing::AssertionResult isImportError(llvm::Expected<T> &ValOrErr,
                                         ImportError::ErrorKind Kind) {
  if (ValOrErr) {
    return ::testing::AssertionFailure() << "Expected<> is expected to contain "
                                            "error but does contain value \""
                                         << (*ValOrErr) << "\"";
  } else {
    std::ostringstream OS;
    bool Result = false;
    auto Err = llvm::handleErrors(
        ValOrErr.takeError(), [&OS, &Result, Kind](clang::ImportError &IE) {
          if (IE.Error == Kind) {
            Result = true;
            OS << "Expected<> contains an ImportError " << IE.toString();
          } else {
            OS << "Expected<> contains an ImportError " << IE.toString()
               << " instead of kind " << Kind;
          }
        });
    if (Err) {
      OS << "Expected<> contains unexpected error: "
         << toString(std::move(Err));
    }
    if (Result)
      return ::testing::AssertionSuccess() << OS.str();
    else
      return ::testing::AssertionFailure() << OS.str();
  }
}

} // end namespace ast_matchers
} // end namespace clang

#endif
