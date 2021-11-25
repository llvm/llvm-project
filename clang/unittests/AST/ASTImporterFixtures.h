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
#include "MatchVerifier.h"

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

// Base class for those tests which use the family of `testImport` functions.
class TestImportBase
    : public CompilerOptionSpecificTest,
      public ::testing::WithParamInterface<std::vector<std::string>> {

  template <typename NodeType>
  llvm::Expected<NodeType> importNode(ASTUnit *From, ASTUnit *To,
                                      ASTImporter &Importer, NodeType Node) {
    ASTContext &ToCtx = To->getASTContext();

    // Add 'From' file to virtual file system so importer can 'find' it
    // while importing SourceLocations. It is safe to add same file multiple
    // times - it just isn't replaced.
    StringRef FromFileName = From->getMainFileName();
    createVirtualFileIfNeeded(To, FromFileName,
                              From->getBufferForFile(FromFileName));

    auto Imported = Importer.Import(Node);

    if (Imported) {
      // This should dump source locations and assert if some source locations
      // were not imported.
      SmallString<1024> ImportChecker;
      llvm::raw_svector_ostream ToNothing(ImportChecker);
      ToCtx.getTranslationUnitDecl()->print(ToNothing);

      // This traverses the AST to catch certain bugs like poorly or not
      // implemented subtrees.
      (*Imported)->dump(ToNothing);
    }

    return Imported;
  }

  template <typename NodeType>
  testing::AssertionResult
  testImport(const std::string &FromCode,
             const std::vector<std::string> &FromArgs,
             const std::string &ToCode, const std::vector<std::string> &ToArgs,
             MatchVerifier<NodeType> &Verifier,
             const internal::BindableMatcher<NodeType> &SearchMatcher,
             const internal::BindableMatcher<NodeType> &VerificationMatcher) {
    const char *const InputFileName = "input.cc";
    const char *const OutputFileName = "output.cc";

    std::unique_ptr<ASTUnit> FromAST = tooling::buildASTFromCodeWithArgs(
                                 FromCode, FromArgs, InputFileName),
                             ToAST = tooling::buildASTFromCodeWithArgs(
                                 ToCode, ToArgs, OutputFileName);

    ASTContext &FromCtx = FromAST->getASTContext(),
               &ToCtx = ToAST->getASTContext();

    ASTImporter Importer(ToCtx, ToAST->getFileManager(), FromCtx,
                         FromAST->getFileManager(), false);

    auto FoundNodes = match(SearchMatcher, FromCtx);
    if (FoundNodes.size() != 1)
      return testing::AssertionFailure()
             << "Multiple potential nodes were found!";

    auto ToImport = selectFirst<NodeType>(DeclToImportID, FoundNodes);
    if (!ToImport)
      return testing::AssertionFailure() << "Node type mismatch!";

    // The node being imported should match in the same way as
    // the result node.
    internal::BindableMatcher<NodeType> WrapperMatcher(VerificationMatcher);
    EXPECT_TRUE(Verifier.match(ToImport, WrapperMatcher));

    auto Imported = importNode(FromAST.get(), ToAST.get(), Importer, ToImport);
    if (!Imported) {
      std::string ErrorText;
      handleAllErrors(
          Imported.takeError(),
          [&ErrorText](const ImportError &Err) { ErrorText = Err.message(); });
      return testing::AssertionFailure()
             << "Import failed, error: \"" << ErrorText << "\"!";
    }

    return Verifier.match(*Imported, WrapperMatcher);
  }

  template <typename NodeType>
  testing::AssertionResult
  testImport(const std::string &FromCode,
             const std::vector<std::string> &FromArgs,
             const std::string &ToCode, const std::vector<std::string> &ToArgs,
             MatchVerifier<NodeType> &Verifier,
             const internal::BindableMatcher<NodeType> &VerificationMatcher) {
    return testImport(
        FromCode, FromArgs, ToCode, ToArgs, Verifier,
        translationUnitDecl(
            has(namedDecl(hasName(DeclToImportID)).bind(DeclToImportID))),
        VerificationMatcher);
  }

protected:
  std::vector<std::string> getExtraArgs() const override { return GetParam(); }

public:
  /// Test how AST node named "declToImport" located in the translation unit
  /// of "FromCode" virtual file is imported to "ToCode" virtual file.
  /// The verification is done by running AMatcher over the imported node.
  template <typename NodeType, typename MatcherType>
  void testImport(const std::string &FromCode, TestLanguage FromLang,
                  const std::string &ToCode, TestLanguage ToLang,
                  MatchVerifier<NodeType> &Verifier,
                  const MatcherType &AMatcher) {
    std::vector<std::string> FromArgs = getCommandLineArgsForLanguage(FromLang);
    std::vector<std::string> ToArgs = getCommandLineArgsForLanguage(ToLang);
    EXPECT_TRUE(
        testImport(FromCode, FromArgs, ToCode, ToArgs, Verifier, AMatcher));
  }

  struct ImportAction {
    StringRef FromFilename;
    StringRef ToFilename;
    // FIXME: Generalize this to support other node kinds.
    internal::BindableMatcher<Decl> ImportPredicate;

    ImportAction(StringRef FromFilename, StringRef ToFilename,
                 DeclarationMatcher ImportPredicate)
        : FromFilename(FromFilename), ToFilename(ToFilename),
          ImportPredicate(ImportPredicate) {}

    ImportAction(StringRef FromFilename, StringRef ToFilename,
                 const std::string &DeclName)
        : FromFilename(FromFilename), ToFilename(ToFilename),
          ImportPredicate(namedDecl(hasName(DeclName))) {}
  };

  using SingleASTUnit = std::unique_ptr<ASTUnit>;
  using AllASTUnits = llvm::StringMap<SingleASTUnit>;

  struct CodeEntry {
    std::string CodeSample;
    TestLanguage Lang;
  };

  using CodeFiles = llvm::StringMap<CodeEntry>;

  /// Builds an ASTUnit for one potential compile options set.
  SingleASTUnit createASTUnit(StringRef FileName, const CodeEntry &CE) const {
    std::vector<std::string> Args = getCommandLineArgsForLanguage(CE.Lang);
    auto AST = tooling::buildASTFromCodeWithArgs(CE.CodeSample, Args, FileName);
    EXPECT_TRUE(AST.get());
    return AST;
  }

  /// Test an arbitrary sequence of imports for a set of given in-memory files.
  /// The verification is done by running VerificationMatcher against a
  /// specified AST node inside of one of given files.
  /// \param CodeSamples Map whose key is the file name and the value is the
  /// file content.
  /// \param ImportActions Sequence of imports. Each import in sequence
  /// specifies "from file" and "to file" and a matcher that is used for
  /// searching a declaration for import in "from file".
  /// \param FileForFinalCheck Name of virtual file for which the final check is
  /// applied.
  /// \param FinalSelectPredicate Matcher that specifies the AST node in the
  /// FileForFinalCheck for which the verification will be done.
  /// \param VerificationMatcher Matcher that will be used for verification
  /// after all imports in sequence are done.
  void testImportSequence(const CodeFiles &CodeSamples,
                          const std::vector<ImportAction> &ImportActions,
                          StringRef FileForFinalCheck,
                          internal::BindableMatcher<Decl> FinalSelectPredicate,
                          internal::BindableMatcher<Decl> VerificationMatcher) {
    AllASTUnits AllASTs;
    using ImporterKey = std::pair<const ASTUnit *, const ASTUnit *>;
    llvm::DenseMap<ImporterKey, std::unique_ptr<ASTImporter>> Importers;

    auto GenASTsIfNeeded = [this, &AllASTs, &CodeSamples](StringRef Filename) {
      if (!AllASTs.count(Filename)) {
        auto Found = CodeSamples.find(Filename);
        assert(Found != CodeSamples.end() && "Wrong file for import!");
        AllASTs[Filename] = createASTUnit(Filename, Found->getValue());
      }
    };

    for (const ImportAction &Action : ImportActions) {
      StringRef FromFile = Action.FromFilename, ToFile = Action.ToFilename;
      GenASTsIfNeeded(FromFile);
      GenASTsIfNeeded(ToFile);

      ASTUnit *From = AllASTs[FromFile].get();
      ASTUnit *To = AllASTs[ToFile].get();

      // Create a new importer if needed.
      std::unique_ptr<ASTImporter> &ImporterRef = Importers[{From, To}];
      if (!ImporterRef)
        ImporterRef.reset(new ASTImporter(
            To->getASTContext(), To->getFileManager(), From->getASTContext(),
            From->getFileManager(), false));

      // Find the declaration and import it.
      auto FoundDecl = match(Action.ImportPredicate.bind(DeclToImportID),
                             From->getASTContext());
      EXPECT_TRUE(FoundDecl.size() == 1);
      const Decl *ToImport = selectFirst<Decl>(DeclToImportID, FoundDecl);
      auto Imported = importNode(From, To, *ImporterRef, ToImport);
      EXPECT_TRUE(static_cast<bool>(Imported));
      if (!Imported)
        llvm::consumeError(Imported.takeError());
    }

    // Find the declaration and import it.
    auto FoundDecl = match(FinalSelectPredicate.bind(DeclToVerifyID),
                           AllASTs[FileForFinalCheck]->getASTContext());
    EXPECT_TRUE(FoundDecl.size() == 1);
    const Decl *ToVerify = selectFirst<Decl>(DeclToVerifyID, FoundDecl);
    MatchVerifier<Decl> Verifier;
    EXPECT_TRUE(Verifier.match(
        ToVerify, internal::BindableMatcher<Decl>(VerificationMatcher)));
  }
};

template <typename T> RecordDecl *getRecordDecl(T *D) {
  auto *ET = cast<ElaboratedType>(D->getType().getTypePtr());
  return cast<RecordType>(ET->getNamedType().getTypePtr())->getDecl();
}

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
