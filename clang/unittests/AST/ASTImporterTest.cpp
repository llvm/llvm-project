//===- unittest/AST/ASTImporterTest.cpp - AST node import test ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for the correct import of AST nodes from one AST context to another.
//
//===----------------------------------------------------------------------===//

// Define this to have ::testing::Combine available.
// FIXME: Better solution for this?
#define GTEST_HAS_COMBINE 1

#include "clang/AST/ASTImporter.h"
#include "MatchVerifier.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclContextInternals.h"
#include "clang/AST/ASTImporter.h"
#include "clang/AST/ASTImporterLookupTable.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"

#include "DeclMatcher.h"
#include "Language.h"
#include "gmock/gmock.h"
#include "llvm/ADT/StringMap.h"

namespace clang {
namespace ast_matchers {

using internal::Matcher;
using internal::BindableMatcher;
using llvm::StringMap;

// Creates a virtual file and assigns that to the context of given AST. If the
// file already exists then the file will not be created again as a duplicate.
static void
createVirtualFileIfNeeded(ASTUnit *ToAST, StringRef FileName,
                          std::unique_ptr<llvm::MemoryBuffer> &&Buffer) {
  assert(ToAST);
  ASTContext &ToCtx = ToAST->getASTContext();
  auto *OFS = static_cast<llvm::vfs::OverlayFileSystem *>(
      ToCtx.getSourceManager().getFileManager().getVirtualFileSystem().get());
  auto *MFS = static_cast<llvm::vfs::InMemoryFileSystem *>(
      OFS->overlays_begin()->get());
  MFS->addFile(FileName, 0, std::move(Buffer));
}

static void createVirtualFileIfNeeded(ASTUnit *ToAST, StringRef FileName,
                                      StringRef Code) {
  return createVirtualFileIfNeeded(ToAST, FileName,
                                   llvm::MemoryBuffer::getMemBuffer(Code));
}

const StringRef DeclToImportID = "declToImport";
const StringRef DeclToVerifyID = "declToVerify";

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

auto DefaultTestValuesForRunOptions = ::testing::Values(
    ArgVector(), ArgVector{"-fdelayed-template-parsing"},
    ArgVector{"-fms-compatibility"},
    ArgVector{"-fdelayed-template-parsing", "-fms-compatibility"});

// Base class for those tests which use the family of `testImport` functions.
class TestImportBase : public CompilerOptionSpecificTest,
                       public ::testing::WithParamInterface<ArgVector> {

  template <typename NodeType>
  NodeType importNode(ASTUnit *From, ASTUnit *To, ASTImporter &Importer,
                      NodeType Node) {
    ASTContext &ToCtx = To->getASTContext();

    // Add 'From' file to virtual file system so importer can 'find' it
    // while importing SourceLocations. It is safe to add same file multiple
    // times - it just isn't replaced.
    StringRef FromFileName = From->getMainFileName();
    createVirtualFileIfNeeded(To, FromFileName,
                              From->getBufferForFile(FromFileName));

    auto Imported = Importer.Import(Node);

    // This should dump source locations and assert if some source locations
    // were not imported.
    SmallString<1024> ImportChecker;
    llvm::raw_svector_ostream ToNothing(ImportChecker);
    ToCtx.getTranslationUnitDecl()->print(ToNothing);

    // This traverses the AST to catch certain bugs like poorly or not
    // implemented subtrees.
    Imported->dump(ToNothing);

    return Imported;
  }

  template <typename NodeType>
  testing::AssertionResult
  testImport(const std::string &FromCode, const ArgVector &FromArgs,
             const std::string &ToCode, const ArgVector &ToArgs,
             MatchVerifier<NodeType> &Verifier,
             const BindableMatcher<NodeType> &SearchMatcher,
             const BindableMatcher<NodeType> &VerificationMatcher) {
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

    // Sanity check: the node being imported should match in the same way as
    // the result node.
    BindableMatcher<NodeType> WrapperMatcher(VerificationMatcher);
    EXPECT_TRUE(Verifier.match(ToImport, WrapperMatcher));

    auto Imported = importNode(FromAST.get(), ToAST.get(), Importer, ToImport);
    if (!Imported)
      return testing::AssertionFailure() << "Import failed, nullptr returned!";


    return Verifier.match(Imported, WrapperMatcher);
  }

  template <typename NodeType>
  testing::AssertionResult
  testImport(const std::string &FromCode, const ArgVector &FromArgs,
             const std::string &ToCode, const ArgVector &ToArgs,
             MatchVerifier<NodeType> &Verifier,
             const BindableMatcher<NodeType> &VerificationMatcher) {
    return testImport(
        FromCode, FromArgs, ToCode, ToArgs, Verifier,
        translationUnitDecl(
            has(namedDecl(hasName(DeclToImportID)).bind(DeclToImportID))),
        VerificationMatcher);
  }

protected:
  ArgVector getExtraArgs() const override { return GetParam(); }

public:

  /// Test how AST node named "declToImport" located in the translation unit
  /// of "FromCode" virtual file is imported to "ToCode" virtual file.
  /// The verification is done by running AMatcher over the imported node.
  template <typename NodeType, typename MatcherType>
  void testImport(const std::string &FromCode, Language FromLang,
                  const std::string &ToCode, Language ToLang,
                  MatchVerifier<NodeType> &Verifier,
                  const MatcherType &AMatcher) {
    ArgVector FromArgs = getArgVectorForLanguage(FromLang),
              ToArgs = getArgVectorForLanguage(ToLang);
    EXPECT_TRUE(
        testImport(FromCode, FromArgs, ToCode, ToArgs, Verifier, AMatcher));
  }

  struct ImportAction {
    StringRef FromFilename;
    StringRef ToFilename;
    // FIXME: Generalize this to support other node kinds.
    BindableMatcher<Decl> ImportPredicate;

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
  using AllASTUnits = StringMap<SingleASTUnit>;

  struct CodeEntry {
    std::string CodeSample;
    Language Lang;
  };

  using CodeFiles = StringMap<CodeEntry>;

  /// Builds an ASTUnit for one potential compile options set.
  SingleASTUnit createASTUnit(StringRef FileName, const CodeEntry &CE) const {
    ArgVector Args = getArgVectorForLanguage(CE.Lang);
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
                          BindableMatcher<Decl> FinalSelectPredicate,
                          BindableMatcher<Decl> VerificationMatcher) {
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
      EXPECT_TRUE(Imported);
    }

    // Find the declaration and import it.
    auto FoundDecl = match(FinalSelectPredicate.bind(DeclToVerifyID),
                           AllASTs[FileForFinalCheck]->getASTContext());
    EXPECT_TRUE(FoundDecl.size() == 1);
    const Decl *ToVerify = selectFirst<Decl>(DeclToVerifyID, FoundDecl);
    MatchVerifier<Decl> Verifier;
    EXPECT_TRUE(
        Verifier.match(ToVerify, BindableMatcher<Decl>(VerificationMatcher)));
  }
};

template <typename T> RecordDecl *getRecordDecl(T *D) {
  auto *ET = cast<ElaboratedType>(D->getType().getTypePtr());
  return cast<RecordType>(ET->getNamedType().getTypePtr())->getDecl();
}

// This class provides generic methods to write tests which can check internal
// attributes of AST nodes like getPreviousDecl(), isVirtual(), etc. Also,
// this fixture makes it possible to import from several "From" contexts.
class ASTImporterTestBase : public CompilerOptionSpecificTest {

  const char *const InputFileName = "input.cc";
  const char *const OutputFileName = "output.cc";

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
    TU(StringRef Code, StringRef FileName, ArgVector Args)
        : Code(Code), FileName(FileName),
          Unit(tooling::buildASTFromCodeWithArgs(this->Code, Args,
                                                 this->FileName)),
          TUDecl(Unit->getASTContext().getTranslationUnitDecl()) {
      Unit->enableSourceFileDiagnostics();
    }

    void lazyInitImporter(ASTImporterLookupTable &LookupTable, ASTUnit *ToAST) {
      assert(ToAST);
      if (!Importer) {
        Importer.reset(
            new ASTImporter(ToAST->getASTContext(), ToAST->getFileManager(),
                            Unit->getASTContext(), Unit->getFileManager(),
                            false, &LookupTable));
      }
      assert(&ToAST->getASTContext() == &Importer->getToContext());
      createVirtualFileIfNeeded(ToAST, FileName, Code);
    }

    Decl *import(ASTImporterLookupTable &LookupTable, ASTUnit *ToAST,
                 Decl *FromDecl) {
      lazyInitImporter(LookupTable, ToAST);
      return Importer->Import(FromDecl);
    }

    QualType import(ASTImporterLookupTable &LookupTable, ASTUnit *ToAST,
                    QualType FromType) {
      lazyInitImporter(LookupTable, ToAST);
      return Importer->Import(FromType);
    }
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
  void lazyInitLookupTable(TranslationUnitDecl *ToTU) {
    assert(ToTU);
    if (!LookupTablePtr)
      LookupTablePtr = llvm::make_unique<ASTImporterLookupTable>(*ToTU);
  }

  void lazyInitToAST(Language ToLang, StringRef ToSrcCode, StringRef FileName) {
    if (ToAST)
      return;
    ArgVector ToArgs = getArgVectorForLanguage(ToLang);
    // Source code must be a valid live buffer through the tests lifetime.
    ToCode = ToSrcCode;
    // Build the AST from an empty file.
    ToAST = tooling::buildASTFromCodeWithArgs(ToCode, ToArgs, FileName);
    ToAST->enableSourceFileDiagnostics();
    lazyInitLookupTable(ToAST->getASTContext().getTranslationUnitDecl());
  }

  TU *findFromTU(Decl *From) {
    // Create a virtual file in the To Ctx which corresponds to the file from
    // which we want to import the `From` Decl. Without this source locations
    // will be invalid in the ToCtx.
    auto It = std::find_if(FromTUs.begin(), FromTUs.end(), [From](const TU &E) {
      return E.TUDecl == From->getTranslationUnitDecl();
    });
    assert(It != FromTUs.end());
    return &*It;
  }

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
                  Language ToLang, StringRef Identifier = DeclToImportID) {
    ArgVector FromArgs = getArgVectorForLanguage(FromLang),
              ToArgs = getArgVectorForLanguage(ToLang);

    FromTUs.emplace_back(FromSrcCode, InputFileName, FromArgs);
    TU &FromTU = FromTUs.back();

    assert(!ToAST);
    lazyInitToAST(ToLang, ToSrcCode, OutputFileName);

    ASTContext &FromCtx = FromTU.Unit->getASTContext();

    IdentifierInfo *ImportedII = &FromCtx.Idents.get(Identifier);
    assert(ImportedII && "Declaration with the given identifier "
                         "should be specified in test!");
    DeclarationName ImportDeclName(ImportedII);
    SmallVector<NamedDecl *, 1> FoundDecls;
    FromCtx.getTranslationUnitDecl()->localUncachedLookup(ImportDeclName,
                                                          FoundDecls);

    assert(FoundDecls.size() == 1);

    Decl *Imported =
        FromTU.import(*LookupTablePtr, ToAST.get(), FoundDecls.front());

    assert(Imported);
    return std::make_tuple(*FoundDecls.begin(), Imported);
  }

  // Creates a TU decl for the given source code which can be used as a From
  // context.  May be called several times in a given test (with different file
  // name).
  TranslationUnitDecl *getTuDecl(StringRef SrcCode, Language Lang,
                                 StringRef FileName = "input.cc") {
    assert(
        std::find_if(FromTUs.begin(), FromTUs.end(), [FileName](const TU &E) {
          return E.FileName == FileName;
        }) == FromTUs.end());

    ArgVector Args = getArgVectorForLanguage(Lang);
    FromTUs.emplace_back(SrcCode, FileName, Args);
    TU &Tu = FromTUs.back();

    return Tu.TUDecl;
  }

  // Creates the To context with the given source code and returns the TU decl.
  TranslationUnitDecl *getToTuDecl(StringRef ToSrcCode, Language ToLang) {
    ArgVector ToArgs = getArgVectorForLanguage(ToLang);
    assert(!ToAST);
    lazyInitToAST(ToLang, ToSrcCode, OutputFileName);
    return ToAST->getASTContext().getTranslationUnitDecl();
  }

  // Import the given Decl into the ToCtx.
  // May be called several times in a given test.
  // The different instances of the param From may have different ASTContext.
  Decl *Import(Decl *From, Language ToLang) {
    lazyInitToAST(ToLang, "", OutputFileName);
    TU *FromTU = findFromTU(From);
    assert(LookupTablePtr);
    return FromTU->import(*LookupTablePtr, ToAST.get(), From);
  }

  template <class DeclT> DeclT *Import(DeclT *From, Language Lang) {
    return cast_or_null<DeclT>(Import(cast<Decl>(From), Lang));
  }

  QualType ImportType(QualType FromType, Decl *TUDecl, Language ToLang) {
    lazyInitToAST(ToLang, "", OutputFileName);
    TU *FromTU = findFromTU(TUDecl);
    assert(LookupTablePtr);
    return FromTU->import(*LookupTablePtr, ToAST.get(), FromType);
   }

  ~ASTImporterTestBase() {
    if (!::testing::Test::HasFailure()) return;

    for (auto &Tu : FromTUs) {
      assert(Tu.Unit);
      llvm::errs() << "FromAST:\n";
      Tu.Unit->getASTContext().getTranslationUnitDecl()->dump();
      llvm::errs() << "\n";
    }
    if (ToAST) {
      llvm::errs() << "ToAST:\n";
      ToAST->getASTContext().getTranslationUnitDecl()->dump();
    }
  }
};

class ASTImporterOptionSpecificTestBase
    : public ASTImporterTestBase,
      public ::testing::WithParamInterface<ArgVector> {
protected:
  ArgVector getExtraArgs() const override { return GetParam(); }
};

struct ImportExpr : TestImportBase {};
struct ImportType : TestImportBase {};
struct ImportDecl : TestImportBase {};

struct CanonicalRedeclChain : ASTImporterOptionSpecificTestBase {};

TEST_P(CanonicalRedeclChain, ShouldBeConsequentWithMatchers) {
  Decl *FromTU = getTuDecl("void f();", Lang_CXX);
  auto Pattern = functionDecl(hasName("f"));
  auto *D0 = FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);

  auto Redecls = getCanonicalForwardRedeclChain(D0);
  ASSERT_EQ(Redecls.size(), 1u);
  EXPECT_EQ(D0, Redecls[0]);
}

TEST_P(CanonicalRedeclChain, ShouldBeConsequentWithMatchers2) {
  Decl *FromTU = getTuDecl("void f(); void f(); void f();", Lang_CXX);
  auto Pattern = functionDecl(hasName("f"));
  auto *D0 = FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);
  auto *D2 = LastDeclMatcher<FunctionDecl>().match(FromTU, Pattern);
  FunctionDecl *D1 = D2->getPreviousDecl();

  auto Redecls = getCanonicalForwardRedeclChain(D0);
  ASSERT_EQ(Redecls.size(), 3u);
  EXPECT_EQ(D0, Redecls[0]);
  EXPECT_EQ(D1, Redecls[1]);
  EXPECT_EQ(D2, Redecls[2]);
}

TEST_P(CanonicalRedeclChain, ShouldBeSameForAllDeclInTheChain) {
  Decl *FromTU = getTuDecl("void f(); void f(); void f();", Lang_CXX);
  auto Pattern = functionDecl(hasName("f"));
  auto *D0 = FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);
  auto *D2 = LastDeclMatcher<FunctionDecl>().match(FromTU, Pattern);
  FunctionDecl *D1 = D2->getPreviousDecl();

  auto RedeclsD0 = getCanonicalForwardRedeclChain(D0);
  auto RedeclsD1 = getCanonicalForwardRedeclChain(D1);
  auto RedeclsD2 = getCanonicalForwardRedeclChain(D2);

  EXPECT_THAT(RedeclsD0, ::testing::ContainerEq(RedeclsD1));
  EXPECT_THAT(RedeclsD1, ::testing::ContainerEq(RedeclsD2));
}

TEST_P(ImportExpr, ImportStringLiteral) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "void declToImport() { (void)\"foo\"; }",
      Lang_CXX, "", Lang_CXX, Verifier,
      functionDecl(hasDescendant(
          stringLiteral(hasType(asString("const char [4]"))))));
  testImport(
      "void declToImport() { (void)L\"foo\"; }",
      Lang_CXX, "", Lang_CXX, Verifier,
      functionDecl(hasDescendant(
          stringLiteral(hasType(asString("const wchar_t [4]"))))));
  testImport(
      "void declToImport() { (void) \"foo\" \"bar\"; }",
      Lang_CXX, "", Lang_CXX, Verifier,
      functionDecl(hasDescendant(
          stringLiteral(hasType(asString("const char [7]"))))));
}

TEST_P(ImportExpr, ImportChooseExpr) {
  MatchVerifier<Decl> Verifier;

  // This case tests C code that is not condition-dependent and has a true
  // condition.
  testImport(
    "void declToImport() { (void)__builtin_choose_expr(1, 2, 3); }",
    Lang_C, "", Lang_C, Verifier,
    functionDecl(hasDescendant(chooseExpr())));
}

TEST_P(ImportExpr, ImportGNUNullExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "void declToImport() { (void)__null; }",
      Lang_CXX, "", Lang_CXX, Verifier,
      functionDecl(hasDescendant(gnuNullExpr(hasType(isInteger())))));
}

TEST_P(ImportExpr, ImportCXXNullPtrLiteralExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "void declToImport() { (void)nullptr; }",
      Lang_CXX11, "", Lang_CXX11, Verifier,
      functionDecl(hasDescendant(cxxNullPtrLiteralExpr())));
}


TEST_P(ImportExpr, ImportFloatinglLiteralExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "void declToImport() { (void)1.0; }",
      Lang_C, "", Lang_C, Verifier,
      functionDecl(hasDescendant(
          floatLiteral(equals(1.0), hasType(asString("double"))))));
  testImport(
      "void declToImport() { (void)1.0e-5f; }",
      Lang_C, "", Lang_C, Verifier,
      functionDecl(hasDescendant(
          floatLiteral(equals(1.0e-5f), hasType(asString("float"))))));
}

TEST_P(ImportExpr, ImportImaginaryLiteralExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "void declToImport() { (void)1.0i; }",
      Lang_CXX14, "", Lang_CXX14, Verifier,
      functionDecl(hasDescendant(imaginaryLiteral())));
}

TEST_P(ImportExpr, ImportCompoundLiteralExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "void declToImport() {"
      "  struct s { int x; long y; unsigned z; }; "
      "  (void)(struct s){ 42, 0L, 1U }; }",
      Lang_CXX, "", Lang_CXX, Verifier,
      functionDecl(hasDescendant(
          compoundLiteralExpr(
              hasType(asString("struct s")),
              has(initListExpr(
                  hasType(asString("struct s")),
                  has(integerLiteral(
                      equals(42), hasType(asString("int")))),
                  has(integerLiteral(
                      equals(0), hasType(asString("long")))),
                  has(integerLiteral(
                      equals(1), hasType(asString("unsigned int"))))))))));
}

TEST_P(ImportExpr, ImportCXXThisExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "class declToImport { void f() { (void)this; } };",
      Lang_CXX, "", Lang_CXX, Verifier,
      cxxRecordDecl(
          hasMethod(
              hasDescendant(
                  cxxThisExpr(
                      hasType(
                          asString("class declToImport *")))))));
}

TEST_P(ImportExpr, ImportAtomicExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "void declToImport() { int *ptr; __atomic_load_n(ptr, 1); }",
      Lang_C, "", Lang_C, Verifier,
      functionDecl(hasDescendant(
          atomicExpr(
              has(ignoringParenImpCasts(
                  declRefExpr(hasDeclaration(varDecl(hasName("ptr"))),
                      hasType(asString("int *"))))),
              has(integerLiteral(equals(1), hasType(asString("int"))))))));
}

TEST_P(ImportExpr, ImportLabelDeclAndAddrLabelExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "void declToImport() { loop: goto loop; (void)&&loop; }",
      Lang_C, "", Lang_C, Verifier,
      functionDecl(
          hasDescendant(
              labelStmt(hasDeclaration(labelDecl(hasName("loop"))))),
          hasDescendant(
              addrLabelExpr(hasDeclaration(labelDecl(hasName("loop")))))));
}

AST_MATCHER_P(TemplateDecl, hasTemplateDecl,
              internal::Matcher<NamedDecl>, InnerMatcher) {
  const NamedDecl *Template = Node.getTemplatedDecl();
  return Template && InnerMatcher.matches(*Template, Finder, Builder);
}

TEST_P(ImportExpr, ImportParenListExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "template<typename T> class dummy { void f() { dummy X(*this); } };"
      "typedef dummy<int> declToImport;"
      "template class dummy<int>;",
      Lang_CXX, "", Lang_CXX, Verifier,
      typedefDecl(hasType(templateSpecializationType(
          hasDeclaration(classTemplateSpecializationDecl(hasSpecializedTemplate(
              classTemplateDecl(hasTemplateDecl(cxxRecordDecl(hasMethod(allOf(
                  hasName("f"),
                  hasBody(compoundStmt(has(declStmt(hasSingleDecl(
                      varDecl(hasInitializer(parenListExpr(has(unaryOperator(
                          hasOperatorName("*"),
                          hasUnaryOperand(cxxThisExpr())))))))))))))))))))))));
}

TEST_P(ImportExpr, ImportSwitch) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "void declToImport() { int b; switch (b) { case 1: break; } }",
      Lang_C, "", Lang_C, Verifier,
      functionDecl(hasDescendant(
          switchStmt(has(compoundStmt(has(caseStmt())))))));
}

TEST_P(ImportExpr, ImportStmtExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
    "void declToImport() { int b; int a = b ?: 1; int C = ({int X=4; X;}); }",
    Lang_C, "", Lang_C, Verifier,
    functionDecl(hasDescendant(
        varDecl(
            hasName("C"),
            hasType(asString("int")),
            hasInitializer(
                stmtExpr(
                    hasAnySubstatement(declStmt(hasSingleDecl(
                        varDecl(
                            hasName("X"),
                            hasType(asString("int")),
                            hasInitializer(
                                integerLiteral(equals(4))))))),
                    hasDescendant(
                        implicitCastExpr())))))));
}

TEST_P(ImportExpr, ImportConditionalOperator) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "void declToImport() { (void)(true ? 1 : -5); }",
      Lang_CXX, "", Lang_CXX, Verifier,
      functionDecl(hasDescendant(
          conditionalOperator(
              hasCondition(cxxBoolLiteral(equals(true))),
              hasTrueExpression(integerLiteral(equals(1))),
              hasFalseExpression(
                  unaryOperator(hasUnaryOperand(integerLiteral(equals(5))))))
          )));
}

TEST_P(ImportExpr, ImportBinaryConditionalOperator) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "void declToImport() { (void)(1 ?: -5); }",
      Lang_CXX, "", Lang_CXX, Verifier,
      functionDecl(hasDescendant(
          binaryConditionalOperator(
              hasCondition(
                  implicitCastExpr(
                      hasSourceExpression(opaqueValueExpr(
                          hasSourceExpression(integerLiteral(equals(1))))),
                      hasType(booleanType()))),
              hasTrueExpression(
                  opaqueValueExpr(
                      hasSourceExpression(integerLiteral(equals(1))))),
              hasFalseExpression(
                  unaryOperator(
                      hasOperatorName("-"),
                      hasUnaryOperand(integerLiteral(equals(5)))))))));
}

TEST_P(ImportExpr, ImportDesignatedInitExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "void declToImport() {"
      "  struct point { double x; double y; };"
      "  struct point ptarray[10] = "
      "{ [2].y = 1.0, [2].x = 2.0, [0].x = 1.0 }; }",
      Lang_C, "", Lang_C, Verifier,
      functionDecl(hasDescendant(
          initListExpr(
              has(designatedInitExpr(
                  designatorCountIs(2),
                  hasDescendant(floatLiteral(equals(1.0))),
                  hasDescendant(integerLiteral(equals(2))))),
              has(designatedInitExpr(
                  designatorCountIs(2),
                  hasDescendant(floatLiteral(equals(2.0))),
                  hasDescendant(integerLiteral(equals(2))))),
              has(designatedInitExpr(
                  designatorCountIs(2),
                  hasDescendant(floatLiteral(equals(1.0))),
                  hasDescendant(integerLiteral(equals(0)))))))));
}

TEST_P(ImportExpr, ImportPredefinedExpr) {
  MatchVerifier<Decl> Verifier;
  // __func__ expands as StringLiteral("declToImport")
  testImport(
      "void declToImport() { (void)__func__; }",
      Lang_CXX, "", Lang_CXX, Verifier,
      functionDecl(hasDescendant(
          predefinedExpr(
              hasType(
                  asString("const char [13]")),
              has(stringLiteral(hasType(
                  asString("const char [13]"))))))));
}

TEST_P(ImportExpr, ImportInitListExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "void declToImport() {"
      "  struct point { double x; double y; };"
      "  point ptarray[10] = { [2].y = 1.0, [2].x = 2.0,"
      "                        [0].x = 1.0 }; }",
      Lang_CXX, "", Lang_CXX, Verifier,
      functionDecl(hasDescendant(
          initListExpr(
              has(
                  cxxConstructExpr(
                  requiresZeroInitialization())),
              has(
                  initListExpr(
                      hasType(asString("struct point")),
                      has(floatLiteral(equals(1.0))),
                      has(implicitValueInitExpr(
                          hasType(asString("double")))))),
              has(
                  initListExpr(
                      hasType(asString("struct point")),
                      has(floatLiteral(equals(2.0))),
                      has(floatLiteral(equals(1.0)))))))));
}


const internal::VariadicDynCastAllOfMatcher<Expr, VAArgExpr> vaArgExpr;

TEST_P(ImportExpr, ImportVAArgExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "void declToImport(__builtin_va_list list, ...) {"
      "  (void)__builtin_va_arg(list, int); }",
      Lang_CXX, "", Lang_CXX, Verifier,
      functionDecl(hasDescendant(
          cStyleCastExpr(hasSourceExpression(vaArgExpr())))));
}

TEST_P(ImportExpr, CXXTemporaryObjectExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "struct C {};"
      "void declToImport() { C c = C(); }",
      Lang_CXX, "", Lang_CXX, Verifier,
      functionDecl(hasDescendant(
          exprWithCleanups(has(cxxConstructExpr(
              has(materializeTemporaryExpr(has(implicitCastExpr(
                  has(cxxTemporaryObjectExpr())))))))))));
}

TEST_P(ImportType, ImportAtomicType) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "void declToImport() { typedef _Atomic(int) a_int; }",
      Lang_CXX11, "", Lang_CXX11, Verifier,
      functionDecl(hasDescendant(typedefDecl(has(atomicType())))));
}

TEST_P(ImportDecl, ImportFunctionTemplateDecl) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "template <typename T> void declToImport() { };",
      Lang_CXX, "", Lang_CXX, Verifier,
      functionTemplateDecl());
}

TEST_P(ImportExpr, ImportCXXDependentScopeMemberExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "template <typename T> struct C { T t; };"
      "template <typename T> void declToImport() {"
      "  C<T> d;"
      "  (void)d.t;"
      "}"
      "void instantiate() { declToImport<int>(); }",
      Lang_CXX, "", Lang_CXX, Verifier,
      functionTemplateDecl(hasDescendant(
          cStyleCastExpr(has(cxxDependentScopeMemberExpr())))));
  testImport(
      "template <typename T> struct C { T t; };"
      "template <typename T> void declToImport() {"
      "  C<T> d;"
      "  (void)(&d)->t;"
      "}"
      "void instantiate() { declToImport<int>(); }",
      Lang_CXX, "", Lang_CXX, Verifier,
      functionTemplateDecl(hasDescendant(
          cStyleCastExpr(has(cxxDependentScopeMemberExpr())))));
}

TEST_P(ImportType, ImportTypeAliasTemplate) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "template <int K>"
      "struct dummy { static const int i = K; };"
      "template <int K> using dummy2 = dummy<K>;"
      "int declToImport() { return dummy2<3>::i; }",
      Lang_CXX11, "", Lang_CXX11, Verifier,
      functionDecl(
          hasDescendant(implicitCastExpr(has(declRefExpr()))),
          unless(hasAncestor(translationUnitDecl(has(typeAliasDecl()))))));
}

const internal::VariadicDynCastAllOfMatcher<Decl, VarTemplateSpecializationDecl>
    varTemplateSpecializationDecl;

TEST_P(ImportDecl, ImportVarTemplate) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "template <typename T>"
      "T pi = T(3.1415926535897932385L);"
      "void declToImport() { (void)pi<int>; }",
      Lang_CXX14, "", Lang_CXX14, Verifier,
      functionDecl(
          hasDescendant(declRefExpr(to(varTemplateSpecializationDecl()))),
          unless(hasAncestor(translationUnitDecl(has(varDecl(
              hasName("pi"), unless(varTemplateSpecializationDecl()))))))));
}

TEST_P(ImportType, ImportPackExpansion) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "template <typename... Args>"
      "struct dummy {"
      "  dummy(Args... args) {}"
      "  static const int i = 4;"
      "};"
      "int declToImport() { return dummy<int>::i; }",
      Lang_CXX11, "", Lang_CXX11, Verifier,
      functionDecl(hasDescendant(
          returnStmt(has(implicitCastExpr(has(declRefExpr())))))));
}

const internal::VariadicDynCastAllOfMatcher<Type,
                                            DependentTemplateSpecializationType>
    dependentTemplateSpecializationType;

TEST_P(ImportType, ImportDependentTemplateSpecialization) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "template<typename T>"
      "struct A;"
      "template<typename T>"
      "struct declToImport {"
      "  typename A<T>::template B<T> a;"
      "};",
      Lang_CXX, "", Lang_CXX, Verifier,
      classTemplateDecl(has(cxxRecordDecl(has(
          fieldDecl(hasType(dependentTemplateSpecializationType())))))));
}

const internal::VariadicDynCastAllOfMatcher<Stmt, SizeOfPackExpr>
    sizeOfPackExpr;

TEST_P(ImportExpr, ImportSizeOfPackExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "template <typename... Ts>"
      "void declToImport() {"
      "  const int i = sizeof...(Ts);"
      "};"
      "void g() { declToImport<int>(); }",
      Lang_CXX11, "", Lang_CXX11, Verifier,
          functionTemplateDecl(hasDescendant(sizeOfPackExpr())));
  testImport(
      "template <typename... Ts>"
      "using X = int[sizeof...(Ts)];"
      "template <typename... Us>"
      "struct Y {"
      "  X<Us..., int, double, int, Us...> f;"
      "};"
      "Y<float, int> declToImport;",
      Lang_CXX11, "", Lang_CXX11, Verifier,
      varDecl(hasType(classTemplateSpecializationDecl(has(fieldDecl(hasType(
          hasUnqualifiedDesugaredType(constantArrayType(hasSize(7))))))))));
}

/// \brief Matches __builtin_types_compatible_p:
/// GNU extension to check equivalent types
/// Given
/// \code
///   __builtin_types_compatible_p(int, int)
/// \endcode
//  will generate TypeTraitExpr <...> 'int'
const internal::VariadicDynCastAllOfMatcher<Stmt, TypeTraitExpr> typeTraitExpr;

TEST_P(ImportExpr, ImportTypeTraitExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "void declToImport() { "
      "  (void)__builtin_types_compatible_p(int, int);"
      "}",
      Lang_C, "", Lang_C, Verifier,
      functionDecl(hasDescendant(typeTraitExpr(hasType(asString("int"))))));
}

const internal::VariadicDynCastAllOfMatcher<Stmt, CXXTypeidExpr> cxxTypeidExpr;

TEST_P(ImportExpr, ImportCXXTypeidExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "namespace std { class type_info {}; }"
      "void declToImport() {"
      "  int x;"
      "  auto a = typeid(int); auto b = typeid(x);"
      "}",
      Lang_CXX11, "", Lang_CXX11, Verifier,
      functionDecl(
          hasDescendant(varDecl(
              hasName("a"), hasInitializer(hasDescendant(cxxTypeidExpr())))),
          hasDescendant(varDecl(
              hasName("b"), hasInitializer(hasDescendant(cxxTypeidExpr()))))));
}

TEST_P(ImportExpr, ImportTypeTraitExprValDep) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "template<typename T> struct declToImport {"
      "  void m() { (void)__is_pod(T); }"
      "};"
      "void f() { declToImport<int>().m(); }",
      Lang_CXX11, "", Lang_CXX11, Verifier,
      classTemplateDecl(has(cxxRecordDecl(has(
          functionDecl(hasDescendant(
              typeTraitExpr(hasType(booleanType())))))))));
}

TEST_P(ImportDecl, ImportRecordDeclInFunc) {
  MatchVerifier<Decl> Verifier;
  testImport("int declToImport() { "
             "  struct data_t {int a;int b;};"
             "  struct data_t d;"
             "  return 0;"
             "}",
             Lang_C, "", Lang_C, Verifier,
             functionDecl(hasBody(compoundStmt(
                 has(declStmt(hasSingleDecl(varDecl(hasName("d")))))))));
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportRecordTypeInFunc) {
  Decl *FromTU = getTuDecl("int declToImport() { "
                           "  struct data_t {int a;int b;};"
                           "  struct data_t d;"
                           "  return 0;"
                           "}",
                           Lang_C, "input.c");
  auto *FromVar =
      FirstDeclMatcher<VarDecl>().match(FromTU, varDecl(hasName("d")));
  ASSERT_TRUE(FromVar);
  auto ToType =
      ImportType(FromVar->getType().getCanonicalType(), FromVar, Lang_C);
  EXPECT_FALSE(ToType.isNull());
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportRecordDeclInFuncParams) {
  // This construct is not supported by ASTImporter.
  Decl *FromTU = getTuDecl(
      "int declToImport(struct data_t{int a;int b;} ***d){ return 0; }",
      Lang_C, "input.c");
  auto *From = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("declToImport")));
  ASSERT_TRUE(From);
  auto *To = Import(From, Lang_C);
  EXPECT_EQ(To, nullptr);
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportRecordDeclInFuncFromMacro) {
  Decl *FromTU = getTuDecl(
      "#define NONAME_SIZEOF(type) sizeof(struct{type *dummy;}) \n"
      "int declToImport(){ return NONAME_SIZEOF(int); }",
      Lang_C, "input.c");
  auto *From = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("declToImport")));
  ASSERT_TRUE(From);
  auto *To = Import(From, Lang_C);
  ASSERT_TRUE(To);
  EXPECT_TRUE(MatchVerifier<FunctionDecl>().match(
      To, functionDecl(hasName("declToImport"),
                       hasDescendant(unaryExprOrTypeTraitExpr()))));
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportRecordDeclInFuncParamsFromMacro) {
  // This construct is not supported by ASTImporter.
  Decl *FromTU = getTuDecl(
      "#define PAIR_STRUCT(type) struct data_t{type a;type b;} \n"
      "int declToImport(PAIR_STRUCT(int) ***d){ return 0; }",
      Lang_C, "input.c");
  auto *From = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("declToImport")));
  ASSERT_TRUE(From);
  auto *To = Import(From, Lang_C);
  EXPECT_EQ(To, nullptr);
}

const internal::VariadicDynCastAllOfMatcher<Expr, CXXPseudoDestructorExpr>
    cxxPseudoDestructorExpr;

TEST_P(ImportExpr, ImportCXXPseudoDestructorExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "typedef int T;"
      "void declToImport(int *p) {"
      "  T t;"
      "  p->T::~T();"
      "}",
      Lang_CXX, "", Lang_CXX, Verifier,
      functionDecl(hasDescendant(
          callExpr(has(cxxPseudoDestructorExpr())))));
}

TEST_P(ImportDecl, ImportUsingDecl) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "namespace foo { int bar; }"
      "void declToImport() { using foo::bar; }",
      Lang_CXX, "", Lang_CXX, Verifier,
      functionDecl(hasDescendant(usingDecl())));
}

/// \brief Matches shadow declarations introduced into a scope by a
///        (resolved) using declaration.
///
/// Given
/// \code
///   namespace n { int f; }
///   namespace declToImport { using n::f; }
/// \endcode
/// usingShadowDecl()
///   matches \code f \endcode
const internal::VariadicDynCastAllOfMatcher<Decl,
                                            UsingShadowDecl> usingShadowDecl;

TEST_P(ImportDecl, ImportUsingShadowDecl) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "namespace foo { int bar; }"
      "namespace declToImport { using foo::bar; }",
      Lang_CXX, "", Lang_CXX, Verifier,
      namespaceDecl(has(usingShadowDecl())));
}

TEST_P(ImportExpr, ImportUnresolvedLookupExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "template<typename T> int foo();"
      "template <typename T> void declToImport() {"
      "  (void)::foo<T>;"
      "  (void)::template foo<T>;"
      "}"
      "void instantiate() { declToImport<int>(); }",
      Lang_CXX, "", Lang_CXX, Verifier,
      functionTemplateDecl(hasDescendant(unresolvedLookupExpr())));
}

TEST_P(ImportExpr, ImportCXXUnresolvedConstructExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "template <typename T> struct C { T t; };"
      "template <typename T> void declToImport() {"
      "  C<T> d;"
      "  d.t = T();"
      "}"
      "void instantiate() { declToImport<int>(); }",
      Lang_CXX, "", Lang_CXX, Verifier,
      functionTemplateDecl(hasDescendant(
          binaryOperator(has(cxxUnresolvedConstructExpr())))));
  testImport(
      "template <typename T> struct C { T t; };"
      "template <typename T> void declToImport() {"
      "  C<T> d;"
      "  (&d)->t = T();"
      "}"
      "void instantiate() { declToImport<int>(); }",
      Lang_CXX, "", Lang_CXX, Verifier,
          functionTemplateDecl(hasDescendant(
              binaryOperator(has(cxxUnresolvedConstructExpr())))));
}

/// Check that function "declToImport()" (which is the templated function
/// for corresponding FunctionTemplateDecl) is not added into DeclContext.
/// Same for class template declarations.
TEST_P(ImportDecl, ImportTemplatedDeclForTemplate) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "template <typename T> void declToImport() { T a = 1; }"
      "void instantiate() { declToImport<int>(); }",
      Lang_CXX, "", Lang_CXX, Verifier,
      functionTemplateDecl(hasAncestor(translationUnitDecl(
          unless(has(functionDecl(hasName("declToImport"))))))));
  testImport(
      "template <typename T> struct declToImport { T t; };"
      "void instantiate() { declToImport<int>(); }",
      Lang_CXX, "", Lang_CXX, Verifier,
      classTemplateDecl(hasAncestor(translationUnitDecl(
          unless(has(cxxRecordDecl(hasName("declToImport"))))))));
}

TEST_P(ImportDecl, ImportClassTemplatePartialSpecialization) {
  MatchVerifier<Decl> Verifier;
  auto Code =
      R"s(
      struct declToImport {
        template <typename T0> struct X;
        template <typename T0> struct X<T0 *> {};
      };
      )s";
  testImport(Code, Lang_CXX, "", Lang_CXX, Verifier,
             recordDecl(has(classTemplateDecl()),
                        has(classTemplateSpecializationDecl())));
}

TEST_P(ImportExpr, CXXOperatorCallExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "class declToImport {"
      "  void f() { *this = declToImport(); }"
      "};",
      Lang_CXX, "", Lang_CXX, Verifier,
      cxxRecordDecl(has(cxxMethodDecl(hasDescendant(
          cxxOperatorCallExpr())))));
}

TEST_P(ImportExpr, DependentSizedArrayType) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "template<typename T, int Size> class declToImport {"
      "  T data[Size];"
      "};",
      Lang_CXX, "", Lang_CXX, Verifier,
      classTemplateDecl(has(cxxRecordDecl(
          has(fieldDecl(hasType(dependentSizedArrayType())))))));
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportOfTemplatedDeclOfClassTemplateDecl) {
  Decl *FromTU = getTuDecl("template<class X> struct S{};", Lang_CXX);
  auto From =
      FirstDeclMatcher<ClassTemplateDecl>().match(FromTU, classTemplateDecl());
  ASSERT_TRUE(From);
  auto To = cast<ClassTemplateDecl>(Import(From, Lang_CXX));
  ASSERT_TRUE(To);
  Decl *ToTemplated = To->getTemplatedDecl();
  Decl *ToTemplated1 = Import(From->getTemplatedDecl(), Lang_CXX);
  EXPECT_TRUE(ToTemplated1);
  EXPECT_EQ(ToTemplated1, ToTemplated);
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportOfTemplatedDeclOfFunctionTemplateDecl) {
  Decl *FromTU = getTuDecl("template<class X> void f(){}", Lang_CXX);
  auto From = FirstDeclMatcher<FunctionTemplateDecl>().match(
      FromTU, functionTemplateDecl());
  ASSERT_TRUE(From);
  auto To = cast<FunctionTemplateDecl>(Import(From, Lang_CXX));
  ASSERT_TRUE(To);
  Decl *ToTemplated = To->getTemplatedDecl();
  Decl *ToTemplated1 = Import(From->getTemplatedDecl(), Lang_CXX);
  EXPECT_TRUE(ToTemplated1);
  EXPECT_EQ(ToTemplated1, ToTemplated);
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportOfTemplatedDeclShouldImportTheClassTemplateDecl) {
  Decl *FromTU = getTuDecl("template<class X> struct S{};", Lang_CXX);
  auto FromFT =
      FirstDeclMatcher<ClassTemplateDecl>().match(FromTU, classTemplateDecl());
  ASSERT_TRUE(FromFT);

  auto ToTemplated =
      cast<CXXRecordDecl>(Import(FromFT->getTemplatedDecl(), Lang_CXX));
  EXPECT_TRUE(ToTemplated);
  auto ToTU = ToTemplated->getTranslationUnitDecl();
  auto ToFT =
      FirstDeclMatcher<ClassTemplateDecl>().match(ToTU, classTemplateDecl());
  EXPECT_TRUE(ToFT);
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportOfTemplatedDeclShouldImportTheFunctionTemplateDecl) {
  Decl *FromTU = getTuDecl("template<class X> void f(){}", Lang_CXX);
  auto FromFT = FirstDeclMatcher<FunctionTemplateDecl>().match(
      FromTU, functionTemplateDecl());
  ASSERT_TRUE(FromFT);

  auto ToTemplated =
      cast<FunctionDecl>(Import(FromFT->getTemplatedDecl(), Lang_CXX));
  EXPECT_TRUE(ToTemplated);
  auto ToTU = ToTemplated->getTranslationUnitDecl();
  auto ToFT = FirstDeclMatcher<FunctionTemplateDecl>().match(
      ToTU, functionTemplateDecl());
  EXPECT_TRUE(ToFT);
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportCorrectTemplatedDecl) {
  auto Code =
        R"(
        namespace x {
          template<class X> struct S1{};
          template<class X> struct S2{};
          template<class X> struct S3{};
        }
        )";
  Decl *FromTU = getTuDecl(Code, Lang_CXX);
  auto FromNs =
      FirstDeclMatcher<NamespaceDecl>().match(FromTU, namespaceDecl());
  auto ToNs = cast<NamespaceDecl>(Import(FromNs, Lang_CXX));
  ASSERT_TRUE(ToNs);
  auto From =
      FirstDeclMatcher<ClassTemplateDecl>().match(FromTU,
                                                  classTemplateDecl(
                                                      hasName("S2")));
  auto To =
      FirstDeclMatcher<ClassTemplateDecl>().match(ToNs,
                                                  classTemplateDecl(
                                                      hasName("S2")));
  ASSERT_TRUE(From);
  ASSERT_TRUE(To);
  auto ToTemplated = To->getTemplatedDecl();
  auto ToTemplated1 =
      cast<CXXRecordDecl>(Import(From->getTemplatedDecl(), Lang_CXX));
  EXPECT_TRUE(ToTemplated1);
  ASSERT_EQ(ToTemplated1, ToTemplated);
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportChooseExpr) {
  // This tests the import of isConditionTrue directly to make sure the importer
  // gets it right.
  Decl *From, *To;
  std::tie(From, To) = getImportedDecl(
    "void declToImport() { (void)__builtin_choose_expr(1, 0, 1); }",
    Lang_C, "", Lang_C);

  auto ToResults = match(chooseExpr().bind("choose"), To->getASTContext());
  auto FromResults = match(chooseExpr().bind("choose"), From->getASTContext());

  const ChooseExpr *FromChooseExpr =
      selectFirst<ChooseExpr>("choose", FromResults);
  ASSERT_TRUE(FromChooseExpr);

  const ChooseExpr *ToChooseExpr = selectFirst<ChooseExpr>("choose", ToResults);
  ASSERT_TRUE(ToChooseExpr);

  EXPECT_EQ(FromChooseExpr->isConditionTrue(), ToChooseExpr->isConditionTrue());
  EXPECT_EQ(FromChooseExpr->isConditionDependent(),
            ToChooseExpr->isConditionDependent());
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportFunctionWithBackReferringParameter) {
  Decl *From, *To;
  std::tie(From, To) = getImportedDecl(
      R"(
      template <typename T> struct X {};

      void declToImport(int y, X<int> &x) {}

      template <> struct X<int> {
        void g() {
          X<int> x;
          declToImport(0, x);
        }
      };
      )",
      Lang_CXX, "", Lang_CXX);

  MatchVerifier<Decl> Verifier;
  auto Matcher = functionDecl(hasName("declToImport"),
                              parameterCountIs(2),
                              hasParameter(0, hasName("y")),
                              hasParameter(1, hasName("x")),
                              hasParameter(1, hasType(asString("X<int> &"))));
  ASSERT_TRUE(Verifier.match(From, Matcher));
  EXPECT_TRUE(Verifier.match(To, Matcher));
}

TEST_P(ASTImporterOptionSpecificTestBase,
       TUshouldNotContainTemplatedDeclOfFunctionTemplates) {
  Decl *From, *To;
  std::tie(From, To) =
      getImportedDecl("template <typename T> void declToImport() { T a = 1; }"
                      "void instantiate() { declToImport<int>(); }",
                      Lang_CXX, "", Lang_CXX);

  auto Check = [](Decl *D) -> bool {
    auto TU = D->getTranslationUnitDecl();
    for (auto Child : TU->decls()) {
      if (auto *FD = dyn_cast<FunctionDecl>(Child)) {
        if (FD->getNameAsString() == "declToImport") {
          GTEST_NONFATAL_FAILURE_(
              "TU should not contain any FunctionDecl with name declToImport");
          return false;
        }
      }
    }
    return true;
  };

  ASSERT_TRUE(Check(From));
  EXPECT_TRUE(Check(To));
}

TEST_P(ASTImporterOptionSpecificTestBase,
       TUshouldNotContainTemplatedDeclOfClassTemplates) {
  Decl *From, *To;
  std::tie(From, To) =
      getImportedDecl("template <typename T> struct declToImport { T t; };"
                      "void instantiate() { declToImport<int>(); }",
                      Lang_CXX, "", Lang_CXX);

  auto Check = [](Decl *D) -> bool {
    auto TU = D->getTranslationUnitDecl();
    for (auto Child : TU->decls()) {
      if (auto *RD = dyn_cast<CXXRecordDecl>(Child)) {
        if (RD->getNameAsString() == "declToImport") {
          GTEST_NONFATAL_FAILURE_(
              "TU should not contain any CXXRecordDecl with name declToImport");
          return false;
        }
      }
    }
    return true;
  };

  ASSERT_TRUE(Check(From));
  EXPECT_TRUE(Check(To));
}

TEST_P(ASTImporterOptionSpecificTestBase,
       TUshouldNotContainTemplatedDeclOfTypeAlias) {
  Decl *From, *To;
  std::tie(From, To) =
      getImportedDecl(
          "template <typename T> struct X {};"
          "template <typename T> using declToImport = X<T>;"
          "void instantiate() { declToImport<int> a; }",
                      Lang_CXX11, "", Lang_CXX11);

  auto Check = [](Decl *D) -> bool {
    auto TU = D->getTranslationUnitDecl();
    for (auto Child : TU->decls()) {
      if (auto *AD = dyn_cast<TypeAliasDecl>(Child)) {
        if (AD->getNameAsString() == "declToImport") {
          GTEST_NONFATAL_FAILURE_(
              "TU should not contain any TypeAliasDecl with name declToImport");
          return false;
        }
      }
    }
    return true;
  };

  ASSERT_TRUE(Check(From));
  EXPECT_TRUE(Check(To));
}

TEST_P(ASTImporterOptionSpecificTestBase,
       TUshouldNotContainClassTemplateSpecializationOfImplicitInstantiation) {

  Decl *From, *To;
  std::tie(From, To) = getImportedDecl(
      R"(
      template<class T>
      class Base {};
      class declToImport : public Base<declToImport> {};
      )",
      Lang_CXX, "", Lang_CXX);

  // Check that the ClassTemplateSpecializationDecl is NOT the child of the TU.
  auto Pattern =
      translationUnitDecl(unless(has(classTemplateSpecializationDecl())));
  ASSERT_TRUE(
      MatchVerifier<Decl>{}.match(From->getTranslationUnitDecl(), Pattern));
  EXPECT_TRUE(
      MatchVerifier<Decl>{}.match(To->getTranslationUnitDecl(), Pattern));

  // Check that the ClassTemplateSpecializationDecl is the child of the
  // ClassTemplateDecl.
  Pattern = translationUnitDecl(has(classTemplateDecl(
      hasName("Base"), has(classTemplateSpecializationDecl()))));
  ASSERT_TRUE(
      MatchVerifier<Decl>{}.match(From->getTranslationUnitDecl(), Pattern));
  EXPECT_TRUE(
      MatchVerifier<Decl>{}.match(To->getTranslationUnitDecl(), Pattern));
}

AST_MATCHER_P(RecordDecl, hasFieldOrder, std::vector<StringRef>, Order) {
  size_t Index = 0;
  for (FieldDecl *Field : Node.fields()) {
    if (Index == Order.size())
      return false;
    if (Field->getName() != Order[Index])
      return false;
    ++Index;
  }
  return Index == Order.size();
}

TEST_P(ASTImporterOptionSpecificTestBase,
       TUshouldContainClassTemplateSpecializationOfExplicitInstantiation) {
  Decl *From, *To;
  std::tie(From, To) = getImportedDecl(
      R"(
      namespace NS {
        template<class T>
        class X {};
        template class X<int>;
      }
      )",
      Lang_CXX, "", Lang_CXX, "NS");

  // Check that the ClassTemplateSpecializationDecl is NOT the child of the
  // ClassTemplateDecl.
  auto Pattern = namespaceDecl(has(classTemplateDecl(
      hasName("X"), unless(has(classTemplateSpecializationDecl())))));
  ASSERT_TRUE(MatchVerifier<Decl>{}.match(From, Pattern));
  EXPECT_TRUE(MatchVerifier<Decl>{}.match(To, Pattern));

  // Check that the ClassTemplateSpecializationDecl is the child of the
  // NamespaceDecl.
  Pattern = namespaceDecl(has(classTemplateSpecializationDecl(hasName("X"))));
  ASSERT_TRUE(MatchVerifier<Decl>{}.match(From, Pattern));
  EXPECT_TRUE(MatchVerifier<Decl>{}.match(To, Pattern));
}

TEST_P(ASTImporterOptionSpecificTestBase,
       CXXRecordDeclFieldsShouldBeInCorrectOrder) {
  Decl *From, *To;
  std::tie(From, To) =
      getImportedDecl(
          "struct declToImport { int a; int b; };",
                      Lang_CXX11, "", Lang_CXX11);

  MatchVerifier<Decl> Verifier;
  ASSERT_TRUE(Verifier.match(From, cxxRecordDecl(hasFieldOrder({"a", "b"}))));
  EXPECT_TRUE(Verifier.match(To, cxxRecordDecl(hasFieldOrder({"a", "b"}))));
}

TEST_P(ASTImporterOptionSpecificTestBase,
       DISABLED_CXXRecordDeclFieldOrderShouldNotDependOnImportOrder) {
  Decl *From, *To;
  std::tie(From, To) = getImportedDecl(
      // The original recursive algorithm of ASTImporter first imports 'c' then
      // 'b' and lastly 'a'.  Therefore we must restore the order somehow.
      R"s(
      struct declToImport {
          int a = c + b;
          int b = 1;
          int c = 2;
      };
      )s",
      Lang_CXX11, "", Lang_CXX11);

  MatchVerifier<Decl> Verifier;
  ASSERT_TRUE(
      Verifier.match(From, cxxRecordDecl(hasFieldOrder({"a", "b", "c"}))));
  EXPECT_TRUE(
      Verifier.match(To, cxxRecordDecl(hasFieldOrder({"a", "b", "c"}))));
}

TEST_P(ASTImporterOptionSpecificTestBase, ShouldImportImplicitCXXRecordDecl) {
  Decl *From, *To;
  std::tie(From, To) = getImportedDecl(
      R"(
      struct declToImport {
      };
      )",
      Lang_CXX, "", Lang_CXX);

  MatchVerifier<Decl> Verifier;
  // Match the implicit Decl.
  auto Matcher = cxxRecordDecl(has(cxxRecordDecl()));
  ASSERT_TRUE(Verifier.match(From, Matcher));
  EXPECT_TRUE(Verifier.match(To, Matcher));
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ShouldImportImplicitCXXRecordDeclOfClassTemplate) {
  Decl *From, *To;
  std::tie(From, To) = getImportedDecl(
      R"(
      template <typename U>
      struct declToImport {
      };
      )",
      Lang_CXX, "", Lang_CXX);

  MatchVerifier<Decl> Verifier;
  // Match the implicit Decl.
  auto Matcher = classTemplateDecl(has(cxxRecordDecl(has(cxxRecordDecl()))));
  ASSERT_TRUE(Verifier.match(From, Matcher));
  EXPECT_TRUE(Verifier.match(To, Matcher));
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ShouldImportImplicitCXXRecordDeclOfClassTemplateSpecializationDecl) {
  Decl *From, *To;
  std::tie(From, To) = getImportedDecl(
      R"(
      template<class T>
      class Base {};
      class declToImport : public Base<declToImport> {};
      )",
      Lang_CXX, "", Lang_CXX);

  auto hasImplicitClass = has(cxxRecordDecl());
  auto Pattern = translationUnitDecl(has(classTemplateDecl(
      hasName("Base"),
      has(classTemplateSpecializationDecl(hasImplicitClass)))));
  ASSERT_TRUE(
      MatchVerifier<Decl>{}.match(From->getTranslationUnitDecl(), Pattern));
  EXPECT_TRUE(
      MatchVerifier<Decl>{}.match(To->getTranslationUnitDecl(), Pattern));
}

TEST_P(ASTImporterOptionSpecificTestBase, IDNSOrdinary) {
  Decl *From, *To;
  std::tie(From, To) =
      getImportedDecl("void declToImport() {}", Lang_CXX, "", Lang_CXX);

  MatchVerifier<Decl> Verifier;
  auto Matcher = functionDecl();
  ASSERT_TRUE(Verifier.match(From, Matcher));
  EXPECT_TRUE(Verifier.match(To, Matcher));
  EXPECT_EQ(From->getIdentifierNamespace(), To->getIdentifierNamespace());
}

TEST_P(ASTImporterOptionSpecificTestBase, IDNSOfNonmemberOperator) {
  Decl *FromTU = getTuDecl(
      R"(
      struct X {};
      void operator<<(int, X);
      )",
      Lang_CXX);
  Decl *From = LastDeclMatcher<Decl>{}.match(FromTU, functionDecl());
  const Decl *To = Import(From, Lang_CXX);
  EXPECT_EQ(From->getIdentifierNamespace(), To->getIdentifierNamespace());
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ShouldImportMembersOfClassTemplateSpecializationDecl) {
  Decl *From, *To;
  std::tie(From, To) = getImportedDecl(
      R"(
      template<class T>
      class Base { int a; };
      class declToImport : Base<declToImport> {};
      )",
      Lang_CXX, "", Lang_CXX);

  auto Pattern = translationUnitDecl(has(classTemplateDecl(
      hasName("Base"),
      has(classTemplateSpecializationDecl(has(fieldDecl(hasName("a"))))))));
  ASSERT_TRUE(
      MatchVerifier<Decl>{}.match(From->getTranslationUnitDecl(), Pattern));
  EXPECT_TRUE(
      MatchVerifier<Decl>{}.match(To->getTranslationUnitDecl(), Pattern));
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportDefinitionOfClassTemplateAfterFwdDecl) {
  {
    Decl *FromTU = getTuDecl(
        R"(
            template <typename T>
            struct B;
            )",
        Lang_CXX, "input0.cc");
    auto *FromD = FirstDeclMatcher<ClassTemplateDecl>().match(
        FromTU, classTemplateDecl(hasName("B")));

    Import(FromD, Lang_CXX);
  }

  {
    Decl *FromTU = getTuDecl(
        R"(
            template <typename T>
            struct B {
              void f();
            };
            )",
        Lang_CXX, "input1.cc");
    FunctionDecl *FromD = FirstDeclMatcher<FunctionDecl>().match(
        FromTU, functionDecl(hasName("f")));
    Import(FromD, Lang_CXX);
    auto *FromCTD = FirstDeclMatcher<ClassTemplateDecl>().match(
        FromTU, classTemplateDecl(hasName("B")));
    auto *ToCTD = cast<ClassTemplateDecl>(Import(FromCTD, Lang_CXX));
    EXPECT_TRUE(ToCTD->isThisDeclarationADefinition());
  }
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportDefinitionOfClassTemplateIfThereIsAnExistingFwdDeclAndDefinition) {
  Decl *ToTU = getToTuDecl(
      R"(
      template <typename T>
      struct B {
        void f();
      };

      template <typename T>
      struct B;
      )",
      Lang_CXX);
  ASSERT_EQ(1u, DeclCounterWithPredicate<ClassTemplateDecl>(
                    [](const ClassTemplateDecl *T) {
                      return T->isThisDeclarationADefinition();
                    })
                    .match(ToTU, classTemplateDecl()));

  Decl *FromTU = getTuDecl(
      R"(
      template <typename T>
      struct B {
        void f();
      };
      )",
      Lang_CXX, "input1.cc");
  ClassTemplateDecl *FromD = FirstDeclMatcher<ClassTemplateDecl>().match(
      FromTU, classTemplateDecl(hasName("B")));

  Import(FromD, Lang_CXX);

  // We should have only one definition.
  EXPECT_EQ(1u, DeclCounterWithPredicate<ClassTemplateDecl>(
                    [](const ClassTemplateDecl *T) {
                      return T->isThisDeclarationADefinition();
                    })
                    .match(ToTU, classTemplateDecl()));
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportDefinitionOfClassIfThereIsAnExistingFwdDeclAndDefinition) {
  Decl *ToTU = getToTuDecl(
      R"(
      struct B {
        void f();
      };

      struct B;
      )",
      Lang_CXX);
  ASSERT_EQ(2u, DeclCounter<CXXRecordDecl>().match(
                    ToTU, cxxRecordDecl(unless(isImplicit()))));

  Decl *FromTU = getTuDecl(
      R"(
      struct B {
        void f();
      };
      )",
      Lang_CXX, "input1.cc");
  auto *FromD = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("B")));

  Import(FromD, Lang_CXX);

  EXPECT_EQ(2u, DeclCounter<CXXRecordDecl>().match(
                    ToTU, cxxRecordDecl(unless(isImplicit()))));
}

static void CompareSourceLocs(FullSourceLoc Loc1, FullSourceLoc Loc2) {
  EXPECT_EQ(Loc1.getExpansionLineNumber(), Loc2.getExpansionLineNumber());
  EXPECT_EQ(Loc1.getExpansionColumnNumber(), Loc2.getExpansionColumnNumber());
  EXPECT_EQ(Loc1.getSpellingLineNumber(), Loc2.getSpellingLineNumber());
  EXPECT_EQ(Loc1.getSpellingColumnNumber(), Loc2.getSpellingColumnNumber());
}
static void CompareSourceRanges(SourceRange Range1, SourceRange Range2,
                                SourceManager &SM1, SourceManager &SM2) {
  CompareSourceLocs(FullSourceLoc{ Range1.getBegin(), SM1 },
                    FullSourceLoc{ Range2.getBegin(), SM2 });
  CompareSourceLocs(FullSourceLoc{ Range1.getEnd(), SM1 },
                    FullSourceLoc{ Range2.getEnd(), SM2 });
}
TEST_P(ASTImporterOptionSpecificTestBase, ImportSourceLocs) {
  Decl *FromTU = getTuDecl(
      R"(
      #define MFOO(arg) arg = arg + 1

      void foo() {
        int a = 5;
        MFOO(a);
      }
      )",
      Lang_CXX);
  auto FromD = FirstDeclMatcher<FunctionDecl>().match(FromTU, functionDecl());
  auto ToD = Import(FromD, Lang_CXX);

  auto ToLHS = LastDeclMatcher<DeclRefExpr>().match(ToD, declRefExpr());
  auto FromLHS = LastDeclMatcher<DeclRefExpr>().match(FromTU, declRefExpr());
  auto ToRHS = LastDeclMatcher<IntegerLiteral>().match(ToD, integerLiteral());
  auto FromRHS =
      LastDeclMatcher<IntegerLiteral>().match(FromTU, integerLiteral());

  SourceManager &ToSM = ToAST->getASTContext().getSourceManager();
  SourceManager &FromSM = FromD->getASTContext().getSourceManager();
  CompareSourceRanges(ToD->getSourceRange(), FromD->getSourceRange(), ToSM,
                      FromSM);
  CompareSourceRanges(ToLHS->getSourceRange(), FromLHS->getSourceRange(), ToSM,
                      FromSM);
  CompareSourceRanges(ToRHS->getSourceRange(), FromRHS->getSourceRange(), ToSM,
                      FromSM);
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportNestedMacro) {
  Decl *FromTU = getTuDecl(
      R"(
      #define FUNC_INT void declToImport
      #define FUNC FUNC_INT
      FUNC(int a);
      )",
      Lang_CXX);
  auto FromD = FirstDeclMatcher<FunctionDecl>().match(FromTU, functionDecl());
  auto ToD = Import(FromD, Lang_CXX);

  SourceManager &ToSM = ToAST->getASTContext().getSourceManager();
  SourceManager &FromSM = FromD->getASTContext().getSourceManager();
  CompareSourceRanges(ToD->getSourceRange(), FromD->getSourceRange(), ToSM,
                      FromSM);
}

TEST_P(
    ASTImporterOptionSpecificTestBase,
    ImportDefinitionOfClassTemplateSpecIfThereIsAnExistingFwdDeclAndDefinition) {
  Decl *ToTU = getToTuDecl(
      R"(
      template <typename T>
      struct B;

      template <>
      struct B<int> {};

      template <>
      struct B<int>;
      )",
      Lang_CXX);
  // We should have only one definition.
  ASSERT_EQ(1u, DeclCounterWithPredicate<ClassTemplateSpecializationDecl>(
                    [](const ClassTemplateSpecializationDecl *T) {
                      return T->isThisDeclarationADefinition();
                    })
                    .match(ToTU, classTemplateSpecializationDecl()));

  Decl *FromTU = getTuDecl(
      R"(
      template <typename T>
      struct B;

      template <>
      struct B<int> {};
      )",
      Lang_CXX, "input1.cc");
  auto *FromD = FirstDeclMatcher<ClassTemplateSpecializationDecl>().match(
      FromTU, classTemplateSpecializationDecl(hasName("B")));

  Import(FromD, Lang_CXX);

  // We should have only one definition.
  EXPECT_EQ(1u, DeclCounterWithPredicate<ClassTemplateSpecializationDecl>(
                    [](const ClassTemplateSpecializationDecl *T) {
                      return T->isThisDeclarationADefinition();
                    })
                    .match(ToTU, classTemplateSpecializationDecl()));
}

TEST_P(ASTImporterOptionSpecificTestBase, ObjectsWithUnnamedStructType) {
  Decl *FromTU = getTuDecl(
      R"(
      struct { int a; int b; } object0 = { 2, 3 };
      struct { int x; int y; int z; } object1;
      )",
      Lang_CXX, "input0.cc");

  auto *Obj0 =
      FirstDeclMatcher<VarDecl>().match(FromTU, varDecl(hasName("object0")));
  auto *From0 = getRecordDecl(Obj0);
  auto *Obj1 =
      FirstDeclMatcher<VarDecl>().match(FromTU, varDecl(hasName("object1")));
  auto *From1 = getRecordDecl(Obj1);

  auto *To0 = Import(From0, Lang_CXX);
  auto *To1 = Import(From1, Lang_CXX);

  EXPECT_TRUE(To0);
  EXPECT_TRUE(To1);
  EXPECT_NE(To0, To1);
  EXPECT_NE(To0->getCanonicalDecl(), To1->getCanonicalDecl());
}

TEST_P(ASTImporterOptionSpecificTestBase, AnonymousRecords) {
  auto *Code =
      R"(
      struct X {
        struct { int a; };
        struct { int b; };
      };
      )";
  Decl *FromTU0 = getTuDecl(Code, Lang_C, "input0.c");

  Decl *FromTU1 = getTuDecl(Code, Lang_C, "input1.c");

  auto *X0 =
      FirstDeclMatcher<RecordDecl>().match(FromTU0, recordDecl(hasName("X")));
  auto *X1 =
      FirstDeclMatcher<RecordDecl>().match(FromTU1, recordDecl(hasName("X")));
  Import(X0, Lang_C);
  Import(X1, Lang_C);

  auto *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  // We expect no (ODR) warning during the import.
  EXPECT_EQ(0u, ToTU->getASTContext().getDiagnostics().getNumWarnings());
  EXPECT_EQ(1u,
            DeclCounter<RecordDecl>().match(ToTU, recordDecl(hasName("X"))));
}

TEST_P(ASTImporterOptionSpecificTestBase, AnonymousRecordsReversed) {
  Decl *FromTU0 = getTuDecl(
      R"(
      struct X {
        struct { int a; };
        struct { int b; };
      };
      )",
      Lang_C, "input0.c");

  Decl *FromTU1 = getTuDecl(
      R"(
      struct X { // reversed order
        struct { int b; };
        struct { int a; };
      };
      )",
      Lang_C, "input1.c");

  auto *X0 =
      FirstDeclMatcher<RecordDecl>().match(FromTU0, recordDecl(hasName("X")));
  auto *X1 =
      FirstDeclMatcher<RecordDecl>().match(FromTU1, recordDecl(hasName("X")));
  Import(X0, Lang_C);
  Import(X1, Lang_C);

  auto *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  // We expect one (ODR) warning during the import.
  EXPECT_EQ(1u, ToTU->getASTContext().getDiagnostics().getNumWarnings());
  EXPECT_EQ(2u,
            DeclCounter<RecordDecl>().match(ToTU, recordDecl(hasName("X"))));
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportDoesUpdateUsedFlag) {
  auto Pattern = varDecl(hasName("x"));
  VarDecl *Imported1;
  {
    Decl *FromTU = getTuDecl("extern int x;", Lang_CXX, "input0.cc");
    auto *FromD = FirstDeclMatcher<VarDecl>().match(FromTU, Pattern);
    Imported1 = cast<VarDecl>(Import(FromD, Lang_CXX));
  }
  VarDecl *Imported2;
  {
    Decl *FromTU = getTuDecl("int x;", Lang_CXX, "input1.cc");
    auto *FromD = FirstDeclMatcher<VarDecl>().match(FromTU, Pattern);
    Imported2 = cast<VarDecl>(Import(FromD, Lang_CXX));
  }
  EXPECT_EQ(Imported1->getCanonicalDecl(), Imported2->getCanonicalDecl());
  EXPECT_FALSE(Imported2->isUsed(false));
  {
    Decl *FromTU =
        getTuDecl("extern int x; int f() { return x; }", Lang_CXX, "input2.cc");
    auto *FromD = FirstDeclMatcher<FunctionDecl>().match(
        FromTU, functionDecl(hasName("f")));
    Import(FromD, Lang_CXX);
  }
  EXPECT_TRUE(Imported2->isUsed(false));
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportDoesUpdateUsedFlag2) {
  auto Pattern = varDecl(hasName("x"));
  VarDecl *ExistingD;
  {
    Decl *ToTU = getToTuDecl("int x = 1;", Lang_CXX);
    ExistingD = FirstDeclMatcher<VarDecl>().match(ToTU, Pattern);
  }
  EXPECT_FALSE(ExistingD->isUsed(false));
  {
    Decl *FromTU = getTuDecl(
        "int x = 1; int f() { return x; }", Lang_CXX, "input1.cc");
    auto *FromD = FirstDeclMatcher<FunctionDecl>().match(
        FromTU, functionDecl(hasName("f")));
    Import(FromD, Lang_CXX);
  }
  EXPECT_TRUE(ExistingD->isUsed(false));
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportDoesUpdateUsedFlag3) {
  auto Pattern = varDecl(hasName("a"));
  VarDecl *ExistingD;
  {
    Decl *ToTU = getToTuDecl(
        R"(
        struct A {
          static const int a = 1;
        };
        )", Lang_CXX);
    ExistingD = FirstDeclMatcher<VarDecl>().match(ToTU, Pattern);
  }
  EXPECT_FALSE(ExistingD->isUsed(false));
  {
    Decl *FromTU = getTuDecl(
        R"(
        struct A {
          static const int a = 1;
        };
        const int *f() { return &A::a; } // requires storage,
                                         // thus used flag will be set
        )", Lang_CXX, "input1.cc");
    auto *FromFunD = FirstDeclMatcher<FunctionDecl>().match(
        FromTU, functionDecl(hasName("f")));
    auto *FromD = FirstDeclMatcher<VarDecl>().match(FromTU, Pattern);
    ASSERT_TRUE(FromD->isUsed(false));
    Import(FromFunD, Lang_CXX);
  }
  EXPECT_TRUE(ExistingD->isUsed(false));
}

TEST_P(ASTImporterOptionSpecificTestBase, ReimportWithUsedFlag) {
  auto Pattern = varDecl(hasName("x"));

  Decl *FromTU = getTuDecl("int x;", Lang_CXX, "input0.cc");
  auto *FromD = FirstDeclMatcher<VarDecl>().match(FromTU, Pattern);

  auto *Imported1 = cast<VarDecl>(Import(FromD, Lang_CXX));

  ASSERT_FALSE(Imported1->isUsed(false));

  FromD->setIsUsed();
  auto *Imported2 = cast<VarDecl>(Import(FromD, Lang_CXX));

  EXPECT_EQ(Imported1, Imported2);
  EXPECT_TRUE(Imported2->isUsed(false));
}

struct ImportFunctions : ASTImporterOptionSpecificTestBase {};

TEST_P(ImportFunctions, ImportPrototypeOfRecursiveFunction) {
  Decl *FromTU = getTuDecl("void f(); void f() { f(); }", Lang_CXX);
  auto Pattern = functionDecl(hasName("f"));
  auto *From =
      FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern); // Proto

  Decl *ImportedD = Import(From, Lang_CXX);
  Decl *ToTU = ImportedD->getTranslationUnitDecl();

  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, Pattern), 2u);
  auto *To0 = FirstDeclMatcher<FunctionDecl>().match(ToTU, Pattern);
  auto *To1 = LastDeclMatcher<FunctionDecl>().match(ToTU, Pattern);
  EXPECT_TRUE(ImportedD == To0);
  EXPECT_FALSE(To0->doesThisDeclarationHaveABody());
  EXPECT_TRUE(To1->doesThisDeclarationHaveABody());
  EXPECT_EQ(To1->getPreviousDecl(), To0);
}

TEST_P(ImportFunctions, ImportDefinitionOfRecursiveFunction) {
  Decl *FromTU = getTuDecl("void f(); void f() { f(); }", Lang_CXX);
  auto Pattern = functionDecl(hasName("f"));
  auto *From =
      LastDeclMatcher<FunctionDecl>().match(FromTU, Pattern); // Def

  Decl *ImportedD = Import(From, Lang_CXX);
  Decl *ToTU = ImportedD->getTranslationUnitDecl();

  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, Pattern), 2u);
  auto *To0 = FirstDeclMatcher<FunctionDecl>().match(ToTU, Pattern);
  auto *To1 = LastDeclMatcher<FunctionDecl>().match(ToTU, Pattern);
  EXPECT_TRUE(ImportedD == To1);
  EXPECT_FALSE(To0->doesThisDeclarationHaveABody());
  EXPECT_TRUE(To1->doesThisDeclarationHaveABody());
  EXPECT_EQ(To1->getPreviousDecl(), To0);
}

TEST_P(ImportFunctions, OverriddenMethodsShouldBeImported) {
  auto Code =
      R"(
      struct B { virtual void f(); };
      void B::f() {}
      struct D : B { void f(); };
      )";
  auto Pattern =
      cxxMethodDecl(hasName("f"), hasParent(cxxRecordDecl(hasName("D"))));
  Decl *FromTU = getTuDecl(Code, Lang_CXX);
  CXXMethodDecl *Proto =
      FirstDeclMatcher<CXXMethodDecl>().match(FromTU, Pattern);

  ASSERT_EQ(Proto->size_overridden_methods(), 1u);
  CXXMethodDecl *To = cast<CXXMethodDecl>(Import(Proto, Lang_CXX));
  EXPECT_EQ(To->size_overridden_methods(), 1u);
}

TEST_P(ImportFunctions, VirtualFlagShouldBePreservedWhenImportingPrototype) {
  auto Code =
      R"(
      struct B { virtual void f(); };
      void B::f() {}
      )";
  auto Pattern =
      cxxMethodDecl(hasName("f"), hasParent(cxxRecordDecl(hasName("B"))));
  Decl *FromTU = getTuDecl(Code, Lang_CXX);
  CXXMethodDecl *Proto =
      FirstDeclMatcher<CXXMethodDecl>().match(FromTU, Pattern);
  CXXMethodDecl *Def = LastDeclMatcher<CXXMethodDecl>().match(FromTU, Pattern);

  ASSERT_TRUE(Proto->isVirtual());
  ASSERT_TRUE(Def->isVirtual());
  CXXMethodDecl *To = cast<CXXMethodDecl>(Import(Proto, Lang_CXX));
  EXPECT_TRUE(To->isVirtual());
}

TEST_P(ImportFunctions,
       ImportDefinitionIfThereIsAnExistingDefinitionAndFwdDecl) {
  Decl *ToTU = getToTuDecl(
      R"(
      void f() {}
      void f();
      )",
      Lang_CXX);
  ASSERT_EQ(1u,
            DeclCounterWithPredicate<FunctionDecl>([](const FunctionDecl *FD) {
              return FD->doesThisDeclarationHaveABody();
            }).match(ToTU, functionDecl()));

  Decl *FromTU = getTuDecl("void f() {}", Lang_CXX, "input0.cc");
  auto *FromD = FirstDeclMatcher<FunctionDecl>().match(FromTU, functionDecl());

  Import(FromD, Lang_CXX);

  EXPECT_EQ(1u,
            DeclCounterWithPredicate<FunctionDecl>([](const FunctionDecl *FD) {
              return FD->doesThisDeclarationHaveABody();
            }).match(ToTU, functionDecl()));
}

TEST_P(ImportFunctions, ImportOverriddenMethodTwice) {
  auto Code =
      R"(
      struct B { virtual void f(); };
      struct D:B { void f(); };
      )";
  auto BFP =
      cxxMethodDecl(hasName("f"), hasParent(cxxRecordDecl(hasName("B"))));
  auto DFP =
      cxxMethodDecl(hasName("f"), hasParent(cxxRecordDecl(hasName("D"))));

  Decl *FromTU0 = getTuDecl(Code, Lang_CXX);
  auto *DF = FirstDeclMatcher<CXXMethodDecl>().match(FromTU0, DFP);
  Import(DF, Lang_CXX);

  Decl *FromTU1 = getTuDecl(Code, Lang_CXX, "input1.cc");
  auto *BF = FirstDeclMatcher<CXXMethodDecl>().match(FromTU1, BFP);
  Import(BF, Lang_CXX);

  auto *ToTU = ToAST->getASTContext().getTranslationUnitDecl();

  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, BFP), 1u);
  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, DFP), 1u);
}

TEST_P(ImportFunctions, ImportOverriddenMethodTwiceDefinitionFirst) {
  auto CodeWithoutDef =
      R"(
      struct B { virtual void f(); };
      struct D:B { void f(); };
      )";
  auto CodeWithDef =
      R"(
    struct B { virtual void f(){}; };
    struct D:B { void f(){}; };
  )";
  auto BFP =
      cxxMethodDecl(hasName("f"), hasParent(cxxRecordDecl(hasName("B"))));
  auto DFP =
      cxxMethodDecl(hasName("f"), hasParent(cxxRecordDecl(hasName("D"))));
  auto BFDefP = cxxMethodDecl(
      hasName("f"), hasParent(cxxRecordDecl(hasName("B"))), isDefinition());
  auto DFDefP = cxxMethodDecl(
      hasName("f"), hasParent(cxxRecordDecl(hasName("D"))), isDefinition());
  auto FDefAllP = cxxMethodDecl(hasName("f"), isDefinition());

  {
    Decl *FromTU = getTuDecl(CodeWithDef, Lang_CXX, "input0.cc");
    auto *FromD = FirstDeclMatcher<CXXMethodDecl>().match(FromTU, DFP);
    Import(FromD, Lang_CXX);
  }
  {
    Decl *FromTU = getTuDecl(CodeWithoutDef, Lang_CXX, "input1.cc");
    auto *FromB = FirstDeclMatcher<CXXMethodDecl>().match(FromTU, BFP);
    Import(FromB, Lang_CXX);
  }

  auto *ToTU = ToAST->getASTContext().getTranslationUnitDecl();

  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, BFP), 1u);
  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, DFP), 1u);
  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, BFDefP), 1u);
  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, DFDefP), 1u);
  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, FDefAllP), 2u);
}

TEST_P(ImportFunctions, ImportOverriddenMethodTwiceOutOfClassDef) {
  auto Code =
      R"(
      struct B { virtual void f(); };
      struct D:B { void f(); };
      void B::f(){};
      )";

  auto BFP =
      cxxMethodDecl(hasName("f"), hasParent(cxxRecordDecl(hasName("B"))));
  auto BFDefP = cxxMethodDecl(
      hasName("f"), hasParent(cxxRecordDecl(hasName("B"))), isDefinition());
  auto DFP = cxxMethodDecl(hasName("f"), hasParent(cxxRecordDecl(hasName("D"))),
                           unless(isDefinition()));

  Decl *FromTU0 = getTuDecl(Code, Lang_CXX);
  auto *D = FirstDeclMatcher<CXXMethodDecl>().match(FromTU0, DFP);
  Import(D, Lang_CXX);

  Decl *FromTU1 = getTuDecl(Code, Lang_CXX, "input1.cc");
  auto *B = FirstDeclMatcher<CXXMethodDecl>().match(FromTU1, BFP);
  Import(B, Lang_CXX);

  auto *ToTU = ToAST->getASTContext().getTranslationUnitDecl();

  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, BFP), 1u);
  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, BFDefP), 0u);

  auto *ToB = FirstDeclMatcher<CXXRecordDecl>().match(
      ToTU, cxxRecordDecl(hasName("B")));
  auto *ToBFInClass = FirstDeclMatcher<CXXMethodDecl>().match(ToTU, BFP);
  auto *ToBFOutOfClass = FirstDeclMatcher<CXXMethodDecl>().match(
      ToTU, cxxMethodDecl(hasName("f"), isDefinition()));

  // The definition should be out-of-class.
  EXPECT_NE(ToBFInClass, ToBFOutOfClass);
  EXPECT_NE(ToBFInClass->getLexicalDeclContext(),
            ToBFOutOfClass->getLexicalDeclContext());
  EXPECT_EQ(ToBFOutOfClass->getDeclContext(), ToB);
  EXPECT_EQ(ToBFOutOfClass->getLexicalDeclContext(), ToTU);

  // Check that the redecl chain is intact.
  EXPECT_EQ(ToBFOutOfClass->getPreviousDecl(), ToBFInClass);
}

TEST_P(ImportFunctions,
       ImportOverriddenMethodTwiceOutOfClassDefInSeparateCode) {
  auto CodeTU0 =
      R"(
      struct B { virtual void f(); };
      struct D:B { void f(); };
      )";
  auto CodeTU1 =
      R"(
      struct B { virtual void f(); };
      struct D:B { void f(); };
      void B::f(){}
      void D::f(){}
      void foo(B &b, D &d) { b.f(); d.f(); }
      )";

  auto BFP =
      cxxMethodDecl(hasName("f"), hasParent(cxxRecordDecl(hasName("B"))));
  auto BFDefP = cxxMethodDecl(
      hasName("f"), hasParent(cxxRecordDecl(hasName("B"))), isDefinition());
  auto DFP =
      cxxMethodDecl(hasName("f"), hasParent(cxxRecordDecl(hasName("D"))));
  auto DFDefP = cxxMethodDecl(
      hasName("f"), hasParent(cxxRecordDecl(hasName("D"))), isDefinition());
  auto FooDef = functionDecl(hasName("foo"));

  {
    Decl *FromTU0 = getTuDecl(CodeTU0, Lang_CXX, "input0.cc");
    auto *D = FirstDeclMatcher<CXXMethodDecl>().match(FromTU0, DFP);
    Import(D, Lang_CXX);
  }

  {
    Decl *FromTU1 = getTuDecl(CodeTU1, Lang_CXX, "input1.cc");
    auto *Foo = FirstDeclMatcher<FunctionDecl>().match(FromTU1, FooDef);
    Import(Foo, Lang_CXX);
  }

  auto *ToTU = ToAST->getASTContext().getTranslationUnitDecl();

  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, BFP), 1u);
  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, DFP), 1u);
  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, BFDefP), 0u);
  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, DFDefP), 0u);

  auto *ToB = FirstDeclMatcher<CXXRecordDecl>().match(
      ToTU, cxxRecordDecl(hasName("B")));
  auto *ToD = FirstDeclMatcher<CXXRecordDecl>().match(
      ToTU, cxxRecordDecl(hasName("D")));
  auto *ToBFInClass = FirstDeclMatcher<CXXMethodDecl>().match(ToTU, BFP);
  auto *ToBFOutOfClass = FirstDeclMatcher<CXXMethodDecl>().match(
      ToTU, cxxMethodDecl(hasName("f"), isDefinition()));
  auto *ToDFInClass = FirstDeclMatcher<CXXMethodDecl>().match(ToTU, DFP);
  auto *ToDFOutOfClass = LastDeclMatcher<CXXMethodDecl>().match(
      ToTU, cxxMethodDecl(hasName("f"), isDefinition()));

  // The definition should be out-of-class.
  EXPECT_NE(ToBFInClass, ToBFOutOfClass);
  EXPECT_NE(ToBFInClass->getLexicalDeclContext(),
            ToBFOutOfClass->getLexicalDeclContext());
  EXPECT_EQ(ToBFOutOfClass->getDeclContext(), ToB);
  EXPECT_EQ(ToBFOutOfClass->getLexicalDeclContext(), ToTU);

  EXPECT_NE(ToDFInClass, ToDFOutOfClass);
  EXPECT_NE(ToDFInClass->getLexicalDeclContext(),
            ToDFOutOfClass->getLexicalDeclContext());
  EXPECT_EQ(ToDFOutOfClass->getDeclContext(), ToD);
  EXPECT_EQ(ToDFOutOfClass->getLexicalDeclContext(), ToTU);

  // Check that the redecl chain is intact.
  EXPECT_EQ(ToBFOutOfClass->getPreviousDecl(), ToBFInClass);
  EXPECT_EQ(ToDFOutOfClass->getPreviousDecl(), ToDFInClass);
}

//FIXME Move these tests to a separate test file.
namespace TypeAndValueParameterizedTests {

// Type parameters for type-parameterized test fixtures.
struct GetFunPattern {
  using DeclTy = FunctionDecl;
  BindableMatcher<Decl> operator()() { return functionDecl(hasName("f")); }
};
struct GetVarPattern {
  using DeclTy = VarDecl;
  BindableMatcher<Decl> operator()() { return varDecl(hasName("v")); }
};

// Values for the value-parameterized test fixtures.
// FunctionDecl:
auto *ExternF = "void f();";
auto *StaticF = "static void f();";
auto *AnonF = "namespace { void f(); }";
// VarDecl:
auto *ExternV = "extern int v;";
auto *StaticV = "static int v;";
auto *AnonV = "namespace { extern int v; }";

// First value in tuple: Compile options.
// Second value in tuple: Source code to be used in the test.
using ImportVisibilityChainParams =
    ::testing::WithParamInterface<std::tuple<ArgVector, const char *>>;
// Fixture to test the redecl chain of Decls with the same visibility.  Gtest
// makes it possible to have either value-parameterized or type-parameterized
// fixtures.  However, we cannot have both value- and type-parameterized test
// fixtures.  This is a value-parameterized test fixture in the gtest sense. We
// intend to mimic gtest's type-parameters via the PatternFactory template
// parameter.  We manually instantiate the different tests with the each types.
template <typename PatternFactory>
class ImportVisibilityChain
    : public ASTImporterTestBase, public ImportVisibilityChainParams {
protected:
  using DeclTy = typename PatternFactory::DeclTy;
  ArgVector getExtraArgs() const override { return std::get<0>(GetParam()); }
  std::string getCode() const { return std::get<1>(GetParam()); }
  BindableMatcher<Decl> getPattern() const { return PatternFactory()(); }

  // Type-parameterized test.
  void TypedTest_ImportChain() {
    std::string Code = getCode() + getCode();
    auto Pattern = getPattern();

    TranslationUnitDecl *FromTu = getTuDecl(Code, Lang_CXX, "input0.cc");

    auto *FromF0 = FirstDeclMatcher<DeclTy>().match(FromTu, Pattern);
    auto *FromF1 = LastDeclMatcher<DeclTy>().match(FromTu, Pattern);

    auto *ToF0 = Import(FromF0, Lang_CXX);
    auto *ToF1 = Import(FromF1, Lang_CXX);

    EXPECT_TRUE(ToF0);
    ASSERT_TRUE(ToF1);
    EXPECT_NE(ToF0, ToF1);
    EXPECT_EQ(ToF1->getPreviousDecl(), ToF0);
  }
};

// Manual instantiation of the fixture with each type.
using ImportFunctionsVisibilityChain = ImportVisibilityChain<GetFunPattern>;
using ImportVariablesVisibilityChain = ImportVisibilityChain<GetVarPattern>;
// Value-parameterized test for the first type.
TEST_P(ImportFunctionsVisibilityChain, ImportChain) {
  TypedTest_ImportChain();
}
// Value-parameterized test for the second type.
TEST_P(ImportVariablesVisibilityChain, ImportChain) {
  TypedTest_ImportChain();
}

// Automatic instantiation of the value-parameterized tests.
INSTANTIATE_TEST_CASE_P(ParameterizedTests, ImportFunctionsVisibilityChain,
                        ::testing::Combine(
                           DefaultTestValuesForRunOptions,
                           ::testing::Values(ExternF, StaticF, AnonF)), );
INSTANTIATE_TEST_CASE_P(
    ParameterizedTests, ImportVariablesVisibilityChain,
    ::testing::Combine(
        DefaultTestValuesForRunOptions,
        // There is no point to instantiate with StaticV, because in C++ we can
        // forward declare a variable only with the 'extern' keyword.
        // Consequently, each fwd declared variable has external linkage.  This
        // is different in the C language where any declaration without an
        // initializer is a tentative definition, subsequent definitions may be
        // provided but they must have the same linkage.  See also the test
        // ImportVariableChainInC which test for this special C Lang case.
        ::testing::Values(ExternV, AnonV)), );

// First value in tuple: Compile options.
// Second value in tuple: Tuple with informations for the test.
// Code for first import (or initial code), code to import, whether the `f`
// functions are expected to be linked in a declaration chain.
// One value of this tuple is combined with every value of compile options.
// The test can have a single tuple as parameter only.
using ImportVisibilityParams = ::testing::WithParamInterface<
    std::tuple<ArgVector, std::tuple<const char *, const char *, bool>>>;

template <typename PatternFactory>
class ImportVisibility
    : public ASTImporterTestBase,
      public ImportVisibilityParams {
protected:
  using DeclTy = typename PatternFactory::DeclTy;
  ArgVector getExtraArgs() const override { return std::get<0>(GetParam()); }
  std::string getCode0() const { return std::get<0>(std::get<1>(GetParam())); }
  std::string getCode1() const { return std::get<1>(std::get<1>(GetParam())); }
  bool shouldBeLinked() const { return std::get<2>(std::get<1>(GetParam())); }
  BindableMatcher<Decl> getPattern() const { return PatternFactory()(); }

  void TypedTest_ImportAfter() {
    TranslationUnitDecl *ToTu = getToTuDecl(getCode0(), Lang_CXX);
    TranslationUnitDecl *FromTu = getTuDecl(getCode1(), Lang_CXX, "input1.cc");

    auto *ToF0 = FirstDeclMatcher<DeclTy>().match(ToTu, getPattern());
    auto *FromF1 = FirstDeclMatcher<DeclTy>().match(FromTu, getPattern());

    auto *ToF1 = Import(FromF1, Lang_CXX);

    ASSERT_TRUE(ToF0);
    ASSERT_TRUE(ToF1);
    EXPECT_NE(ToF0, ToF1);

    if (shouldBeLinked())
      EXPECT_EQ(ToF1->getPreviousDecl(), ToF0);
    else
      EXPECT_FALSE(ToF1->getPreviousDecl());
  }

  void TypedTest_ImportAfterImport() {
    TranslationUnitDecl *FromTu0 = getTuDecl(getCode0(), Lang_CXX, "input0.cc");
    TranslationUnitDecl *FromTu1 = getTuDecl(getCode1(), Lang_CXX, "input1.cc");
    auto *FromF0 =
        FirstDeclMatcher<DeclTy>().match(FromTu0, getPattern());
    auto *FromF1 =
        FirstDeclMatcher<DeclTy>().match(FromTu1, getPattern());
    auto *ToF0 = Import(FromF0, Lang_CXX);
    auto *ToF1 = Import(FromF1, Lang_CXX);
    ASSERT_TRUE(ToF0);
    ASSERT_TRUE(ToF1);
    EXPECT_NE(ToF0, ToF1);
    if (shouldBeLinked())
      EXPECT_EQ(ToF1->getPreviousDecl(), ToF0);
    else
      EXPECT_FALSE(ToF1->getPreviousDecl());
  }
};
using ImportFunctionsVisibility = ImportVisibility<GetFunPattern>;
using ImportVariablesVisibility = ImportVisibility<GetVarPattern>;

// FunctionDecl.
TEST_P(ImportFunctionsVisibility, ImportAfter) {
  TypedTest_ImportAfter();
}
TEST_P(ImportFunctionsVisibility, ImportAfterImport) {
  TypedTest_ImportAfterImport();
}
// VarDecl.
TEST_P(ImportVariablesVisibility, ImportAfter) {
  TypedTest_ImportAfter();
}
TEST_P(ImportVariablesVisibility, ImportAfterImport) {
  TypedTest_ImportAfterImport();
}

bool ExpectLink = true;
bool ExpectNotLink = false;

INSTANTIATE_TEST_CASE_P(
    ParameterizedTests, ImportFunctionsVisibility,
    ::testing::Combine(
        DefaultTestValuesForRunOptions,
        ::testing::Values(std::make_tuple(ExternF, ExternF, ExpectLink),
                          std::make_tuple(ExternF, StaticF, ExpectNotLink),
                          std::make_tuple(ExternF, AnonF, ExpectNotLink),
                          std::make_tuple(StaticF, ExternF, ExpectNotLink),
                          std::make_tuple(StaticF, StaticF, ExpectNotLink),
                          std::make_tuple(StaticF, AnonF, ExpectNotLink),
                          std::make_tuple(AnonF, ExternF, ExpectNotLink),
                          std::make_tuple(AnonF, StaticF, ExpectNotLink),
                          std::make_tuple(AnonF, AnonF, ExpectNotLink))), );
INSTANTIATE_TEST_CASE_P(
    ParameterizedTests, ImportVariablesVisibility,
    ::testing::Combine(
        DefaultTestValuesForRunOptions,
        ::testing::Values(std::make_tuple(ExternV, ExternV, ExpectLink),
                          std::make_tuple(ExternV, StaticV, ExpectNotLink),
                          std::make_tuple(ExternV, AnonV, ExpectNotLink),
                          std::make_tuple(StaticV, ExternV, ExpectNotLink),
                          std::make_tuple(StaticV, StaticV, ExpectNotLink),
                          std::make_tuple(StaticV, AnonV, ExpectNotLink),
                          std::make_tuple(AnonV, ExternV, ExpectNotLink),
                          std::make_tuple(AnonV, StaticV, ExpectNotLink),
                          std::make_tuple(AnonV, AnonV, ExpectNotLink))), );

} // namespace TypeAndValueParameterizedTests

TEST_P(ASTImporterOptionSpecificTestBase, ImportVariableChainInC) {
    std::string Code = "static int v; static int v = 0;";
    auto Pattern = varDecl(hasName("v"));

    TranslationUnitDecl *FromTu = getTuDecl(Code, Lang_C, "input0.c");

    auto *From0 = FirstDeclMatcher<VarDecl>().match(FromTu, Pattern);
    auto *From1 = LastDeclMatcher<VarDecl>().match(FromTu, Pattern);

    auto *To0 = Import(From0, Lang_C);
    auto *To1 = Import(From1, Lang_C);

    EXPECT_TRUE(To0);
    ASSERT_TRUE(To1);
    EXPECT_NE(To0, To1);
    EXPECT_EQ(To1->getPreviousDecl(), To0);
}

TEST_P(ImportFunctions, ImportFromDifferentScopedAnonNamespace) {
  TranslationUnitDecl *FromTu = getTuDecl(
      "namespace NS0 { namespace { void f(); } }"
      "namespace NS1 { namespace { void f(); } }",
      Lang_CXX, "input0.cc");
  auto Pattern = functionDecl(hasName("f"));

  auto *FromF0 = FirstDeclMatcher<FunctionDecl>().match(FromTu, Pattern);
  auto *FromF1 = LastDeclMatcher<FunctionDecl>().match(FromTu, Pattern);

  auto *ToF0 = Import(FromF0, Lang_CXX);
  auto *ToF1 = Import(FromF1, Lang_CXX);

  EXPECT_TRUE(ToF0);
  ASSERT_TRUE(ToF1);
  EXPECT_NE(ToF0, ToF1);
  EXPECT_FALSE(ToF1->getPreviousDecl());
}

TEST_P(ImportFunctions, ImportFunctionFromUnnamedNamespace) {
  {
    Decl *FromTU = getTuDecl("namespace { void f() {} } void g0() { f(); }",
                             Lang_CXX, "input0.cc");
    auto *FromD = FirstDeclMatcher<FunctionDecl>().match(
        FromTU, functionDecl(hasName("g0")));

    Import(FromD, Lang_CXX);
  }
  {
    Decl *FromTU =
        getTuDecl("namespace { void f() { int a; } } void g1() { f(); }",
                  Lang_CXX, "input1.cc");
    auto *FromD = FirstDeclMatcher<FunctionDecl>().match(
        FromTU, functionDecl(hasName("g1")));
    Import(FromD, Lang_CXX);
  }

  Decl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  ASSERT_EQ(DeclCounter<FunctionDecl>().match(ToTU, functionDecl(hasName("f"))),
            2u);
}

TEST_P(ImportFunctions, ImportImplicitFunctionsInLambda) {
  Decl *FromTU = getTuDecl(
      R"(
      void foo() {
        (void)[]() { ; };
      }
      )",
      Lang_CXX11);
  auto *FromD = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("foo")));
  auto *ToD = Import(FromD, Lang_CXX);
  EXPECT_TRUE(ToD);
  CXXRecordDecl *LambdaRec =
      cast<LambdaExpr>(cast<CStyleCastExpr>(
                           *cast<CompoundStmt>(ToD->getBody())->body_begin())
                           ->getSubExpr())
          ->getLambdaClass();
  EXPECT_TRUE(LambdaRec->getDestructor());
}

TEST_P(ImportFunctions,
       CallExprOfMemberFunctionTemplateWithExplicitTemplateArgs) {
  Decl *FromTU = getTuDecl(
      R"(
      struct X {
        template <typename T>
        void foo(){}
      };
      void f() {
        X x;
        x.foo<int>();
      }
      )",
      Lang_CXX);
  auto *FromD = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("f")));
  auto *ToD = Import(FromD, Lang_CXX);
  EXPECT_TRUE(ToD);
  EXPECT_TRUE(MatchVerifier<FunctionDecl>().match(
      ToD, functionDecl(hasName("f"), hasDescendant(declRefExpr()))));
}

TEST_P(ImportFunctions,
       DependentCallExprOfMemberFunctionTemplateWithExplicitTemplateArgs) {
  Decl *FromTU = getTuDecl(
      R"(
      struct X {
        template <typename T>
        void foo(){}
      };
      template <typename T>
      void f() {
        X x;
        x.foo<T>();
      }
      void g() {
        f<int>();
      }
      )",
      Lang_CXX);
  auto *FromD = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("g")));
  auto *ToD = Import(FromD, Lang_CXX);
  EXPECT_TRUE(ToD);
  Decl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  EXPECT_TRUE(MatchVerifier<TranslationUnitDecl>().match(
      ToTU, translationUnitDecl(hasDescendant(
                functionDecl(hasName("f"), hasDescendant(declRefExpr()))))));
}

struct ImportFriendFunctions : ImportFunctions {};

TEST_P(ImportFriendFunctions, ImportFriendFunctionRedeclChainProto) {
  auto Pattern = functionDecl(hasName("f"));

  Decl *FromTU = getTuDecl("struct X { friend void f(); };"
                           "void f();",
                           Lang_CXX,
                           "input0.cc");
  auto *FromD = FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);

  auto *ImportedD = cast<FunctionDecl>(Import(FromD, Lang_CXX));
  Decl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  ASSERT_EQ(DeclCounter<FunctionDecl>().match(ToTU, Pattern), 2u);
  EXPECT_FALSE(ImportedD->doesThisDeclarationHaveABody());
  auto *ToFD = LastDeclMatcher<FunctionDecl>().match(ToTU, Pattern);
  EXPECT_FALSE(ToFD->doesThisDeclarationHaveABody());
  EXPECT_EQ(ToFD->getPreviousDecl(), ImportedD);
}

TEST_P(ImportFriendFunctions,
       ImportFriendFunctionRedeclChainProto_OutOfClassProtoFirst) {
  auto Pattern = functionDecl(hasName("f"));

  Decl *FromTU = getTuDecl("void f();"
                           "struct X { friend void f(); };",
                           Lang_CXX, "input0.cc");
  auto FromD = FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);

  auto *ImportedD = cast<FunctionDecl>(Import(FromD, Lang_CXX));
  Decl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  ASSERT_EQ(DeclCounter<FunctionDecl>().match(ToTU, Pattern), 2u);
  EXPECT_FALSE(ImportedD->doesThisDeclarationHaveABody());
  auto *ToFD = LastDeclMatcher<FunctionDecl>().match(ToTU, Pattern);
  EXPECT_FALSE(ToFD->doesThisDeclarationHaveABody());
  EXPECT_EQ(ToFD->getPreviousDecl(), ImportedD);
}

TEST_P(ImportFriendFunctions, ImportFriendFunctionRedeclChainDef) {
  auto Pattern = functionDecl(hasName("f"));

  Decl *FromTU = getTuDecl("struct X { friend void f(){} };"
                           "void f();",
                           Lang_CXX,
                           "input0.cc");
  auto *FromD = FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);

  auto *ImportedD = cast<FunctionDecl>(Import(FromD, Lang_CXX));
  Decl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  ASSERT_EQ(DeclCounter<FunctionDecl>().match(ToTU, Pattern), 2u);
  EXPECT_TRUE(ImportedD->doesThisDeclarationHaveABody());
  auto *ToFD = LastDeclMatcher<FunctionDecl>().match(ToTU, Pattern);
  EXPECT_FALSE(ToFD->doesThisDeclarationHaveABody());
  EXPECT_EQ(ToFD->getPreviousDecl(), ImportedD);
}

TEST_P(ImportFriendFunctions,
       ImportFriendFunctionRedeclChainDef_OutOfClassDef) {
  auto Pattern = functionDecl(hasName("f"));

  Decl *FromTU = getTuDecl("struct X { friend void f(); };"
                           "void f(){}",
                           Lang_CXX, "input0.cc");
  auto *FromD = FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);

  auto *ImportedD = cast<FunctionDecl>(Import(FromD, Lang_CXX));
  Decl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  ASSERT_EQ(DeclCounter<FunctionDecl>().match(ToTU, Pattern), 2u);
  EXPECT_FALSE(ImportedD->doesThisDeclarationHaveABody());
  auto *ToFD = LastDeclMatcher<FunctionDecl>().match(ToTU, Pattern);
  EXPECT_TRUE(ToFD->doesThisDeclarationHaveABody());
  EXPECT_EQ(ToFD->getPreviousDecl(), ImportedD);
}

// Disabled temporarily, because the new structural equivalence check
// (https://reviews.llvm.org/D48628) breaks it.
// PreviousDecl is not set because there is no structural match.
// FIXME Enable!
TEST_P(ImportFriendFunctions,
    DISABLED_ImportFriendFunctionRedeclChainDefWithClass) {
  auto Pattern = functionDecl(hasName("f"));

  Decl *FromTU = getTuDecl(
      R"(
        class X;
        void f(X *x){}
        class X{
        friend void f(X *x);
        };
      )",
      Lang_CXX, "input0.cc");
  auto *FromD = FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);

  auto *ImportedD = cast<FunctionDecl>(Import(FromD, Lang_CXX));
  Decl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  ASSERT_EQ(DeclCounter<FunctionDecl>().match(ToTU, Pattern), 2u);
  EXPECT_TRUE(ImportedD->doesThisDeclarationHaveABody());
  auto *InClassFD = cast<FunctionDecl>(FirstDeclMatcher<FriendDecl>()
                                              .match(ToTU, friendDecl())
                                              ->getFriendDecl());
  EXPECT_FALSE(InClassFD->doesThisDeclarationHaveABody());
  EXPECT_EQ(InClassFD->getPreviousDecl(), ImportedD);
  // The parameters must refer the same type
  EXPECT_EQ((*InClassFD->param_begin())->getOriginalType(),
            (*ImportedD->param_begin())->getOriginalType());
}

// Disabled temporarily, because the new structural equivalence check
// (https://reviews.llvm.org/D48628) breaks it.
// PreviousDecl is not set because there is no structural match.
// FIXME Enable!
TEST_P(ImportFriendFunctions,
    DISABLED_ImportFriendFunctionRedeclChainDefWithClass_ImportTheProto) {
  auto Pattern = functionDecl(hasName("f"));

  Decl *FromTU = getTuDecl(
      R"(
        class X;
        void f(X *x){}
        class X{
        friend void f(X *x);
        };
      )",
      Lang_CXX, "input0.cc");
  auto *FromD = LastDeclMatcher<FunctionDecl>().match(FromTU, Pattern);

  auto *ImportedD = cast<FunctionDecl>(Import(FromD, Lang_CXX));
  Decl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  ASSERT_EQ(DeclCounter<FunctionDecl>().match(ToTU, Pattern), 2u);
  EXPECT_FALSE(ImportedD->doesThisDeclarationHaveABody());
  auto *OutOfClassFD = FirstDeclMatcher<FunctionDecl>().match(
      ToTU, functionDecl(unless(hasParent(friendDecl()))));

  EXPECT_TRUE(OutOfClassFD->doesThisDeclarationHaveABody());
  EXPECT_EQ(ImportedD->getPreviousDecl(), OutOfClassFD);
  // The parameters must refer the same type
  EXPECT_EQ((*OutOfClassFD->param_begin())->getOriginalType(),
            (*ImportedD->param_begin())->getOriginalType());
}

TEST_P(ImportFriendFunctions, ImportFriendFunctionFromMultipleTU) {
  auto Pattern = functionDecl(hasName("f"));

  FunctionDecl *ImportedD;
  {
    Decl *FromTU =
        getTuDecl("struct X { friend void f(){} };", Lang_CXX, "input0.cc");
    auto *FromD = FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);
    ImportedD = cast<FunctionDecl>(Import(FromD, Lang_CXX));
  }
  FunctionDecl *ImportedD1;
  {
    Decl *FromTU = getTuDecl("void f();", Lang_CXX, "input1.cc");
    auto *FromD = FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);
    ImportedD1 = cast<FunctionDecl>(Import(FromD, Lang_CXX));
  }

  Decl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  ASSERT_EQ(DeclCounter<FunctionDecl>().match(ToTU, Pattern), 2u);
  EXPECT_TRUE(ImportedD->doesThisDeclarationHaveABody());
  EXPECT_FALSE(ImportedD1->doesThisDeclarationHaveABody());
  EXPECT_EQ(ImportedD1->getPreviousDecl(), ImportedD);
}

TEST_P(ImportFriendFunctions, Lookup) {
  auto FunctionPattern = functionDecl(hasName("f"));
  auto ClassPattern = cxxRecordDecl(hasName("X"));

  TranslationUnitDecl *FromTU =
      getTuDecl("struct X { friend void f(); };", Lang_CXX, "input0.cc");
  auto *FromD = FirstDeclMatcher<FunctionDecl>().match(FromTU, FunctionPattern);
  ASSERT_TRUE(FromD->isInIdentifierNamespace(Decl::IDNS_OrdinaryFriend));
  ASSERT_FALSE(FromD->isInIdentifierNamespace(Decl::IDNS_Ordinary));
  {
    auto FromName = FromD->getDeclName();
    auto *Class = FirstDeclMatcher<CXXRecordDecl>().match(FromTU, ClassPattern);
    auto LookupRes = Class->noload_lookup(FromName);
    ASSERT_EQ(LookupRes.size(), 0u);
    LookupRes = FromTU->noload_lookup(FromName);
    ASSERT_EQ(LookupRes.size(), 1u);
  }

  auto *ToD = cast<FunctionDecl>(Import(FromD, Lang_CXX));
  auto ToName = ToD->getDeclName();

  TranslationUnitDecl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  auto *Class = FirstDeclMatcher<CXXRecordDecl>().match(ToTU, ClassPattern);
  auto LookupRes = Class->noload_lookup(ToName);
  EXPECT_EQ(LookupRes.size(), 0u);
  LookupRes = ToTU->noload_lookup(ToName);
  EXPECT_EQ(LookupRes.size(), 1u);

  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, FunctionPattern), 1u);
  auto *To0 = FirstDeclMatcher<FunctionDecl>().match(ToTU, FunctionPattern);
  EXPECT_TRUE(To0->isInIdentifierNamespace(Decl::IDNS_OrdinaryFriend));
  EXPECT_FALSE(To0->isInIdentifierNamespace(Decl::IDNS_Ordinary));
}

TEST_P(ImportFriendFunctions, DISABLED_LookupWithProtoAfter) {
  auto FunctionPattern = functionDecl(hasName("f"));
  auto ClassPattern = cxxRecordDecl(hasName("X"));

  TranslationUnitDecl *FromTU = getTuDecl(
      "struct X { friend void f(); };"
      // This proto decl makes f available to normal
      // lookup, otherwise it is hidden.
      // Normal C++ lookup (implemented in
      // `clang::Sema::CppLookupName()` and in `LookupDirect()`)
      // returns the found `NamedDecl` only if the set IDNS is matched
      "void f();",
      Lang_CXX, "input0.cc");
  auto *FromFriend =
      FirstDeclMatcher<FunctionDecl>().match(FromTU, FunctionPattern);
  auto *FromNormal =
      LastDeclMatcher<FunctionDecl>().match(FromTU, FunctionPattern);
  ASSERT_TRUE(FromFriend->isInIdentifierNamespace(Decl::IDNS_OrdinaryFriend));
  ASSERT_FALSE(FromFriend->isInIdentifierNamespace(Decl::IDNS_Ordinary));
  ASSERT_FALSE(FromNormal->isInIdentifierNamespace(Decl::IDNS_OrdinaryFriend));
  ASSERT_TRUE(FromNormal->isInIdentifierNamespace(Decl::IDNS_Ordinary));

  auto FromName = FromFriend->getDeclName();
  auto *FromClass =
      FirstDeclMatcher<CXXRecordDecl>().match(FromTU, ClassPattern);
  auto LookupRes = FromClass->noload_lookup(FromName);
  ASSERT_EQ(LookupRes.size(), 0u);
  LookupRes = FromTU->noload_lookup(FromName);
  ASSERT_EQ(LookupRes.size(), 1u);

  auto *ToFriend = cast<FunctionDecl>(Import(FromFriend, Lang_CXX));
  auto ToName = ToFriend->getDeclName();

  TranslationUnitDecl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  auto *ToClass = FirstDeclMatcher<CXXRecordDecl>().match(ToTU, ClassPattern);
  LookupRes = ToClass->noload_lookup(ToName);
  EXPECT_EQ(LookupRes.size(), 0u);
  LookupRes = ToTU->noload_lookup(ToName);
  // Test is disabled because this result is 2.
  EXPECT_EQ(LookupRes.size(), 1u);

  ASSERT_EQ(DeclCounter<FunctionDecl>().match(ToTU, FunctionPattern), 2u);
  ToFriend = FirstDeclMatcher<FunctionDecl>().match(ToTU, FunctionPattern);
  auto *ToNormal = LastDeclMatcher<FunctionDecl>().match(ToTU, FunctionPattern);
  EXPECT_TRUE(ToFriend->isInIdentifierNamespace(Decl::IDNS_OrdinaryFriend));
  EXPECT_FALSE(ToFriend->isInIdentifierNamespace(Decl::IDNS_Ordinary));
  EXPECT_FALSE(ToNormal->isInIdentifierNamespace(Decl::IDNS_OrdinaryFriend));
  EXPECT_TRUE(ToNormal->isInIdentifierNamespace(Decl::IDNS_Ordinary));
}

TEST_P(ImportFriendFunctions, LookupWithProtoBefore) {
  auto FunctionPattern = functionDecl(hasName("f"));
  auto ClassPattern = cxxRecordDecl(hasName("X"));

  TranslationUnitDecl *FromTU = getTuDecl(
      "void f();"
      "struct X { friend void f(); };",
      Lang_CXX, "input0.cc");
  auto *FromNormal =
      FirstDeclMatcher<FunctionDecl>().match(FromTU, FunctionPattern);
  auto *FromFriend =
      LastDeclMatcher<FunctionDecl>().match(FromTU, FunctionPattern);
  ASSERT_FALSE(FromNormal->isInIdentifierNamespace(Decl::IDNS_OrdinaryFriend));
  ASSERT_TRUE(FromNormal->isInIdentifierNamespace(Decl::IDNS_Ordinary));
  ASSERT_TRUE(FromFriend->isInIdentifierNamespace(Decl::IDNS_OrdinaryFriend));
  ASSERT_TRUE(FromFriend->isInIdentifierNamespace(Decl::IDNS_Ordinary));

  auto FromName = FromNormal->getDeclName();
  auto *FromClass =
      FirstDeclMatcher<CXXRecordDecl>().match(FromTU, ClassPattern);
  auto LookupRes = FromClass->noload_lookup(FromName);
  ASSERT_EQ(LookupRes.size(), 0u);
  LookupRes = FromTU->noload_lookup(FromName);
  ASSERT_EQ(LookupRes.size(), 1u);

  auto *ToNormal = cast<FunctionDecl>(Import(FromNormal, Lang_CXX));
  auto ToName = ToNormal->getDeclName();
  TranslationUnitDecl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();

  auto *ToClass = FirstDeclMatcher<CXXRecordDecl>().match(ToTU, ClassPattern);
  LookupRes = ToClass->noload_lookup(ToName);
  EXPECT_EQ(LookupRes.size(), 0u);
  LookupRes = ToTU->noload_lookup(ToName);
  EXPECT_EQ(LookupRes.size(), 1u);

  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, FunctionPattern), 2u);
  ToNormal = FirstDeclMatcher<FunctionDecl>().match(ToTU, FunctionPattern);
  auto *ToFriend = LastDeclMatcher<FunctionDecl>().match(ToTU, FunctionPattern);
  EXPECT_FALSE(ToNormal->isInIdentifierNamespace(Decl::IDNS_OrdinaryFriend));
  EXPECT_TRUE(ToNormal->isInIdentifierNamespace(Decl::IDNS_Ordinary));
  EXPECT_TRUE(ToFriend->isInIdentifierNamespace(Decl::IDNS_OrdinaryFriend));
  EXPECT_TRUE(ToFriend->isInIdentifierNamespace(Decl::IDNS_Ordinary));
}

TEST_P(ImportFriendFunctions, ImportFriendChangesLookup) {
  auto Pattern = functionDecl(hasName("f"));

  TranslationUnitDecl *FromNormalTU =
      getTuDecl("void f();", Lang_CXX, "input0.cc");
  auto *FromNormalF =
      FirstDeclMatcher<FunctionDecl>().match(FromNormalTU, Pattern);
  TranslationUnitDecl *FromFriendTU =
      getTuDecl("class X { friend void f(); };", Lang_CXX, "input1.cc");
  auto *FromFriendF =
      FirstDeclMatcher<FunctionDecl>().match(FromFriendTU, Pattern);
  auto FromNormalName = FromNormalF->getDeclName();
  auto FromFriendName = FromFriendF->getDeclName();

  ASSERT_TRUE(FromNormalF->isInIdentifierNamespace(Decl::IDNS_Ordinary));
  ASSERT_FALSE(FromNormalF->isInIdentifierNamespace(Decl::IDNS_OrdinaryFriend));
  ASSERT_FALSE(FromFriendF->isInIdentifierNamespace(Decl::IDNS_Ordinary));
  ASSERT_TRUE(FromFriendF->isInIdentifierNamespace(Decl::IDNS_OrdinaryFriend));
  auto LookupRes = FromNormalTU->noload_lookup(FromNormalName);
  ASSERT_EQ(LookupRes.size(), 1u);
  LookupRes = FromFriendTU->noload_lookup(FromFriendName);
  ASSERT_EQ(LookupRes.size(), 1u);

  auto *ToNormalF = cast<FunctionDecl>(Import(FromNormalF, Lang_CXX));
  TranslationUnitDecl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  auto ToName = ToNormalF->getDeclName();
  EXPECT_TRUE(ToNormalF->isInIdentifierNamespace(Decl::IDNS_Ordinary));
  EXPECT_FALSE(ToNormalF->isInIdentifierNamespace(Decl::IDNS_OrdinaryFriend));
  LookupRes = ToTU->noload_lookup(ToName);
  EXPECT_EQ(LookupRes.size(), 1u);
  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, Pattern), 1u);

  auto *ToFriendF = cast<FunctionDecl>(Import(FromFriendF, Lang_CXX));
  LookupRes = ToTU->noload_lookup(ToName);
  EXPECT_EQ(LookupRes.size(), 1u);
  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, Pattern), 2u);

  EXPECT_TRUE(ToNormalF->isInIdentifierNamespace(Decl::IDNS_Ordinary));
  EXPECT_FALSE(ToNormalF->isInIdentifierNamespace(Decl::IDNS_OrdinaryFriend));

  EXPECT_TRUE(ToFriendF->isInIdentifierNamespace(Decl::IDNS_Ordinary));
  EXPECT_TRUE(ToFriendF->isInIdentifierNamespace(Decl::IDNS_OrdinaryFriend));
}

TEST_P(ImportFriendFunctions, ImportFriendList) {
  TranslationUnitDecl *FromTU = getTuDecl(
      "struct X { friend void f(); };"
      "void f();",
      Lang_CXX, "input0.cc");
  auto *FromFriendF = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("f")));

  auto *FromClass = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("X")));
  auto *FromFriend = FirstDeclMatcher<FriendDecl>().match(FromTU, friendDecl());
  auto FromFriends = FromClass->friends();
  unsigned int FrN = 0;
  for (auto Fr : FromFriends) {
    ASSERT_EQ(Fr, FromFriend);
    ++FrN;
  }
  ASSERT_EQ(FrN, 1u);

  Import(FromFriendF, Lang_CXX);
  TranslationUnitDecl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  auto *ToClass = FirstDeclMatcher<CXXRecordDecl>().match(
      ToTU, cxxRecordDecl(hasName("X")));
  auto *ToFriend = FirstDeclMatcher<FriendDecl>().match(ToTU, friendDecl());
  auto ToFriends = ToClass->friends();
  FrN = 0;
  for (auto Fr : ToFriends) {
    EXPECT_EQ(Fr, ToFriend);
    ++FrN;
  }
  EXPECT_EQ(FrN, 1u);
}

AST_MATCHER_P(TagDecl, hasTypedefForAnonDecl, Matcher<TypedefNameDecl>,
              InnerMatcher) {
  if (auto *Typedef = Node.getTypedefNameForAnonDecl())
    return InnerMatcher.matches(*Typedef, Finder, Builder);
  return false;
}

TEST_P(ImportDecl, ImportEnumSequential) {
  CodeFiles Samples{{"main.c",
                     {"void foo();"
                      "void moo();"
                      "int main() { foo(); moo(); }",
                      Lang_C}},

                    {"foo.c",
                     {"typedef enum { THING_VALUE } thing_t;"
                      "void conflict(thing_t type);"
                      "void foo() { (void)THING_VALUE; }"
                      "void conflict(thing_t type) {}",
                      Lang_C}},

                    {"moo.c",
                     {"typedef enum { THING_VALUE } thing_t;"
                      "void conflict(thing_t type);"
                      "void moo() { conflict(THING_VALUE); }",
                      Lang_C}}};

  auto VerificationMatcher =
      enumDecl(has(enumConstantDecl(hasName("THING_VALUE"))),
               hasTypedefForAnonDecl(hasName("thing_t")));

  ImportAction ImportFoo{"foo.c", "main.c", functionDecl(hasName("foo"))},
      ImportMoo{"moo.c", "main.c", functionDecl(hasName("moo"))};

  testImportSequence(
      Samples, {ImportFoo, ImportMoo}, // "foo", them "moo".
      // Just check that there is only one enum decl in the result AST.
      "main.c", enumDecl(), VerificationMatcher);

  // For different import order, result should be the same.
  testImportSequence(
      Samples, {ImportMoo, ImportFoo}, // "moo", them "foo".
      // Check that there is only one enum decl in the result AST.
      "main.c", enumDecl(), VerificationMatcher);
}

const internal::VariadicDynCastAllOfMatcher<Expr, DependentScopeDeclRefExpr>
    dependentScopeDeclRefExpr;

TEST_P(ImportExpr, DependentScopeDeclRefExpr) {
  MatchVerifier<Decl> Verifier;
  testImport("template <typename T> struct S { static T foo; };"
             "template <typename T> void declToImport() {"
             "  (void) S<T>::foo;"
             "}"
             "void instantiate() { declToImport<int>(); }"
             "template <typename T> T S<T>::foo;",
             Lang_CXX11, "", Lang_CXX11, Verifier,
             functionTemplateDecl(has(functionDecl(has(compoundStmt(
                 has(cStyleCastExpr(has(dependentScopeDeclRefExpr())))))))));

  testImport("template <typename T> struct S {"
             "template<typename S> static void foo(){};"
             "};"
             "template <typename T> void declToImport() {"
             "  S<T>::template foo<T>();"
             "}"
             "void instantiate() { declToImport<int>(); }",
             Lang_CXX11, "", Lang_CXX11, Verifier,
             functionTemplateDecl(has(functionDecl(has(compoundStmt(
                 has(callExpr(has(dependentScopeDeclRefExpr())))))))));
}

const internal::VariadicDynCastAllOfMatcher<Type, DependentNameType>
    dependentNameType;

TEST_P(ImportExpr, DependentNameType) {
  MatchVerifier<Decl> Verifier;
  testImport("template <typename T> struct declToImport {"
             "  typedef typename T::type dependent_name;"
             "};",
             Lang_CXX11, "", Lang_CXX11, Verifier,
             classTemplateDecl(has(
                 cxxRecordDecl(has(typedefDecl(has(dependentNameType())))))));
}

TEST_P(ImportExpr, UnresolvedMemberExpr) {
  MatchVerifier<Decl> Verifier;
  testImport("struct S { template <typename T> void mem(); };"
             "template <typename U> void declToImport() {"
             "  S s;"
             "  s.mem<U>();"
             "}"
             "void instantiate() { declToImport<int>(); }",
             Lang_CXX11, "", Lang_CXX11, Verifier,
             functionTemplateDecl(has(functionDecl(has(
                 compoundStmt(has(callExpr(has(unresolvedMemberExpr())))))))));
}

class ImportImplicitMethods : public ASTImporterOptionSpecificTestBase {
public:
  static constexpr auto DefaultCode = R"(
      struct A { int x; };
      void f() {
        A a;
        A a1(a);
        A a2(A{});
        a = a1;
        a = A{};
        a.~A();
      })";

  template <typename MatcherType>
  void testImportOf(
      const MatcherType &MethodMatcher, const char *Code = DefaultCode) {
    test(MethodMatcher, Code, /*ExpectedCount=*/1u);
  }

  template <typename MatcherType>
  void testNoImportOf(
      const MatcherType &MethodMatcher, const char *Code = DefaultCode) {
    test(MethodMatcher, Code, /*ExpectedCount=*/0u);
  }

private:
  template <typename MatcherType>
  void test(const MatcherType &MethodMatcher,
      const char *Code, unsigned int ExpectedCount) {
    auto ClassMatcher = cxxRecordDecl(unless(isImplicit()));

    Decl *ToTU = getToTuDecl(Code, Lang_CXX11);
    auto *ToClass = FirstDeclMatcher<CXXRecordDecl>().match(
        ToTU, ClassMatcher);

    ASSERT_EQ(DeclCounter<CXXMethodDecl>().match(ToClass, MethodMatcher), 1u);

    {
      CXXMethodDecl *Method =
          FirstDeclMatcher<CXXMethodDecl>().match(ToClass, MethodMatcher);
      ToClass->removeDecl(Method);
      LookupTablePtr->remove(Method);
    }

    ASSERT_EQ(DeclCounter<CXXMethodDecl>().match(ToClass, MethodMatcher), 0u);

    Decl *ImportedClass = nullptr;
    {
      Decl *FromTU = getTuDecl(Code, Lang_CXX11, "input1.cc");
      auto *FromClass = FirstDeclMatcher<CXXRecordDecl>().match(
          FromTU, ClassMatcher);
      ImportedClass = Import(FromClass, Lang_CXX11);
    }

    EXPECT_EQ(ToClass, ImportedClass);
    EXPECT_EQ(DeclCounter<CXXMethodDecl>().match(ToClass, MethodMatcher),
        ExpectedCount);
  }
};

TEST_P(ImportImplicitMethods, DefaultConstructor) {
  testImportOf(cxxConstructorDecl(isDefaultConstructor()));
}

TEST_P(ImportImplicitMethods, CopyConstructor) {
  testImportOf(cxxConstructorDecl(isCopyConstructor()));
}

TEST_P(ImportImplicitMethods, MoveConstructor) {
  testImportOf(cxxConstructorDecl(isMoveConstructor()));
}

TEST_P(ImportImplicitMethods, Destructor) {
  testImportOf(cxxDestructorDecl());
}

TEST_P(ImportImplicitMethods, CopyAssignment) {
  testImportOf(cxxMethodDecl(isCopyAssignmentOperator()));
}

TEST_P(ImportImplicitMethods, MoveAssignment) {
  testImportOf(cxxMethodDecl(isMoveAssignmentOperator()));
}

TEST_P(ImportImplicitMethods, DoNotImportUserProvided) {
  auto Code = R"(
      struct A { A() { int x; } };
      )";
  testNoImportOf(cxxConstructorDecl(isDefaultConstructor()), Code);
}

TEST_P(ImportImplicitMethods, DoNotImportDefault) {
  auto Code = R"(
      struct A { A() = default; };
      )";
  testNoImportOf(cxxConstructorDecl(isDefaultConstructor()), Code);
}

TEST_P(ImportImplicitMethods, DoNotImportDeleted) {
  auto Code = R"(
      struct A { A() = delete; };
      )";
  testNoImportOf(cxxConstructorDecl(isDefaultConstructor()), Code);
}

TEST_P(ImportImplicitMethods, DoNotImportOtherMethod) {
  auto Code = R"(
      struct A { void f() { } };
      )";
  testNoImportOf(cxxMethodDecl(hasName("f")), Code);
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportOfEquivalentRecord) {
  Decl *ToR1;
  {
    Decl *FromTU = getTuDecl(
        "struct A { };", Lang_CXX, "input0.cc");
    auto *FromR = FirstDeclMatcher<CXXRecordDecl>().match(
        FromTU, cxxRecordDecl(hasName("A")));

    ToR1 = Import(FromR, Lang_CXX);
  }

  Decl *ToR2;
  {
    Decl *FromTU = getTuDecl(
        "struct A { };", Lang_CXX, "input1.cc");
    auto *FromR = FirstDeclMatcher<CXXRecordDecl>().match(
        FromTU, cxxRecordDecl(hasName("A")));

    ToR2 = Import(FromR, Lang_CXX);
  }

  EXPECT_EQ(ToR1, ToR2);
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportOfNonEquivalentRecord) {
  Decl *ToR1;
  {
    Decl *FromTU = getTuDecl(
        "struct A { int x; };", Lang_CXX, "input0.cc");
    auto *FromR = FirstDeclMatcher<CXXRecordDecl>().match(
        FromTU, cxxRecordDecl(hasName("A")));
    ToR1 = Import(FromR, Lang_CXX);
  }
  Decl *ToR2;
  {
    Decl *FromTU = getTuDecl(
        "struct A { unsigned x; };", Lang_CXX, "input1.cc");
    auto *FromR = FirstDeclMatcher<CXXRecordDecl>().match(
        FromTU, cxxRecordDecl(hasName("A")));
    ToR2 = Import(FromR, Lang_CXX);
  }
  EXPECT_NE(ToR1, ToR2);
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportOfEquivalentField) {
  Decl *ToF1;
  {
    Decl *FromTU = getTuDecl(
        "struct A { int x; };", Lang_CXX, "input0.cc");
    auto *FromF = FirstDeclMatcher<FieldDecl>().match(
        FromTU, fieldDecl(hasName("x")));
    ToF1 = Import(FromF, Lang_CXX);
  }
  Decl *ToF2;
  {
    Decl *FromTU = getTuDecl(
        "struct A { int x; };", Lang_CXX, "input1.cc");
    auto *FromF = FirstDeclMatcher<FieldDecl>().match(
        FromTU, fieldDecl(hasName("x")));
    ToF2 = Import(FromF, Lang_CXX);
  }
  EXPECT_EQ(ToF1, ToF2);
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportOfNonEquivalentField) {
  Decl *ToF1;
  {
    Decl *FromTU = getTuDecl(
        "struct A { int x; };", Lang_CXX, "input0.cc");
    auto *FromF = FirstDeclMatcher<FieldDecl>().match(
        FromTU, fieldDecl(hasName("x")));
    ToF1 = Import(FromF, Lang_CXX);
  }
  Decl *ToF2;
  {
    Decl *FromTU = getTuDecl(
        "struct A { unsigned x; };", Lang_CXX, "input1.cc");
    auto *FromF = FirstDeclMatcher<FieldDecl>().match(
        FromTU, fieldDecl(hasName("x")));
    ToF2 = Import(FromF, Lang_CXX);
  }
  EXPECT_NE(ToF1, ToF2);
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportOfEquivalentMethod) {
  Decl *ToM1;
  {
    Decl *FromTU = getTuDecl(
        "struct A { void x(); }; void A::x() { }", Lang_CXX, "input0.cc");
    auto *FromM = FirstDeclMatcher<FunctionDecl>().match(
        FromTU, functionDecl(hasName("x"), isDefinition()));
    ToM1 = Import(FromM, Lang_CXX);
  }
  Decl *ToM2;
  {
    Decl *FromTU = getTuDecl(
        "struct A { void x(); }; void A::x() { }", Lang_CXX, "input1.cc");
    auto *FromM = FirstDeclMatcher<FunctionDecl>().match(
        FromTU, functionDecl(hasName("x"), isDefinition()));
    ToM2 = Import(FromM, Lang_CXX);
  }
  EXPECT_EQ(ToM1, ToM2);
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportOfNonEquivalentMethod) {
  Decl *ToM1;
  {
    Decl *FromTU = getTuDecl(
        "struct A { void x(); }; void A::x() { }",
        Lang_CXX, "input0.cc");
    auto *FromM = FirstDeclMatcher<FunctionDecl>().match(
        FromTU, functionDecl(hasName("x"), isDefinition()));
    ToM1 = Import(FromM, Lang_CXX);
  }
  Decl *ToM2;
  {
    Decl *FromTU = getTuDecl(
        "struct A { void x() const; }; void A::x() const { }",
        Lang_CXX, "input1.cc");
    auto *FromM = FirstDeclMatcher<FunctionDecl>().match(
        FromTU, functionDecl(hasName("x"), isDefinition()));
    ToM2 = Import(FromM, Lang_CXX);
  }
  EXPECT_NE(ToM1, ToM2);
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportUnnamedStructsWithRecursingField) {
  Decl *FromTU = getTuDecl(
      R"(
      struct A {
        struct {
          struct A *next;
        } entry0;
        struct {
          struct A *next;
        } entry1;
      };
      )",
      Lang_C, "input0.cc");
  auto *From =
      FirstDeclMatcher<RecordDecl>().match(FromTU, recordDecl(hasName("A")));

  Import(From, Lang_C);

  auto *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  auto *Entry0 =
      FirstDeclMatcher<FieldDecl>().match(ToTU, fieldDecl(hasName("entry0")));
  auto *Entry1 =
      FirstDeclMatcher<FieldDecl>().match(ToTU, fieldDecl(hasName("entry1")));
  auto *R0 = getRecordDecl(Entry0);
  auto *R1 = getRecordDecl(Entry1);
  EXPECT_NE(R0, R1);
  EXPECT_TRUE(MatchVerifier<RecordDecl>().match(
      R0, recordDecl(has(fieldDecl(hasName("next"))))));
  EXPECT_TRUE(MatchVerifier<RecordDecl>().match(
      R1, recordDecl(has(fieldDecl(hasName("next"))))));
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportUnnamedFieldsInCorrectOrder) {
  Decl *FromTU = getTuDecl(
      R"(
      void f(int X, int Y, bool Z) {
        (void)[X, Y, Z] { (void)Z; };
      }
      )",
      Lang_CXX11, "input0.cc");
  auto *FromF = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("f")));
  auto *ToF = cast_or_null<FunctionDecl>(Import(FromF, Lang_CXX11));
  EXPECT_TRUE(ToF);

  CXXRecordDecl *FromLambda =
      cast<LambdaExpr>(cast<CStyleCastExpr>(cast<CompoundStmt>(
          FromF->getBody())->body_front())->getSubExpr())->getLambdaClass();

  auto *ToLambda = cast_or_null<CXXRecordDecl>(Import(FromLambda, Lang_CXX11));
  EXPECT_TRUE(ToLambda);

  // Check if the fields of the lambda class are imported in correct order.
  unsigned FromIndex = 0u;
  for (auto *FromField : FromLambda->fields()) {
    ASSERT_FALSE(FromField->getDeclName());
    auto *ToField = cast_or_null<FieldDecl>(Import(FromField, Lang_CXX11));
    EXPECT_TRUE(ToField);
    Optional<unsigned> ToIndex = ASTImporter::getFieldIndex(ToField);
    EXPECT_TRUE(ToIndex);
    EXPECT_EQ(*ToIndex, FromIndex);
    ++FromIndex;
  }

  EXPECT_EQ(FromIndex, 3u);
}

TEST_P(ASTImporterOptionSpecificTestBase,
       MergeFieldDeclsOfClassTemplateSpecialization) {
  std::string ClassTemplate =
      R"(
      template <typename T>
      struct X {
          int a{0}; // FieldDecl with InitListExpr
          X(char) : a(3) {}     // (1)
          X(int) {}             // (2)
      };
      )";
  Decl *ToTU = getToTuDecl(ClassTemplate +
      R"(
      void foo() {
          // ClassTemplateSpec with ctor (1): FieldDecl without InitlistExpr
          X<char> xc('c');
      }
      )", Lang_CXX11);
  auto *ToSpec = FirstDeclMatcher<ClassTemplateSpecializationDecl>().match(
      ToTU, classTemplateSpecializationDecl(hasName("X")));
  // FieldDecl without InitlistExpr:
  auto *ToField = *ToSpec->field_begin();
  ASSERT_TRUE(ToField);
  ASSERT_FALSE(ToField->getInClassInitializer());
  Decl *FromTU = getTuDecl(ClassTemplate +
      R"(
      void bar() {
          // ClassTemplateSpec with ctor (2): FieldDecl WITH InitlistExpr
          X<char> xc(1);
      }
      )", Lang_CXX11);
  auto *FromSpec = FirstDeclMatcher<ClassTemplateSpecializationDecl>().match(
      FromTU, classTemplateSpecializationDecl(hasName("X")));
  // FieldDecl with InitlistExpr:
  auto *FromField = *FromSpec->field_begin();
  ASSERT_TRUE(FromField);
  ASSERT_TRUE(FromField->getInClassInitializer());

  auto *ImportedSpec = Import(FromSpec, Lang_CXX11);
  ASSERT_TRUE(ImportedSpec);
  EXPECT_EQ(ImportedSpec, ToSpec);
  // After the import, the FieldDecl has to be merged, thus it should have the
  // InitListExpr.
  EXPECT_TRUE(ToField->getInClassInitializer());
}

TEST_P(ASTImporterOptionSpecificTestBase,
       MergeFunctionOfClassTemplateSpecialization) {
  std::string ClassTemplate =
      R"(
      template <typename T>
      struct X {
        void f() {}
        void g() {}
      };
      )";
  Decl *ToTU = getToTuDecl(ClassTemplate +
      R"(
      void foo() {
          X<char> x;
          x.f();
      }
      )", Lang_CXX11);
  Decl *FromTU = getTuDecl(ClassTemplate +
      R"(
      void bar() {
          X<char> x;
          x.g();
      }
      )", Lang_CXX11);
  auto *FromSpec = FirstDeclMatcher<ClassTemplateSpecializationDecl>().match(
      FromTU, classTemplateSpecializationDecl(hasName("X")));
  auto FunPattern = functionDecl(hasName("g"),
                         hasParent(classTemplateSpecializationDecl()));
  auto *FromFun =
      FirstDeclMatcher<FunctionDecl>().match(FromTU, FunPattern);
  auto *ToFun =
      FirstDeclMatcher<FunctionDecl>().match(ToTU, FunPattern);
  ASSERT_TRUE(FromFun->hasBody());
  ASSERT_FALSE(ToFun->hasBody());
  auto *ImportedSpec = Import(FromSpec, Lang_CXX11);
  ASSERT_TRUE(ImportedSpec);
  auto *ToSpec = FirstDeclMatcher<ClassTemplateSpecializationDecl>().match(
      ToTU, classTemplateSpecializationDecl(hasName("X")));
  EXPECT_EQ(ImportedSpec, ToSpec);
  EXPECT_TRUE(ToFun->hasBody());
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ODRViolationOfClassTemplateSpecializationsShouldBeReported) {
  std::string ClassTemplate =
      R"(
      template <typename T>
      struct X {};
      )";
  Decl *ToTU = getToTuDecl(ClassTemplate +
                               R"(
      template <>
      struct X<char> {
          int a;
      };
      void foo() {
          X<char> x;
      }
      )",
                           Lang_CXX11);
  Decl *FromTU = getTuDecl(ClassTemplate +
                               R"(
      template <>
      struct X<char> {
          int b;
      };
      void foo() {
          X<char> x;
      }
      )",
                           Lang_CXX11);
  auto *FromSpec = FirstDeclMatcher<ClassTemplateSpecializationDecl>().match(
      FromTU, classTemplateSpecializationDecl(hasName("X")));
  auto *ImportedSpec = Import(FromSpec, Lang_CXX11);

  // We expect one (ODR) warning during the import.
  EXPECT_EQ(1u, ToTU->getASTContext().getDiagnostics().getNumWarnings());

  // The second specialization is different from the first, thus it violates
  // ODR, consequently we expect to keep the first specialization only, which is
  // already in the "To" context.
  EXPECT_TRUE(ImportedSpec);
  auto *ToSpec = FirstDeclMatcher<ClassTemplateSpecializationDecl>().match(
      ToTU, classTemplateSpecializationDecl(hasName("X")));
  EXPECT_EQ(ImportedSpec, ToSpec);
  EXPECT_EQ(1u, DeclCounter<ClassTemplateSpecializationDecl>().match(
                    ToTU, classTemplateSpecializationDecl()));
}

TEST_P(ASTImporterOptionSpecificTestBase,
       MergeCtorOfClassTemplateSpecialization) {
  std::string ClassTemplate =
      R"(
      template <typename T>
      struct X {
          X(char) {}
          X(int) {}
      };
      )";
  Decl *ToTU = getToTuDecl(ClassTemplate +
      R"(
      void foo() {
          X<char> x('c');
      }
      )", Lang_CXX11);
  Decl *FromTU = getTuDecl(ClassTemplate +
      R"(
      void bar() {
          X<char> x(1);
      }
      )", Lang_CXX11);
  auto *FromSpec = FirstDeclMatcher<ClassTemplateSpecializationDecl>().match(
      FromTU, classTemplateSpecializationDecl(hasName("X")));
  // Match the void(int) ctor.
  auto CtorPattern =
      cxxConstructorDecl(hasParameter(0, varDecl(hasType(asString("int")))),
                         hasParent(classTemplateSpecializationDecl()));
  auto *FromCtor =
      FirstDeclMatcher<CXXConstructorDecl>().match(FromTU, CtorPattern);
  auto *ToCtor =
      FirstDeclMatcher<CXXConstructorDecl>().match(ToTU, CtorPattern);
  ASSERT_TRUE(FromCtor->hasBody());
  ASSERT_FALSE(ToCtor->hasBody());
  auto *ImportedSpec = Import(FromSpec, Lang_CXX11);
  ASSERT_TRUE(ImportedSpec);
  auto *ToSpec = FirstDeclMatcher<ClassTemplateSpecializationDecl>().match(
      ToTU, classTemplateSpecializationDecl(hasName("X")));
  EXPECT_EQ(ImportedSpec, ToSpec);
  EXPECT_TRUE(ToCtor->hasBody());
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ClassTemplatePartialSpecializationsShouldNotBeDuplicated) {
  auto Code =
      R"(
    // primary template
    template<class T1, class T2, int I>
    class A {};

    // partial specialization
    template<class T, int I>
    class A<T, T*, I> {};
    )";
  Decl *ToTU = getToTuDecl(Code, Lang_CXX11);
  Decl *FromTU = getTuDecl(Code, Lang_CXX11);
  auto *FromSpec =
      FirstDeclMatcher<ClassTemplatePartialSpecializationDecl>().match(
          FromTU, classTemplatePartialSpecializationDecl());
  auto *ToSpec =
      FirstDeclMatcher<ClassTemplatePartialSpecializationDecl>().match(
          ToTU, classTemplatePartialSpecializationDecl());

  auto *ImportedSpec = Import(FromSpec, Lang_CXX11);
  EXPECT_EQ(ImportedSpec, ToSpec);
  EXPECT_EQ(1u, DeclCounter<ClassTemplatePartialSpecializationDecl>().match(
                    ToTU, classTemplatePartialSpecializationDecl()));
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ClassTemplateSpecializationsShouldNotBeDuplicated) {
  auto Code =
      R"(
    // primary template
    template<class T1, class T2, int I>
    class A {};

    // full specialization
    template<>
    class A<int, int, 1> {};
    )";
  Decl *ToTU = getToTuDecl(Code, Lang_CXX11);
  Decl *FromTU = getTuDecl(Code, Lang_CXX11);
  auto *FromSpec = FirstDeclMatcher<ClassTemplateSpecializationDecl>().match(
      FromTU, classTemplateSpecializationDecl());
  auto *ToSpec = FirstDeclMatcher<ClassTemplateSpecializationDecl>().match(
      ToTU, classTemplateSpecializationDecl());

  auto *ImportedSpec = Import(FromSpec, Lang_CXX11);
  EXPECT_EQ(ImportedSpec, ToSpec);
  EXPECT_EQ(1u, DeclCounter<ClassTemplateSpecializationDecl>().match(
                   ToTU, classTemplateSpecializationDecl()));
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ClassTemplateFullAndPartialSpecsShouldNotBeMixed) {
  std::string PrimaryTemplate =
      R"(
    template<class T1, class T2, int I>
    class A {};
    )";
  auto PartialSpec =
      R"(
    template<class T, int I>
    class A<T, T*, I> {};
    )";
  auto FullSpec =
      R"(
    template<>
    class A<int, int, 1> {};
    )";
  Decl *ToTU = getToTuDecl(PrimaryTemplate + FullSpec, Lang_CXX11);
  Decl *FromTU = getTuDecl(PrimaryTemplate + PartialSpec, Lang_CXX11);
  auto *FromSpec = FirstDeclMatcher<ClassTemplateSpecializationDecl>().match(
      FromTU, classTemplateSpecializationDecl());

  auto *ImportedSpec = Import(FromSpec, Lang_CXX11);
  EXPECT_TRUE(ImportedSpec);
  // Check the number of partial specializations.
  EXPECT_EQ(1u, DeclCounter<ClassTemplatePartialSpecializationDecl>().match(
                    ToTU, classTemplatePartialSpecializationDecl()));
  // Check the number of full specializations.
  EXPECT_EQ(1u, DeclCounter<ClassTemplateSpecializationDecl>().match(
                    ToTU, classTemplateSpecializationDecl(
                              unless(classTemplatePartialSpecializationDecl()))));
}

TEST_P(ASTImporterOptionSpecificTestBase,
       InitListExprValueKindShouldBeImported) {
  Decl *TU = getTuDecl(
      R"(
      const int &init();
      void foo() { const int &a{init()}; }
      )", Lang_CXX11, "input0.cc");
  auto *FromD = FirstDeclMatcher<VarDecl>().match(TU, varDecl(hasName("a")));
  ASSERT_TRUE(FromD->getAnyInitializer());
  auto *InitExpr = FromD->getAnyInitializer();
  ASSERT_TRUE(InitExpr);
  ASSERT_TRUE(InitExpr->isGLValue());

  auto *ToD = Import(FromD, Lang_CXX11);
  EXPECT_TRUE(ToD);
  auto *ToInitExpr = cast<VarDecl>(ToD)->getAnyInitializer();
  EXPECT_TRUE(ToInitExpr);
  EXPECT_TRUE(ToInitExpr->isGLValue());
}

struct ImportVariables : ASTImporterOptionSpecificTestBase {};

TEST_P(ImportVariables, ImportOfOneDeclBringsInTheWholeChain) {
  Decl *FromTU = getTuDecl(
      R"(
      struct A {
        static const int a = 1 + 2;
      };
      const int A::a;
      )", Lang_CXX, "input1.cc");

  auto *FromDWithInit = FirstDeclMatcher<VarDecl>().match(
      FromTU, varDecl(hasName("a"))); // Decl with init
  auto *FromDWithDef = LastDeclMatcher<VarDecl>().match(
      FromTU, varDecl(hasName("a"))); // Decl with definition
  ASSERT_NE(FromDWithInit, FromDWithDef);
  ASSERT_EQ(FromDWithDef->getPreviousDecl(), FromDWithInit);

  auto *ToD0 = cast<VarDecl>(Import(FromDWithInit, Lang_CXX11));
  auto *ToD1 = cast<VarDecl>(Import(FromDWithDef, Lang_CXX11));
  ASSERT_TRUE(ToD0);
  ASSERT_TRUE(ToD1);
  EXPECT_NE(ToD0, ToD1);
  EXPECT_EQ(ToD1->getPreviousDecl(), ToD0);
}

TEST_P(ImportVariables, InitAndDefinitionAreInDifferentTUs) {
  auto StructA =
      R"(
      struct A {
        static const int a = 1 + 2;
      };
      )";
  Decl *ToTU = getToTuDecl(StructA, Lang_CXX);
  Decl *FromTU = getTuDecl(std::string(StructA) + "const int A::a;", Lang_CXX,
                           "input1.cc");

  auto *FromDWithInit = FirstDeclMatcher<VarDecl>().match(
      FromTU, varDecl(hasName("a"))); // Decl with init
  auto *FromDWithDef = LastDeclMatcher<VarDecl>().match(
      FromTU, varDecl(hasName("a"))); // Decl with definition
  ASSERT_EQ(FromDWithInit, FromDWithDef->getPreviousDecl());
  ASSERT_TRUE(FromDWithInit->getInit());
  ASSERT_FALSE(FromDWithInit->isThisDeclarationADefinition());
  ASSERT_TRUE(FromDWithDef->isThisDeclarationADefinition());
  ASSERT_FALSE(FromDWithDef->getInit());

  auto *ToD = FirstDeclMatcher<VarDecl>().match(
      ToTU, varDecl(hasName("a"))); // Decl with init
  ASSERT_TRUE(ToD->getInit());
  ASSERT_FALSE(ToD->getDefinition());

  auto *ImportedD = cast<VarDecl>(Import(FromDWithDef, Lang_CXX11));
  EXPECT_TRUE(ImportedD->getAnyInitializer());
  EXPECT_TRUE(ImportedD->getDefinition());
}

TEST_P(ImportVariables, InitAndDefinitionAreInTheFromContext) {
  auto StructA =
      R"(
      struct A {
        static const int a;
      };
      )";
  Decl *ToTU = getToTuDecl(StructA, Lang_CXX);
  Decl *FromTU = getTuDecl(std::string(StructA) + "const int A::a = 1 + 2;",
                           Lang_CXX, "input1.cc");

  auto *FromDDeclarationOnly = FirstDeclMatcher<VarDecl>().match(
      FromTU, varDecl(hasName("a")));
  auto *FromDWithDef = LastDeclMatcher<VarDecl>().match(
      FromTU, varDecl(hasName("a"))); // Decl with definition and with init.
  ASSERT_EQ(FromDDeclarationOnly, FromDWithDef->getPreviousDecl());
  ASSERT_FALSE(FromDDeclarationOnly->getInit());
  ASSERT_FALSE(FromDDeclarationOnly->isThisDeclarationADefinition());
  ASSERT_TRUE(FromDWithDef->isThisDeclarationADefinition());
  ASSERT_TRUE(FromDWithDef->getInit());

  auto *ToD = FirstDeclMatcher<VarDecl>().match(
      ToTU, varDecl(hasName("a")));
  ASSERT_FALSE(ToD->getInit());
  ASSERT_FALSE(ToD->getDefinition());

  auto *ImportedD = cast<VarDecl>(Import(FromDWithDef, Lang_CXX11));
  EXPECT_TRUE(ImportedD->getAnyInitializer());
  EXPECT_TRUE(ImportedD->getDefinition());
}

struct ImportClasses : ASTImporterOptionSpecificTestBase {};

TEST_P(ImportClasses, ImportDefinitionWhenProtoIsInNestedToContext) {
  Decl *ToTU = getToTuDecl("struct A { struct X *Xp; };", Lang_C);
  Decl *FromTU1 = getTuDecl("struct X {};", Lang_C, "input1.cc");
  auto Pattern = recordDecl(hasName("X"), unless(isImplicit()));
  auto ToProto = FirstDeclMatcher<RecordDecl>().match(ToTU, Pattern);
  auto FromDef = FirstDeclMatcher<RecordDecl>().match(FromTU1, Pattern);

  Decl *ImportedDef = Import(FromDef, Lang_C);

  EXPECT_NE(ImportedDef, ToProto);
  EXPECT_EQ(DeclCounter<RecordDecl>().match(ToTU, Pattern), 2u);
  auto ToDef = LastDeclMatcher<RecordDecl>().match(ToTU, Pattern);
  EXPECT_TRUE(ImportedDef == ToDef);
  EXPECT_TRUE(ToDef->isThisDeclarationADefinition());
  EXPECT_FALSE(ToProto->isThisDeclarationADefinition());
  EXPECT_EQ(ToDef->getPreviousDecl(), ToProto);
}

TEST_P(ImportClasses, ImportDefinitionWhenProtoIsInNestedToContextCXX) {
  Decl *ToTU = getToTuDecl("struct A { struct X *Xp; };", Lang_CXX);
  Decl *FromTU1 = getTuDecl("struct X {};", Lang_CXX, "input1.cc");
  auto Pattern = recordDecl(hasName("X"), unless(isImplicit()));
  auto ToProto = FirstDeclMatcher<RecordDecl>().match(ToTU, Pattern);
  auto FromDef = FirstDeclMatcher<RecordDecl>().match(FromTU1, Pattern);

  Decl *ImportedDef = Import(FromDef, Lang_CXX);

  EXPECT_NE(ImportedDef, ToProto);
  EXPECT_EQ(DeclCounter<RecordDecl>().match(ToTU, Pattern), 2u);
  auto ToDef = LastDeclMatcher<RecordDecl>().match(ToTU, Pattern);
  EXPECT_TRUE(ImportedDef == ToDef);
  EXPECT_TRUE(ToDef->isThisDeclarationADefinition());
  EXPECT_FALSE(ToProto->isThisDeclarationADefinition());
  EXPECT_EQ(ToDef->getPreviousDecl(), ToProto);
}

TEST_P(ImportClasses, ImportNestedPrototypeThenDefinition) {
  Decl *FromTU0 = getTuDecl("struct A { struct X *Xp; };", Lang_C, "input0.cc");
  Decl *FromTU1 = getTuDecl("struct X {};", Lang_C, "input1.cc");
  auto Pattern = recordDecl(hasName("X"), unless(isImplicit()));
  auto FromProto = FirstDeclMatcher<RecordDecl>().match(FromTU0, Pattern);
  auto FromDef = FirstDeclMatcher<RecordDecl>().match(FromTU1, Pattern);

  Decl *ImportedProto = Import(FromProto, Lang_C);
  Decl *ImportedDef = Import(FromDef, Lang_C);
  Decl *ToTU = ImportedDef->getTranslationUnitDecl();

  EXPECT_NE(ImportedDef, ImportedProto);
  EXPECT_EQ(DeclCounter<RecordDecl>().match(ToTU, Pattern), 2u);
  auto ToProto = FirstDeclMatcher<RecordDecl>().match(ToTU, Pattern);
  auto ToDef = LastDeclMatcher<RecordDecl>().match(ToTU, Pattern);
  EXPECT_TRUE(ImportedDef == ToDef);
  EXPECT_TRUE(ImportedProto == ToProto);
  EXPECT_TRUE(ToDef->isThisDeclarationADefinition());
  EXPECT_FALSE(ToProto->isThisDeclarationADefinition());
  EXPECT_EQ(ToDef->getPreviousDecl(), ToProto);
}

// FIXME put these structs and the tests rely on them into their own separate
// test file!
struct Function {
  using DeclTy = FunctionDecl;
  static constexpr auto *Prototype = "void X();";
  static constexpr auto *Definition = "void X() {}";
  BindableMatcher<Decl> getPattern() {
    return functionDecl(hasName("X"), unless(isImplicit()));
  }
};

struct Class {
  using DeclTy = CXXRecordDecl;
  static constexpr auto *Prototype = "class X;";
  static constexpr auto *Definition = "class X {};";
  BindableMatcher<Decl> getPattern() {
    return cxxRecordDecl(hasName("X"), unless(isImplicit()));
  }
};

struct Variable {
  using DeclTy = VarDecl;
  static constexpr auto *Prototype = "extern int X;";
  static constexpr auto *Definition = "int X;";
  BindableMatcher<Decl> getPattern() {
    return varDecl(hasName("X"));
  }
};

struct FunctionTemplate {
  using DeclTy = FunctionTemplateDecl;
  static constexpr auto *Prototype = "template <class T> void X();";
  static constexpr auto *Definition =
      R"(
      template <class T> void X() {};
      // Explicit instantiation is a must because of -fdelayed-template-parsing:
      template void X<int>();
      )";
  BindableMatcher<Decl> getPattern() {
    return functionTemplateDecl(hasName("X"), unless(isImplicit()));
  }
};

struct ClassTemplate {
  using DeclTy = ClassTemplateDecl;
  static constexpr auto *Prototype = "template <class T> class X;";
  static constexpr auto *Definition = "template <class T> class X {};";
  BindableMatcher<Decl> getPattern() {
    return classTemplateDecl(hasName("X"), unless(isImplicit()));
  }
};

struct FunctionTemplateSpec {
  using DeclTy = FunctionDecl;
  static constexpr auto *Prototype =
      R"(
      // Proto of the primary template.
      template <class T>
      void X();
      // Proto of the specialization.
      template <>
      void X<int>();
      )";
  static constexpr auto *Definition =
      R"(
      // Proto of the primary template.
      template <class T>
      void X();
      // Specialization and definition.
      template <>
      void X<int>() {}
      )";
  BindableMatcher<Decl> getPattern() {
    return functionDecl(hasName("X"), isExplicitTemplateSpecialization());
  }
};

template <typename TypeParam>
struct RedeclChain : ASTImporterOptionSpecificTestBase {

  using DeclTy = typename TypeParam::DeclTy;
  std::string getPrototype() { return TypeParam::Prototype; }
  std::string getDefinition() { return TypeParam::Definition; }
  BindableMatcher<Decl> getPattern() const { return TypeParam().getPattern(); }

  void
  TypedTest_PrototypeShouldBeImportedAsAPrototypeWhenThereIsNoDefinition() {
    Decl *FromTU = getTuDecl(getPrototype(), Lang_CXX);
    auto *FromD = FirstDeclMatcher<DeclTy>().match(FromTU, getPattern());
    ASSERT_FALSE(FromD->isThisDeclarationADefinition());

    Decl *ImportedD = Import(FromD, Lang_CXX);
    Decl *ToTU = ImportedD->getTranslationUnitDecl();

    EXPECT_EQ(DeclCounter<DeclTy>().match(ToTU, getPattern()), 1u);
    auto *ToD = LastDeclMatcher<DeclTy>().match(ToTU, getPattern());
    EXPECT_TRUE(ImportedD == ToD);
    EXPECT_FALSE(ToD->isThisDeclarationADefinition());
    if (auto *ToT = dyn_cast<TemplateDecl>(ToD))
      EXPECT_TRUE(ToT->getTemplatedDecl());
  }

  void TypedTest_DefinitionShouldBeImportedAsADefinition() {
    Decl *FromTU = getTuDecl(getDefinition(), Lang_CXX);
    auto *FromD = FirstDeclMatcher<DeclTy>().match(FromTU, getPattern());
    ASSERT_TRUE(FromD->isThisDeclarationADefinition());

    Decl *ImportedD = Import(FromD, Lang_CXX);
    Decl *ToTU = ImportedD->getTranslationUnitDecl();

    EXPECT_EQ(DeclCounter<DeclTy>().match(ToTU, getPattern()), 1u);
    auto *ToD = LastDeclMatcher<DeclTy>().match(ToTU, getPattern());
    EXPECT_TRUE(ToD->isThisDeclarationADefinition());
    if (auto *ToT = dyn_cast<TemplateDecl>(ToD))
      EXPECT_TRUE(ToT->getTemplatedDecl());
  }

  void TypedTest_ImportPrototypeAfterImportedPrototype() {
    Decl *FromTU = getTuDecl(
        getPrototype() + getPrototype(), Lang_CXX);
    auto *From0 =
        FirstDeclMatcher<DeclTy>().match(FromTU, getPattern());
    auto *From1 = LastDeclMatcher<DeclTy>().match(FromTU, getPattern());
    ASSERT_FALSE(From0->isThisDeclarationADefinition());
    ASSERT_FALSE(From1->isThisDeclarationADefinition());

    Decl *Imported0 = Import(From0, Lang_CXX);
    Decl *Imported1 = Import(From1, Lang_CXX);
    Decl *ToTU = Imported0->getTranslationUnitDecl();

    EXPECT_EQ(DeclCounter<DeclTy>().match(ToTU, getPattern()), 2u);
    auto *To0 = FirstDeclMatcher<DeclTy>().match(ToTU, getPattern());
    auto *To1 = LastDeclMatcher<DeclTy>().match(ToTU, getPattern());
    EXPECT_TRUE(Imported0 == To0);
    EXPECT_TRUE(Imported1 == To1);
    EXPECT_FALSE(To0->isThisDeclarationADefinition());
    EXPECT_FALSE(To1->isThisDeclarationADefinition());
    EXPECT_EQ(To1->getPreviousDecl(), To0);
    if (auto *ToT0 = dyn_cast<TemplateDecl>(To0)) {
      auto *ToT1 = cast<TemplateDecl>(To1);
      ASSERT_TRUE(ToT0->getTemplatedDecl());
      ASSERT_TRUE(ToT1->getTemplatedDecl());
      EXPECT_EQ(ToT1->getTemplatedDecl()->getPreviousDecl(),
                ToT0->getTemplatedDecl());
    }
  }

  void TypedTest_ImportDefinitionAfterImportedPrototype() {
    Decl *FromTU = getTuDecl(
        getPrototype() + getDefinition(), Lang_CXX);
    auto *FromProto = FirstDeclMatcher<DeclTy>().match(FromTU, getPattern());
    auto *FromDef = LastDeclMatcher<DeclTy>().match(FromTU, getPattern());
    ASSERT_FALSE(FromProto->isThisDeclarationADefinition());
    ASSERT_TRUE(FromDef->isThisDeclarationADefinition());

    Decl *ImportedProto = Import(FromProto, Lang_CXX);
    Decl *ImportedDef = Import(FromDef, Lang_CXX);
    Decl *ToTU = ImportedProto->getTranslationUnitDecl();

    EXPECT_EQ(DeclCounter<DeclTy>().match(ToTU, getPattern()), 2u);
    auto *ToProto = FirstDeclMatcher<DeclTy>().match(ToTU, getPattern());
    auto *ToDef = LastDeclMatcher<DeclTy>().match(ToTU, getPattern());
    EXPECT_TRUE(ImportedProto == ToProto);
    EXPECT_TRUE(ImportedDef == ToDef);
    EXPECT_FALSE(ToProto->isThisDeclarationADefinition());
    EXPECT_TRUE(ToDef->isThisDeclarationADefinition());
    EXPECT_EQ(ToDef->getPreviousDecl(), ToProto);
    if (auto *ToProtoT = dyn_cast<TemplateDecl>(ToProto)) {
      auto *ToDefT = cast<TemplateDecl>(ToDef);
      ASSERT_TRUE(ToProtoT->getTemplatedDecl());
      ASSERT_TRUE(ToDefT->getTemplatedDecl());
      EXPECT_EQ(ToDefT->getTemplatedDecl()->getPreviousDecl(),
                ToProtoT->getTemplatedDecl());
    }
  }

  void TypedTest_ImportPrototypeAfterImportedDefinition() {
    Decl *FromTU = getTuDecl(
        getDefinition() + getPrototype(), Lang_CXX);
    auto *FromDef = FirstDeclMatcher<DeclTy>().match(FromTU, getPattern());
    auto *FromProto = LastDeclMatcher<DeclTy>().match(FromTU, getPattern());
    ASSERT_TRUE(FromDef->isThisDeclarationADefinition());
    ASSERT_FALSE(FromProto->isThisDeclarationADefinition());

    Decl *ImportedDef = Import(FromDef, Lang_CXX);
    Decl *ImportedProto = Import(FromProto, Lang_CXX);
    Decl *ToTU = ImportedDef->getTranslationUnitDecl();

    EXPECT_EQ(DeclCounter<DeclTy>().match(ToTU, getPattern()), 2u);
    auto *ToDef = FirstDeclMatcher<DeclTy>().match(ToTU, getPattern());
    auto *ToProto = LastDeclMatcher<DeclTy>().match(ToTU, getPattern());
    EXPECT_TRUE(ImportedDef == ToDef);
    EXPECT_TRUE(ImportedProto == ToProto);
    EXPECT_TRUE(ToDef->isThisDeclarationADefinition());
    EXPECT_FALSE(ToProto->isThisDeclarationADefinition());
    EXPECT_EQ(ToProto->getPreviousDecl(), ToDef);
    if (auto *ToDefT = dyn_cast<TemplateDecl>(ToDef)) {
      auto *ToProtoT = cast<TemplateDecl>(ToProto);
      ASSERT_TRUE(ToDefT->getTemplatedDecl());
      ASSERT_TRUE(ToProtoT->getTemplatedDecl());
      EXPECT_EQ(ToProtoT->getTemplatedDecl()->getPreviousDecl(),
                ToDefT->getTemplatedDecl());
    }
  }

  void TypedTest_ImportPrototypes() {
    Decl *FromTU0 = getTuDecl(getPrototype(), Lang_CXX, "input0.cc");
    Decl *FromTU1 = getTuDecl(getPrototype(), Lang_CXX, "input1.cc");
    auto *From0 = FirstDeclMatcher<DeclTy>().match(FromTU0, getPattern());
    auto *From1 = FirstDeclMatcher<DeclTy>().match(FromTU1, getPattern());
    ASSERT_FALSE(From0->isThisDeclarationADefinition());
    ASSERT_FALSE(From1->isThisDeclarationADefinition());

    Decl *Imported0 = Import(From0, Lang_CXX);
    Decl *Imported1 = Import(From1, Lang_CXX);
    Decl *ToTU = Imported0->getTranslationUnitDecl();

    EXPECT_EQ(DeclCounter<DeclTy>().match(ToTU, getPattern()), 2u);
    auto *To0 = FirstDeclMatcher<DeclTy>().match(ToTU, getPattern());
    auto *To1 = LastDeclMatcher<DeclTy>().match(ToTU, getPattern());
    EXPECT_TRUE(Imported0 == To0);
    EXPECT_TRUE(Imported1 == To1);
    EXPECT_FALSE(To0->isThisDeclarationADefinition());
    EXPECT_FALSE(To1->isThisDeclarationADefinition());
    EXPECT_EQ(To1->getPreviousDecl(), To0);
    if (auto *ToT0 = dyn_cast<TemplateDecl>(To0)) {
      auto *ToT1 = cast<TemplateDecl>(To1);
      ASSERT_TRUE(ToT0->getTemplatedDecl());
      ASSERT_TRUE(ToT1->getTemplatedDecl());
      EXPECT_EQ(ToT1->getTemplatedDecl()->getPreviousDecl(),
                ToT0->getTemplatedDecl());
    }
    // Extra check for specializations.
    // FIXME Add this check to other tests too (possibly factor out into a
    // function), when they start to pass.
    if (auto *From0F = dyn_cast<FunctionDecl>(From0)) {
      auto *To0F = cast<FunctionDecl>(To0);
      if (From0F->getTemplatedKind() ==
          FunctionDecl::TK_FunctionTemplateSpecialization) {
        auto *TemplateD = FirstDeclMatcher<FunctionTemplateDecl>().match(
            ToTU, functionTemplateDecl());
        auto *FirstSpecD = *(TemplateD->spec_begin());
        EXPECT_EQ(FirstSpecD->getCanonicalDecl(), To0F->getCanonicalDecl());
      }
    }
  }

  void TypedTest_ImportDefinitions() {
    Decl *FromTU0 = getTuDecl(getDefinition(), Lang_CXX, "input0.cc");
    Decl *FromTU1 = getTuDecl(getDefinition(), Lang_CXX, "input1.cc");
    auto *From0 = FirstDeclMatcher<DeclTy>().match(FromTU0, getPattern());
    auto *From1 = FirstDeclMatcher<DeclTy>().match(FromTU1, getPattern());
    ASSERT_TRUE(From0->isThisDeclarationADefinition());
    ASSERT_TRUE(From1->isThisDeclarationADefinition());

    Decl *Imported0 = Import(From0, Lang_CXX);
    Decl *Imported1 = Import(From1, Lang_CXX);
    Decl *ToTU = Imported0->getTranslationUnitDecl();

    EXPECT_EQ(Imported0, Imported1);
    EXPECT_EQ(DeclCounter<DeclTy>().match(ToTU, getPattern()), 1u);
    auto *To0 = FirstDeclMatcher<DeclTy>().match(ToTU, getPattern());
    EXPECT_TRUE(Imported0 == To0);
    EXPECT_TRUE(To0->isThisDeclarationADefinition());
    if (auto *ToT0 = dyn_cast<TemplateDecl>(To0))
      EXPECT_TRUE(ToT0->getTemplatedDecl());
  }

  void TypedTest_ImportDefinitionThenPrototype() {
    Decl *FromTUDef = getTuDecl(getDefinition(), Lang_CXX, "input0.cc");
    Decl *FromTUProto = getTuDecl(getPrototype(), Lang_CXX, "input1.cc");
    auto *FromDef = FirstDeclMatcher<DeclTy>().match(FromTUDef, getPattern());
    auto *FromProto =
        FirstDeclMatcher<DeclTy>().match(FromTUProto, getPattern());
    ASSERT_TRUE(FromDef->isThisDeclarationADefinition());
    ASSERT_FALSE(FromProto->isThisDeclarationADefinition());

    Decl *ImportedDef = Import(FromDef, Lang_CXX);
    Decl *ImportedProto = Import(FromProto, Lang_CXX);
    Decl *ToTU = ImportedDef->getTranslationUnitDecl();

    EXPECT_NE(ImportedDef, ImportedProto);
    EXPECT_EQ(DeclCounter<DeclTy>().match(ToTU, getPattern()), 2u);
    auto *ToDef = FirstDeclMatcher<DeclTy>().match(ToTU, getPattern());
    auto *ToProto = LastDeclMatcher<DeclTy>().match(ToTU, getPattern());
    EXPECT_TRUE(ImportedDef == ToDef);
    EXPECT_TRUE(ImportedProto == ToProto);
    EXPECT_TRUE(ToDef->isThisDeclarationADefinition());
    EXPECT_FALSE(ToProto->isThisDeclarationADefinition());
    EXPECT_EQ(ToProto->getPreviousDecl(), ToDef);
    if (auto *ToDefT = dyn_cast<TemplateDecl>(ToDef)) {
      auto *ToProtoT = cast<TemplateDecl>(ToProto);
      ASSERT_TRUE(ToDefT->getTemplatedDecl());
      ASSERT_TRUE(ToProtoT->getTemplatedDecl());
      EXPECT_EQ(ToProtoT->getTemplatedDecl()->getPreviousDecl(),
                ToDefT->getTemplatedDecl());
    }
  }

  void TypedTest_ImportPrototypeThenDefinition() {
    Decl *FromTUProto = getTuDecl(getPrototype(), Lang_CXX, "input0.cc");
    Decl *FromTUDef = getTuDecl(getDefinition(), Lang_CXX, "input1.cc");
    auto *FromProto =
        FirstDeclMatcher<DeclTy>().match(FromTUProto, getPattern());
    auto *FromDef = FirstDeclMatcher<DeclTy>().match(FromTUDef, getPattern());
    ASSERT_TRUE(FromDef->isThisDeclarationADefinition());
    ASSERT_FALSE(FromProto->isThisDeclarationADefinition());

    Decl *ImportedProto = Import(FromProto, Lang_CXX);
    Decl *ImportedDef = Import(FromDef, Lang_CXX);
    Decl *ToTU = ImportedDef->getTranslationUnitDecl();

    EXPECT_NE(ImportedDef, ImportedProto);
    EXPECT_EQ(DeclCounter<DeclTy>().match(ToTU, getPattern()), 2u);
    auto *ToProto = FirstDeclMatcher<DeclTy>().match(ToTU, getPattern());
    auto *ToDef = LastDeclMatcher<DeclTy>().match(ToTU, getPattern());
    EXPECT_TRUE(ImportedDef == ToDef);
    EXPECT_TRUE(ImportedProto == ToProto);
    EXPECT_TRUE(ToDef->isThisDeclarationADefinition());
    EXPECT_FALSE(ToProto->isThisDeclarationADefinition());
    EXPECT_EQ(ToDef->getPreviousDecl(), ToProto);
    if (auto *ToDefT = dyn_cast<TemplateDecl>(ToDef)) {
      auto *ToProtoT = cast<TemplateDecl>(ToProto);
      ASSERT_TRUE(ToDefT->getTemplatedDecl());
      ASSERT_TRUE(ToProtoT->getTemplatedDecl());
      EXPECT_EQ(ToDefT->getTemplatedDecl()->getPreviousDecl(),
                ToProtoT->getTemplatedDecl());
    }
  }

  void TypedTest_WholeRedeclChainIsImportedAtOnce() {
    Decl *FromTU = getTuDecl(getPrototype() + getDefinition(), Lang_CXX);
    auto *FromD = // Definition
        LastDeclMatcher<DeclTy>().match(FromTU, getPattern());
    ASSERT_TRUE(FromD->isThisDeclarationADefinition());

    Decl *ImportedD = Import(FromD, Lang_CXX);
    Decl *ToTU = ImportedD->getTranslationUnitDecl();

    // The whole redecl chain is imported at once.
    EXPECT_EQ(DeclCounter<DeclTy>().match(ToTU, getPattern()), 2u);
    EXPECT_TRUE(cast<DeclTy>(ImportedD)->isThisDeclarationADefinition());
  }

  void TypedTest_ImportPrototypeThenProtoAndDefinition() {
    {
      Decl *FromTU = getTuDecl(getPrototype(), Lang_CXX, "input0.cc");
      auto *FromD = FirstDeclMatcher<DeclTy>().match(FromTU, getPattern());
      Import(FromD, Lang_CXX);
    }
    {
      Decl *FromTU =
          getTuDecl(getPrototype() + getDefinition(), Lang_CXX, "input1.cc");
      auto *FromD = FirstDeclMatcher<DeclTy>().match(FromTU, getPattern());
      Import(FromD, Lang_CXX);
    }

    Decl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();

    ASSERT_EQ(DeclCounter<DeclTy>().match(ToTU, getPattern()), 3u);
    DeclTy *ProtoD = FirstDeclMatcher<DeclTy>().match(ToTU, getPattern());
    EXPECT_FALSE(ProtoD->isThisDeclarationADefinition());

    DeclTy *DefinitionD = LastDeclMatcher<DeclTy>().match(ToTU, getPattern());
    EXPECT_TRUE(DefinitionD->isThisDeclarationADefinition());

    EXPECT_TRUE(DefinitionD->getPreviousDecl());
    EXPECT_FALSE(
        DefinitionD->getPreviousDecl()->isThisDeclarationADefinition());
    EXPECT_EQ(DefinitionD->getPreviousDecl()->getPreviousDecl(), ProtoD);
  }
};

#define ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(BaseTemplate, TypeParam,       \
                                                NamePrefix, TestCase)          \
  using BaseTemplate##TypeParam = BaseTemplate<TypeParam>;                     \
  TEST_P(BaseTemplate##TypeParam, NamePrefix##TestCase) {                      \
    TypedTest_##TestCase();                                                    \
  }

ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(
    RedeclChain, Function, ,
    PrototypeShouldBeImportedAsAPrototypeWhenThereIsNoDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(
    RedeclChain, Class, ,
    PrototypeShouldBeImportedAsAPrototypeWhenThereIsNoDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(
    RedeclChain, Variable, ,
    PrototypeShouldBeImportedAsAPrototypeWhenThereIsNoDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(
    RedeclChain, FunctionTemplate, ,
    PrototypeShouldBeImportedAsAPrototypeWhenThereIsNoDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(
    RedeclChain, ClassTemplate, ,
    PrototypeShouldBeImportedAsAPrototypeWhenThereIsNoDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(
    RedeclChain, FunctionTemplateSpec, ,
    PrototypeShouldBeImportedAsAPrototypeWhenThereIsNoDefinition)

ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(
    RedeclChain, Function, , DefinitionShouldBeImportedAsADefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(
    RedeclChain, Class, , DefinitionShouldBeImportedAsADefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(
    RedeclChain, Variable, , DefinitionShouldBeImportedAsADefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(
    RedeclChain, FunctionTemplate, ,
    DefinitionShouldBeImportedAsADefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(
    RedeclChain, ClassTemplate, , DefinitionShouldBeImportedAsADefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(
    RedeclChain, FunctionTemplateSpec, ,
    DefinitionShouldBeImportedAsADefinition)

ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, Function, ,
                                        ImportPrototypeAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, Class, ,
                                        ImportPrototypeAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, Variable, ,
                                        ImportPrototypeAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, FunctionTemplate, ,
                                        ImportPrototypeAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, ClassTemplate, ,
                                        ImportPrototypeAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, FunctionTemplateSpec, ,
                                        ImportPrototypeAfterImportedPrototype)

ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, Function, ,
                                        ImportDefinitionAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, Class, ,
                                        ImportDefinitionAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, Variable, ,
                                        ImportDefinitionAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, FunctionTemplate, ,
                                        ImportDefinitionAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, ClassTemplate, ,
                                        ImportDefinitionAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, FunctionTemplateSpec, ,
                                        ImportDefinitionAfterImportedPrototype)

ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, Function, ,
                                        ImportPrototypeAfterImportedDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, Class, ,
                                        ImportPrototypeAfterImportedDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, Variable, ,
                                        ImportPrototypeAfterImportedDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, FunctionTemplate, ,
                                        ImportPrototypeAfterImportedDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, ClassTemplate, ,
                                        ImportPrototypeAfterImportedDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, FunctionTemplateSpec, ,
                                        ImportPrototypeAfterImportedDefinition)

ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, Function, ,
                                        ImportPrototypes)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, Class, , ImportPrototypes)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, Variable, ,
                                        ImportPrototypes)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, FunctionTemplate, ,
                                        ImportPrototypes)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, ClassTemplate, ,
                                        ImportPrototypes)
// FIXME This does not pass, possible error with Spec import.
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, FunctionTemplateSpec,
                                        DISABLED_, ImportPrototypes)

ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, Function, ,
                                        ImportDefinitions)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, Class, ,
                                        ImportDefinitions)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, Variable, ,
                                        ImportDefinitions)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, FunctionTemplate, ,
                                        ImportDefinitions)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, ClassTemplate, ,
                                        ImportDefinitions)
// FIXME This does not pass, possible error with Spec import.
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, FunctionTemplateSpec,
                                        DISABLED_, ImportDefinitions)

ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, Function, ,
                                        ImportDefinitionThenPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, Class, ,
                                        ImportDefinitionThenPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, Variable, ,
                                        ImportDefinitionThenPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, FunctionTemplate, ,
                                        ImportDefinitionThenPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, ClassTemplate, ,
                                        ImportDefinitionThenPrototype)
// FIXME This does not pass, possible error with Spec import.
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, FunctionTemplateSpec,
                                        DISABLED_,
                                        ImportDefinitionThenPrototype)

ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, Function, ,
                                        ImportPrototypeThenDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, Class, ,
                                        ImportPrototypeThenDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, Variable, ,
                                        ImportPrototypeThenDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, FunctionTemplate, ,
                                        ImportPrototypeThenDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, ClassTemplate, ,
                                        ImportPrototypeThenDefinition)
// FIXME This does not pass, possible error with Spec import.
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, FunctionTemplateSpec,
                                        DISABLED_,
                                        ImportPrototypeThenDefinition)

ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, Function, ,
                                        WholeRedeclChainIsImportedAtOnce)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, Variable, ,
                                        WholeRedeclChainIsImportedAtOnce)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, FunctionTemplate, ,
                                        WholeRedeclChainIsImportedAtOnce)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, FunctionTemplateSpec, ,
                                        WholeRedeclChainIsImportedAtOnce)

ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, Function, ,
                                        ImportPrototypeThenProtoAndDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, Variable, ,
                                        ImportPrototypeThenProtoAndDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, FunctionTemplate, ,
                                        ImportPrototypeThenProtoAndDefinition)
// FIXME This does not pass, possible error with Spec import.
ASTIMPORTER_INSTANTIATE_TYPED_TEST_CASE(RedeclChain, FunctionTemplateSpec,
                                        DISABLED_,
                                        ImportPrototypeThenProtoAndDefinition)

INSTANTIATE_TEST_CASE_P(ParameterizedTests, RedeclChainFunction,
                        DefaultTestValuesForRunOptions, );
INSTANTIATE_TEST_CASE_P(ParameterizedTests, RedeclChainClass,
                        DefaultTestValuesForRunOptions, );
INSTANTIATE_TEST_CASE_P(ParameterizedTests, RedeclChainVariable,
                        DefaultTestValuesForRunOptions, );
INSTANTIATE_TEST_CASE_P(ParameterizedTests, RedeclChainFunctionTemplate,
                        DefaultTestValuesForRunOptions, );
INSTANTIATE_TEST_CASE_P(ParameterizedTests, RedeclChainClassTemplate,
                        DefaultTestValuesForRunOptions, );
INSTANTIATE_TEST_CASE_P(ParameterizedTests, RedeclChainFunctionTemplateSpec,
                        DefaultTestValuesForRunOptions, );



struct ImportFriendClasses : ASTImporterOptionSpecificTestBase {};

TEST_P(ImportFriendClasses, ImportOfFriendRecordDoesNotMergeDefinition) {
  Decl *FromTU = getTuDecl(
      R"(
      class A {
        template <int I> class F {};
        class X {
          template <int I> friend class F;
        };
      };
      )",
      Lang_CXX, "input0.cc");

  auto *FromClass = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("F"), isDefinition()));
  auto *FromFriendClass = LastDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("F")));

  ASSERT_TRUE(FromClass);
  ASSERT_TRUE(FromFriendClass);
  ASSERT_NE(FromClass, FromFriendClass);
  ASSERT_EQ(FromFriendClass->getDefinition(), FromClass);
  ASSERT_EQ(FromFriendClass->getPreviousDecl(), FromClass);
  ASSERT_EQ(FromFriendClass->getDescribedClassTemplate()->getPreviousDecl(),
            FromClass->getDescribedClassTemplate());

  auto *ToClass = cast<CXXRecordDecl>(Import(FromClass, Lang_CXX));
  auto *ToFriendClass = cast<CXXRecordDecl>(Import(FromFriendClass, Lang_CXX));

  EXPECT_TRUE(ToClass);
  EXPECT_TRUE(ToFriendClass);
  EXPECT_NE(ToClass, ToFriendClass);
  EXPECT_EQ(ToFriendClass->getDefinition(), ToClass);
  EXPECT_EQ(ToFriendClass->getPreviousDecl(), ToClass);
  EXPECT_EQ(ToFriendClass->getDescribedClassTemplate()->getPreviousDecl(),
            ToClass->getDescribedClassTemplate());
}

TEST_P(ImportFriendClasses, ImportOfRecursiveFriendClass) {
  Decl *FromTu = getTuDecl(
      R"(
      class declToImport {
        friend class declToImport;
      };
      )",
      Lang_CXX, "input.cc");

  auto *FromD = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTu, cxxRecordDecl(hasName("declToImport")));
  auto *ToD = Import(FromD, Lang_CXX);
  auto Pattern = cxxRecordDecl(has(friendDecl()));
  ASSERT_TRUE(MatchVerifier<Decl>{}.match(FromD, Pattern));
  EXPECT_TRUE(MatchVerifier<Decl>{}.match(ToD, Pattern));
}

TEST_P(ImportFriendClasses, ImportOfRecursiveFriendClassTemplate) {
  Decl *FromTu = getTuDecl(
      R"(
      template<class A> class declToImport {
        template<class A1> friend class declToImport;
      };
      )",
      Lang_CXX, "input.cc");

  auto *FromD =
      FirstDeclMatcher<ClassTemplateDecl>().match(FromTu, classTemplateDecl());
  auto *ToD = Import(FromD, Lang_CXX);

  auto Pattern = classTemplateDecl(
      has(cxxRecordDecl(has(friendDecl(has(classTemplateDecl()))))));
  ASSERT_TRUE(MatchVerifier<Decl>{}.match(FromD, Pattern));
  EXPECT_TRUE(MatchVerifier<Decl>{}.match(ToD, Pattern));

  auto *Class =
      FirstDeclMatcher<ClassTemplateDecl>().match(ToD, classTemplateDecl());
  auto *Friend = FirstDeclMatcher<FriendDecl>().match(ToD, friendDecl());
  EXPECT_NE(Friend->getFriendDecl(), Class);
  EXPECT_EQ(Friend->getFriendDecl()->getPreviousDecl(), Class);
}

TEST_P(ImportFriendClasses, ProperPrevDeclForClassTemplateDecls) {
  auto Pattern = classTemplateSpecializationDecl(hasName("X"));

  ClassTemplateSpecializationDecl *Imported1;
  {
    Decl *FromTU = getTuDecl("template<class T> class X;"
                             "struct Y { friend class X<int>; };",
                             Lang_CXX, "input0.cc");
    auto *FromD = FirstDeclMatcher<ClassTemplateSpecializationDecl>().match(
        FromTU, Pattern);

    Imported1 = cast<ClassTemplateSpecializationDecl>(Import(FromD, Lang_CXX));
  }
  ClassTemplateSpecializationDecl *Imported2;
  {
    Decl *FromTU = getTuDecl("template<class T> class X;"
                             "template<> class X<int>{};"
                             "struct Z { friend class X<int>; };",
                             Lang_CXX, "input1.cc");
    auto *FromD = FirstDeclMatcher<ClassTemplateSpecializationDecl>().match(
        FromTU, Pattern);

    Imported2 = cast<ClassTemplateSpecializationDecl>(Import(FromD, Lang_CXX));
  }

  Decl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  EXPECT_EQ(DeclCounter<ClassTemplateSpecializationDecl>().match(ToTU, Pattern),
            2u);
  ASSERT_TRUE(Imported2->getPreviousDecl());
  EXPECT_EQ(Imported2->getPreviousDecl(), Imported1);
}

TEST_P(ImportFriendClasses, TypeForDeclShouldBeSetInTemplated) {
  Decl *FromTU0 = getTuDecl(
      R"(
      class X {
        class Y;
      };
      class X::Y {
        template <typename T>
        friend class F; // The decl context of F is the global namespace.
      };
      )",
      Lang_CXX, "input0.cc");
  auto *Fwd = FirstDeclMatcher<ClassTemplateDecl>().match(
      FromTU0, classTemplateDecl(hasName("F")));
  auto *Imported0 = cast<ClassTemplateDecl>(Import(Fwd, Lang_CXX));
  Decl *FromTU1 = getTuDecl(
      R"(
      template <typename T>
      class F {};
      )",
      Lang_CXX, "input1.cc");
  auto *Definition = FirstDeclMatcher<ClassTemplateDecl>().match(
      FromTU1, classTemplateDecl(hasName("F")));
  auto *Imported1 = cast<ClassTemplateDecl>(Import(Definition, Lang_CXX));
  EXPECT_EQ(Imported0->getTemplatedDecl()->getTypeForDecl(),
            Imported1->getTemplatedDecl()->getTypeForDecl());
}

TEST_P(ImportFriendClasses, DeclsFromFriendsShouldBeInRedeclChains) {
  Decl *From, *To;
  std::tie(From, To) =
      getImportedDecl("class declToImport {};", Lang_CXX,
                      "class Y { friend class declToImport; };", Lang_CXX);
  auto *Imported = cast<CXXRecordDecl>(To);

  EXPECT_TRUE(Imported->getPreviousDecl());
}

TEST_P(ImportFriendClasses,
       ImportOfClassTemplateDefinitionShouldConnectToFwdFriend) {
  Decl *ToTU = getToTuDecl(
      R"(
      class X {
        class Y;
      };
      class X::Y {
        template <typename T>
        friend class F; // The decl context of F is the global namespace.
      };
      )",
      Lang_CXX);
  auto *ToDecl = FirstDeclMatcher<ClassTemplateDecl>().match(
      ToTU, classTemplateDecl(hasName("F")));
  Decl *FromTU = getTuDecl(
      R"(
      template <typename T>
      class F {};
      )",
      Lang_CXX, "input0.cc");
  auto *Definition = FirstDeclMatcher<ClassTemplateDecl>().match(
      FromTU, classTemplateDecl(hasName("F")));
  auto *ImportedDef = cast<ClassTemplateDecl>(Import(Definition, Lang_CXX));
  EXPECT_TRUE(ImportedDef->getPreviousDecl());
  EXPECT_EQ(ToDecl, ImportedDef->getPreviousDecl());
  EXPECT_EQ(ToDecl->getTemplatedDecl(),
            ImportedDef->getTemplatedDecl()->getPreviousDecl());
}

TEST_P(ImportFriendClasses,
       ImportOfClassTemplateDefinitionAndFwdFriendShouldBeLinked) {
  Decl *FromTU0 = getTuDecl(
      R"(
      class X {
        class Y;
      };
      class X::Y {
        template <typename T>
        friend class F; // The decl context of F is the global namespace.
      };
      )",
      Lang_CXX, "input0.cc");
  auto *Fwd = FirstDeclMatcher<ClassTemplateDecl>().match(
      FromTU0, classTemplateDecl(hasName("F")));
  auto *ImportedFwd = cast<ClassTemplateDecl>(Import(Fwd, Lang_CXX));
  Decl *FromTU1 = getTuDecl(
      R"(
      template <typename T>
      class F {};
      )",
      Lang_CXX, "input1.cc");
  auto *Definition = FirstDeclMatcher<ClassTemplateDecl>().match(
      FromTU1, classTemplateDecl(hasName("F")));
  auto *ImportedDef = cast<ClassTemplateDecl>(Import(Definition, Lang_CXX));
  EXPECT_TRUE(ImportedDef->getPreviousDecl());
  EXPECT_EQ(ImportedFwd, ImportedDef->getPreviousDecl());
  EXPECT_EQ(ImportedFwd->getTemplatedDecl(),
            ImportedDef->getTemplatedDecl()->getPreviousDecl());
}

TEST_P(ImportFriendClasses, ImportOfClassDefinitionAndFwdFriendShouldBeLinked) {
  Decl *FromTU0 = getTuDecl(
      R"(
      class X {
        class Y;
      };
      class X::Y {
        friend class F; // The decl context of F is the global namespace.
      };
      )",
      Lang_CXX, "input0.cc");
  auto *Friend = FirstDeclMatcher<FriendDecl>().match(FromTU0, friendDecl());
  QualType FT = Friend->getFriendType()->getType();
  FT = FromTU0->getASTContext().getCanonicalType(FT);
  auto *Fwd = cast<TagType>(FT)->getDecl();
  auto *ImportedFwd = Import(Fwd, Lang_CXX);
  Decl *FromTU1 = getTuDecl(
      R"(
      class F {};
      )",
      Lang_CXX, "input1.cc");
  auto *Definition = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU1, cxxRecordDecl(hasName("F")));
  auto *ImportedDef = Import(Definition, Lang_CXX);
  EXPECT_TRUE(ImportedDef->getPreviousDecl());
  EXPECT_EQ(ImportedFwd, ImportedDef->getPreviousDecl());
}

TEST_P(ASTImporterOptionSpecificTestBase, FriendFunInClassTemplate) {
  auto *Code = R"(
  template <class T>
  struct X {
    friend void foo(){}
  };
      )";
  TranslationUnitDecl *ToTU = getToTuDecl(Code, Lang_CXX);
  auto *ToFoo = FirstDeclMatcher<FunctionDecl>().match(
      ToTU, functionDecl(hasName("foo")));

  TranslationUnitDecl *FromTU = getTuDecl(Code, Lang_CXX, "input.cc");
  auto *FromFoo = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("foo")));
  auto *ImportedFoo = Import(FromFoo, Lang_CXX);
  EXPECT_EQ(ImportedFoo, ToFoo);
}

struct DeclContextTest : ASTImporterOptionSpecificTestBase {};

TEST_P(DeclContextTest, removeDeclOfClassTemplateSpecialization) {
  Decl *TU = getTuDecl(
      R"(
      namespace NS {

      template <typename T>
      struct S {};
      template struct S<int>;

      inline namespace INS {
        template <typename T>
        struct S {};
        template struct S<int>;
      }

      }
      )", Lang_CXX11, "input0.cc");
  auto *NS = FirstDeclMatcher<NamespaceDecl>().match(
      TU, namespaceDecl());
  auto *Spec = FirstDeclMatcher<ClassTemplateSpecializationDecl>().match(
      TU, classTemplateSpecializationDecl());
  ASSERT_TRUE(NS->containsDecl(Spec));

  NS->removeDecl(Spec);
  EXPECT_FALSE(NS->containsDecl(Spec));
}

TEST_P(DeclContextTest,
       removeDeclShouldNotFailEvenIfWeHaveExternalVisibleStorage) {
  Decl *TU = getTuDecl("extern int A; int A;", Lang_CXX);
  auto *A0 = FirstDeclMatcher<VarDecl>().match(TU, varDecl(hasName("A")));
  auto *A1 = LastDeclMatcher<VarDecl>().match(TU, varDecl(hasName("A")));

  // Investigate the list.
  auto *DC = A0->getDeclContext();
  ASSERT_TRUE(DC->containsDecl(A0));
  ASSERT_TRUE(DC->containsDecl(A1));

  // Investigate the lookup table.
  auto *Map = DC->getLookupPtr();
  ASSERT_TRUE(Map);
  auto I = Map->find(A0->getDeclName());
  ASSERT_NE(I, Map->end());
  StoredDeclsList &L = I->second;
  // The lookup table contains the most recent decl of A.
  ASSERT_NE(L.getAsDecl(), A0);
  ASSERT_EQ(L.getAsDecl(), A1);

  ASSERT_TRUE(L.getAsDecl());
  // Simulate the private function DeclContext::reconcileExternalVisibleStorage.
  // The point here is to have a Vec with only one element, which is not the
  // one we are going to delete from the DC later.
  L.setHasExternalDecls();
  ASSERT_TRUE(L.getAsVector());
  ASSERT_EQ(1u, L.getAsVector()->size());

  // This asserts in the old implementation.
  DC->removeDecl(A0);
  EXPECT_FALSE(DC->containsDecl(A0));
}

struct ImportFunctionTemplateSpecializations
    : ASTImporterOptionSpecificTestBase {};

TEST_P(ImportFunctionTemplateSpecializations,
       TUshouldNotContainFunctionTemplateImplicitInstantiation) {

  Decl *FromTU = getTuDecl(
      R"(
      template<class T>
      int f() { return 0; }
      void foo() { f<int>(); }
      )",
      Lang_CXX, "input0.cc");

  // Check that the function template instantiation is NOT the child of the TU.
  auto Pattern = translationUnitDecl(
      unless(has(functionDecl(hasName("f"), isTemplateInstantiation()))));
  ASSERT_TRUE(MatchVerifier<Decl>{}.match(FromTU, Pattern));

  auto *Foo = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("foo")));
  ASSERT_TRUE(Import(Foo, Lang_CXX));

  auto *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  EXPECT_TRUE(MatchVerifier<Decl>{}.match(ToTU, Pattern));
}

TEST_P(ImportFunctionTemplateSpecializations,
       TUshouldNotContainFunctionTemplateExplicitInstantiation) {

  Decl *FromTU = getTuDecl(
      R"(
      template<class T>
      int f() { return 0; }
      template int f<int>();
      )",
      Lang_CXX, "input0.cc");

  // Check that the function template instantiation is NOT the child of the TU.
  auto Instantiation = functionDecl(hasName("f"), isTemplateInstantiation());
  auto Pattern = translationUnitDecl(unless(has(Instantiation)));
  ASSERT_TRUE(MatchVerifier<Decl>{}.match(FromTU, Pattern));

  ASSERT_TRUE(
      Import(FirstDeclMatcher<Decl>().match(FromTU, Instantiation), Lang_CXX));

  auto *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  EXPECT_TRUE(MatchVerifier<Decl>{}.match(ToTU, Pattern));
}

TEST_P(ImportFunctionTemplateSpecializations,
       TUshouldContainFunctionTemplateSpecialization) {

  Decl *FromTU = getTuDecl(
      R"(
      template<class T>
      int f() { return 0; }
      template <> int f<int>() { return 4; }
      )",
      Lang_CXX, "input0.cc");

  // Check that the function template specialization is the child of the TU.
  auto Specialization =
      functionDecl(hasName("f"), isExplicitTemplateSpecialization());
  auto Pattern = translationUnitDecl(has(Specialization));
  ASSERT_TRUE(MatchVerifier<Decl>{}.match(FromTU, Pattern));

  ASSERT_TRUE(
      Import(FirstDeclMatcher<Decl>().match(FromTU, Specialization), Lang_CXX));

  auto *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  EXPECT_TRUE(MatchVerifier<Decl>{}.match(ToTU, Pattern));
}

TEST_P(ImportFunctionTemplateSpecializations,
       FunctionTemplateSpecializationRedeclChain) {

  Decl *FromTU = getTuDecl(
      R"(
      template<class T>
      int f() { return 0; }
      template <> int f<int>() { return 4; }
      )",
      Lang_CXX, "input0.cc");

  auto Spec = functionDecl(hasName("f"), isExplicitTemplateSpecialization(),
                           hasParent(translationUnitDecl()));
  auto *FromSpecD = FirstDeclMatcher<Decl>().match(FromTU, Spec);
  {
    auto *TU = FromTU;
    auto *SpecD = FromSpecD;
    auto *TemplateD = FirstDeclMatcher<FunctionTemplateDecl>().match(
        TU, functionTemplateDecl());
    auto *FirstSpecD = *(TemplateD->spec_begin());
    ASSERT_EQ(SpecD, FirstSpecD);
    ASSERT_TRUE(SpecD->getPreviousDecl());
    ASSERT_FALSE(cast<FunctionDecl>(SpecD->getPreviousDecl())
                     ->doesThisDeclarationHaveABody());
  }

  ASSERT_TRUE(Import(FromSpecD, Lang_CXX));

  {
    auto *TU = ToAST->getASTContext().getTranslationUnitDecl();
    auto *SpecD = FirstDeclMatcher<Decl>().match(TU, Spec);
    auto *TemplateD = FirstDeclMatcher<FunctionTemplateDecl>().match(
        TU, functionTemplateDecl());
    auto *FirstSpecD = *(TemplateD->spec_begin());
    EXPECT_EQ(SpecD, FirstSpecD);
    ASSERT_TRUE(SpecD->getPreviousDecl());
    EXPECT_FALSE(cast<FunctionDecl>(SpecD->getPreviousDecl())
                     ->doesThisDeclarationHaveABody());
  }
}

TEST_P(ImportFunctionTemplateSpecializations,
       MatchNumberOfFunctionTemplateSpecializations) {

  Decl *FromTU = getTuDecl(
      R"(
      template <typename T> constexpr int f() { return 0; }
      template <> constexpr int f<int>() { return 4; }
      void foo() {
        static_assert(f<char>() == 0, "");
        static_assert(f<int>() == 4, "");
      }
      )",
      Lang_CXX11, "input0.cc");
  auto *FromD = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("foo")));

  Import(FromD, Lang_CXX11);
  auto *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  EXPECT_EQ(
      DeclCounter<FunctionDecl>().match(FromTU, functionDecl(hasName("f"))),
      DeclCounter<FunctionDecl>().match(ToTU, functionDecl(hasName("f"))));
}

TEST_P(ASTImporterOptionSpecificTestBase,
    ImportShouldNotReportFalseODRErrorWhenRecordIsBeingDefined) {
  {
    Decl *FromTU = getTuDecl(
        R"(
            template <typename T>
            struct B;
            )",
        Lang_CXX, "input0.cc");
    auto *FromD = FirstDeclMatcher<ClassTemplateDecl>().match(
        FromTU, classTemplateDecl(hasName("B")));

    Import(FromD, Lang_CXX);
  }

  {
    Decl *FromTU = getTuDecl(
        R"(
            template <typename T>
            struct B {
              void f();
              B* b;
            };
            )",
        Lang_CXX, "input1.cc");
    FunctionDecl *FromD = FirstDeclMatcher<FunctionDecl>().match(
        FromTU, functionDecl(hasName("f")));
    Import(FromD, Lang_CXX);
    auto *FromCTD = FirstDeclMatcher<ClassTemplateDecl>().match(
        FromTU, classTemplateDecl(hasName("B")));
    auto *ToCTD = cast<ClassTemplateDecl>(Import(FromCTD, Lang_CXX));
    EXPECT_TRUE(ToCTD->isThisDeclarationADefinition());

    // We expect no (ODR) warning during the import.
    auto *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
    EXPECT_EQ(0u, ToTU->getASTContext().getDiagnostics().getNumWarnings());
  }
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportingTypedefShouldImportTheCompleteType) {
  // We already have an incomplete underlying type in the "To" context.
  auto Code =
      R"(
      template <typename T>
      struct S {
        void foo();
      };
      using U = S<int>;
      )";
  Decl *ToTU = getToTuDecl(Code, Lang_CXX11);
  auto *ToD = FirstDeclMatcher<TypedefNameDecl>().match(ToTU,
      typedefNameDecl(hasName("U")));
  ASSERT_TRUE(ToD->getUnderlyingType()->isIncompleteType());

  // The "From" context has the same typedef, but the underlying type is
  // complete this time.
  Decl *FromTU = getTuDecl(std::string(Code) +
      R"(
      void foo(U* u) {
        u->foo();
      }
      )", Lang_CXX11);
  auto *FromD = FirstDeclMatcher<TypedefNameDecl>().match(FromTU,
      typedefNameDecl(hasName("U")));
  ASSERT_FALSE(FromD->getUnderlyingType()->isIncompleteType());

  // The imported type should be complete.
  auto *ImportedD = cast<TypedefNameDecl>(Import(FromD, Lang_CXX11));
  EXPECT_FALSE(ImportedD->getUnderlyingType()->isIncompleteType());
}

struct ASTImporterLookupTableTest : ASTImporterOptionSpecificTestBase {};

TEST_P(ASTImporterLookupTableTest, OneDecl) {
  auto *ToTU = getToTuDecl("int a;", Lang_CXX);
  auto *D = FirstDeclMatcher<VarDecl>().match(ToTU, varDecl(hasName("a")));
  ASTImporterLookupTable LT(*ToTU);
  auto Res = LT.lookup(ToTU, D->getDeclName());
  ASSERT_EQ(Res.size(), 1u);
  EXPECT_EQ(*Res.begin(), D);
}

static Decl *findInDeclListOfDC(DeclContext *DC, DeclarationName Name) {
  for (Decl *D : DC->decls()) {
    if (auto *ND = dyn_cast<NamedDecl>(D))
      if (ND->getDeclName() == Name)
        return ND;
  }
  return nullptr;
}

TEST_P(ASTImporterLookupTableTest,
    FriendWhichIsnotFoundByNormalLookupShouldBeFoundByImporterSpecificLookup) {
  auto *Code = R"(
  template <class T>
  struct X {
    friend void foo(){}
  };
      )";
  TranslationUnitDecl *ToTU = getToTuDecl(Code, Lang_CXX);
  auto *X = FirstDeclMatcher<ClassTemplateDecl>().match(
      ToTU, classTemplateDecl(hasName("X")));
  auto *Foo = FirstDeclMatcher<FunctionDecl>().match(
      ToTU, functionDecl(hasName("foo")));
  DeclContext *FooDC = Foo->getDeclContext();
  DeclContext *FooLexicalDC = Foo->getLexicalDeclContext();
  ASSERT_EQ(cast<Decl>(FooLexicalDC), X->getTemplatedDecl());
  ASSERT_EQ(cast<Decl>(FooDC), ToTU);
  DeclarationName FooName = Foo->getDeclName();

  // Cannot find in the LookupTable of its DC (TUDecl)
  SmallVector<NamedDecl *, 2> FoundDecls;
  FooDC->getRedeclContext()->localUncachedLookup(FooName, FoundDecls);
  EXPECT_EQ(FoundDecls.size(), 0u);

  // Cannot find in the LookupTable of its LexicalDC (X)
  FooLexicalDC->getRedeclContext()->localUncachedLookup(FooName, FoundDecls);
  EXPECT_EQ(FoundDecls.size(), 0u);

  // Can't find in the list of Decls of the DC.
  EXPECT_EQ(findInDeclListOfDC(FooDC, FooName), nullptr);

  // Can't find in the list of Decls of the LexicalDC
  EXPECT_EQ(findInDeclListOfDC(FooLexicalDC, FooName), nullptr);

  // ASTImporter specific lookup finds it.
  ASTImporterLookupTable LT(*ToTU);
  auto Res = LT.lookup(FooDC, Foo->getDeclName());
  ASSERT_EQ(Res.size(), 1u);
  EXPECT_EQ(*Res.begin(), Foo);
}

TEST_P(ASTImporterLookupTableTest,
       FwdDeclStructShouldBeFoundByImporterSpecificLookup) {
  TranslationUnitDecl *ToTU =
      getToTuDecl("struct A { struct Foo *p; };", Lang_C);
  auto *Foo =
      FirstDeclMatcher<RecordDecl>().match(ToTU, recordDecl(hasName("Foo")));
  auto *A =
      FirstDeclMatcher<RecordDecl>().match(ToTU, recordDecl(hasName("A")));
  DeclContext *FooDC = Foo->getDeclContext();
  DeclContext *FooLexicalDC = Foo->getLexicalDeclContext();
  ASSERT_EQ(cast<Decl>(FooLexicalDC), A);
  ASSERT_EQ(cast<Decl>(FooDC), ToTU);
  DeclarationName FooName = Foo->getDeclName();

  // Cannot find in the LookupTable of its DC (TUDecl).
  SmallVector<NamedDecl *, 2> FoundDecls;
  FooDC->getRedeclContext()->localUncachedLookup(FooName, FoundDecls);
  EXPECT_EQ(FoundDecls.size(), 0u);

  // Cannot find in the LookupTable of its LexicalDC (A).
  FooLexicalDC->getRedeclContext()->localUncachedLookup(FooName, FoundDecls);
  EXPECT_EQ(FoundDecls.size(), 0u);

  // Can't find in the list of Decls of the DC.
  EXPECT_EQ(findInDeclListOfDC(FooDC, FooName), nullptr);

  // Can find in the list of Decls of the LexicalDC.
  EXPECT_EQ(findInDeclListOfDC(FooLexicalDC, FooName), Foo);

  // ASTImporter specific lookup finds it.
  ASTImporterLookupTable LT(*ToTU);
  auto Res = LT.lookup(FooDC, Foo->getDeclName());
  ASSERT_EQ(Res.size(), 1u);
  EXPECT_EQ(*Res.begin(), Foo);
}

TEST_P(ASTImporterLookupTableTest, LookupFindsNamesInDifferentDC) {
  TranslationUnitDecl *ToTU =
      getToTuDecl("int V; struct A { int V; }; struct B { int V; };", Lang_C);
  DeclarationName VName = FirstDeclMatcher<VarDecl>()
                              .match(ToTU, varDecl(hasName("V")))
                              ->getDeclName();
  auto *A =
      FirstDeclMatcher<RecordDecl>().match(ToTU, recordDecl(hasName("A")));
  auto *B =
      FirstDeclMatcher<RecordDecl>().match(ToTU, recordDecl(hasName("B")));

  ASTImporterLookupTable LT(*ToTU);

  auto Res = LT.lookup(cast<DeclContext>(A), VName);
  ASSERT_EQ(Res.size(), 1u);
  EXPECT_EQ(*Res.begin(), FirstDeclMatcher<FieldDecl>().match(
                        ToTU, fieldDecl(hasName("V"),
                                        hasParent(recordDecl(hasName("A"))))));
  Res = LT.lookup(cast<DeclContext>(B), VName);
  ASSERT_EQ(Res.size(), 1u);
  EXPECT_EQ(*Res.begin(), FirstDeclMatcher<FieldDecl>().match(
                        ToTU, fieldDecl(hasName("V"),
                                        hasParent(recordDecl(hasName("B"))))));
  Res = LT.lookup(ToTU, VName);
  ASSERT_EQ(Res.size(), 1u);
  EXPECT_EQ(*Res.begin(), FirstDeclMatcher<VarDecl>().match(
                        ToTU, varDecl(hasName("V"),
                                        hasParent(translationUnitDecl()))));
}

TEST_P(ASTImporterLookupTableTest, LookupFindsOverloadedNames) {
  TranslationUnitDecl *ToTU = getToTuDecl(
      R"(
      void foo();
      void foo(int);
      void foo(int, int);
      )",
      Lang_CXX);

  ASTImporterLookupTable LT(*ToTU);
  auto *F0 = FirstDeclMatcher<FunctionDecl>().match(ToTU, functionDecl());
  auto *F2 = LastDeclMatcher<FunctionDecl>().match(ToTU, functionDecl());
  DeclarationName Name = F0->getDeclName();
  auto Res = LT.lookup(ToTU, Name);
  EXPECT_EQ(Res.size(), 3u);
  EXPECT_EQ(Res.count(F0), 1u);
  EXPECT_EQ(Res.count(F2), 1u);
}

TEST_P(ASTImporterLookupTableTest,
       DifferentOperatorsShouldHaveDifferentResultSet) {
  TranslationUnitDecl *ToTU = getToTuDecl(
      R"(
      struct X{};
      void operator+(X, X);
      void operator-(X, X);
      )",
      Lang_CXX);

  ASTImporterLookupTable LT(*ToTU);
  auto *FPlus = FirstDeclMatcher<FunctionDecl>().match(
      ToTU, functionDecl(hasOverloadedOperatorName("+")));
  auto *FMinus = FirstDeclMatcher<FunctionDecl>().match(
      ToTU, functionDecl(hasOverloadedOperatorName("-")));
  DeclarationName NamePlus = FPlus->getDeclName();
  auto ResPlus = LT.lookup(ToTU, NamePlus);
  EXPECT_EQ(ResPlus.size(), 1u);
  EXPECT_EQ(ResPlus.count(FPlus), 1u);
  EXPECT_EQ(ResPlus.count(FMinus), 0u);
  DeclarationName NameMinus = FMinus->getDeclName();
  auto ResMinus = LT.lookup(ToTU, NameMinus);
  EXPECT_EQ(ResMinus.size(), 1u);
  EXPECT_EQ(ResMinus.count(FMinus), 1u);
  EXPECT_EQ(ResMinus.count(FPlus), 0u);
  EXPECT_NE(*ResMinus.begin(), *ResPlus.begin());
}

TEST_P(ASTImporterLookupTableTest, LookupDeclNamesFromDifferentTUs) {
  TranslationUnitDecl *ToTU = getToTuDecl(
      R"(
      struct X {};
      void operator+(X, X);
      )",
      Lang_CXX);
  auto *ToPlus = FirstDeclMatcher<FunctionDecl>().match(
      ToTU, functionDecl(hasOverloadedOperatorName("+")));

  Decl *FromTU = getTuDecl(
      R"(
      struct X {};
      void operator+(X, X);
      )",
      Lang_CXX);
  auto *FromPlus = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasOverloadedOperatorName("+")));

  // FromPlus have a different TU, thus its DeclarationName is different too.
  ASSERT_NE(ToPlus->getDeclName(), FromPlus->getDeclName());

  ASTImporterLookupTable LT(*ToTU);
  auto Res = LT.lookup(ToTU, ToPlus->getDeclName());
  ASSERT_EQ(Res.size(), 1u);
  EXPECT_EQ(*Res.begin(), ToPlus);

  // FromPlus have a different TU, thus its DeclarationName is different too.
  Res = LT.lookup(ToTU, FromPlus->getDeclName());
  ASSERT_EQ(Res.size(), 0u);
}

static const RecordDecl * getRecordDeclOfFriend(FriendDecl *FD) {
  QualType Ty = FD->getFriendType()->getType();
  QualType NamedTy = cast<ElaboratedType>(Ty)->getNamedType();
  return cast<RecordType>(NamedTy)->getDecl();
}

TEST_P(ASTImporterLookupTableTest, LookupFindsFwdFriendClassDecl) {
  TranslationUnitDecl *ToTU = getToTuDecl(
      R"(
      class Y { friend class F; };
      )",
      Lang_CXX);

  // In this case, the CXXRecordDecl is hidden, the FriendDecl is not a parent.
  // So we must dig up the underlying CXXRecordDecl.
  ASTImporterLookupTable LT(*ToTU);
  auto *FriendD = FirstDeclMatcher<FriendDecl>().match(ToTU, friendDecl());
  const RecordDecl *RD = getRecordDeclOfFriend(FriendD);
  auto *Y = FirstDeclMatcher<CXXRecordDecl>().match(
      ToTU, cxxRecordDecl(hasName("Y")));

  DeclarationName Name = RD->getDeclName();
  auto Res = LT.lookup(ToTU, Name);
  EXPECT_EQ(Res.size(), 1u);
  EXPECT_EQ(*Res.begin(), RD);

  Res = LT.lookup(Y, Name);
  EXPECT_EQ(Res.size(), 0u);
}

TEST_P(ASTImporterLookupTableTest, LookupFindsFwdFriendClassTemplateDecl) {
  TranslationUnitDecl *ToTU = getToTuDecl(
      R"(
      class Y { template <class T> friend class F; };
      )",
      Lang_CXX);

  ASTImporterLookupTable LT(*ToTU);
  auto *F = FirstDeclMatcher<ClassTemplateDecl>().match(
      ToTU, classTemplateDecl(hasName("F")));
  DeclarationName Name = F->getDeclName();
  auto Res = LT.lookup(ToTU, Name);
  EXPECT_EQ(Res.size(), 2u);
  EXPECT_EQ(Res.count(F), 1u);
  EXPECT_EQ(Res.count(F->getTemplatedDecl()), 1u);
}

TEST_P(ASTImporterLookupTableTest, DependentFriendClass) {
  TranslationUnitDecl *ToTU = getToTuDecl(
      R"(
      template <typename T>
      class F;

      template <typename T>
      class Y {
        friend class F<T>;
      };
      )",
      Lang_CXX);

  ASTImporterLookupTable LT(*ToTU);
  auto *F = FirstDeclMatcher<ClassTemplateDecl>().match(
      ToTU, classTemplateDecl(hasName("F")));
  DeclarationName Name = F->getDeclName();
  auto Res = LT.lookup(ToTU, Name);
  EXPECT_EQ(Res.size(), 2u);
  EXPECT_EQ(Res.count(F), 1u);
  EXPECT_EQ(Res.count(F->getTemplatedDecl()), 1u);
}

TEST_P(ASTImporterLookupTableTest, FriendClassTemplateSpecialization) {
  TranslationUnitDecl *ToTU = getToTuDecl(
      R"(
      template <typename T>
      class F;

      class Y {
        friend class F<int>;
      };
      )",
      Lang_CXX);

  ASTImporterLookupTable LT(*ToTU);
  auto *F = FirstDeclMatcher<ClassTemplateDecl>().match(
      ToTU, classTemplateDecl(hasName("F")));
  DeclarationName Name = F->getDeclName();
  auto Res = LT.lookup(ToTU, Name);
  ASSERT_EQ(Res.size(), 3u);
  EXPECT_EQ(Res.count(F), 1u);
  EXPECT_EQ(Res.count(F->getTemplatedDecl()), 1u);
  EXPECT_EQ(Res.count(*F->spec_begin()), 1u);
}

TEST_P(ASTImporterLookupTableTest, LookupFindsFwdFriendFunctionDecl) {
  TranslationUnitDecl *ToTU = getToTuDecl(
      R"(
      class Y { friend void F(); };
      )",
      Lang_CXX);

  ASTImporterLookupTable LT(*ToTU);
  auto *F =
      FirstDeclMatcher<FunctionDecl>().match(ToTU, functionDecl(hasName("F")));
  DeclarationName Name = F->getDeclName();
  auto Res = LT.lookup(ToTU, Name);
  EXPECT_EQ(Res.size(), 1u);
  EXPECT_EQ(*Res.begin(), F);
}

TEST_P(ASTImporterLookupTableTest,
       LookupFindsDeclsInClassTemplateSpecialization) {
  TranslationUnitDecl *ToTU = getToTuDecl(
      R"(
      template <typename T>
      struct X {
        int F;
      };
      void foo() {
        X<char> xc;
      }
      )",
      Lang_CXX);

  ASTImporterLookupTable LT(*ToTU);

  auto *Template = FirstDeclMatcher<ClassTemplateDecl>().match(
      ToTU, classTemplateDecl(hasName("X")));
  auto *FieldInTemplate = FirstDeclMatcher<FieldDecl>().match(
      ToTU,
      fieldDecl(hasParent(cxxRecordDecl(hasParent(classTemplateDecl())))));

  auto *Spec = FirstDeclMatcher<ClassTemplateSpecializationDecl>().match(
      ToTU, classTemplateSpecializationDecl(hasName("X")));
  FieldDecl *FieldInSpec = *Spec->field_begin();
  ASSERT_TRUE(FieldInSpec);

  DeclarationName Name = FieldInSpec->getDeclName();
  auto TemplateDC = cast<DeclContext>(Template->getTemplatedDecl());

  SmallVector<NamedDecl *, 2> FoundDecls;
  TemplateDC->getRedeclContext()->localUncachedLookup(Name, FoundDecls);
  EXPECT_EQ(FoundDecls.size(), 1u);
  EXPECT_EQ(FoundDecls[0], FieldInTemplate);

  auto Res = LT.lookup(TemplateDC, Name);
  ASSERT_EQ(Res.size(), 1u);
  EXPECT_EQ(*Res.begin(), FieldInTemplate);

  cast<DeclContext>(Spec)->getRedeclContext()->localUncachedLookup(Name,
                                                                   FoundDecls);
  EXPECT_EQ(FoundDecls.size(), 1u);
  EXPECT_EQ(FoundDecls[0], FieldInSpec);

  Res = LT.lookup(cast<DeclContext>(Spec), Name);
  ASSERT_EQ(Res.size(), 1u);
  EXPECT_EQ(*Res.begin(), FieldInSpec);
}

TEST_P(ASTImporterLookupTableTest, LookupFindsFwdFriendFunctionTemplateDecl) {
  TranslationUnitDecl *ToTU = getToTuDecl(
      R"(
      class Y { template <class T> friend void F(); };
      )",
      Lang_CXX);

  ASTImporterLookupTable LT(*ToTU);
  auto *F = FirstDeclMatcher<FunctionTemplateDecl>().match(
      ToTU, functionTemplateDecl(hasName("F")));
  DeclarationName Name = F->getDeclName();
  auto Res = LT.lookup(ToTU, Name);
  EXPECT_EQ(Res.size(), 2u);
  EXPECT_EQ(Res.count(F), 1u);
  EXPECT_EQ(Res.count(F->getTemplatedDecl()), 1u);
}

TEST_P(ASTImporterLookupTableTest, MultipleBefriendingClasses) {
  TranslationUnitDecl *ToTU = getToTuDecl(
      R"(
      struct X;
      struct A {
        friend struct X;
      };
      struct B {
        friend struct X;
      };
      )",
      Lang_CXX);

  ASTImporterLookupTable LT(*ToTU);
  auto *X = FirstDeclMatcher<CXXRecordDecl>().match(
      ToTU, cxxRecordDecl(hasName("X")));
  auto *FriendD0 = FirstDeclMatcher<FriendDecl>().match(ToTU, friendDecl());
  auto *FriendD1 = LastDeclMatcher<FriendDecl>().match(ToTU, friendDecl());
  const RecordDecl *RD0 = getRecordDeclOfFriend(FriendD0);
  const RecordDecl *RD1 = getRecordDeclOfFriend(FriendD1);
  ASSERT_EQ(RD0, RD1);
  ASSERT_EQ(RD1, X);

  DeclarationName Name = X->getDeclName();
  auto Res = LT.lookup(ToTU, Name);
  EXPECT_EQ(Res.size(), 1u);
  EXPECT_EQ(*Res.begin(), X);
}

TEST_P(ASTImporterLookupTableTest, EnumConstantDecl) {
  TranslationUnitDecl *ToTU = getToTuDecl(
      R"(
      enum E {
        A,
        B
      };
      )",
      Lang_C);

  ASTImporterLookupTable LT(*ToTU);
  auto *E = FirstDeclMatcher<EnumDecl>().match(ToTU, enumDecl(hasName("E")));
  auto *A = FirstDeclMatcher<EnumConstantDecl>().match(
      ToTU, enumConstantDecl(hasName("A")));

  DeclarationName Name = A->getDeclName();
  // Redecl context is the TU.
  ASSERT_EQ(E->getRedeclContext(), ToTU);

  SmallVector<NamedDecl *, 2> FoundDecls;
  // Normal lookup finds in the DC.
  E->localUncachedLookup(Name, FoundDecls);
  EXPECT_EQ(FoundDecls.size(), 1u);

  // Normal lookup finds in the Redecl context.
  ToTU->localUncachedLookup(Name, FoundDecls);
  EXPECT_EQ(FoundDecls.size(), 1u);

  // Import specific lookup finds in the DC.
  auto Res = LT.lookup(E, Name);
  ASSERT_EQ(Res.size(), 1u);
  EXPECT_EQ(*Res.begin(), A);

  // Import specific lookup finds in the Redecl context.
  Res = LT.lookup(ToTU, Name);
  ASSERT_EQ(Res.size(), 1u);
  EXPECT_EQ(*Res.begin(), A);
}

TEST_P(ASTImporterLookupTableTest, LookupSearchesInTheWholeRedeclChain) {
  TranslationUnitDecl *ToTU = getToTuDecl(
      R"(
      namespace N {
        int A;
      }
      namespace N {
      }
      )",
      Lang_CXX);
  auto *N1 =
      LastDeclMatcher<NamespaceDecl>().match(ToTU, namespaceDecl(hasName("N")));
  auto *A = FirstDeclMatcher<VarDecl>().match(ToTU, varDecl(hasName("A")));
  DeclarationName Name = A->getDeclName();

  ASTImporterLookupTable LT(*ToTU);
  auto Res = LT.lookup(N1, Name);
  ASSERT_EQ(Res.size(), 1u);
  EXPECT_EQ(*Res.begin(), A);
}

INSTANTIATE_TEST_CASE_P(ParameterizedTests, DeclContextTest,
                        ::testing::Values(ArgVector()), );

INSTANTIATE_TEST_CASE_P(
    ParameterizedTests, CanonicalRedeclChain,
    ::testing::Values(ArgVector()),);

// FIXME This test is disabled currently, upcoming patches will make it
// possible to enable.
TEST_P(ASTImporterOptionSpecificTestBase,
       DISABLED_RedeclChainShouldBeCorrectAmongstNamespaces) {
  Decl *FromTU = getTuDecl(
      R"(
      namespace NS {
        struct X;
        struct Y {
          static const int I = 3;
        };
      }
      namespace NS {
        struct X {  // <--- To be imported
          void method(int i = Y::I) {}
          int f;
        };
      }
      )",
      Lang_CXX);
  auto *FromFwd = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("X"), unless(isImplicit())));
  auto *FromDef = LastDeclMatcher<CXXRecordDecl>().match(
      FromTU,
      cxxRecordDecl(hasName("X"), isDefinition(), unless(isImplicit())));
  ASSERT_NE(FromFwd, FromDef);
  ASSERT_FALSE(FromFwd->isThisDeclarationADefinition());
  ASSERT_TRUE(FromDef->isThisDeclarationADefinition());
  ASSERT_EQ(FromFwd->getCanonicalDecl(), FromDef->getCanonicalDecl());

  auto *ToDef = cast_or_null<CXXRecordDecl>(Import(FromDef, Lang_CXX));
  auto *ToFwd = cast_or_null<CXXRecordDecl>(Import(FromFwd, Lang_CXX));
  EXPECT_NE(ToFwd, ToDef);
  EXPECT_FALSE(ToFwd->isThisDeclarationADefinition());
  EXPECT_TRUE(ToDef->isThisDeclarationADefinition());
  EXPECT_EQ(ToFwd->getCanonicalDecl(), ToDef->getCanonicalDecl());
  auto *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  // We expect no (ODR) warning during the import.
  EXPECT_EQ(0u, ToTU->getASTContext().getDiagnostics().getNumWarnings());
}

struct ImportFriendFunctionTemplates : ASTImporterOptionSpecificTestBase {};

TEST_P(ImportFriendFunctionTemplates, LookupShouldFindPreviousFriend) {
  Decl *ToTU = getToTuDecl(
      R"(
      class X {
        template <typename T> friend void foo();
      };
      )",
      Lang_CXX);
  auto *Friend = FirstDeclMatcher<FunctionTemplateDecl>().match(
      ToTU, functionTemplateDecl(hasName("foo")));

  Decl *FromTU = getTuDecl(
      R"(
      template <typename T> void foo();
      )",
      Lang_CXX);
  auto *FromFoo = FirstDeclMatcher<FunctionTemplateDecl>().match(
      FromTU, functionTemplateDecl(hasName("foo")));
  auto *Imported = Import(FromFoo, Lang_CXX);

  EXPECT_EQ(Imported->getPreviousDecl(), Friend);
}

INSTANTIATE_TEST_CASE_P(ParameterizedTests, ASTImporterLookupTableTest,
                        DefaultTestValuesForRunOptions, );

INSTANTIATE_TEST_CASE_P(ParameterizedTests, ImportExpr,
                        DefaultTestValuesForRunOptions, );

INSTANTIATE_TEST_CASE_P(ParameterizedTests, ImportType,
                        DefaultTestValuesForRunOptions, );

INSTANTIATE_TEST_CASE_P(ParameterizedTests, ImportDecl,
                        DefaultTestValuesForRunOptions, );

INSTANTIATE_TEST_CASE_P(ParameterizedTests, ASTImporterOptionSpecificTestBase,
                        DefaultTestValuesForRunOptions, );

INSTANTIATE_TEST_CASE_P(ParameterizedTests, ImportFunctions,
                        DefaultTestValuesForRunOptions, );

INSTANTIATE_TEST_CASE_P(ParameterizedTests, ImportFriendFunctionTemplates,
                        DefaultTestValuesForRunOptions, );

INSTANTIATE_TEST_CASE_P(ParameterizedTests, ImportClasses,
                        DefaultTestValuesForRunOptions, );

INSTANTIATE_TEST_CASE_P(ParameterizedTests, ImportFriendFunctions,
                        DefaultTestValuesForRunOptions, );

INSTANTIATE_TEST_CASE_P(ParameterizedTests, ImportFriendClasses,
                        DefaultTestValuesForRunOptions, );

INSTANTIATE_TEST_CASE_P(ParameterizedTests,
                        ImportFunctionTemplateSpecializations,
                        DefaultTestValuesForRunOptions, );

INSTANTIATE_TEST_CASE_P(ParameterizedTests, ImportImplicitMethods,
                        DefaultTestValuesForRunOptions, );

INSTANTIATE_TEST_CASE_P(ParameterizedTests, ImportVariables,
                        DefaultTestValuesForRunOptions, );

} // end namespace ast_matchers
} // end namespace clang
