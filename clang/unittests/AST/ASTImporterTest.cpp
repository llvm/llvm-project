//===- unittest/AST/ASTImporterTest.cpp - AST node import test ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Tests for the correct import of AST nodes from one AST context to another.
//
//===----------------------------------------------------------------------===//

#include "MatchVerifier.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTImporter.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"

#include "DeclMatcher.h"
#include "Language.h"
#include "gtest/gtest.h"
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
  auto *OFS = static_cast<vfs::OverlayFileSystem *>(
      ToCtx.getSourceManager().getFileManager().getVirtualFileSystem().get());
  auto *MFS =
      static_cast<vfs::InMemoryFileSystem *>(OFS->overlays_begin()->get());
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
struct ParameterizedTestsFixture : ::testing::TestWithParam<ArgVector> {

  // Returns the argument vector used for a specific language option, this set
  // can be tweaked by the test parameters.
  ArgVector getArgVectorForLanguage(Language Lang) const {
    ArgVector Args = getBasicRunOptionsForLanguage(Lang);
    ArgVector ExtraArgs = GetParam();
    for (const auto &Arg : ExtraArgs) {
      Args.push_back(Arg);
    }
    return Args;
  }

};

// Base class for those tests which use the family of `testImport` functions.
class TestImportBase : public ParameterizedTestsFixture {

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

// This class provides generic methods to write tests which can check internal
// attributes of AST nodes like getPreviousDecl(), isVirtual(), etc.  Also,
// this fixture makes it possible to import from several "From" contexts.
class ASTImporterTestBase : public ParameterizedTestsFixture {

  const char *const InputFileName = "input.cc";
  const char *const OutputFileName = "output.cc";

  // Buffer for the To context, must live in the test scope.
  std::string ToCode;

  struct TU {
    // Buffer for the context, must live in the test scope.
    std::string Code;
    std::string FileName;
    std::unique_ptr<ASTUnit> Unit;
    TranslationUnitDecl *TUDecl = nullptr;
    TU(StringRef Code, StringRef FileName, ArgVector Args)
        : Code(Code), FileName(FileName),
          Unit(tooling::buildASTFromCodeWithArgs(this->Code, Args,
                                                 this->FileName)),
          TUDecl(Unit->getASTContext().getTranslationUnitDecl()) {
      Unit->enableSourceFileDiagnostics();
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

    ToCode = ToSrcCode;
    assert(!ToAST);
    ToAST = tooling::buildASTFromCodeWithArgs(ToCode, ToArgs, OutputFileName);
    ToAST->enableSourceFileDiagnostics();

    ASTContext &FromCtx = FromTU.Unit->getASTContext(),
               &ToCtx = ToAST->getASTContext();

    createVirtualFileIfNeeded(ToAST.get(), InputFileName, FromTU.Code);

    ASTImporter Importer(ToCtx, ToAST->getFileManager(), FromCtx,
                         FromTU.Unit->getFileManager(), false);

    IdentifierInfo *ImportedII = &FromCtx.Idents.get(Identifier);
    assert(ImportedII && "Declaration with the given identifier "
                         "should be specified in test!");
    DeclarationName ImportDeclName(ImportedII);
    SmallVector<NamedDecl *, 4> FoundDecls;
    FromCtx.getTranslationUnitDecl()->localUncachedLookup(ImportDeclName,
                                                          FoundDecls);

    assert(FoundDecls.size() == 1);

    Decl *Imported = Importer.Import(FoundDecls.front());
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
    ToCode = ToSrcCode;
    assert(!ToAST);
    ToAST = tooling::buildASTFromCodeWithArgs(ToCode, ToArgs, OutputFileName);
    ToAST->enableSourceFileDiagnostics();

    return ToAST->getASTContext().getTranslationUnitDecl();
  }

  // Import the given Decl into the ToCtx.
  // May be called several times in a given test.
  // The different instances of the param From may have different ASTContext.
  Decl *Import(Decl *From, Language ToLang) {
    if (!ToAST) {
      ArgVector ToArgs = getArgVectorForLanguage(ToLang);
      // Build the AST from an empty file.
      ToAST =
          tooling::buildASTFromCodeWithArgs(/*Code=*/"", ToArgs, "empty.cc");
      ToAST->enableSourceFileDiagnostics();
    }

    // Create a virtual file in the To Ctx which corresponds to the file from
    // which we want to import the `From` Decl. Without this source locations
    // will be invalid in the ToCtx.
    auto It = std::find_if(FromTUs.begin(), FromTUs.end(), [From](const TU &E) {
      return E.TUDecl == From->getTranslationUnitDecl();
    });
    assert(It != FromTUs.end());
    createVirtualFileIfNeeded(ToAST.get(), It->FileName, It->Code);

    ASTContext &FromCtx = From->getASTContext(),
               &ToCtx = ToAST->getASTContext();
    ASTImporter Importer(ToCtx, ToAST->getFileManager(), FromCtx,
                         FromCtx.getSourceManager().getFileManager(), false);
    return Importer.Import(From);
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

struct ImportExpr : TestImportBase {};
struct ImportType : TestImportBase {};
struct ImportDecl : TestImportBase {};

TEST_P(ImportExpr, ImportStringLiteral) {
  MatchVerifier<Decl> Verifier;
  testImport("void declToImport() { \"foo\"; }",
             Lang_CXX, "", Lang_CXX, Verifier,
             functionDecl(
               hasBody(
                 compoundStmt(
                   has(
                     stringLiteral(
                       hasType(
                         asString("const char [4]"))))))));
  testImport("void declToImport() { L\"foo\"; }",
             Lang_CXX, "", Lang_CXX, Verifier,
             functionDecl(
               hasBody(
                 compoundStmt(
                   has(
                     stringLiteral(
                       hasType(
                        asString("const wchar_t [4]"))))))));
  testImport("void declToImport() { \"foo\" \"bar\"; }",
             Lang_CXX, "", Lang_CXX, Verifier,
             functionDecl(
               hasBody(
                 compoundStmt(
                   has(
                     stringLiteral(
                       hasType(
                         asString("const char [7]"))))))));
}

TEST_P(ImportExpr, ImportGNUNullExpr) {
  MatchVerifier<Decl> Verifier;
  testImport("void declToImport() { __null; }",
             Lang_CXX, "", Lang_CXX, Verifier,
             functionDecl(
               hasBody(
                 compoundStmt(
                   has(
                     gnuNullExpr(
                       hasType(isInteger())))))));
}

TEST_P(ImportExpr, ImportCXXNullPtrLiteralExpr) {
  MatchVerifier<Decl> Verifier;
  testImport("void declToImport() { nullptr; }",
             Lang_CXX11, "", Lang_CXX11, Verifier,
             functionDecl(
               hasBody(
                 compoundStmt(
                   has(
                     cxxNullPtrLiteralExpr())))));
}


TEST_P(ImportExpr, ImportFloatinglLiteralExpr) {
  MatchVerifier<Decl> Verifier;
  testImport("void declToImport() { 1.0; }",
             Lang_C, "", Lang_C, Verifier,
             functionDecl(
               hasBody(
                 compoundStmt(
                   has(
                     floatLiteral(
                       equals(1.0),
                       hasType(asString("double"))))))));
  testImport("void declToImport() { 1.0e-5f; }",
              Lang_C, "", Lang_C, Verifier,
              functionDecl(
                hasBody(
                  compoundStmt(
                    has(
                      floatLiteral(
                        equals(1.0e-5f),
                        hasType(asString("float"))))))));
}

TEST_P(ImportExpr, ImportCompoundLiteralExpr) {
  MatchVerifier<Decl> Verifier;
  testImport("void declToImport() {"
             "  struct s { int x; long y; unsigned z; }; "
             "  (struct s){ 42, 0L, 1U }; }",
             Lang_CXX, "", Lang_CXX, Verifier,
             functionDecl(
               hasBody(
                 compoundStmt(
                   has(
                     compoundLiteralExpr(
                       hasType(asString("struct s")),
                       has(initListExpr(
                         hasType(asString("struct s")),
                         has(integerLiteral(
                               equals(42), hasType(asString("int")))),
                         has(integerLiteral(
                               equals(0), hasType(asString("long")))),
                         has(integerLiteral(
                               equals(1),
                               hasType(asString("unsigned int"))))
                         ))))))));
}

TEST_P(ImportExpr, ImportCXXThisExpr) {
  MatchVerifier<Decl> Verifier;
  testImport("class declToImport { void f() { this; } };",
             Lang_CXX, "", Lang_CXX, Verifier,
             cxxRecordDecl(
               hasMethod(
                 hasBody(
                   compoundStmt(
                     has(
                       cxxThisExpr(
                         hasType(
                           asString("class declToImport *")))))))));
}

TEST_P(ImportExpr, ImportAtomicExpr) {
  MatchVerifier<Decl> Verifier;
  testImport("void declToImport() { int *ptr; __atomic_load_n(ptr, 1); }",
             Lang_C, "", Lang_C, Verifier,
             functionDecl(hasBody(compoundStmt(has(atomicExpr(
                 has(ignoringParenImpCasts(
                     declRefExpr(hasDeclaration(varDecl(hasName("ptr"))),
                                 hasType(asString("int *"))))),
                 has(integerLiteral(equals(1), hasType(asString("int"))))))))));
}

TEST_P(ImportExpr, ImportLabelDeclAndAddrLabelExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "void declToImport() { loop: goto loop; &&loop; }", Lang_C, "", Lang_C,
      Verifier,
      functionDecl(hasBody(compoundStmt(
          has(labelStmt(hasDeclaration(labelDecl(hasName("loop"))))),
          has(addrLabelExpr(hasDeclaration(labelDecl(hasName("loop")))))))));
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
  testImport("void declToImport() { int b; switch (b) { case 1: break; } }",
             Lang_C, "", Lang_C, Verifier,
             functionDecl(hasBody(compoundStmt(
                 has(switchStmt(has(compoundStmt(has(caseStmt())))))))));
}

TEST_P(ImportExpr, ImportStmtExpr) {
  MatchVerifier<Decl> Verifier;
  // NOTE: has() ignores implicit casts, using hasDescendant() to match it
  testImport(
    "void declToImport() { int b; int a = b ?: 1; int C = ({int X=4; X;}); }",
    Lang_C, "", Lang_C, Verifier,
    functionDecl(
      hasBody(
        compoundStmt(
          has(
            declStmt(
              hasSingleDecl(
                varDecl(
                  hasName("C"),
                  hasType(asString("int")),
                  hasInitializer(
                    stmtExpr(
                      hasAnySubstatement(
                        declStmt(
                          hasSingleDecl(
                            varDecl(
                              hasName("X"),
                              hasType(asString("int")),
                              hasInitializer(
                                integerLiteral(equals(4))))))),
                      hasDescendant(
                        implicitCastExpr()
                        )))))))))));
}

TEST_P(ImportExpr, ImportConditionalOperator) {
  MatchVerifier<Decl> Verifier;
  testImport(
    "void declToImport() { true ? 1 : -5; }",
    Lang_CXX, "", Lang_CXX, Verifier,
    functionDecl(
      hasBody(
        compoundStmt(
          has(
            conditionalOperator(
              hasCondition(cxxBoolLiteral(equals(true))),
              hasTrueExpression(integerLiteral(equals(1))),
              hasFalseExpression(
                unaryOperator(hasUnaryOperand(integerLiteral(equals(5))))
                )))))));
}

TEST_P(ImportExpr, ImportBinaryConditionalOperator) {
  MatchVerifier<Decl> Verifier;
  testImport(
    "void declToImport() { 1 ?: -5; }", Lang_CXX, "", Lang_CXX, Verifier,
    functionDecl(
      hasBody(
        compoundStmt(
          has(
            binaryConditionalOperator(
              hasCondition(
                implicitCastExpr(
                  hasSourceExpression(
                    opaqueValueExpr(
                      hasSourceExpression(integerLiteral(equals(1))))),
                  hasType(booleanType()))),
              hasTrueExpression(
                opaqueValueExpr(hasSourceExpression(
                                  integerLiteral(equals(1))))),
              hasFalseExpression(
                unaryOperator(hasOperatorName("-"),
                              hasUnaryOperand(integerLiteral(equals(5)))))
                ))))));
}

TEST_P(ImportExpr, ImportDesignatedInitExpr) {
  MatchVerifier<Decl> Verifier;
  testImport("void declToImport() {"
             "  struct point { double x; double y; };"
             "  struct point ptarray[10] = "
                    "{ [2].y = 1.0, [2].x = 2.0, [0].x = 1.0 }; }",
             Lang_C, "", Lang_C, Verifier,
             functionDecl(
               hasBody(
                 compoundStmt(
                   has(
                     declStmt(
                       hasSingleDecl(
                         varDecl(
                           hasInitializer(
                             initListExpr(
                               hasSyntacticForm(
                                 initListExpr(
                                   has(
                                     designatedInitExpr(
                                       designatorCountIs(2),
                                       has(floatLiteral(
                                             equals(1.0))),
                                       has(integerLiteral(
                                             equals(2))))),
                                   has(
                                     designatedInitExpr(
                                       designatorCountIs(2),
                                       has(floatLiteral(
                                             equals(2.0))),
                                       has(integerLiteral(
                                             equals(2))))),
                                   has(
                                     designatedInitExpr(
                                       designatorCountIs(2),
                                       has(floatLiteral(
                                             equals(1.0))),
                                       has(integerLiteral(
                                             equals(0)))))
                                   ))))))))))));
}


TEST_P(ImportExpr, ImportPredefinedExpr) {
  MatchVerifier<Decl> Verifier;
  // __func__ expands as StringLiteral("declToImport")
  testImport("void declToImport() { __func__; }",
             Lang_CXX, "", Lang_CXX, Verifier,
             functionDecl(
               hasBody(
                 compoundStmt(
                   has(
                     predefinedExpr(
                       hasType(
                         asString("const char [13]")),
                       has(
                         stringLiteral(
                           hasType(
                             asString("const char [13]"))))))))));
}

TEST_P(ImportExpr, ImportInitListExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
    "void declToImport() {"
    "  struct point { double x; double y; };"
    "  point ptarray[10] = { [2].y = 1.0, [2].x = 2.0,"
    "                        [0].x = 1.0 }; }",
    Lang_CXX, "", Lang_CXX, Verifier,
    functionDecl(
      hasBody(
        compoundStmt(
          has(
            declStmt(
              hasSingleDecl(
                varDecl(
                  hasInitializer(
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
                          has(floatLiteral(equals(1.0)))))
                        ))))))))));
}


const internal::VariadicDynCastAllOfMatcher<Expr, VAArgExpr> vaArgExpr;

TEST_P(ImportExpr, ImportVAArgExpr) {
  MatchVerifier<Decl> Verifier;
  testImport("void declToImport(__builtin_va_list list, ...) {"
             "  (void)__builtin_va_arg(list, int); }",
             Lang_CXX, "", Lang_CXX, Verifier,
             functionDecl(
               hasBody(
                 compoundStmt(
                   has(
                     cStyleCastExpr(
                       hasSourceExpression(
                         vaArgExpr())))))));
}

TEST_P(ImportExpr, CXXTemporaryObjectExpr) {
  MatchVerifier<Decl> Verifier;
  testImport("struct C {};"
             "void declToImport() { C c = C(); }",
             Lang_CXX, "", Lang_CXX, Verifier,
             functionDecl(hasBody(compoundStmt(has(
                 declStmt(has(varDecl(has(exprWithCleanups(has(cxxConstructExpr(
                     has(materializeTemporaryExpr(has(implicitCastExpr(
                         has(cxxTemporaryObjectExpr())))))))))))))))));
}

TEST_P(ImportType, ImportAtomicType) {
  MatchVerifier<Decl> Verifier;
  testImport("void declToImport() { typedef _Atomic(int) a_int; }",
             Lang_CXX11, "", Lang_CXX11, Verifier,
             functionDecl(
               hasBody(
                 compoundStmt(
                   has(
                     declStmt(
                       has(
                         typedefDecl(
                           has(atomicType())))))))));
}

TEST_P(ImportDecl, ImportFunctionTemplateDecl) {
  MatchVerifier<Decl> Verifier;
  testImport("template <typename T> void declToImport() { };", Lang_CXX, "",
             Lang_CXX, Verifier, functionTemplateDecl());
}

const internal::VariadicDynCastAllOfMatcher<Expr, CXXDependentScopeMemberExpr>
    cxxDependentScopeMemberExpr;

TEST_P(ImportExpr, ImportCXXDependentScopeMemberExpr) {
  MatchVerifier<Decl> Verifier;
  testImport("template <typename T> struct C { T t; };"
             "template <typename T> void declToImport() {"
             "  C<T> d;"
             "  d.t;"
             "}"
             "void instantiate() { declToImport<int>(); }",
             Lang_CXX, "", Lang_CXX, Verifier,
             functionTemplateDecl(has(functionDecl(
                 has(compoundStmt(has(cxxDependentScopeMemberExpr())))))));
  testImport("template <typename T> struct C { T t; };"
             "template <typename T> void declToImport() {"
             "  C<T> d;"
             "  (&d)->t;"
             "}"
             "void instantiate() { declToImport<int>(); }",
             Lang_CXX, "", Lang_CXX, Verifier,
             functionTemplateDecl(has(functionDecl(
                 has(compoundStmt(has(cxxDependentScopeMemberExpr())))))));
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
          hasBody(compoundStmt(
              has(returnStmt(has(implicitCastExpr(has(declRefExpr()))))))),
          unless(hasAncestor(translationUnitDecl(has(typeAliasDecl()))))));
}

const internal::VariadicDynCastAllOfMatcher<Decl, VarTemplateSpecializationDecl>
    varTemplateSpecializationDecl;

TEST_P(ImportDecl, ImportVarTemplate) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "template <typename T>"
      "T pi = T(3.1415926535897932385L);"
      "void declToImport() { pi<int>; }",
      Lang_CXX11, "", Lang_CXX11, Verifier,
      functionDecl(
          hasBody(has(declRefExpr(to(varTemplateSpecializationDecl())))),
          unless(hasAncestor(translationUnitDecl(has(varDecl(
              hasName("pi"), unless(varTemplateSpecializationDecl()))))))));
}

TEST_P(ImportType, ImportPackExpansion) {
  MatchVerifier<Decl> Verifier;
  testImport("template <typename... Args>"
             "struct dummy {"
             "  dummy(Args... args) {}"
             "  static const int i = 4;"
             "};"
             "int declToImport() { return dummy<int>::i; }",
             Lang_CXX11, "", Lang_CXX11, Verifier,
             functionDecl(
               hasBody(
                 compoundStmt(
                   has(
                     returnStmt(
                       has(
                         implicitCastExpr(
                           has(
                             declRefExpr())))))))));
}

const internal::VariadicDynCastAllOfMatcher<Type,
                                            DependentTemplateSpecializationType>
    dependentTemplateSpecializationType;

TEST_P(ImportType, ImportDependentTemplateSpecialization) {
  MatchVerifier<Decl> Verifier;
  testImport("template<typename T>"
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
  testImport("template <typename... Ts>"
             "void declToImport() {"
             "  const int i = sizeof...(Ts);"
             "};"
             "void g() { declToImport<int>(); }",
             Lang_CXX11, "", Lang_CXX11, Verifier,
             functionTemplateDecl(has(functionDecl(
                 hasBody(compoundStmt(has(declStmt(has(varDecl(hasInitializer(
                     implicitCastExpr(has(sizeOfPackExpr())))))))))))));
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
  testImport("void declToImport() { "
             "  __builtin_types_compatible_p(int, int);"
             "}",
             Lang_C, "", Lang_C, Verifier,
             functionDecl(
               hasBody(
                 compoundStmt(
                   has(
                     typeTraitExpr(hasType(asString("int"))))))));
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
  testImport("template<typename T> struct declToImport {"
             "  void m() { __is_pod(T); }"
             "};"
             "void f() { declToImport<int>().m(); }",
             Lang_CXX11, "", Lang_CXX11, Verifier,
             classTemplateDecl(
               has(
                 cxxRecordDecl(
                   has(
                     functionDecl(
                       hasBody(
                         compoundStmt(
                           has(
                             typeTraitExpr(
                               hasType(booleanType())
                               ))))))))));
}

const internal::VariadicDynCastAllOfMatcher<Expr, CXXPseudoDestructorExpr>
    cxxPseudoDestructorExpr;

TEST_P(ImportExpr, ImportCXXPseudoDestructorExpr) {
  MatchVerifier<Decl> Verifier;
  testImport("typedef int T;"
             "void declToImport(int *p) {"
             "  T t;"
             "  p->T::~T();"
             "}",
             Lang_CXX, "", Lang_CXX, Verifier,
             functionDecl(has(compoundStmt(has(
                 callExpr(has(cxxPseudoDestructorExpr())))))));
}

TEST_P(ImportDecl, ImportUsingDecl) {
  MatchVerifier<Decl> Verifier;
  testImport("namespace foo { int bar; }"
             "void declToImport() { using foo::bar; }",
             Lang_CXX, "", Lang_CXX, Verifier,
             functionDecl(
               has(
                 compoundStmt(
                   has(
                     declStmt(
                       has(
                         usingDecl())))))));
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
  testImport("namespace foo { int bar; }"
             "namespace declToImport { using foo::bar; }",
             Lang_CXX, "", Lang_CXX, Verifier,
             namespaceDecl(has(usingShadowDecl())));
}

TEST_P(ImportExpr, ImportUnresolvedLookupExpr) {
  MatchVerifier<Decl> Verifier;
  testImport("template<typename T> int foo();"
             "template <typename T> void declToImport() {"
             "  ::foo<T>;"
             "  ::template foo<T>;"
             "}"
             "void instantiate() { declToImport<int>(); }",
             Lang_CXX, "", Lang_CXX, Verifier,
             functionTemplateDecl(has(functionDecl(
                 has(compoundStmt(has(unresolvedLookupExpr())))))));
}

TEST_P(ImportExpr, ImportCXXUnresolvedConstructExpr) {
  MatchVerifier<Decl> Verifier;
  testImport("template <typename T> struct C { T t; };"
             "template <typename T> void declToImport() {"
             "  C<T> d;"
             "  d.t = T();"
             "}"
             "void instantiate() { declToImport<int>(); }",
             Lang_CXX, "", Lang_CXX, Verifier,
             functionTemplateDecl(has(functionDecl(has(compoundStmt(has(
                 binaryOperator(has(cxxUnresolvedConstructExpr())))))))));
  testImport("template <typename T> struct C { T t; };"
             "template <typename T> void declToImport() {"
             "  C<T> d;"
             "  (&d)->t = T();"
             "}"
             "void instantiate() { declToImport<int>(); }",
             Lang_CXX, "", Lang_CXX, Verifier,
             functionTemplateDecl(has(functionDecl(has(compoundStmt(has(
                 binaryOperator(has(cxxUnresolvedConstructExpr())))))))));
}

/// Check that function "declToImport()" (which is the templated function
/// for corresponding FunctionTemplateDecl) is not added into DeclContext.
/// Same for class template declarations.
TEST_P(ImportDecl, ImportTemplatedDeclForTemplate) {
  MatchVerifier<Decl> Verifier;
  testImport("template <typename T> void declToImport() { T a = 1; }"
             "void instantiate() { declToImport<int>(); }",
             Lang_CXX, "", Lang_CXX, Verifier,
             functionTemplateDecl(hasAncestor(translationUnitDecl(
                 unless(has(functionDecl(hasName("declToImport"))))))));
  testImport("template <typename T> struct declToImport { T t; };"
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
  testImport("class declToImport {"
             "  void f() { *this = declToImport(); }"
             "};",
             Lang_CXX, "", Lang_CXX, Verifier,
             cxxRecordDecl(has(cxxMethodDecl(hasBody(compoundStmt(
                 has(exprWithCleanups(has(cxxOperatorCallExpr())))))))));
}

TEST_P(ImportExpr, DependentSizedArrayType) {
  MatchVerifier<Decl> Verifier;
  testImport("template<typename T, int Size> class declToImport {"
             "  T data[Size];"
             "};",
             Lang_CXX, "", Lang_CXX, Verifier,
             classTemplateDecl(has(cxxRecordDecl(
                 has(fieldDecl(hasType(dependentSizedArrayType())))))));
}

TEST_P(ASTImporterTestBase, ImportOfTemplatedDeclOfClassTemplateDecl) {
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

TEST_P(ASTImporterTestBase,
       DISABLED_ImportOfTemplatedDeclOfFunctionTemplateDecl) {
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

TEST_P(ASTImporterTestBase,
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

TEST_P(ASTImporterTestBase,
       DISABLED_ImportOfTemplatedDeclShouldImportTheFunctionTemplateDecl) {
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

TEST_P(ASTImporterTestBase, ImportCorrectTemplatedDecl) {
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

TEST_P(ASTImporterTestBase, DISABLED_ImportFunctionWithBackReferringParameter) {
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

TEST_P(ASTImporterTestBase,
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

TEST_P(ASTImporterTestBase, TUshouldNotContainTemplatedDeclOfClassTemplates) {
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

TEST_P(ASTImporterTestBase, TUshouldNotContainTemplatedDeclOfTypeAlias) {
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

TEST_P(
    ASTImporterTestBase,
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

TEST_P(ASTImporterTestBase,
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

TEST_P(ASTImporterTestBase, CXXRecordDeclFieldsShouldBeInCorrectOrder) {
  Decl *From, *To;
  std::tie(From, To) =
      getImportedDecl(
          "struct declToImport { int a; int b; };",
                      Lang_CXX11, "", Lang_CXX11);

  MatchVerifier<Decl> Verifier;
  ASSERT_TRUE(Verifier.match(From, cxxRecordDecl(hasFieldOrder({"a", "b"}))));
  EXPECT_TRUE(Verifier.match(To, cxxRecordDecl(hasFieldOrder({"a", "b"}))));
}

TEST_P(ASTImporterTestBase,
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

TEST_P(ASTImporterTestBase, ShouldImportImplicitCXXRecordDecl) {
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

TEST_P(ASTImporterTestBase, ShouldImportImplicitCXXRecordDeclOfClassTemplate) {
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

TEST_P(
    ASTImporterTestBase,
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

TEST_P(ASTImporterTestBase, IDNSOrdinary) {
  Decl *From, *To;
  std::tie(From, To) =
      getImportedDecl("void declToImport() {}", Lang_CXX, "", Lang_CXX);

  MatchVerifier<Decl> Verifier;
  auto Matcher = functionDecl();
  ASSERT_TRUE(Verifier.match(From, Matcher));
  EXPECT_TRUE(Verifier.match(To, Matcher));
  EXPECT_EQ(From->getIdentifierNamespace(), To->getIdentifierNamespace());
}

TEST_P(ASTImporterTestBase, IDNSOfNonmemberOperator) {
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

TEST_P(ASTImporterTestBase,
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

TEST_P(ASTImporterTestBase, ImportDefinitionOfClassTemplateAfterFwdDecl) {
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

TEST_P(ASTImporterTestBase,
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

TEST_P(ASTImporterTestBase,
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

TEST_P(
    ASTImporterTestBase,
    ImportDefinitionOfClassTemplateSpecIfThereIsAnExistingFwdDeclAndDefinition)
{
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

struct ImportFunctions : ASTImporterTestBase {};

TEST_P(ImportFunctions,
       PrototypeShouldBeImportedAsAPrototypeWhenThereIsNoDefinition) {
  Decl *FromTU = getTuDecl("void f();", Lang_CXX);
  auto Pattern = functionDecl(hasName("f"));
  FunctionDecl *FromD =
      FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);

  Decl *ImportedD = Import(FromD, Lang_CXX);
  Decl *ToTU = ImportedD->getTranslationUnitDecl();

  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, Pattern), 1u);
  EXPECT_TRUE(!cast<FunctionDecl>(ImportedD)->doesThisDeclarationHaveABody());
}

TEST_P(ImportFunctions,
       PrototypeShouldBeImportedAsDefintionWhenThereIsADefinition) {
  Decl *FromTU = getTuDecl("void f(); void f() {}", Lang_CXX);
  auto Pattern = functionDecl(hasName("f"));
  FunctionDecl *FromD = // Prototype
      FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);

  Decl *ImportedD = Import(FromD, Lang_CXX);
  Decl *ToTU = ImportedD->getTranslationUnitDecl();

  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, Pattern), 1u);
  EXPECT_TRUE(cast<FunctionDecl>(ImportedD)->doesThisDeclarationHaveABody());
}

TEST_P(ImportFunctions,
       DefinitionShouldBeImportedAsDefintionWhenThereIsAPrototype) {
  Decl *FromTU = getTuDecl("void f(); void f() {}", Lang_CXX);
  auto Pattern = functionDecl(hasName("f"));
  FunctionDecl *FromD = // Definition
      LastDeclMatcher<FunctionDecl>().match(FromTU, Pattern);

  Decl *ImportedD = Import(FromD, Lang_CXX);
  Decl *ToTU = ImportedD->getTranslationUnitDecl();

  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, Pattern), 1u);
  EXPECT_TRUE(cast<FunctionDecl>(ImportedD)->doesThisDeclarationHaveABody());
}

TEST_P(ImportFunctions, DefinitionShouldBeImportedAsADefinition) {
  Decl *FromTU = getTuDecl("void f() {}", Lang_CXX);
  auto Pattern = functionDecl(hasName("f"));
  FunctionDecl *FromD =
      FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);

  Decl *ImportedD = Import(FromD, Lang_CXX);
  Decl *ToTU = ImportedD->getTranslationUnitDecl();

  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, Pattern), 1u);
  EXPECT_TRUE(cast<FunctionDecl>(ImportedD)->doesThisDeclarationHaveABody());
}

TEST_P(ImportFunctions, DISABLED_ImportPrototypeOfRecursiveFunction) {
  Decl *FromTU = getTuDecl("void f(); void f() { f(); }", Lang_CXX);
  auto Pattern = functionDecl(hasName("f"));
  FunctionDecl *PrototypeFD =
      FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);

  Decl *ImportedD = Import(PrototypeFD, Lang_CXX);
  Decl *ToTU = ImportedD->getTranslationUnitDecl();

  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, Pattern), 1u);
  EXPECT_TRUE(cast<FunctionDecl>(ImportedD)->doesThisDeclarationHaveABody());
}

TEST_P(ImportFunctions, ImportDefinitionOfRecursiveFunction) {
  Decl *FromTU = getTuDecl("void f(); void f() { f(); }", Lang_CXX);
  auto Pattern = functionDecl(hasName("f"));
  FunctionDecl *DefinitionFD =
      LastDeclMatcher<FunctionDecl>().match(FromTU, Pattern);

  Decl *ImportedD = Import(DefinitionFD, Lang_CXX);
  Decl *ToTU = ImportedD->getTranslationUnitDecl();

  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, Pattern), 1u);
  EXPECT_TRUE(cast<FunctionDecl>(ImportedD)->doesThisDeclarationHaveABody());
}

TEST_P(ImportFunctions, ImportPrototypes) {
  auto Pattern = functionDecl(hasName("f"));

  Decl *ImportedD;
  {
    Decl *FromTU = getTuDecl("void f();", Lang_CXX, "input0.cc");
    FunctionDecl *FromD =
        FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);

    ImportedD = Import(FromD, Lang_CXX);
  }
  Decl *ImportedD1;
  {
    Decl *FromTU = getTuDecl("void f();", Lang_CXX, "input1.cc");
    FunctionDecl *FromD =
        FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);
    ImportedD1 = Import(FromD, Lang_CXX);
  }

  Decl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, Pattern), 1u);
  EXPECT_EQ(ImportedD, ImportedD1);
  EXPECT_TRUE(!cast<FunctionDecl>(ImportedD)->doesThisDeclarationHaveABody());
}

TEST_P(ImportFunctions, ImportDefinitionThenPrototype) {
  auto Pattern = functionDecl(hasName("f"));

  Decl *ImportedD;
  {
    Decl *FromTU = getTuDecl("void f(){}", Lang_CXX, "input0.cc");
    FunctionDecl *FromD =
        FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);

    ImportedD = Import(FromD, Lang_CXX);
  }
  Decl *ImportedD1;
  {
    Decl *FromTU = getTuDecl("void f();", Lang_CXX, "input1.cc");
    FunctionDecl *FromD =
        FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);
    ImportedD1 = Import(FromD, Lang_CXX);
  }

  Decl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, Pattern), 1u);
  EXPECT_EQ(ImportedD, ImportedD1);
  EXPECT_TRUE(cast<FunctionDecl>(ImportedD)->doesThisDeclarationHaveABody());
}

TEST_P(ImportFunctions, ImportPrototypeThenDefinition) {
  auto Pattern = functionDecl(hasName("f"));

  {
    Decl *FromTU = getTuDecl("void f();", Lang_CXX, "input0.cc");
    FunctionDecl *FromD =
        FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);

    Import(FromD, Lang_CXX);
  }
  {
    Decl *FromTU = getTuDecl("void f(){}", Lang_CXX, "input1.cc");
    FunctionDecl *FromD =
        FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);
    Import(FromD, Lang_CXX);
  }

  Decl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  ASSERT_EQ(DeclCounter<FunctionDecl>().match(ToTU, Pattern), 2u);
  FunctionDecl *ProtoD = FirstDeclMatcher<FunctionDecl>().match(ToTU, Pattern);
  EXPECT_TRUE(!ProtoD->doesThisDeclarationHaveABody());
  FunctionDecl *DefinitionD =
      LastDeclMatcher<FunctionDecl>().match(ToTU, Pattern);
  EXPECT_TRUE(DefinitionD->doesThisDeclarationHaveABody());
  EXPECT_EQ(DefinitionD->getPreviousDecl(), ProtoD);
}

TEST_P(ImportFunctions, DISABLED_ImportPrototypeThenProtoAndDefinition) {
  auto Pattern = functionDecl(hasName("f"));

  {
    Decl *FromTU = getTuDecl("void f();", Lang_CXX, "input0.cc");
    FunctionDecl *FromD =
        FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);

    Import(FromD, Lang_CXX);
  }
  {
    Decl *FromTU = getTuDecl("void f(); void f(){}", Lang_CXX, "input1.cc");
    FunctionDecl *FromD =
        FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);
    Import(FromD, Lang_CXX);
  }

  Decl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  ASSERT_EQ(DeclCounter<FunctionDecl>().match(ToTU, Pattern), 2u);
  FunctionDecl *ProtoD = FirstDeclMatcher<FunctionDecl>().match(ToTU, Pattern);
  EXPECT_TRUE(!ProtoD->doesThisDeclarationHaveABody());
  FunctionDecl *DefinitionD =
      LastDeclMatcher<FunctionDecl>().match(ToTU, Pattern);
  EXPECT_TRUE(DefinitionD->doesThisDeclarationHaveABody());
  EXPECT_EQ(DefinitionD->getPreviousDecl(), ProtoD);
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
             "void instantiate() { declToImport<int>(); }",
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

const internal::VariadicDynCastAllOfMatcher<Expr, UnresolvedMemberExpr>
    unresolvedMemberExpr;

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

struct DeclContextTest : ASTImporterTestBase {};

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

INSTANTIATE_TEST_CASE_P(ParameterizedTests, DeclContextTest,
                        ::testing::Values(ArgVector()), );

auto DefaultTestValuesForRunOptions = ::testing::Values(
    ArgVector(),
    ArgVector{"-fdelayed-template-parsing"},
    ArgVector{"-fms-compatibility"},
    ArgVector{"-fdelayed-template-parsing", "-fms-compatibility"});

INSTANTIATE_TEST_CASE_P(ParameterizedTests, ImportExpr,
                        DefaultTestValuesForRunOptions, );

INSTANTIATE_TEST_CASE_P(ParameterizedTests, ImportType,
                        DefaultTestValuesForRunOptions, );

INSTANTIATE_TEST_CASE_P(ParameterizedTests, ImportDecl,
                        DefaultTestValuesForRunOptions, );

INSTANTIATE_TEST_CASE_P(ParameterizedTests, ASTImporterTestBase,
                        DefaultTestValuesForRunOptions, );

INSTANTIATE_TEST_CASE_P(ParameterizedTests, ImportFunctions,
                        DefaultTestValuesForRunOptions, );

} // end namespace ast_matchers
} // end namespace clang
