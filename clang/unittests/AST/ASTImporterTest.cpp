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

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTImporter.h"
#include "MatchVerifier.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

namespace clang {
namespace ast_matchers {

typedef std::vector<std::string> StringVector;

void getLangArgs(Language Lang, StringVector &Args) {
  switch (Lang) {
  case Lang_C:
    Args.insert(Args.end(), { "-x", "c", "-std=c99" });
    break;
  case Lang_C89:
    Args.insert(Args.end(), { "-x", "c", "-std=c89" });
    break;
  case Lang_CXX:
    Args.push_back("-std=c++98");
    break;
  case Lang_CXX11:
    Args.push_back("-std=c++11");
    break;
  case Lang_OpenCL:
  case Lang_OBJCXX:
    break;
  }
}

template<typename NodeType, typename MatcherType>
testing::AssertionResult
testImport(const std::string &FromCode, Language FromLang,
           const std::string &ToCode, Language ToLang,
           MatchVerifier<NodeType> &Verifier,
           const MatcherType &AMatcher) {
  StringVector FromArgs, ToArgs;
  getLangArgs(FromLang, FromArgs);
  getLangArgs(ToLang, ToArgs);

  const char *const InputFileName = "input.cc";
  const char *const OutputFileName = "output.cc";

  std::unique_ptr<ASTUnit>
      FromAST = tooling::buildASTFromCodeWithArgs(
        FromCode, FromArgs, InputFileName),
      ToAST = tooling::buildASTFromCodeWithArgs(ToCode, ToArgs, OutputFileName);

  ASTContext &FromCtx = FromAST->getASTContext(),
      &ToCtx = ToAST->getASTContext();

  // Add input.cc to virtual file system so importer can 'find' it
  // while importing SourceLocations.
  vfs::OverlayFileSystem *OFS = static_cast<vfs::OverlayFileSystem *>(
        ToCtx.getSourceManager().getFileManager().getVirtualFileSystem().get());
  vfs::InMemoryFileSystem *MFS = static_cast<vfs::InMemoryFileSystem *>(
        OFS->overlays_begin()->get());
  MFS->addFile(InputFileName, 0, llvm::MemoryBuffer::getMemBuffer(FromCode));

  ASTImporter Importer(ToCtx, ToAST->getFileManager(),
                       FromCtx, FromAST->getFileManager(), false);

  IdentifierInfo *ImportedII = &FromCtx.Idents.get("declToImport");
  assert(ImportedII && "Declaration with 'declToImport' name"
                       "should be specified in test!");
  DeclarationName ImportDeclName(ImportedII);
  SmallVector<NamedDecl *, 4> FoundDecls;
  FromCtx.getTranslationUnitDecl()->localUncachedLookup(
        ImportDeclName, FoundDecls);

  if (FoundDecls.size() != 1)
    return testing::AssertionFailure() << "Multiple declarations were found!";

  auto Imported = Importer.Import(*FoundDecls.begin());
  if (!Imported)
    return testing::AssertionFailure() << "Import failed, nullptr returned!";

  // This should dump source locations and assert if some source locations
  // were not imported
  SmallString<1024> ImportChecker;
  llvm::raw_svector_ostream ToNothing(ImportChecker);
  ToCtx.getTranslationUnitDecl()->print(ToNothing);

  return Verifier.match(Imported, AMatcher);
}

TEST(ImportExpr, ImportStringLiteral) {
  MatchVerifier<Decl> Verifier;
  EXPECT_TRUE(testImport("void declToImport() { \"foo\"; }",
                         Lang_CXX, "", Lang_CXX, Verifier,
                         functionDecl(
                           hasBody(
                             compoundStmt(
                               has(
                                 stringLiteral(
                                   hasType(
                                     asString("const char [4]")))))))));
  EXPECT_TRUE(testImport("void declToImport() { L\"foo\"; }",
                         Lang_CXX, "", Lang_CXX, Verifier,
                         functionDecl(
                           hasBody(
                             compoundStmt(
                               has(
                                 stringLiteral(
                                   hasType(
                                     asString("const wchar_t [4]")))))))));
  EXPECT_TRUE(testImport("void declToImport() { \"foo\" \"bar\"; }",
                         Lang_CXX, "", Lang_CXX, Verifier,
                         functionDecl(
                           hasBody(
                             compoundStmt(
                               has(
                                 stringLiteral(
                                   hasType(
                                     asString("const char [7]")))))))));
}

TEST(ImportExpr, ImportGNUNullExpr) {
  MatchVerifier<Decl> Verifier;
  EXPECT_TRUE(testImport("void declToImport() { __null; }",
                         Lang_CXX, "", Lang_CXX, Verifier,
                         functionDecl(
                           hasBody(
                             compoundStmt(
                               has(
                                 gnuNullExpr(
                                   hasType(isInteger()))))))));
}

TEST(ImportExpr, ImportCXXNullPtrLiteralExpr) {
  MatchVerifier<Decl> Verifier;
  EXPECT_TRUE(testImport("void declToImport() { nullptr; }",
                         Lang_CXX11, "", Lang_CXX11, Verifier,
                         functionDecl(
                           hasBody(
                             compoundStmt(
                               has(
                                 cxxNullPtrLiteralExpr()))))));
}


TEST(ImportExpr, ImportFloatinglLiteralExpr) {
  MatchVerifier<Decl> Verifier;
  EXPECT_TRUE(testImport("void declToImport() { 1.0; }",
                         Lang_CXX, "", Lang_CXX, Verifier,
                         functionDecl(
                           hasBody(
                             compoundStmt(
                               has(
                                 floatLiteral(
                                   equals(1.0),
                                   hasType(asString("double")))))))));
  EXPECT_TRUE(testImport("void declToImport() { 1.0e-5f; }",
                         Lang_CXX, "", Lang_CXX, Verifier,
                         functionDecl(
                           hasBody(
                             compoundStmt(
                               has(
                                 floatLiteral(
                                   equals(1.0e-5f),
                                   hasType(asString("float")))))))));
}

TEST(ImportExpr, ImportCompoundLiteralExpr) {
  MatchVerifier<Decl> Verifier;
  EXPECT_TRUE(
        testImport(
          "void declToImport() {"
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
                      )))))))));
}

TEST(ImportExpr, ImportCXXThisExpr) {
  MatchVerifier<Decl> Verifier;
  EXPECT_TRUE(
        testImport("class declToImport { void f() { this; } };",
                   Lang_CXX, "", Lang_CXX, Verifier,
                   cxxRecordDecl(
                     hasMethod(
                       hasBody(
                         compoundStmt(
                           has(
                             cxxThisExpr(
                               hasType(
                                 asString("class declToImport *"))))))))));
}

TEST(ImportExpr, ImportAtomicExpr) {
  MatchVerifier<Decl> Verifier;
  EXPECT_TRUE(testImport(
      "void declToImport() { int *ptr; __atomic_load_n(ptr, 1); }", Lang_CXX,
      "", Lang_CXX, Verifier,
      functionDecl(hasBody(compoundStmt(has(atomicExpr(
          has(ignoringParenImpCasts(
              declRefExpr(hasDeclaration(varDecl(hasName("ptr"))),
                          hasType(asString("int *"))))),
          has(integerLiteral(equals(1), hasType(asString("int")))))))))));
}

TEST(ImportExpr, ImportLabelDeclAndAddrLabelExpr) {
  MatchVerifier<Decl> Verifier;
  EXPECT_TRUE(
        testImport(
          "void declToImport() { loop: goto loop; &&loop; }",
          Lang_CXX, "", Lang_CXX, Verifier,
          functionDecl(
            hasBody(
              compoundStmt(
                has(labelStmt(hasDeclaration(labelDecl(hasName("loop"))))),
                has(addrLabelExpr(hasDeclaration(labelDecl(hasName("loop")))))
                )))));
}

AST_MATCHER_P(TemplateDecl, hasTemplateDecl,
              internal::Matcher<NamedDecl>, InnerMatcher) {
  const NamedDecl *Template = Node.getTemplatedDecl();
  return Template && InnerMatcher.matches(*Template, Finder, Builder);
}

TEST(ImportExpr, ImportParenListExpr) {
  MatchVerifier<Decl> Verifier;
  EXPECT_TRUE(
        testImport(
          "template<typename T> class dummy { void f() { dummy X(*this); } };"
          "typedef dummy<int> declToImport;"
          "template class dummy<int>;",
          Lang_CXX, "", Lang_CXX, Verifier,
          typedefDecl(
            hasType(
              templateSpecializationType(
                hasDeclaration(
                  classTemplateDecl(
                    hasTemplateDecl(
                      cxxRecordDecl(
                        hasMethod(
                        allOf(
                          hasName("f"),
                          hasBody(
                            compoundStmt(
                              has(
                                declStmt(
                                  hasSingleDecl(
                                    varDecl(
                                      hasInitializer(
                                        parenListExpr(
                                          has(
                                            unaryOperator(
                                              hasOperatorName("*"),
                                              hasUnaryOperand(cxxThisExpr())
                                              )))))))))))))))))))));
}

TEST(ImportExpr, ImportStmtExpr) {
  MatchVerifier<Decl> Verifier;
  // NOTE: has() ignores implicit casts, using hasDescendant() to match it
  EXPECT_TRUE(
        testImport(
          "void declToImport() { int b; int a = b ?: 1; int C = ({int X=4; X;}); }",
          Lang_CXX, "", Lang_CXX, Verifier,
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
                              ))))))))))));
}

TEST(ImportExpr, ImportConditionalOperator) {
  MatchVerifier<Decl> Verifier;
  EXPECT_TRUE(
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
                      ))))))));
}

TEST(ImportExpr, ImportBinaryConditionalOperator) {
  MatchVerifier<Decl> Verifier;
  EXPECT_TRUE(
        testImport(
          "void declToImport() { 1 ?: -5; }",
          Lang_CXX, "", Lang_CXX, Verifier,
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
                      )))))));
}

TEST(ImportExpr, ImportDesignatedInitExpr) {
  MatchVerifier<Decl> Verifier;
  EXPECT_TRUE(testImport("void declToImport() {"
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
                                               )))))))))))));
}


TEST(ImportExpr, ImportPredefinedExpr) {
  MatchVerifier<Decl> Verifier;
  // __func__ expands as StringLiteral("declToImport")
  EXPECT_TRUE(testImport("void declToImport() { __func__; }",
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
                                         asString("const char [13]")))))))))));
}

TEST(ImportExpr, ImportInitListExpr) {
  MatchVerifier<Decl> Verifier;
  EXPECT_TRUE(
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
                              )))))))))));
}


const internal::VariadicDynCastAllOfMatcher<Expr, VAArgExpr> vaArgExpr;

TEST(ImportExpr, ImportVAArgExpr) {
  MatchVerifier<Decl> Verifier;
  EXPECT_TRUE(
        testImport(
          "void declToImport(__builtin_va_list list, ...) {"
          "  (void)__builtin_va_arg(list, int); }",
          Lang_CXX, "", Lang_CXX, Verifier,
          functionDecl(
            hasBody(
              compoundStmt(
                has(
                  cStyleCastExpr(
                    hasSourceExpression(
                      vaArgExpr()))))))));
}


TEST(ImportType, ImportAtomicType) {
  MatchVerifier<Decl> Verifier;
  EXPECT_TRUE(testImport("void declToImport() { typedef _Atomic(int) a_int; }",
                         Lang_CXX11, "", Lang_CXX11, Verifier,
                         functionDecl(
                           hasBody(
                             compoundStmt(
                               has(
                                 declStmt(
                                   has(
                                     typedefDecl(
                                       has(atomicType()))))))))));
}


} // end namespace ast_matchers
} // end namespace clang
