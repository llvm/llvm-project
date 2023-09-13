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

#include "clang/AST/RecordLayout.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"

#include "clang/AST/DeclContextInternals.h"
#include "gtest/gtest.h"

#include "ASTImporterFixtures.h"
#include <optional>

namespace clang {
namespace ast_matchers {

using internal::Matcher;
using internal::BindableMatcher;
using llvm::StringMap;

static const RecordDecl *getRecordDeclOfFriend(FriendDecl *FD) {
  QualType Ty = FD->getFriendType()->getType().getCanonicalType();
  return cast<RecordType>(Ty)->getDecl();
}

struct ImportExpr : TestImportBase {};
struct ImportType : TestImportBase {};
struct ImportDecl : TestImportBase {};
struct ImportFixedPointExpr : ImportExpr {};

struct CanonicalRedeclChain : ASTImporterOptionSpecificTestBase {};

TEST_P(CanonicalRedeclChain, ShouldBeConsequentWithMatchers) {
  Decl *FromTU = getTuDecl("void f();", Lang_CXX03);
  auto Pattern = functionDecl(hasName("f"));
  auto *D0 = FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);

  auto Redecls = getCanonicalForwardRedeclChain(D0);
  ASSERT_EQ(Redecls.size(), 1u);
  EXPECT_EQ(D0, Redecls[0]);
}

TEST_P(CanonicalRedeclChain, ShouldBeConsequentWithMatchers2) {
  Decl *FromTU = getTuDecl("void f(); void f(); void f();", Lang_CXX03);
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
  Decl *FromTU = getTuDecl("void f(); void f(); void f();", Lang_CXX03);
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

namespace {
struct RedirectingImporter : public ASTImporter {
  using ASTImporter::ASTImporter;

protected:
  llvm::Expected<Decl *> ImportImpl(Decl *FromD) override {
    auto *ND = dyn_cast<NamedDecl>(FromD);
    if (!ND || ND->getName() != "shouldNotBeImported")
      return ASTImporter::ImportImpl(FromD);
    for (Decl *D : getToContext().getTranslationUnitDecl()->decls()) {
      if (auto *ND = dyn_cast<NamedDecl>(D))
        if (ND->getName() == "realDecl") {
          RegisterImportedDecl(FromD, ND);
          return ND;
        }
    }
    return ASTImporter::ImportImpl(FromD);
  }
};

} // namespace

struct RedirectingImporterTest : ASTImporterOptionSpecificTestBase {
  RedirectingImporterTest() {
    Creator = [](ASTContext &ToContext, FileManager &ToFileManager,
                 ASTContext &FromContext, FileManager &FromFileManager,
                 bool MinimalImport,
                 const std::shared_ptr<ASTImporterSharedState> &SharedState) {
      return new RedirectingImporter(ToContext, ToFileManager, FromContext,
                                     FromFileManager, MinimalImport,
                                     SharedState);
    };
  }
};

// Test that an ASTImporter subclass can intercept an import call.
TEST_P(RedirectingImporterTest, InterceptImport) {
  Decl *From, *To;
  std::tie(From, To) =
      getImportedDecl("class shouldNotBeImported {};", Lang_CXX03,
                      "class realDecl {};", Lang_CXX03, "shouldNotBeImported");
  auto *Imported = cast<CXXRecordDecl>(To);
  EXPECT_EQ(Imported->getQualifiedNameAsString(), "realDecl");

  // Make sure our importer prevented the importing of the decl.
  auto *ToTU = Imported->getTranslationUnitDecl();
  auto Pattern = functionDecl(hasName("shouldNotBeImported"));
  unsigned count =
      DeclCounterWithPredicate<CXXRecordDecl>().match(ToTU, Pattern);
  EXPECT_EQ(0U, count);
}

// Test that when we indirectly import a declaration the custom ASTImporter
// is still intercepting the import.
TEST_P(RedirectingImporterTest, InterceptIndirectImport) {
  Decl *From, *To;
  std::tie(From, To) =
      getImportedDecl("class shouldNotBeImported {};"
                      "class F { shouldNotBeImported f; };",
                      Lang_CXX03, "class realDecl {};", Lang_CXX03, "F");

  // Make sure our ASTImporter prevented the importing of the decl.
  auto *ToTU = To->getTranslationUnitDecl();
  auto Pattern = functionDecl(hasName("shouldNotBeImported"));
  unsigned count =
      DeclCounterWithPredicate<CXXRecordDecl>().match(ToTU, Pattern);
  EXPECT_EQ(0U, count);
}

struct ImportPath : ASTImporterOptionSpecificTestBase {
  Decl *FromTU;
  FunctionDecl *D0, *D1, *D2;
  ImportPath() {
    FromTU = getTuDecl("void f(); void f(); void f();", Lang_CXX03);
    auto Pattern = functionDecl(hasName("f"));
    D0 = FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);
    D2 = LastDeclMatcher<FunctionDecl>().match(FromTU, Pattern);
    D1 = D2->getPreviousDecl();
  }
};

TEST_P(ImportPath, Push) {
  ASTImporter::ImportPathTy path;
  path.push(D0);
  EXPECT_FALSE(path.hasCycleAtBack());
}

TEST_P(ImportPath, SmallCycle) {
  ASTImporter::ImportPathTy path;
  path.push(D0);
  path.push(D0);
  EXPECT_TRUE(path.hasCycleAtBack());
  path.pop();
  EXPECT_FALSE(path.hasCycleAtBack());
  path.push(D0);
  EXPECT_TRUE(path.hasCycleAtBack());
}

TEST_P(ImportPath, GetSmallCycle) {
  ASTImporter::ImportPathTy path;
  path.push(D0);
  path.push(D0);
  EXPECT_TRUE(path.hasCycleAtBack());
  std::array<Decl* ,2> Res;
  int i = 0;
  for (Decl *Di : path.getCycleAtBack()) {
    Res[i++] = Di;
  }
  ASSERT_EQ(i, 2);
  EXPECT_EQ(Res[0], D0);
  EXPECT_EQ(Res[1], D0);
}

TEST_P(ImportPath, GetCycle) {
  ASTImporter::ImportPathTy path;
  path.push(D0);
  path.push(D1);
  path.push(D2);
  path.push(D0);
  EXPECT_TRUE(path.hasCycleAtBack());
  std::array<Decl* ,4> Res;
  int i = 0;
  for (Decl *Di : path.getCycleAtBack()) {
    Res[i++] = Di;
  }
  ASSERT_EQ(i, 4);
  EXPECT_EQ(Res[0], D0);
  EXPECT_EQ(Res[1], D2);
  EXPECT_EQ(Res[2], D1);
  EXPECT_EQ(Res[3], D0);
}

TEST_P(ImportPath, CycleAfterCycle) {
  ASTImporter::ImportPathTy path;
  path.push(D0);
  path.push(D1);
  path.push(D0);
  path.push(D1);
  path.push(D2);
  path.push(D0);
  EXPECT_TRUE(path.hasCycleAtBack());
  std::array<Decl* ,4> Res;
  int i = 0;
  for (Decl *Di : path.getCycleAtBack()) {
    Res[i++] = Di;
  }
  ASSERT_EQ(i, 4);
  EXPECT_EQ(Res[0], D0);
  EXPECT_EQ(Res[1], D2);
  EXPECT_EQ(Res[2], D1);
  EXPECT_EQ(Res[3], D0);

  path.pop();
  path.pop();
  path.pop();
  EXPECT_TRUE(path.hasCycleAtBack());
  i = 0;
  for (Decl *Di : path.getCycleAtBack()) {
    Res[i++] = Di;
  }
  ASSERT_EQ(i, 3);
  EXPECT_EQ(Res[0], D0);
  EXPECT_EQ(Res[1], D1);
  EXPECT_EQ(Res[2], D0);

  path.pop();
  EXPECT_FALSE(path.hasCycleAtBack());
}

const internal::VariadicDynCastAllOfMatcher<Stmt, SourceLocExpr> sourceLocExpr;

AST_MATCHER_P(SourceLocExpr, hasBuiltinStr, StringRef, Str) {
  return Node.getBuiltinStr() == Str;
}

TEST_P(ImportExpr, ImportSourceLocExpr) {
  MatchVerifier<Decl> Verifier;
  testImport("void declToImport() { (void)__builtin_FILE(); }", Lang_CXX03, "",
             Lang_CXX03, Verifier,
             functionDecl(hasDescendant(
                 sourceLocExpr(hasBuiltinStr("__builtin_FILE")))));
  testImport("void declToImport() { (void)__builtin_FILE_NAME(); }", Lang_CXX03,
             "", Lang_CXX03, Verifier,
             functionDecl(hasDescendant(
                 sourceLocExpr(hasBuiltinStr("__builtin_FILE_NAME")))));
  testImport("void declToImport() { (void)__builtin_COLUMN(); }", Lang_CXX03,
             "", Lang_CXX03, Verifier,
             functionDecl(hasDescendant(
                 sourceLocExpr(hasBuiltinStr("__builtin_COLUMN")))));
}

TEST_P(ImportExpr, ImportStringLiteral) {
  MatchVerifier<Decl> Verifier;
  testImport("void declToImport() { (void)\"foo\"; }", Lang_CXX03, "",
             Lang_CXX03, Verifier,
             functionDecl(hasDescendant(
                 stringLiteral(hasType(asString("const char[4]"))))));
  testImport("void declToImport() { (void)L\"foo\"; }", Lang_CXX03, "",
             Lang_CXX03, Verifier,
             functionDecl(hasDescendant(
                 stringLiteral(hasType(asString("const wchar_t[4]"))))));
  testImport("void declToImport() { (void) \"foo\" \"bar\"; }", Lang_CXX03, "",
             Lang_CXX03, Verifier,
             functionDecl(hasDescendant(
                 stringLiteral(hasType(asString("const char[7]"))))));
}

TEST_P(ImportExpr, ImportChooseExpr) {
  MatchVerifier<Decl> Verifier;

  // This case tests C code that is not condition-dependent and has a true
  // condition.
  testImport("void declToImport() { (void)__builtin_choose_expr(1, 2, 3); }",
             Lang_C99, "", Lang_C99, Verifier,
             functionDecl(hasDescendant(chooseExpr())));
}

const internal::VariadicDynCastAllOfMatcher<Stmt, ShuffleVectorExpr>
    shuffleVectorExpr;

TEST_P(ImportExpr, ImportShuffleVectorExpr) {
  MatchVerifier<Decl> Verifier;
  constexpr auto Code = R"code(
    typedef double vector4double __attribute__((__vector_size__(32)));
    vector4double declToImport(vector4double a, vector4double b) {
      return __builtin_shufflevector(a, b, 0, 1, 2, 3);
    }
  )code";
  const auto Pattern = functionDecl(hasDescendant(shuffleVectorExpr(
      allOf(has(declRefExpr(to(parmVarDecl(hasName("a"))))),
            has(declRefExpr(to(parmVarDecl(hasName("b"))))),
            has(integerLiteral(equals(0))), has(integerLiteral(equals(1))),
            has(integerLiteral(equals(2))), has(integerLiteral(equals(3)))))));
  testImport(Code, Lang_C99, "", Lang_C99, Verifier, Pattern);
}

TEST_P(ImportExpr, ImportGNUNullExpr) {
  MatchVerifier<Decl> Verifier;
  testImport("void declToImport() { (void)__null; }", Lang_CXX03, "",
             Lang_CXX03, Verifier,
             functionDecl(hasDescendant(gnuNullExpr(hasType(isInteger())))));
}

TEST_P(ImportExpr, ImportGenericSelectionExpr) {
  MatchVerifier<Decl> Verifier;

  testImport(
      "void declToImport() { int x; (void)_Generic(x, int: 0, float: 1); }",
      Lang_C99, "", Lang_C99, Verifier,
      functionDecl(hasDescendant(genericSelectionExpr())));
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
  testImport("void declToImport() { (void)1.0; }", Lang_C99, "", Lang_C99,
             Verifier,
             functionDecl(hasDescendant(
                 floatLiteral(equals(1.0), hasType(asString("double"))))));
  testImport("void declToImport() { (void)1.0e-5f; }", Lang_C99, "", Lang_C99,
             Verifier,
             functionDecl(hasDescendant(
                 floatLiteral(equals(1.0e-5f), hasType(asString("float"))))));
}

TEST_P(ImportFixedPointExpr, ImportFixedPointerLiteralExpr) {
  MatchVerifier<Decl> Verifier;
  testImport("void declToImport() { (void)1.0k; }", Lang_C99, "", Lang_C99,
             Verifier, functionDecl(hasDescendant(fixedPointLiteral())));
  testImport("void declToImport() { (void)0.75r; }", Lang_C99, "", Lang_C99,
             Verifier, functionDecl(hasDescendant(fixedPointLiteral())));
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
  testImport("void declToImport() {"
             "  struct s { int x; long y; unsigned z; }; "
             "  (void)(struct s){ 42, 0L, 1U }; }",
             Lang_CXX03, "", Lang_CXX03, Verifier,
             functionDecl(hasDescendant(compoundLiteralExpr(
                 hasType(asString("struct s")),
                 has(initListExpr(
                     hasType(asString("struct s")),
                     has(integerLiteral(equals(42), hasType(asString("int")))),
                     has(integerLiteral(equals(0), hasType(asString("long")))),
                     has(integerLiteral(
                         equals(1), hasType(asString("unsigned int"))))))))));
}

TEST_P(ImportExpr, ImportCXXThisExpr) {
  MatchVerifier<Decl> Verifier;
  testImport("class declToImport { void f() { (void)this; } };", Lang_CXX03, "",
             Lang_CXX03, Verifier,
             cxxRecordDecl(hasMethod(hasDescendant(
                 cxxThisExpr(hasType(asString("class declToImport *")))))));
}

TEST_P(ImportExpr, ImportAtomicExpr) {
  MatchVerifier<Decl> Verifier;
  testImport("void declToImport() { int *ptr; __atomic_load_n(ptr, 1); }",
             Lang_C99, "", Lang_C99, Verifier,
             functionDecl(hasDescendant(atomicExpr(
                 has(ignoringParenImpCasts(
                     declRefExpr(hasDeclaration(varDecl(hasName("ptr"))),
                                 hasType(asString("int *"))))),
                 has(integerLiteral(equals(1), hasType(asString("int"))))))));
}

TEST_P(ImportExpr, ImportLabelDeclAndAddrLabelExpr) {
  MatchVerifier<Decl> Verifier;
  testImport("void declToImport() { loop: goto loop; (void)&&loop; }", Lang_C99,
             "", Lang_C99, Verifier,
             functionDecl(hasDescendant(labelStmt(
                              hasDeclaration(labelDecl(hasName("loop"))))),
                          hasDescendant(addrLabelExpr(
                              hasDeclaration(labelDecl(hasName("loop")))))));
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
      Lang_CXX03, "", Lang_CXX03, Verifier,
      typedefDecl(hasType(elaboratedType(namesType(templateSpecializationType(
          hasDeclaration(classTemplateSpecializationDecl(hasSpecializedTemplate(
              classTemplateDecl(hasTemplateDecl(cxxRecordDecl(hasMethod(
                  allOf(hasName("f"),
                        hasBody(compoundStmt(has(declStmt(hasSingleDecl(varDecl(
                            hasInitializer(parenListExpr(has(unaryOperator(
                                hasOperatorName("*"),
                                hasUnaryOperand(
                                    cxxThisExpr())))))))))))))))))))))))));
}

TEST_P(ImportExpr, ImportSwitch) {
  MatchVerifier<Decl> Verifier;
  testImport("void declToImport() { int b; switch (b) { case 1: break; } }",
             Lang_C99, "", Lang_C99, Verifier,
             functionDecl(hasDescendant(
                 switchStmt(has(compoundStmt(has(caseStmt())))))));
}

TEST_P(ImportExpr, ImportStmtExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "void declToImport() { int b; int a = b ?: 1; int C = ({int X=4; X;}); }",
      Lang_C99, "", Lang_C99, Verifier,
      traverse(TK_AsIs,
               functionDecl(hasDescendant(varDecl(
                   hasName("C"), hasType(asString("int")),
                   hasInitializer(stmtExpr(
                       hasAnySubstatement(declStmt(hasSingleDecl(varDecl(
                           hasName("X"), hasType(asString("int")),
                           hasInitializer(integerLiteral(equals(4))))))),
                       hasDescendant(implicitCastExpr()))))))));
}

TEST_P(ImportExpr, ImportConditionalOperator) {
  MatchVerifier<Decl> Verifier;
  testImport("void declToImport() { (void)(true ? 1 : -5); }", Lang_CXX03, "",
             Lang_CXX03, Verifier,
             functionDecl(hasDescendant(conditionalOperator(
                 hasCondition(cxxBoolLiteral(equals(true))),
                 hasTrueExpression(integerLiteral(equals(1))),
                 hasFalseExpression(unaryOperator(
                     hasUnaryOperand(integerLiteral(equals(5)))))))));
}

TEST_P(ImportExpr, ImportBinaryConditionalOperator) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "void declToImport() { (void)(1 ?: -5); }", Lang_CXX03, "", Lang_CXX03,
      Verifier,
      traverse(TK_AsIs,
               functionDecl(hasDescendant(binaryConditionalOperator(
                   hasCondition(implicitCastExpr(
                       hasSourceExpression(opaqueValueExpr(
                           hasSourceExpression(integerLiteral(equals(1))))),
                       hasType(booleanType()))),
                   hasTrueExpression(opaqueValueExpr(
                       hasSourceExpression(integerLiteral(equals(1))))),
                   hasFalseExpression(unaryOperator(
                       hasOperatorName("-"),
                       hasUnaryOperand(integerLiteral(equals(5))))))))));
}

TEST_P(ImportExpr, ImportDesignatedInitExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "void declToImport() {"
      "  struct point { double x; double y; };"
      "  struct point ptarray[10] = "
      "{ [2].y = 1.0, [2].x = 2.0, [0].x = 1.0 }; }",
      Lang_C99, "", Lang_C99, Verifier,
      functionDecl(hasDescendant(initListExpr(
          has(designatedInitExpr(designatorCountIs(2),
                                 hasDescendant(floatLiteral(equals(1.0))),
                                 hasDescendant(integerLiteral(equals(2))))),
          has(designatedInitExpr(designatorCountIs(2),
                                 hasDescendant(floatLiteral(equals(2.0))),
                                 hasDescendant(integerLiteral(equals(2))))),
          has(designatedInitExpr(designatorCountIs(2),
                                 hasDescendant(floatLiteral(equals(1.0))),
                                 hasDescendant(integerLiteral(equals(0)))))))));
}

TEST_P(ImportExpr, ImportPredefinedExpr) {
  MatchVerifier<Decl> Verifier;
  // __func__ expands as StringLiteral("declToImport")
  testImport("void declToImport() { (void)__func__; }", Lang_CXX03, "",
             Lang_CXX03, Verifier,
             functionDecl(hasDescendant(predefinedExpr(
                 hasType(asString("const char[13]")),
                 has(stringLiteral(hasType(asString("const char[13]"))))))));
}

TEST_P(ImportExpr, ImportInitListExpr) {
  MatchVerifier<Decl> Verifier;
  testImport("void declToImport() {"
             "  struct point { double x; double y; };"
             "  point ptarray[10] = { [2].y = 1.0, [2].x = 2.0,"
             "                        [0].x = 1.0 }; }",
             Lang_CXX03, "", Lang_CXX03, Verifier,
             functionDecl(hasDescendant(initListExpr(
                 has(cxxConstructExpr(requiresZeroInitialization())),
                 has(initListExpr(
                     hasType(asString("point")), has(floatLiteral(equals(1.0))),
                     has(implicitValueInitExpr(hasType(asString("double")))))),
                 has(initListExpr(hasType(asString("point")),
                                  has(floatLiteral(equals(2.0))),
                                  has(floatLiteral(equals(1.0)))))))));
}

const internal::VariadicDynCastAllOfMatcher<Expr, CXXDefaultInitExpr>
    cxxDefaultInitExpr;

TEST_P(ImportExpr, ImportCXXDefaultInitExpr) {
  MatchVerifier<Decl> Verifier;
  testImport("class declToImport { int DefInit = 5; }; declToImport X;",
             Lang_CXX11, "", Lang_CXX11, Verifier,
             cxxRecordDecl(hasDescendant(cxxConstructorDecl(
                 hasAnyConstructorInitializer(cxxCtorInitializer(
                     withInitializer(cxxDefaultInitExpr())))))));
  testImport(
      "struct X { int A = 5; }; X declToImport{};", Lang_CXX17, "", Lang_CXX17,
      Verifier,
      varDecl(hasInitializer(initListExpr(hasInit(0, cxxDefaultInitExpr())))));
}

const internal::VariadicDynCastAllOfMatcher<Expr, VAArgExpr> vaArgExpr;

TEST_P(ImportExpr, ImportVAArgExpr) {
  MatchVerifier<Decl> Verifier;
  testImport("void declToImport(__builtin_va_list list, ...) {"
             "  (void)__builtin_va_arg(list, int); }",
             Lang_CXX03, "", Lang_CXX03, Verifier,
             functionDecl(hasDescendant(
                 cStyleCastExpr(hasSourceExpression(vaArgExpr())))));
}

TEST_P(ImportExpr, CXXTemporaryObjectExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "struct C {};"
      "void declToImport() { C c = C(); }",
      Lang_CXX03, "", Lang_CXX03, Verifier,
      traverse(TK_AsIs,
               functionDecl(hasDescendant(exprWithCleanups(has(cxxConstructExpr(
                   has(materializeTemporaryExpr(has(implicitCastExpr(
                       has(cxxTemporaryObjectExpr()))))))))))));
}

TEST_P(ImportType, ImportAtomicType) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "void declToImport() { typedef _Atomic(int) a_int; }",
      Lang_CXX11, "", Lang_CXX11, Verifier,
      functionDecl(hasDescendant(typedefDecl(has(atomicType())))));
}

TEST_P(ImportType, ImportBitIntType) {
  const AstTypeMatcher<BitIntType> bitIntType;
  MatchVerifier<Decl> Verifier;
  testImport("_BitInt(10) declToImport;", Lang_CXX11, "", Lang_CXX11, Verifier,
             varDecl(hasType(bitIntType())));
}

TEST_P(ImportType, ImportDependentBitIntType) {
  const AstTypeMatcher<DependentBitIntType> dependentBitIntType;
  MatchVerifier<Decl> Verifier;
  testImport("template<int Width> using declToImport = _BitInt(Width);",
             Lang_CXX11, "", Lang_CXX11, Verifier,
             typeAliasTemplateDecl(
                 has(typeAliasDecl(hasType(dependentBitIntType())))));
}

TEST_P(ImportType, ImportDependentAddressSpaceType) {
  const AstTypeMatcher<DependentAddressSpaceType> dependentAddressSpaceType;
  MatchVerifier<Decl> Verifier;
  testImport(
      R"(
        template<typename T, int AddrSpace>
        using declToImport = T __attribute__((address_space(AddrSpace)));
      )",
      Lang_CXX11, "", Lang_CXX11, Verifier,
      typeAliasTemplateDecl(
          has(typeAliasDecl(hasType(dependentAddressSpaceType())))));
}

TEST_P(ImportType, ImportVectorType) {
  const AstTypeMatcher<VectorType> vectorType;
  MatchVerifier<Decl> Verifier;
  testImport("typedef int __attribute__((vector_size(12))) declToImport;",
             Lang_CXX11, "", Lang_CXX11, Verifier,
             typedefDecl(hasType(vectorType())));
}

TEST_P(ImportType, ImportDependentVectorType) {
  const AstTypeMatcher<DependentVectorType> dependentVectorType;
  MatchVerifier<Decl> Verifier;
  testImport(
      R"(
        template<typename T, int Size>
        using declToImport = T __attribute__((vector_size(Size)));
      )",
      Lang_CXX11, "", Lang_CXX11, Verifier,
      typeAliasTemplateDecl(
          has(typeAliasDecl(hasType(dependentVectorType())))));
}

struct ImportOpenCLPipe : ImportType {
  std::vector<std::string> getExtraArgs() const override {
    return {"-x", "cl", "-cl-no-stdinc", "-cl-std=CL2.0"};
  }
};

TEST_P(ImportOpenCLPipe, ImportPipeType) {
  const AstTypeMatcher<PipeType> pipeType;
  MatchVerifier<Decl> Verifier;
  testImport("typedef pipe int declToImport;", Lang_OpenCL, "", Lang_OpenCL,
             Verifier, typedefDecl(hasType(pipeType())));
}

struct ImportMatrixType : ImportType {
  std::vector<std::string> getExtraArgs() const override {
    return {"-fenable-matrix"};
  }
};

TEST_P(ImportMatrixType, ImportConstantMatrixType) {
  const AstTypeMatcher<ConstantMatrixType> constantMatrixType;
  MatchVerifier<Decl> Verifier;
  testImport("typedef int __attribute__((matrix_type(5, 5))) declToImport;",
             Lang_CXX11, "", Lang_CXX11, Verifier,
             typedefDecl(hasType(constantMatrixType())));
}

TEST_P(ImportMatrixType, ImportDependentSizedMatrixType) {
  const AstTypeMatcher<DependentSizedMatrixType> dependentSizedMatrixType;
  MatchVerifier<Decl> Verifier;
  testImport(
      R"(
        template<typename T, int Rows, int Cols>
        using declToImport = T __attribute__((matrix_type(Rows, Cols)));
      )",
      Lang_CXX11, "", Lang_CXX11, Verifier,
      typeAliasTemplateDecl(
          has(typeAliasDecl(hasType(dependentSizedMatrixType())))));
}

TEST_P(ImportType, ImportUsingType) {
  MatchVerifier<Decl> Verifier;
  testImport("struct C {};"
             "void declToImport() { using ::C; new C{}; }",
             Lang_CXX11, "", Lang_CXX11, Verifier,
             functionDecl(hasDescendant(cxxNewExpr(hasType(pointerType(
                 pointee(elaboratedType(namesType(usingType())))))))));
}

TEST_P(ImportDecl, ImportFunctionTemplateDecl) {
  MatchVerifier<Decl> Verifier;
  testImport("template <typename T> void declToImport() { };", Lang_CXX03, "",
             Lang_CXX03, Verifier, functionTemplateDecl());
}

TEST_P(ImportExpr, ImportCXXDependentScopeMemberExpr) {
  MatchVerifier<Decl> Verifier;
  testImport("template <typename T> struct C { T t; };"
             "template <typename T> void declToImport() {"
             "  C<T> d;"
             "  (void)d.t;"
             "}"
             "void instantiate() { declToImport<int>(); }",
             Lang_CXX03, "", Lang_CXX03, Verifier,
             functionTemplateDecl(hasDescendant(
                 cStyleCastExpr(has(cxxDependentScopeMemberExpr())))));
  testImport("template <typename T> struct C { T t; };"
             "template <typename T> void declToImport() {"
             "  C<T> d;"
             "  (void)(&d)->t;"
             "}"
             "void instantiate() { declToImport<int>(); }",
             Lang_CXX03, "", Lang_CXX03, Verifier,
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
      traverse(TK_AsIs,
               functionDecl(hasDescendant(implicitCastExpr(has(declRefExpr()))),
                            unless(hasAncestor(
                                translationUnitDecl(has(typeAliasDecl())))))));
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
  testImport("template <typename... Args>"
             "struct dummy {"
             "  dummy(Args... args) {}"
             "  static const int i = 4;"
             "};"
             "int declToImport() { return dummy<int>::i; }",
             Lang_CXX11, "", Lang_CXX11, Verifier,
             traverse(TK_AsIs, functionDecl(hasDescendant(returnStmt(has(
                                   implicitCastExpr(has(declRefExpr()))))))));
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
             Lang_CXX03, "", Lang_CXX03, Verifier,
             classTemplateDecl(has(cxxRecordDecl(has(
                 fieldDecl(hasType(dependentTemplateSpecializationType())))))));
}

TEST_P(ImportType, ImportDeducedTemplateSpecialization) {
  MatchVerifier<Decl> Verifier;
  testImport("template <typename T>"
             "class C { public: C(T); };"
             "C declToImport(123);",
             Lang_CXX17, "", Lang_CXX17, Verifier,
             varDecl(hasType(elaboratedType(
                 namesType(deducedTemplateSpecializationType())))));
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

const internal::VariadicDynCastAllOfMatcher<Stmt, CXXFoldExpr> cxxFoldExpr;

AST_MATCHER_P(CXXFoldExpr, hasOperator, BinaryOperatorKind, Op) {
  return Node.getOperator() == Op;
}
AST_MATCHER(CXXFoldExpr, hasInit) { return Node.getInit(); }
AST_MATCHER(CXXFoldExpr, isRightFold) { return Node.isRightFold(); }
AST_MATCHER(CXXFoldExpr, isLeftFold) { return Node.isLeftFold(); }

TEST_P(ImportExpr, ImportCXXFoldExpr) {
  auto Match1 =
      cxxFoldExpr(hasOperator(BO_Add), isLeftFold(), unless(hasInit()));
  auto Match2 = cxxFoldExpr(hasOperator(BO_Sub), isLeftFold(), hasInit());
  auto Match3 =
      cxxFoldExpr(hasOperator(BO_Mul), isRightFold(), unless(hasInit()));
  auto Match4 = cxxFoldExpr(hasOperator(BO_Div), isRightFold(), hasInit());

  MatchVerifier<Decl> Verifier;
  testImport("template <typename... Ts>"
             "void declToImport(Ts... args) {"
             "  const int i1 = (... + args);"
             "  const int i2 = (1 - ... - args);"
             "  const int i3 = (args * ...);"
             "  const int i4 = (args / ... / 1);"
             "};"
             "void g() { declToImport(1, 2, 3, 4, 5); }",
             Lang_CXX17, "", Lang_CXX17, Verifier,
             functionTemplateDecl(hasDescendant(Match1), hasDescendant(Match2),
                                  hasDescendant(Match3),
                                  hasDescendant(Match4)));
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
      Lang_C99, "", Lang_C99, Verifier,
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
      traverse(
          TK_AsIs,
          functionDecl(
              hasDescendant(varDecl(hasName("a"), hasInitializer(hasDescendant(
                                                      cxxTypeidExpr())))),
              hasDescendant(varDecl(hasName("b"), hasInitializer(hasDescendant(
                                                      cxxTypeidExpr())))))));
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
             Lang_C99, "", Lang_C99, Verifier,
             functionDecl(hasBody(compoundStmt(
                 has(declStmt(hasSingleDecl(varDecl(hasName("d")))))))));
}

TEST_P(ImportDecl, ImportedVarDeclPreservesThreadLocalStorage) {
  MatchVerifier<Decl> Verifier;
  testImport("thread_local int declToImport;", Lang_CXX11, "", Lang_CXX11,
             Verifier, varDecl(hasThreadStorageDuration()));
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportRecordTypeInFunc) {
  Decl *FromTU = getTuDecl("int declToImport() { "
                           "  struct data_t {int a;int b;};"
                           "  struct data_t d;"
                           "  return 0;"
                           "}",
                           Lang_C99, "input.c");
  auto *FromVar =
      FirstDeclMatcher<VarDecl>().match(FromTU, varDecl(hasName("d")));
  ASSERT_TRUE(FromVar);
  auto ToType =
      ImportType(FromVar->getType().getCanonicalType(), FromVar, Lang_C99);
  EXPECT_FALSE(ToType.isNull());
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportRecordDeclInFuncParams) {
  // This construct is not supported by ASTImporter.
  Decl *FromTU = getTuDecl(
      "int declToImport(struct data_t{int a;int b;} ***d){ return 0; }",
      Lang_C99, "input.c");
  auto *From = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("declToImport")));
  ASSERT_TRUE(From);
  auto *To = Import(From, Lang_C99);
  EXPECT_EQ(To, nullptr);
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportRecordDeclInFuncFromMacro) {
  Decl *FromTU =
      getTuDecl("#define NONAME_SIZEOF(type) sizeof(struct{type *dummy;}) \n"
                "int declToImport(){ return NONAME_SIZEOF(int); }",
                Lang_C99, "input.c");
  auto *From = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("declToImport")));
  ASSERT_TRUE(From);
  auto *To = Import(From, Lang_C99);
  ASSERT_TRUE(To);
  EXPECT_TRUE(MatchVerifier<FunctionDecl>().match(
      To, functionDecl(hasName("declToImport"),
                       hasDescendant(unaryExprOrTypeTraitExpr()))));
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportRecordDeclInFuncParamsFromMacro) {
  // This construct is not supported by ASTImporter.
  Decl *FromTU =
      getTuDecl("#define PAIR_STRUCT(type) struct data_t{type a;type b;} \n"
                "int declToImport(PAIR_STRUCT(int) ***d){ return 0; }",
                Lang_C99, "input.c");
  auto *From = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("declToImport")));
  ASSERT_TRUE(From);
  auto *To = Import(From, Lang_C99);
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
      Lang_CXX03, "", Lang_CXX03, Verifier,
      functionDecl(hasDescendant(callExpr(has(cxxPseudoDestructorExpr())))));
}

TEST_P(ImportDecl, ImportUsingDecl) {
  MatchVerifier<Decl> Verifier;
  testImport("namespace foo { int bar; }"
             "void declToImport() { using foo::bar; }",
             Lang_CXX03, "", Lang_CXX03, Verifier,
             functionDecl(hasDescendant(usingDecl(hasName("bar")))));
}

TEST_P(ImportDecl, ImportUsingTemplate) {
  MatchVerifier<Decl> Verifier;
  testImport("namespace ns { template <typename T> struct S {}; }"
             "template <template <typename> class T> class X {};"
             "void declToImport() {"
             "using ns::S;  X<S> xi; }",
             Lang_CXX11, "", Lang_CXX11, Verifier,
             functionDecl(hasDescendant(varDecl(hasTypeLoc(elaboratedTypeLoc(
                 hasNamedTypeLoc(templateSpecializationTypeLoc(
                     hasAnyTemplateArgumentLoc(templateArgumentLoc())))))))));
}

TEST_P(ImportDecl, ImportUsingEnumDecl) {
  MatchVerifier<Decl> Verifier;
  testImport("namespace foo { enum bar { baz, toto, quux }; }"
             "void declToImport() { using enum foo::bar; }",
             Lang_CXX20, "", Lang_CXX20, Verifier,
             functionDecl(hasDescendant(usingEnumDecl(hasName("bar")))));
}

const internal::VariadicDynCastAllOfMatcher<Decl, UsingPackDecl> usingPackDecl;

TEST_P(ImportDecl, ImportUsingPackDecl) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "struct A { int operator()() { return 1; } };"
      "struct B { int operator()() { return 2; } };"
      "template<typename ...T> struct C : T... { using T::operator()...; };"
      "C<A, B> declToImport;",
      Lang_CXX20, "", Lang_CXX20, Verifier,
      varDecl(hasType(elaboratedType(namesType(templateSpecializationType(
          hasDeclaration(classTemplateSpecializationDecl(
              hasDescendant(usingPackDecl())))))))));
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
  // from using-decl
  testImport("namespace foo { int bar; }"
             "namespace declToImport { using foo::bar; }",
             Lang_CXX03, "", Lang_CXX03, Verifier,
             namespaceDecl(has(usingShadowDecl(hasName("bar")))));
  // from using-enum-decl
  testImport("namespace foo { enum bar {baz, toto, quux }; }"
             "namespace declToImport { using enum foo::bar; }",
             Lang_CXX20, "", Lang_CXX20, Verifier,
             namespaceDecl(has(usingShadowDecl(hasName("baz")))));
}

TEST_P(ImportExpr, ImportUnresolvedLookupExpr) {
  MatchVerifier<Decl> Verifier;
  testImport("template<typename T> int foo();"
             "template <typename T> void declToImport() {"
             "  (void)::foo<T>;"
             "  (void)::template foo<T>;"
             "}"
             "void instantiate() { declToImport<int>(); }",
             Lang_CXX03, "", Lang_CXX03, Verifier,
             functionTemplateDecl(hasDescendant(unresolvedLookupExpr())));
}

TEST_P(ImportExpr, ImportCXXUnresolvedConstructExpr) {
  MatchVerifier<Decl> Verifier;
  testImport("template <typename T> struct C { T t; };"
             "template <typename T> void declToImport() {"
             "  C<T> d;"
             "  d.t = T();"
             "}"
             "void instantiate() { declToImport<int>(); }",
             Lang_CXX03, "", Lang_CXX03, Verifier,
             functionTemplateDecl(hasDescendant(
                 binaryOperator(has(cxxUnresolvedConstructExpr())))));
  testImport("template <typename T> struct C { T t; };"
             "template <typename T> void declToImport() {"
             "  C<T> d;"
             "  (&d)->t = T();"
             "}"
             "void instantiate() { declToImport<int>(); }",
             Lang_CXX03, "", Lang_CXX03, Verifier,
             functionTemplateDecl(hasDescendant(
                 binaryOperator(has(cxxUnresolvedConstructExpr())))));
}

/// Check that function "declToImport()" (which is the templated function
/// for corresponding FunctionTemplateDecl) is not added into DeclContext.
/// Same for class template declarations.
TEST_P(ImportDecl, ImportTemplatedDeclForTemplate) {
  MatchVerifier<Decl> Verifier;
  testImport("template <typename T> void declToImport() { T a = 1; }"
             "void instantiate() { declToImport<int>(); }",
             Lang_CXX03, "", Lang_CXX03, Verifier,
             functionTemplateDecl(hasAncestor(translationUnitDecl(
                 unless(has(functionDecl(hasName("declToImport"))))))));
  testImport("template <typename T> struct declToImport { T t; };"
             "void instantiate() { declToImport<int>(); }",
             Lang_CXX03, "", Lang_CXX03, Verifier,
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
  testImport(Code, Lang_CXX03, "", Lang_CXX03, Verifier,
             recordDecl(has(classTemplateDecl()),
                        has(classTemplateSpecializationDecl())));
}

TEST_P(ImportExpr, CXXOperatorCallExpr) {
  MatchVerifier<Decl> Verifier;
  testImport(
      "class declToImport {"
      "  void f() { *this = declToImport(); }"
      "};",
      Lang_CXX03, "", Lang_CXX03, Verifier,
      cxxRecordDecl(has(cxxMethodDecl(hasDescendant(cxxOperatorCallExpr())))));
}

TEST_P(ImportExpr, DependentSizedArrayType) {
  MatchVerifier<Decl> Verifier;
  testImport("template<typename T, int Size> class declToImport {"
             "  T data[Size];"
             "};",
             Lang_CXX03, "", Lang_CXX03, Verifier,
             classTemplateDecl(has(cxxRecordDecl(
                 has(fieldDecl(hasType(dependentSizedArrayType())))))));
}

TEST_P(ImportExpr, DependentSizedExtVectorType) {
  MatchVerifier<Decl> Verifier;
  testImport("template<typename T, int Size>"
             "class declToImport {"
             "  typedef T __attribute__((ext_vector_type(Size))) type;"
             "};",
             Lang_CXX03, "", Lang_CXX03, Verifier,
             classTemplateDecl(has(cxxRecordDecl(
                 has(typedefDecl(hasType(dependentSizedExtVectorType())))))));
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportUsingPackDecl) {
  Decl *FromTU = getTuDecl(
      "struct A { int operator()() { return 1; } };"
      "struct B { int operator()() { return 2; } };"
      "template<typename ...T> struct C : T... { using T::operator()...; };"
      "C<A, B> Var;",
      Lang_CXX20);

  auto From = FirstDeclMatcher<UsingPackDecl>().match(FromTU, usingPackDecl());
  ASSERT_TRUE(From);
  auto To = cast<UsingPackDecl>(Import(From, Lang_CXX20));
  ASSERT_TRUE(To);

  ArrayRef<NamedDecl *> FromExpansions = From->expansions();
  ArrayRef<NamedDecl *> ToExpansions = To->expansions();
  ASSERT_EQ(FromExpansions.size(), ToExpansions.size());
  for (unsigned int I = 0; I < FromExpansions.size(); ++I) {
    auto ImportedExpansion = Import(FromExpansions[I], Lang_CXX20);
    EXPECT_EQ(ImportedExpansion, ToExpansions[I]);
  }

  auto ImportedDC = cast<Decl>(Import(From->getDeclContext(), Lang_CXX20));
  EXPECT_EQ(ImportedDC, cast<Decl>(To->getDeclContext()));
}

TEST_P(ASTImporterOptionSpecificTestBase, TemplateTypeParmDeclNoDefaultArg) {
  Decl *FromTU = getTuDecl("template<typename T> struct X {};", Lang_CXX03);
  auto From = FirstDeclMatcher<TemplateTypeParmDecl>().match(
      FromTU, templateTypeParmDecl(hasName("T")));
  TemplateTypeParmDecl *To = Import(From, Lang_CXX03);
  ASSERT_FALSE(To->hasDefaultArgument());
}

TEST_P(ASTImporterOptionSpecificTestBase, TemplateTypeParmDeclDefaultArg) {
  Decl *FromTU =
      getTuDecl("template<typename T = int> struct X {};", Lang_CXX03);
  auto From = FirstDeclMatcher<TemplateTypeParmDecl>().match(
      FromTU, templateTypeParmDecl(hasName("T")));
  TemplateTypeParmDecl *To = Import(From, Lang_CXX03);
  ASSERT_TRUE(To->hasDefaultArgument());
  QualType ToArg = To->getDefaultArgument();
  ASSERT_EQ(ToArg, QualType(To->getASTContext().IntTy));
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportBeginLocOfDeclRefExpr) {
  Decl *FromTU =
      getTuDecl("class A { public: static int X; }; void f() { (void)A::X; }",
                Lang_CXX03);
  auto From = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("f")));
  ASSERT_TRUE(From);
  ASSERT_TRUE(
      cast<CStyleCastExpr>(cast<CompoundStmt>(From->getBody())->body_front())
          ->getSubExpr()
          ->getBeginLoc()
          .isValid());
  FunctionDecl *To = Import(From, Lang_CXX03);
  ASSERT_TRUE(To);
  ASSERT_TRUE(
      cast<CStyleCastExpr>(cast<CompoundStmt>(To->getBody())->body_front())
          ->getSubExpr()
          ->getBeginLoc()
          .isValid());
}

TEST_P(ASTImporterOptionSpecificTestBase,
       TemplateTemplateParmDeclNoDefaultArg) {
  Decl *FromTU = getTuDecl(R"(
                           template<template<typename> typename TT> struct Y {};
                           )",
                           Lang_CXX17);
  auto From = FirstDeclMatcher<TemplateTemplateParmDecl>().match(
      FromTU, templateTemplateParmDecl(hasName("TT")));
  TemplateTemplateParmDecl *To = Import(From, Lang_CXX17);
  ASSERT_FALSE(To->hasDefaultArgument());
}

TEST_P(ASTImporterOptionSpecificTestBase, TemplateTemplateParmDeclDefaultArg) {
  Decl *FromTU = getTuDecl(R"(
                           template<typename T> struct X {};
                           template<template<typename> typename TT = X> struct Y {};
                           )",
                           Lang_CXX17);
  auto From = FirstDeclMatcher<TemplateTemplateParmDecl>().match(
      FromTU, templateTemplateParmDecl(hasName("TT")));
  TemplateTemplateParmDecl *To = Import(From, Lang_CXX17);
  ASSERT_TRUE(To->hasDefaultArgument());
  const TemplateArgument &ToDefaultArg = To->getDefaultArgument().getArgument();
  ASSERT_TRUE(To->isTemplateDecl());
  TemplateDecl *ToTemplate = ToDefaultArg.getAsTemplate().getAsTemplateDecl();

  // Find the default argument template 'X' in the AST and compare it against
  // the default argument we got.
  auto ToExpectedDecl = FirstDeclMatcher<ClassTemplateDecl>().match(
      To->getTranslationUnitDecl(), classTemplateDecl(hasName("X")));
  ASSERT_EQ(ToTemplate, ToExpectedDecl);
}

TEST_P(ASTImporterOptionSpecificTestBase, NonTypeTemplateParmDeclNoDefaultArg) {
  Decl *FromTU = getTuDecl("template<int N> struct X {};", Lang_CXX03);
  auto From = FirstDeclMatcher<NonTypeTemplateParmDecl>().match(
      FromTU, nonTypeTemplateParmDecl(hasName("N")));
  NonTypeTemplateParmDecl *To = Import(From, Lang_CXX03);
  ASSERT_FALSE(To->hasDefaultArgument());
}

TEST_P(ASTImporterOptionSpecificTestBase, NonTypeTemplateParmDeclDefaultArg) {
  Decl *FromTU = getTuDecl("template<int S = 1> struct X {};", Lang_CXX03);
  auto From = FirstDeclMatcher<NonTypeTemplateParmDecl>().match(
      FromTU, nonTypeTemplateParmDecl(hasName("S")));
  NonTypeTemplateParmDecl *To = Import(From, Lang_CXX03);
  ASSERT_TRUE(To->hasDefaultArgument());
  Stmt *ToArg = To->getDefaultArgument();
  ASSERT_TRUE(isa<IntegerLiteral>(ToArg));
  ASSERT_EQ(cast<IntegerLiteral>(ToArg)->getValue().getLimitedValue(), 1U);
}

TEST_P(ASTImporterOptionSpecificTestBase, TemplateArgumentsDefaulted) {
  Decl *FromTU = getTuDecl(R"(
                           template<typename T> struct X {};
                           template<typename TP = double,
                                    int NTTP = 50,
                                    template<typename> typename TT = X> struct S {};
                           S<> s;
                           )",
                           Lang_CXX17);
  auto *FromSpec = FirstDeclMatcher<ClassTemplateSpecializationDecl>().match(
      FromTU, classTemplateSpecializationDecl(hasName("S")));
  ASSERT_TRUE(FromSpec);
  auto *ToSpec = Import(FromSpec, Lang_CXX03);
  ASSERT_TRUE(ToSpec);
  auto const &TList = ToSpec->getTemplateArgs();
  for (auto const &Arg : TList.asArray()) {
    ASSERT_TRUE(Arg.getIsDefaulted());
  }
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportOfTemplatedDeclOfClassTemplateDecl) {
  Decl *FromTU = getTuDecl("template<class X> struct S{};", Lang_CXX03);
  auto From =
      FirstDeclMatcher<ClassTemplateDecl>().match(FromTU, classTemplateDecl());
  ASSERT_TRUE(From);
  auto To = cast<ClassTemplateDecl>(Import(From, Lang_CXX03));
  ASSERT_TRUE(To);
  Decl *ToTemplated = To->getTemplatedDecl();
  Decl *ToTemplated1 = Import(From->getTemplatedDecl(), Lang_CXX03);
  EXPECT_TRUE(ToTemplated1);
  EXPECT_EQ(ToTemplated1, ToTemplated);
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportOfTemplatedDeclOfFunctionTemplateDecl) {
  Decl *FromTU = getTuDecl("template<class X> void f(){}", Lang_CXX03);
  auto From = FirstDeclMatcher<FunctionTemplateDecl>().match(
      FromTU, functionTemplateDecl());
  ASSERT_TRUE(From);
  auto To = cast<FunctionTemplateDecl>(Import(From, Lang_CXX03));
  ASSERT_TRUE(To);
  Decl *ToTemplated = To->getTemplatedDecl();
  Decl *ToTemplated1 = Import(From->getTemplatedDecl(), Lang_CXX03);
  EXPECT_TRUE(ToTemplated1);
  EXPECT_EQ(ToTemplated1, ToTemplated);
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportOfTemplatedDeclShouldImportTheClassTemplateDecl) {
  Decl *FromTU = getTuDecl("template<class X> struct S{};", Lang_CXX03);
  auto FromFT =
      FirstDeclMatcher<ClassTemplateDecl>().match(FromTU, classTemplateDecl());
  ASSERT_TRUE(FromFT);

  auto ToTemplated =
      cast<CXXRecordDecl>(Import(FromFT->getTemplatedDecl(), Lang_CXX03));
  EXPECT_TRUE(ToTemplated);
  auto ToTU = ToTemplated->getTranslationUnitDecl();
  auto ToFT =
      FirstDeclMatcher<ClassTemplateDecl>().match(ToTU, classTemplateDecl());
  EXPECT_TRUE(ToFT);
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportOfTemplatedDeclShouldImportTheFunctionTemplateDecl) {
  Decl *FromTU = getTuDecl("template<class X> void f(){}", Lang_CXX03);
  auto FromFT = FirstDeclMatcher<FunctionTemplateDecl>().match(
      FromTU, functionTemplateDecl());
  ASSERT_TRUE(FromFT);

  auto ToTemplated =
      cast<FunctionDecl>(Import(FromFT->getTemplatedDecl(), Lang_CXX03));
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
  Decl *FromTU = getTuDecl(Code, Lang_CXX03);
  auto FromNs =
      FirstDeclMatcher<NamespaceDecl>().match(FromTU, namespaceDecl());
  auto ToNs = cast<NamespaceDecl>(Import(FromNs, Lang_CXX03));
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
      cast<CXXRecordDecl>(Import(From->getTemplatedDecl(), Lang_CXX03));
  EXPECT_TRUE(ToTemplated1);
  ASSERT_EQ(ToTemplated1, ToTemplated);
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportChooseExpr) {
  // This tests the import of isConditionTrue directly to make sure the importer
  // gets it right.
  Decl *From, *To;
  std::tie(From, To) = getImportedDecl(
      "void declToImport() { (void)__builtin_choose_expr(1, 0, 1); }", Lang_C99,
      "", Lang_C99);

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

TEST_P(ASTImporterOptionSpecificTestBase, ImportConvertVectorExpr) {
  Decl *From, *To;
  std::tie(From, To) = getImportedDecl(
      "typedef double v4double __attribute__((__vector_size__(32)));"
      "typedef float  v4float  __attribute__((__vector_size__(16)));"
      "v4float vf;"
      "void declToImport() { (void)__builtin_convertvector(vf, v4double); }",
      Lang_CXX03, "", Lang_CXX03);

  auto ToResults =
      match(convertVectorExpr().bind("convert"), To->getASTContext());
  auto FromResults =
      match(convertVectorExpr().bind("convert"), From->getASTContext());

  const ConvertVectorExpr *FromConvertVectorExpr =
      selectFirst<ConvertVectorExpr>("convert", FromResults);
  ASSERT_TRUE(FromConvertVectorExpr);

  const ConvertVectorExpr *ToConvertVectorExpr =
      selectFirst<ConvertVectorExpr>("convert", ToResults);
  ASSERT_TRUE(ToConvertVectorExpr);
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportGenericSelectionExpr) {
  Decl *From, *To;
  std::tie(From, To) = getImportedDecl(
      R"(
      int declToImport() {
        int x;
        return _Generic(x, int: 0, default: 1);
      }
      )",
      Lang_C99, "", Lang_C99);

  auto ToResults =
      match(genericSelectionExpr().bind("expr"), To->getASTContext());
  auto FromResults =
      match(genericSelectionExpr().bind("expr"), From->getASTContext());

  const GenericSelectionExpr *FromGenericSelectionExpr =
      selectFirst<GenericSelectionExpr>("expr", FromResults);
  ASSERT_TRUE(FromGenericSelectionExpr);

  const GenericSelectionExpr *ToGenericSelectionExpr =
      selectFirst<GenericSelectionExpr>("expr", ToResults);
  ASSERT_TRUE(ToGenericSelectionExpr);

  EXPECT_EQ(FromGenericSelectionExpr->isResultDependent(),
            ToGenericSelectionExpr->isResultDependent());
  EXPECT_EQ(FromGenericSelectionExpr->getResultIndex(),
            ToGenericSelectionExpr->getResultIndex());
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
      Lang_CXX03, "", Lang_CXX03);

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
                      Lang_CXX03, "", Lang_CXX03);

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
                      Lang_CXX03, "", Lang_CXX03);

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
      Lang_CXX03, "", Lang_CXX03);

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
  for (Decl *D : Node.decls()) {
    if (isa<FieldDecl>(D) || isa<IndirectFieldDecl>(D)) {
      auto *ND = cast<NamedDecl>(D);
      if (Index == Order.size())
        return false;
      if (ND->getName() != Order[Index])
        return false;
      ++Index;
    }
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
      Lang_CXX03, "", Lang_CXX03, "NS");

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
       CXXRecordDeclFieldOrderShouldNotDependOnImportOrder) {
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

TEST_P(ASTImporterOptionSpecificTestBase,
       CXXRecordDeclFieldAndIndirectFieldOrder) {
  Decl *From, *To;
  std::tie(From, To) = getImportedDecl(
      // First field is "a", then the field for unnamed union, then "b" and "c"
      // from it (indirect fields), then "d".
      R"s(
      struct declToImport {
        int a = d;
        union { 
          int b;
          int c;
        };
        int d;
      };
      )s",
      Lang_CXX11, "", Lang_CXX11);

  MatchVerifier<Decl> Verifier;
  ASSERT_TRUE(Verifier.match(
      From, cxxRecordDecl(hasFieldOrder({"a", "", "b", "c", "d"}))));
  EXPECT_TRUE(Verifier.match(
      To, cxxRecordDecl(hasFieldOrder({"a", "", "b", "c", "d"}))));
}

TEST_P(ASTImporterOptionSpecificTestBase, ShouldImportImplicitCXXRecordDecl) {
  Decl *From, *To;
  std::tie(From, To) = getImportedDecl(
      R"(
      struct declToImport {
      };
      )",
      Lang_CXX03, "", Lang_CXX03);

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
      Lang_CXX03, "", Lang_CXX03);

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
      Lang_CXX03, "", Lang_CXX03);

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
      getImportedDecl("void declToImport() {}", Lang_CXX03, "", Lang_CXX03);

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
      Lang_CXX03);
  Decl *From = LastDeclMatcher<Decl>{}.match(FromTU, functionDecl());
  const Decl *To = Import(From, Lang_CXX03);
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
      Lang_CXX03, "", Lang_CXX03);

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
        Lang_CXX03, "input0.cc");
    auto *FromD = FirstDeclMatcher<ClassTemplateDecl>().match(
        FromTU, classTemplateDecl(hasName("B")));

    Import(FromD, Lang_CXX03);
  }

  {
    Decl *FromTU = getTuDecl(
        R"(
            template <typename T>
            struct B {
              void f();
            };
            )",
        Lang_CXX03, "input1.cc");
    FunctionDecl *FromD = FirstDeclMatcher<FunctionDecl>().match(
        FromTU, functionDecl(hasName("f")));
    Import(FromD, Lang_CXX03);
    auto *FromCTD = FirstDeclMatcher<ClassTemplateDecl>().match(
        FromTU, classTemplateDecl(hasName("B")));
    auto *ToCTD = cast<ClassTemplateDecl>(Import(FromCTD, Lang_CXX03));
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
      Lang_CXX03);
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
      Lang_CXX03, "input1.cc");
  ClassTemplateDecl *FromD = FirstDeclMatcher<ClassTemplateDecl>().match(
      FromTU, classTemplateDecl(hasName("B")));

  Import(FromD, Lang_CXX03);

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
      Lang_CXX03);
  ASSERT_EQ(2u, DeclCounter<CXXRecordDecl>().match(
                    ToTU, cxxRecordDecl(unless(isImplicit()))));

  Decl *FromTU = getTuDecl(
      R"(
      struct B {
        void f();
      };
      )",
      Lang_CXX03, "input1.cc");
  auto *FromD = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("B")));

  Import(FromD, Lang_CXX03);

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
      Lang_CXX03);
  auto FromD = FirstDeclMatcher<FunctionDecl>().match(FromTU, functionDecl());
  auto ToD = Import(FromD, Lang_CXX03);

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
      Lang_CXX03);
  auto FromD = FirstDeclMatcher<FunctionDecl>().match(FromTU, functionDecl());
  auto ToD = Import(FromD, Lang_CXX03);

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
      Lang_CXX03);
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
      Lang_CXX03, "input1.cc");
  auto *FromD = FirstDeclMatcher<ClassTemplateSpecializationDecl>().match(
      FromTU, classTemplateSpecializationDecl(hasName("B")));

  Import(FromD, Lang_CXX03);

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
      Lang_CXX03, "input0.cc");

  auto *Obj0 =
      FirstDeclMatcher<VarDecl>().match(FromTU, varDecl(hasName("object0")));
  auto *From0 = getRecordDecl(Obj0);
  auto *Obj1 =
      FirstDeclMatcher<VarDecl>().match(FromTU, varDecl(hasName("object1")));
  auto *From1 = getRecordDecl(Obj1);

  auto *To0 = Import(From0, Lang_CXX03);
  auto *To1 = Import(From1, Lang_CXX03);

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
  Decl *FromTU0 = getTuDecl(Code, Lang_C99, "input0.c");

  Decl *FromTU1 = getTuDecl(Code, Lang_C99, "input1.c");

  auto *X0 =
      FirstDeclMatcher<RecordDecl>().match(FromTU0, recordDecl(hasName("X")));
  auto *X1 =
      FirstDeclMatcher<RecordDecl>().match(FromTU1, recordDecl(hasName("X")));
  Import(X0, Lang_C99);
  Import(X1, Lang_C99);

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
      Lang_C99, "input0.c");

  Decl *FromTU1 = getTuDecl(
      R"(
      struct X { // reversed order
        struct { int b; };
        struct { int a; };
      };
      )",
      Lang_C99, "input1.c");

  auto *X0 =
      FirstDeclMatcher<RecordDecl>().match(FromTU0, recordDecl(hasName("X")));
  auto *X1 =
      FirstDeclMatcher<RecordDecl>().match(FromTU1, recordDecl(hasName("X")));
  Import(X0, Lang_C99);
  Import(X1, Lang_C99);

  auto *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  // We expect one (ODR) warning during the import.
  EXPECT_EQ(1u, ToTU->getASTContext().getDiagnostics().getNumWarnings());
  EXPECT_EQ(1u,
            DeclCounter<RecordDecl>().match(ToTU, recordDecl(hasName("X"))));
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportDoesUpdateUsedFlag) {
  auto Pattern = varDecl(hasName("x"));
  VarDecl *Imported1;
  {
    Decl *FromTU = getTuDecl("extern int x;", Lang_CXX03, "input0.cc");
    auto *FromD = FirstDeclMatcher<VarDecl>().match(FromTU, Pattern);
    Imported1 = cast<VarDecl>(Import(FromD, Lang_CXX03));
  }
  VarDecl *Imported2;
  {
    Decl *FromTU = getTuDecl("int x;", Lang_CXX03, "input1.cc");
    auto *FromD = FirstDeclMatcher<VarDecl>().match(FromTU, Pattern);
    Imported2 = cast<VarDecl>(Import(FromD, Lang_CXX03));
  }
  EXPECT_EQ(Imported1->getCanonicalDecl(), Imported2->getCanonicalDecl());
  EXPECT_FALSE(Imported2->isUsed(false));
  {
    Decl *FromTU = getTuDecl("extern int x; int f() { return x; }", Lang_CXX03,
                             "input2.cc");
    auto *FromD = FirstDeclMatcher<FunctionDecl>().match(
        FromTU, functionDecl(hasName("f")));
    Import(FromD, Lang_CXX03);
  }
  EXPECT_TRUE(Imported2->isUsed(false));
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportDoesUpdateUsedFlag2) {
  auto Pattern = varDecl(hasName("x"));
  VarDecl *ExistingD;
  {
    Decl *ToTU = getToTuDecl("int x = 1;", Lang_CXX03);
    ExistingD = FirstDeclMatcher<VarDecl>().match(ToTU, Pattern);
  }
  EXPECT_FALSE(ExistingD->isUsed(false));
  {
    Decl *FromTU =
        getTuDecl("int x = 1; int f() { return x; }", Lang_CXX03, "input1.cc");
    auto *FromD = FirstDeclMatcher<FunctionDecl>().match(
        FromTU, functionDecl(hasName("f")));
    Import(FromD, Lang_CXX03);
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
        )",
        Lang_CXX03);
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
        )",
        Lang_CXX03, "input1.cc");
    auto *FromFunD = FirstDeclMatcher<FunctionDecl>().match(
        FromTU, functionDecl(hasName("f")));
    auto *FromD = FirstDeclMatcher<VarDecl>().match(FromTU, Pattern);
    ASSERT_TRUE(FromD->isUsed(false));
    Import(FromFunD, Lang_CXX03);
  }
  EXPECT_TRUE(ExistingD->isUsed(false));
}

TEST_P(ASTImporterOptionSpecificTestBase, ReimportWithUsedFlag) {
  auto Pattern = varDecl(hasName("x"));

  Decl *FromTU = getTuDecl("int x;", Lang_CXX03, "input0.cc");
  auto *FromD = FirstDeclMatcher<VarDecl>().match(FromTU, Pattern);

  auto *Imported1 = cast<VarDecl>(Import(FromD, Lang_CXX03));

  ASSERT_FALSE(Imported1->isUsed(false));

  FromD->setIsUsed();
  auto *Imported2 = cast<VarDecl>(Import(FromD, Lang_CXX03));

  EXPECT_EQ(Imported1, Imported2);
  EXPECT_TRUE(Imported2->isUsed(false));
}

struct ImportFunctions : ASTImporterOptionSpecificTestBase {};

TEST_P(ImportFunctions, ImportPrototypeOfRecursiveFunction) {
  Decl *FromTU = getTuDecl("void f(); void f() { f(); }", Lang_CXX03);
  auto Pattern = functionDecl(hasName("f"));
  auto *From =
      FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern); // Proto

  Decl *ImportedD = Import(From, Lang_CXX03);
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
  Decl *FromTU = getTuDecl("void f(); void f() { f(); }", Lang_CXX03);
  auto Pattern = functionDecl(hasName("f"));
  auto *From =
      LastDeclMatcher<FunctionDecl>().match(FromTU, Pattern); // Def

  Decl *ImportedD = Import(From, Lang_CXX03);
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
  Decl *FromTU = getTuDecl(Code, Lang_CXX03);
  CXXMethodDecl *Proto =
      FirstDeclMatcher<CXXMethodDecl>().match(FromTU, Pattern);

  ASSERT_EQ(Proto->size_overridden_methods(), 1u);
  CXXMethodDecl *To = cast<CXXMethodDecl>(Import(Proto, Lang_CXX03));
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
  Decl *FromTU = getTuDecl(Code, Lang_CXX03);
  CXXMethodDecl *Proto =
      FirstDeclMatcher<CXXMethodDecl>().match(FromTU, Pattern);
  CXXMethodDecl *Def = LastDeclMatcher<CXXMethodDecl>().match(FromTU, Pattern);

  ASSERT_TRUE(Proto->isVirtual());
  ASSERT_TRUE(Def->isVirtual());
  CXXMethodDecl *To = cast<CXXMethodDecl>(Import(Proto, Lang_CXX03));
  EXPECT_TRUE(To->isVirtual());
}

TEST_P(ImportFunctions,
       ImportDefinitionIfThereIsAnExistingDefinitionAndFwdDecl) {
  Decl *ToTU = getToTuDecl(
      R"(
      void f() {}
      void f();
      )",
      Lang_CXX03);
  ASSERT_EQ(1u,
            DeclCounterWithPredicate<FunctionDecl>([](const FunctionDecl *FD) {
              return FD->doesThisDeclarationHaveABody();
            }).match(ToTU, functionDecl()));

  Decl *FromTU = getTuDecl("void f() {}", Lang_CXX03, "input0.cc");
  auto *FromD = FirstDeclMatcher<FunctionDecl>().match(FromTU, functionDecl());

  Import(FromD, Lang_CXX03);

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

  Decl *FromTU0 = getTuDecl(Code, Lang_CXX03);
  auto *DF = FirstDeclMatcher<CXXMethodDecl>().match(FromTU0, DFP);
  Import(DF, Lang_CXX03);

  Decl *FromTU1 = getTuDecl(Code, Lang_CXX03, "input1.cc");
  auto *BF = FirstDeclMatcher<CXXMethodDecl>().match(FromTU1, BFP);
  Import(BF, Lang_CXX03);

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
    Decl *FromTU = getTuDecl(CodeWithDef, Lang_CXX03, "input0.cc");
    auto *FromD = FirstDeclMatcher<CXXMethodDecl>().match(FromTU, DFP);
    Import(FromD, Lang_CXX03);
  }
  {
    Decl *FromTU = getTuDecl(CodeWithoutDef, Lang_CXX03, "input1.cc");
    auto *FromB = FirstDeclMatcher<CXXMethodDecl>().match(FromTU, BFP);
    Import(FromB, Lang_CXX03);
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

  Decl *FromTU0 = getTuDecl(Code, Lang_CXX03);
  auto *D = FirstDeclMatcher<CXXMethodDecl>().match(FromTU0, DFP);
  Import(D, Lang_CXX03);

  Decl *FromTU1 = getTuDecl(Code, Lang_CXX03, "input1.cc");
  auto *B = FirstDeclMatcher<CXXMethodDecl>().match(FromTU1, BFP);
  Import(B, Lang_CXX03);

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
    Decl *FromTU0 = getTuDecl(CodeTU0, Lang_CXX03, "input0.cc");
    auto *D = FirstDeclMatcher<CXXMethodDecl>().match(FromTU0, DFP);
    Import(D, Lang_CXX03);
  }

  {
    Decl *FromTU1 = getTuDecl(CodeTU1, Lang_CXX03, "input1.cc");
    auto *Foo = FirstDeclMatcher<FunctionDecl>().match(FromTU1, FooDef);
    Import(Foo, Lang_CXX03);
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

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportVirtualOverriddenMethodOnALoop) {
  // B::f() calls => f1() ==> C ==> C::f()
  //     \
  //      \---- A::f()
  //
  // C::f()'s ImportOverriddenMethods() asserts B::isVirtual(), so B::f()'s
  // ImportOverriddenMethods() should be completed before B::f()'s body
  const char *Code =
      R"(
      void f1();
      class A {
        virtual void f(){}
      };
      class B: public A {
        void f() override {
          f1();
        }
      };
      class C: public B {
        void f() override {}
      };
      void f1() { C c; }
      )";
  Decl *FromTU = getTuDecl(Code, Lang_CXX11);

  auto *FromF = FirstDeclMatcher<CXXMethodDecl>().match(
      FromTU, cxxMethodDecl(hasName("B::f")));

  auto *ToBF = Import(FromF, Lang_CXX11);
  EXPECT_TRUE(ToBF->isVirtual());

  auto *ToCF = FirstDeclMatcher<CXXMethodDecl>().match(
      ToBF->getTranslationUnitDecl(), cxxMethodDecl(hasName("C::f")));
  EXPECT_TRUE(ToCF->isVirtual());
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportVariableChainInC) {
    std::string Code = "static int v; static int v = 0;";
    auto Pattern = varDecl(hasName("v"));

    TranslationUnitDecl *FromTu = getTuDecl(Code, Lang_C99, "input0.c");

    auto *From0 = FirstDeclMatcher<VarDecl>().match(FromTu, Pattern);
    auto *From1 = LastDeclMatcher<VarDecl>().match(FromTu, Pattern);

    auto *To0 = Import(From0, Lang_C99);
    auto *To1 = Import(From1, Lang_C99);

    EXPECT_TRUE(To0);
    ASSERT_TRUE(To1);
    EXPECT_NE(To0, To1);
    EXPECT_EQ(To1->getPreviousDecl(), To0);
}

TEST_P(ImportFunctions, ImportFromDifferentScopedAnonNamespace) {
  TranslationUnitDecl *FromTu =
      getTuDecl("namespace NS0 { namespace { void f(); } }"
                "namespace NS1 { namespace { void f(); } }",
                Lang_CXX03, "input0.cc");
  auto Pattern = functionDecl(hasName("f"));

  auto *FromF0 = FirstDeclMatcher<FunctionDecl>().match(FromTu, Pattern);
  auto *FromF1 = LastDeclMatcher<FunctionDecl>().match(FromTu, Pattern);

  auto *ToF0 = Import(FromF0, Lang_CXX03);
  auto *ToF1 = Import(FromF1, Lang_CXX03);

  EXPECT_TRUE(ToF0);
  ASSERT_TRUE(ToF1);
  EXPECT_NE(ToF0, ToF1);
  EXPECT_FALSE(ToF1->getPreviousDecl());
}

TEST_P(ImportFunctions, ImportFunctionFromUnnamedNamespace) {
  {
    Decl *FromTU = getTuDecl("namespace { void f() {} } void g0() { f(); }",
                             Lang_CXX03, "input0.cc");
    auto *FromD = FirstDeclMatcher<FunctionDecl>().match(
        FromTU, functionDecl(hasName("g0")));

    Import(FromD, Lang_CXX03);
  }
  {
    Decl *FromTU =
        getTuDecl("namespace { void f() { int a; } } void g1() { f(); }",
                  Lang_CXX03, "input1.cc");
    auto *FromD = FirstDeclMatcher<FunctionDecl>().match(
        FromTU, functionDecl(hasName("g1")));
    Import(FromD, Lang_CXX03);
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
  auto *ToD = Import(FromD, Lang_CXX03);
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
      Lang_CXX03);
  auto *FromD = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("f")));
  auto *ToD = Import(FromD, Lang_CXX03);
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
      Lang_CXX03);
  auto *FromD = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("g")));
  auto *ToD = Import(FromD, Lang_CXX03);
  EXPECT_TRUE(ToD);
  Decl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  EXPECT_TRUE(MatchVerifier<TranslationUnitDecl>().match(
      ToTU, translationUnitDecl(hasDescendant(
                functionDecl(hasName("f"), hasDescendant(declRefExpr()))))));
}

struct ImportFunctionTemplates : ASTImporterOptionSpecificTestBase {};

TEST_P(ImportFunctionTemplates, ImportFunctionTemplateInRecordDeclTwice) {
  auto Code =
      R"(
      class X {
        template <class T>
        void f(T t);
      };
      )";
  Decl *FromTU1 = getTuDecl(Code, Lang_CXX03, "input1.cc");
  auto *FromD1 = FirstDeclMatcher<FunctionTemplateDecl>().match(
      FromTU1, functionTemplateDecl(hasName("f")));
  auto *ToD1 = Import(FromD1, Lang_CXX03);
  Decl *FromTU2 = getTuDecl(Code, Lang_CXX03, "input2.cc");
  auto *FromD2 = FirstDeclMatcher<FunctionTemplateDecl>().match(
      FromTU2, functionTemplateDecl(hasName("f")));
  auto *ToD2 = Import(FromD2, Lang_CXX03);
  EXPECT_EQ(ToD1, ToD2);
}

TEST_P(ImportFunctionTemplates,
       ImportFunctionTemplateWithDefInRecordDeclTwice) {
  auto Code =
      R"(
      class X {
        template <class T>
        void f(T t);
      };
      template <class T>
      void X::f(T t) {};
      )";
  Decl *FromTU1 = getTuDecl(Code, Lang_CXX03, "input1.cc");
  auto *FromD1 = FirstDeclMatcher<FunctionTemplateDecl>().match(
      FromTU1, functionTemplateDecl(hasName("f")));
  auto *ToD1 = Import(FromD1, Lang_CXX03);
  Decl *FromTU2 = getTuDecl(Code, Lang_CXX03, "input2.cc");
  auto *FromD2 = FirstDeclMatcher<FunctionTemplateDecl>().match(
      FromTU2, functionTemplateDecl(hasName("f")));
  auto *ToD2 = Import(FromD2, Lang_CXX03);
  EXPECT_EQ(ToD1, ToD2);
}

TEST_P(ImportFunctionTemplates,
       ImportFunctionWhenThereIsAFunTemplateWithSameName) {
  getToTuDecl(
      R"(
      template <typename T>
      void foo(T) {}
      void foo();
      )",
      Lang_CXX03);
  Decl *FromTU = getTuDecl("void foo();", Lang_CXX03);
  auto *FromD = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("foo")));
  auto *ImportedD = Import(FromD, Lang_CXX03);
  EXPECT_TRUE(ImportedD);
}

TEST_P(ImportFunctionTemplates,
       ImportConstructorWhenThereIsAFunTemplateWithSameName) {
  auto Code =
      R"(
      struct Foo {
        template <typename T>
        Foo(T) {}
        Foo();
      };
      )";
  getToTuDecl(Code, Lang_CXX03);
  Decl *FromTU = getTuDecl(Code, Lang_CXX03);
  auto *FromD =
      LastDeclMatcher<CXXConstructorDecl>().match(FromTU, cxxConstructorDecl());
  auto *ImportedD = Import(FromD, Lang_CXX03);
  EXPECT_TRUE(ImportedD);
}

TEST_P(ImportFunctionTemplates,
       ImportOperatorWhenThereIsAFunTemplateWithSameName) {
  getToTuDecl(
      R"(
      template <typename T>
      void operator<(T,T) {}
      struct X{};
      void operator<(X, X);
      )",
      Lang_CXX03);
  Decl *FromTU = getTuDecl(
      R"(
      struct X{};
      void operator<(X, X);
      )",
      Lang_CXX03);
  auto *FromD = LastDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasOverloadedOperatorName("<")));
  auto *ImportedD = Import(FromD, Lang_CXX03);
  EXPECT_TRUE(ImportedD);
}

struct ImportFriendFunctions : ImportFunctions {};

TEST_P(ImportFriendFunctions, ImportFriendFunctionRedeclChainProto) {
  auto Pattern = functionDecl(hasName("f"));

  Decl *FromTU = getTuDecl("struct X { friend void f(); };"
                           "void f();",
                           Lang_CXX03, "input0.cc");
  auto *FromD = FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);

  auto *ImportedD = cast<FunctionDecl>(Import(FromD, Lang_CXX03));
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
                           Lang_CXX03, "input0.cc");
  auto FromD = FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);

  auto *ImportedD = cast<FunctionDecl>(Import(FromD, Lang_CXX03));
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
                           Lang_CXX03, "input0.cc");
  auto *FromD = FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);

  auto *ImportedD = cast<FunctionDecl>(Import(FromD, Lang_CXX03));
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
                           Lang_CXX03, "input0.cc");
  auto *FromD = FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);

  auto *ImportedD = cast<FunctionDecl>(Import(FromD, Lang_CXX03));
  Decl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  ASSERT_EQ(DeclCounter<FunctionDecl>().match(ToTU, Pattern), 2u);
  EXPECT_FALSE(ImportedD->doesThisDeclarationHaveABody());
  auto *ToFD = LastDeclMatcher<FunctionDecl>().match(ToTU, Pattern);
  EXPECT_TRUE(ToFD->doesThisDeclarationHaveABody());
  EXPECT_EQ(ToFD->getPreviousDecl(), ImportedD);
}

TEST_P(ImportFriendFunctions, ImportFriendFunctionRedeclChainDefWithClass) {
  auto Pattern = functionDecl(hasName("f"));

  Decl *FromTU = getTuDecl(
      R"(
        class X;
        void f(X *x){}
        class X{
        friend void f(X *x);
        };
      )",
      Lang_CXX03, "input0.cc");
  auto *FromD = FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);

  auto *ImportedD = cast<FunctionDecl>(Import(FromD, Lang_CXX03));
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

TEST_P(ImportFriendFunctions,
       ImportFriendFunctionRedeclChainDefWithClass_ImportTheProto) {
  auto Pattern = functionDecl(hasName("f"));

  Decl *FromTU = getTuDecl(
      R"(
        class X;
        void f(X *x){}
        class X{
        friend void f(X *x);
        };
      )",
      Lang_CXX03, "input0.cc");
  auto *FromD = LastDeclMatcher<FunctionDecl>().match(FromTU, Pattern);

  auto *ImportedD = cast<FunctionDecl>(Import(FromD, Lang_CXX03));
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
        getTuDecl("struct X { friend void f(){} };", Lang_CXX03, "input0.cc");
    auto *FromD = FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);
    ImportedD = cast<FunctionDecl>(Import(FromD, Lang_CXX03));
  }
  FunctionDecl *ImportedD1;
  {
    Decl *FromTU = getTuDecl("void f();", Lang_CXX03, "input1.cc");
    auto *FromD = FirstDeclMatcher<FunctionDecl>().match(FromTU, Pattern);
    ImportedD1 = cast<FunctionDecl>(Import(FromD, Lang_CXX03));
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
      getTuDecl("struct X { friend void f(); };", Lang_CXX03, "input0.cc");
  auto *FromD = FirstDeclMatcher<FunctionDecl>().match(FromTU, FunctionPattern);
  ASSERT_TRUE(FromD->isInIdentifierNamespace(Decl::IDNS_OrdinaryFriend));
  ASSERT_FALSE(FromD->isInIdentifierNamespace(Decl::IDNS_Ordinary));
  {
    auto FromName = FromD->getDeclName();
    auto *Class = FirstDeclMatcher<CXXRecordDecl>().match(FromTU, ClassPattern);
    auto LookupRes = Class->noload_lookup(FromName);
    ASSERT_TRUE(LookupRes.empty());
    LookupRes = FromTU->noload_lookup(FromName);
    ASSERT_TRUE(LookupRes.isSingleResult());
  }

  auto *ToD = cast<FunctionDecl>(Import(FromD, Lang_CXX03));
  auto ToName = ToD->getDeclName();

  TranslationUnitDecl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  auto *Class = FirstDeclMatcher<CXXRecordDecl>().match(ToTU, ClassPattern);
  auto LookupRes = Class->noload_lookup(ToName);
  EXPECT_TRUE(LookupRes.empty());
  LookupRes = ToTU->noload_lookup(ToName);
  EXPECT_TRUE(LookupRes.isSingleResult());

  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, FunctionPattern), 1u);
  auto *To0 = FirstDeclMatcher<FunctionDecl>().match(ToTU, FunctionPattern);
  EXPECT_TRUE(To0->isInIdentifierNamespace(Decl::IDNS_OrdinaryFriend));
  EXPECT_FALSE(To0->isInIdentifierNamespace(Decl::IDNS_Ordinary));
}

TEST_P(ImportFriendFunctions, LookupWithProtoAfter) {
  auto FunctionPattern = functionDecl(hasName("f"));
  auto ClassPattern = cxxRecordDecl(hasName("X"));

  TranslationUnitDecl *FromTU =
      getTuDecl("struct X { friend void f(); };"
                // This proto decl makes f available to normal
                // lookup, otherwise it is hidden.
                // Normal C++ lookup (implemented in
                // `clang::Sema::CppLookupName()` and in `LookupDirect()`)
                // returns the found `NamedDecl` only if the set IDNS is matched
                "void f();",
                Lang_CXX03, "input0.cc");
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
  ASSERT_TRUE(LookupRes.empty());
  LookupRes = FromTU->noload_lookup(FromName);
  ASSERT_TRUE(LookupRes.isSingleResult());

  auto *ToFriend = cast<FunctionDecl>(Import(FromFriend, Lang_CXX03));
  auto ToName = ToFriend->getDeclName();

  TranslationUnitDecl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  auto *ToClass = FirstDeclMatcher<CXXRecordDecl>().match(ToTU, ClassPattern);
  LookupRes = ToClass->noload_lookup(ToName);
  EXPECT_TRUE(LookupRes.empty());
  LookupRes = ToTU->noload_lookup(ToName);
  // Test is disabled because this result is 2.
  EXPECT_TRUE(LookupRes.isSingleResult());

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

  TranslationUnitDecl *FromTU = getTuDecl("void f();"
                                          "struct X { friend void f(); };",
                                          Lang_CXX03, "input0.cc");
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
  ASSERT_TRUE(LookupRes.empty());
  LookupRes = FromTU->noload_lookup(FromName);
  ASSERT_TRUE(LookupRes.isSingleResult());

  auto *ToNormal = cast<FunctionDecl>(Import(FromNormal, Lang_CXX03));
  auto ToName = ToNormal->getDeclName();
  TranslationUnitDecl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();

  auto *ToClass = FirstDeclMatcher<CXXRecordDecl>().match(ToTU, ClassPattern);
  LookupRes = ToClass->noload_lookup(ToName);
  EXPECT_TRUE(LookupRes.empty());
  LookupRes = ToTU->noload_lookup(ToName);
  EXPECT_TRUE(LookupRes.isSingleResult());

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
      getTuDecl("void f();", Lang_CXX03, "input0.cc");
  auto *FromNormalF =
      FirstDeclMatcher<FunctionDecl>().match(FromNormalTU, Pattern);
  TranslationUnitDecl *FromFriendTU =
      getTuDecl("class X { friend void f(); };", Lang_CXX03, "input1.cc");
  auto *FromFriendF =
      FirstDeclMatcher<FunctionDecl>().match(FromFriendTU, Pattern);
  auto FromNormalName = FromNormalF->getDeclName();
  auto FromFriendName = FromFriendF->getDeclName();

  ASSERT_TRUE(FromNormalF->isInIdentifierNamespace(Decl::IDNS_Ordinary));
  ASSERT_FALSE(FromNormalF->isInIdentifierNamespace(Decl::IDNS_OrdinaryFriend));
  ASSERT_FALSE(FromFriendF->isInIdentifierNamespace(Decl::IDNS_Ordinary));
  ASSERT_TRUE(FromFriendF->isInIdentifierNamespace(Decl::IDNS_OrdinaryFriend));
  auto LookupRes = FromNormalTU->noload_lookup(FromNormalName);
  ASSERT_TRUE(LookupRes.isSingleResult());
  LookupRes = FromFriendTU->noload_lookup(FromFriendName);
  ASSERT_TRUE(LookupRes.isSingleResult());

  auto *ToNormalF = cast<FunctionDecl>(Import(FromNormalF, Lang_CXX03));
  TranslationUnitDecl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  auto ToName = ToNormalF->getDeclName();
  EXPECT_TRUE(ToNormalF->isInIdentifierNamespace(Decl::IDNS_Ordinary));
  EXPECT_FALSE(ToNormalF->isInIdentifierNamespace(Decl::IDNS_OrdinaryFriend));
  LookupRes = ToTU->noload_lookup(ToName);
  EXPECT_TRUE(LookupRes.isSingleResult());
  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, Pattern), 1u);

  auto *ToFriendF = cast<FunctionDecl>(Import(FromFriendF, Lang_CXX03));
  LookupRes = ToTU->noload_lookup(ToName);
  EXPECT_TRUE(LookupRes.isSingleResult());
  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, Pattern), 2u);

  EXPECT_TRUE(ToNormalF->isInIdentifierNamespace(Decl::IDNS_Ordinary));
  EXPECT_FALSE(ToNormalF->isInIdentifierNamespace(Decl::IDNS_OrdinaryFriend));

  EXPECT_TRUE(ToFriendF->isInIdentifierNamespace(Decl::IDNS_Ordinary));
  EXPECT_TRUE(ToFriendF->isInIdentifierNamespace(Decl::IDNS_OrdinaryFriend));
}

TEST_P(ImportFriendFunctions, ImportFriendList) {
  TranslationUnitDecl *FromTU = getTuDecl("struct X { friend void f(); };"
                                          "void f();",
                                          Lang_CXX03, "input0.cc");
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

  Import(FromFriendF, Lang_CXX03);
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
                      Lang_C99}},

                    {"foo.c",
                     {"typedef enum { THING_VALUE } thing_t;"
                      "void conflict(thing_t type);"
                      "void foo() { (void)THING_VALUE; }"
                      "void conflict(thing_t type) {}",
                      Lang_C99}},

                    {"moo.c",
                     {"typedef enum { THING_VALUE } thing_t;"
                      "void conflict(thing_t type);"
                      "void moo() { conflict(THING_VALUE); }",
                      Lang_C99}}};

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

TEST_P(ImportDecl, ImportFieldOrder) {
  MatchVerifier<Decl> Verifier;
  testImport("struct declToImport {"
             "  int b = a + 2;"
             "  int a = 5;"
             "};",
             Lang_CXX11, "", Lang_CXX11, Verifier,
             recordDecl(hasFieldOrder({"b", "a"})));
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
      SharedStatePtr->getLookupTable()->remove(Method);
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
    Decl *FromTU = getTuDecl("struct A { };", Lang_CXX03, "input0.cc");
    auto *FromR = FirstDeclMatcher<CXXRecordDecl>().match(
        FromTU, cxxRecordDecl(hasName("A")));

    ToR1 = Import(FromR, Lang_CXX03);
  }

  Decl *ToR2;
  {
    Decl *FromTU = getTuDecl("struct A { };", Lang_CXX03, "input1.cc");
    auto *FromR = FirstDeclMatcher<CXXRecordDecl>().match(
        FromTU, cxxRecordDecl(hasName("A")));

    ToR2 = Import(FromR, Lang_CXX03);
  }

  EXPECT_EQ(ToR1, ToR2);
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportOfNonEquivalentRecord) {
  Decl *ToR1;
  {
    Decl *FromTU = getTuDecl("struct A { int x; };", Lang_CXX03, "input0.cc");
    auto *FromR = FirstDeclMatcher<CXXRecordDecl>().match(
        FromTU, cxxRecordDecl(hasName("A")));
    ToR1 = Import(FromR, Lang_CXX03);
  }
  Decl *ToR2;
  {
    Decl *FromTU =
        getTuDecl("struct A { unsigned x; };", Lang_CXX03, "input1.cc");
    auto *FromR = FirstDeclMatcher<CXXRecordDecl>().match(
        FromTU, cxxRecordDecl(hasName("A")));
    ToR2 = Import(FromR, Lang_CXX03);
  }
  EXPECT_NE(ToR1, ToR2);
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportOfEquivalentField) {
  Decl *ToF1;
  {
    Decl *FromTU = getTuDecl("struct A { int x; };", Lang_CXX03, "input0.cc");
    auto *FromF = FirstDeclMatcher<FieldDecl>().match(
        FromTU, fieldDecl(hasName("x")));
    ToF1 = Import(FromF, Lang_CXX03);
  }
  Decl *ToF2;
  {
    Decl *FromTU = getTuDecl("struct A { int x; };", Lang_CXX03, "input1.cc");
    auto *FromF = FirstDeclMatcher<FieldDecl>().match(
        FromTU, fieldDecl(hasName("x")));
    ToF2 = Import(FromF, Lang_CXX03);
  }
  EXPECT_EQ(ToF1, ToF2);
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportBitfields) {
  Decl *FromTU = getTuDecl("struct A { unsigned x : 3; };", Lang_CXX03);
  auto *FromF =
      FirstDeclMatcher<FieldDecl>().match(FromTU, fieldDecl(hasName("x")));

  ASSERT_TRUE(FromF->isBitField());
  ASSERT_EQ(3u, FromF->getBitWidthValue(FromTU->getASTContext()));
  auto *ToField = Import(FromF, Lang_CXX03);
  auto *ToTU = ToField->getTranslationUnitDecl();

  EXPECT_TRUE(ToField->isBitField());
  EXPECT_EQ(3u, ToField->getBitWidthValue(ToTU->getASTContext()));

  const auto *FromBT = FromF->getBitWidth()->getType()->getAs<BuiltinType>();
  const auto *ToBT = ToField->getBitWidth()->getType()->getAs<BuiltinType>();
  ASSERT_TRUE(FromBT);
  ASSERT_EQ(BuiltinType::Int, FromBT->getKind());
  EXPECT_TRUE(ToBT);
  EXPECT_EQ(BuiltinType::Int, ToBT->getKind());
}

struct ImportBlock : ASTImporterOptionSpecificTestBase {};
TEST_P(ImportBlock, ImportBlocksAreUnsupported) {
  const auto *Code = R"(
    void test_block__capture_null() {
      int *p = 0;
      ^(){
        *p = 1;
      }();
    })";
  Decl *FromTU = getTuDecl(Code, Lang_CXX03);
  auto *FromBlock = FirstDeclMatcher<BlockDecl>().match(FromTU, blockDecl());
  ASSERT_TRUE(FromBlock);

  auto ToBlockOrError = importOrError(FromBlock, Lang_CXX03);

  const auto ExpectUnsupportedConstructError = [](const ASTImportError &Error) {
    EXPECT_EQ(ASTImportError::UnsupportedConstruct, Error.Error);
  };
  llvm::handleAllErrors(ToBlockOrError.takeError(),
                        ExpectUnsupportedConstructError);
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportParmVarDecl) {
  const auto *Code = R"(
    template <typename T> struct Wrapper {
      Wrapper(T Value = {}) {}
    };
    template class Wrapper<int>;
    )";
  Decl *FromTU = getTuDecl(Code, Lang_CXX11);
  auto *FromVar = FirstDeclMatcher<ParmVarDecl>().match(
      FromTU, parmVarDecl(hasType(asString("int"))));
  ASSERT_TRUE(FromVar);
  ASSERT_TRUE(FromVar->hasUninstantiatedDefaultArg());
  ASSERT_TRUE(FromVar->getUninstantiatedDefaultArg());

  const auto *ToVar = Import(FromVar, Lang_CXX11);
  EXPECT_TRUE(ToVar);
  EXPECT_TRUE(ToVar->hasUninstantiatedDefaultArg());
  EXPECT_TRUE(ToVar->getUninstantiatedDefaultArg());
  EXPECT_NE(FromVar->getUninstantiatedDefaultArg(),
            ToVar->getUninstantiatedDefaultArg());
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportOfNonEquivalentField) {
  Decl *ToF1;
  {
    Decl *FromTU = getTuDecl("struct A { int x; };", Lang_CXX03, "input0.cc");
    auto *FromF = FirstDeclMatcher<FieldDecl>().match(
        FromTU, fieldDecl(hasName("x")));
    ToF1 = Import(FromF, Lang_CXX03);
  }
  Decl *ToF2;
  {
    Decl *FromTU =
        getTuDecl("struct A { unsigned x; };", Lang_CXX03, "input1.cc");
    auto *FromF = FirstDeclMatcher<FieldDecl>().match(
        FromTU, fieldDecl(hasName("x")));
    ToF2 = Import(FromF, Lang_CXX03);
  }
  EXPECT_NE(ToF1, ToF2);
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportOfEquivalentMethod) {
  Decl *ToM1;
  {
    Decl *FromTU = getTuDecl("struct A { void x(); }; void A::x() { }",
                             Lang_CXX03, "input0.cc");
    auto *FromM = FirstDeclMatcher<FunctionDecl>().match(
        FromTU, functionDecl(hasName("x"), isDefinition()));
    ToM1 = Import(FromM, Lang_CXX03);
  }
  Decl *ToM2;
  {
    Decl *FromTU = getTuDecl("struct A { void x(); }; void A::x() { }",
                             Lang_CXX03, "input1.cc");
    auto *FromM = FirstDeclMatcher<FunctionDecl>().match(
        FromTU, functionDecl(hasName("x"), isDefinition()));
    ToM2 = Import(FromM, Lang_CXX03);
  }
  EXPECT_EQ(ToM1, ToM2);
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportOfNonEquivalentMethod) {
  Decl *ToM1;
  {
    Decl *FromTU = getTuDecl("struct A { void x(); }; void A::x() { }",
                             Lang_CXX03, "input0.cc");
    auto *FromM = FirstDeclMatcher<FunctionDecl>().match(
        FromTU, functionDecl(hasName("x"), isDefinition()));
    ToM1 = Import(FromM, Lang_CXX03);
  }
  Decl *ToM2;
  {
    Decl *FromTU =
        getTuDecl("struct A { void x() const; }; void A::x() const { }",
                  Lang_CXX03, "input1.cc");
    auto *FromM = FirstDeclMatcher<FunctionDecl>().match(
        FromTU, functionDecl(hasName("x"), isDefinition()));
    ToM2 = Import(FromM, Lang_CXX03);
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
      Lang_C99, "input0.cc");
  auto *From =
      FirstDeclMatcher<RecordDecl>().match(FromTU, recordDecl(hasName("A")));

  Import(From, Lang_C99);

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
    std::optional<unsigned> ToIndex = ASTImporter::getFieldIndex(ToField);
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

TEST_P(ASTImporterOptionSpecificTestBase, MergeTemplateSpecWithForwardDecl) {
  std::string ClassTemplate =
      R"(
      template<typename T>
      struct X { int m; };
      template<>
      struct X<int> { int m; };
      )";
  // Append a forward decl for our template specialization.
  getToTuDecl(ClassTemplate + "template<> struct X<int>;", Lang_CXX11);
  Decl *FromTU = getTuDecl(ClassTemplate, Lang_CXX11);
  auto *FromSpec = FirstDeclMatcher<ClassTemplateSpecializationDecl>().match(
      FromTU, classTemplateSpecializationDecl(hasName("X"), isDefinition()));
  auto *ImportedSpec = Import(FromSpec, Lang_CXX11);
  // Check that our definition got merged with the existing definition.
  EXPECT_TRUE(FromSpec->isThisDeclarationADefinition());
  EXPECT_TRUE(ImportedSpec->isThisDeclarationADefinition());
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
  EXPECT_FALSE(ImportedSpec);
  EXPECT_EQ(1u,
            DeclCounter<ClassTemplateSpecializationDecl>().match(
                ToTU, classTemplateSpecializationDecl(hasName("X"))));
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

TEST_P(ASTImporterOptionSpecificTestBase, ClassTemplateFriendDecl) {
  const auto *Code =
      R"(
      template <class T> class X {  friend T; };
      struct Y {};
      template class X<Y>;
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
      )",
      Lang_CXX03, "input1.cc");

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
  Decl *ToTU = getToTuDecl(StructA, Lang_CXX03);
  Decl *FromTU = getTuDecl(std::string(StructA) + "const int A::a;", Lang_CXX03,
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
  Decl *ToTU = getToTuDecl(StructA, Lang_CXX03);
  Decl *FromTU = getTuDecl(std::string(StructA) + "const int A::a = 1 + 2;",
                           Lang_CXX03, "input1.cc");

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

TEST_P(ImportVariables, ImportBindingDecl) {
  Decl *From, *To;
  std::tie(From, To) = getImportedDecl(
      R"(
      void declToImport() {
        int a[2] = {1,2};
        auto [x1,y1] = a;
        auto& [x2,y2] = a;
        
        struct S {
          mutable int x1 : 2;
          volatile double y1;
        };
        S b;
        const auto [x3, y3] = b;
      };
      )",
      Lang_CXX17, "", Lang_CXX17);

  TranslationUnitDecl *FromTU = From->getTranslationUnitDecl();
  auto *FromF = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("declToImport")));
  auto *ToF = Import(FromF, Lang_CXX17);
  EXPECT_TRUE(ToF);

  auto VerifyImport = [&](llvm::StringRef BindName) {
    auto *FromB = FirstDeclMatcher<BindingDecl>().match(
        FromF, bindingDecl(hasName(BindName)));
    ASSERT_TRUE(FromB);
    auto *ToB = Import(FromB, Lang_CXX17);
    EXPECT_TRUE(ToB);
    EXPECT_EQ(FromB->getBinding() != nullptr, ToB->getBinding() != nullptr);
    EXPECT_EQ(FromB->getDecomposedDecl() != nullptr,
              ToB->getDecomposedDecl() != nullptr);
    EXPECT_EQ(FromB->getHoldingVar() != nullptr,
              ToB->getHoldingVar() != nullptr);
  };

  VerifyImport("x1");
  VerifyImport("y1");
  VerifyImport("x2");
  VerifyImport("y2");
  VerifyImport("x3");
  VerifyImport("y3");
}

TEST_P(ImportVariables, ImportDecompositionDeclArray) {
  Decl *From, *To;
  std::tie(From, To) = getImportedDecl(
      R"(
      void declToImport() {
        int a[2] = {1,2};
        auto [x1,y1] = a;
      };
      )",
      Lang_CXX17, "", Lang_CXX17);

  TranslationUnitDecl *FromTU = From->getTranslationUnitDecl();
  auto *FromDecomp =
      FirstDeclMatcher<DecompositionDecl>().match(FromTU, decompositionDecl());
  auto *ToDecomp = Import(FromDecomp, Lang_CXX17);
  EXPECT_TRUE(ToDecomp);

  ArrayRef<BindingDecl *> FromB = FromDecomp->bindings();
  ArrayRef<BindingDecl *> ToB = ToDecomp->bindings();
  EXPECT_EQ(FromB.size(), ToB.size());
  for (unsigned int I = 0; I < FromB.size(); ++I) {
    auto *ToBI = Import(FromB[I], Lang_CXX17);
    EXPECT_EQ(ToBI, ToB[I]);
  }
}

struct ImportClasses : ASTImporterOptionSpecificTestBase {};

TEST_P(ImportClasses, ImportDefinitionWhenProtoIsInNestedToContext) {
  Decl *ToTU = getToTuDecl("struct A { struct X *Xp; };", Lang_C99);
  Decl *FromTU1 = getTuDecl("struct X {};", Lang_C99, "input1.cc");
  auto Pattern = recordDecl(hasName("X"), unless(isImplicit()));
  auto ToProto = FirstDeclMatcher<RecordDecl>().match(ToTU, Pattern);
  auto FromDef = FirstDeclMatcher<RecordDecl>().match(FromTU1, Pattern);

  Decl *ImportedDef = Import(FromDef, Lang_C99);

  EXPECT_NE(ImportedDef, ToProto);
  EXPECT_EQ(DeclCounter<RecordDecl>().match(ToTU, Pattern), 2u);
  auto ToDef = LastDeclMatcher<RecordDecl>().match(ToTU, Pattern);
  EXPECT_TRUE(ImportedDef == ToDef);
  EXPECT_TRUE(ToDef->isThisDeclarationADefinition());
  EXPECT_FALSE(ToProto->isThisDeclarationADefinition());
  EXPECT_EQ(ToDef->getPreviousDecl(), ToProto);
}

TEST_P(ImportClasses, ImportDefinitionWhenProtoIsInNestedToContextCXX) {
  Decl *ToTU = getToTuDecl("struct A { struct X *Xp; };", Lang_CXX03);
  Decl *FromTU1 = getTuDecl("struct X {};", Lang_CXX03, "input1.cc");
  auto Pattern = recordDecl(hasName("X"), unless(isImplicit()));
  auto ToProto = FirstDeclMatcher<RecordDecl>().match(ToTU, Pattern);
  auto FromDef = FirstDeclMatcher<RecordDecl>().match(FromTU1, Pattern);

  Decl *ImportedDef = Import(FromDef, Lang_CXX03);

  EXPECT_NE(ImportedDef, ToProto);
  EXPECT_EQ(DeclCounter<RecordDecl>().match(ToTU, Pattern), 2u);
  auto ToDef = LastDeclMatcher<RecordDecl>().match(ToTU, Pattern);
  EXPECT_TRUE(ImportedDef == ToDef);
  EXPECT_TRUE(ToDef->isThisDeclarationADefinition());
  EXPECT_FALSE(ToProto->isThisDeclarationADefinition());
  EXPECT_EQ(ToDef->getPreviousDecl(), ToProto);
}

TEST_P(ImportClasses, ImportNestedPrototypeThenDefinition) {
  Decl *FromTU0 =
      getTuDecl("struct A { struct X *Xp; };", Lang_C99, "input0.cc");
  Decl *FromTU1 = getTuDecl("struct X {};", Lang_C99, "input1.cc");
  auto Pattern = recordDecl(hasName("X"), unless(isImplicit()));
  auto FromProto = FirstDeclMatcher<RecordDecl>().match(FromTU0, Pattern);
  auto FromDef = FirstDeclMatcher<RecordDecl>().match(FromTU1, Pattern);

  Decl *ImportedProto = Import(FromProto, Lang_C99);
  Decl *ImportedDef = Import(FromDef, Lang_C99);
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

struct ImportFriendClasses : ASTImporterOptionSpecificTestBase {
  void testRecursiveFriendClassTemplate(Decl *FromTu) {
    auto *FromD = FirstDeclMatcher<ClassTemplateDecl>().match(
        FromTu, classTemplateDecl());

    auto Pattern = classTemplateDecl(
        has(cxxRecordDecl(has(friendDecl(has(classTemplateDecl()))))));
    ASSERT_TRUE(MatchVerifier<Decl>{}.match(FromD, Pattern));

    auto *FromFriend =
        FirstDeclMatcher<FriendDecl>().match(FromD, friendDecl());
    auto *FromRecordOfFriend =
        cast<ClassTemplateDecl>(FromFriend->getFriendDecl())
            ->getTemplatedDecl();
    EXPECT_NE(FromRecordOfFriend, FromD->getTemplatedDecl());
    EXPECT_TRUE(FromRecordOfFriend->getPreviousDecl() == nullptr);

    auto *FromDC = FromRecordOfFriend->getDeclContext();
    auto *FromLexicalDC = FromRecordOfFriend->getLexicalDeclContext();
    ASSERT_EQ(FromDC, cast<DeclContext>(FromTu));
    ASSERT_EQ(FromLexicalDC, cast<DeclContext>(FromD->getTemplatedDecl()));

    ASSERT_FALSE(FromDC->containsDecl(FromRecordOfFriend));
    ASSERT_FALSE(FromLexicalDC->containsDecl(FromRecordOfFriend));
    ASSERT_FALSE(cast<RecordDecl>(FromRecordOfFriend)
                     ->getLookupParent()
                     ->lookup(FromRecordOfFriend->getDeclName())
                     .empty());

    auto *ToD = Import(FromD, Lang_CXX03);
    EXPECT_TRUE(MatchVerifier<Decl>{}.match(ToD, Pattern));

    auto *ToFriend = FirstDeclMatcher<FriendDecl>().match(ToD, friendDecl());
    auto *ToRecordOfFriend =
        cast<ClassTemplateDecl>(ToFriend->getFriendDecl())->getTemplatedDecl();

    EXPECT_NE(ToRecordOfFriend, ToD->getTemplatedDecl());
    EXPECT_TRUE(ToRecordOfFriend->getPreviousDecl() == nullptr);

    auto *ToDC = ToRecordOfFriend->getDeclContext();
    auto *ToLexicalDC = ToRecordOfFriend->getLexicalDeclContext();
    ASSERT_EQ(ToDC, cast<DeclContext>(ToD->getTranslationUnitDecl()));
    ASSERT_EQ(ToLexicalDC, cast<DeclContext>(ToD->getTemplatedDecl()));

    ASSERT_FALSE(ToDC->containsDecl(ToRecordOfFriend));
    ASSERT_FALSE(ToLexicalDC->containsDecl(ToRecordOfFriend));
    ASSERT_FALSE(cast<RecordDecl>(ToRecordOfFriend)
                     ->getLookupParent()
                     ->lookup(ToRecordOfFriend->getDeclName())
                     .empty());
  }

  void testRepeatedFriendImport(const char *Code) {
    Decl *ToTu = getToTuDecl(Code, Lang_CXX03);
    Decl *FromTu = getTuDecl(Code, Lang_CXX03, "from.cc");

    auto *ToFriend1 = FirstDeclMatcher<FriendDecl>().match(ToTu, friendDecl());
    auto *ToFriend2 = LastDeclMatcher<FriendDecl>().match(ToTu, friendDecl());
    auto *FromFriend1 =
        FirstDeclMatcher<FriendDecl>().match(FromTu, friendDecl());
    auto *FromFriend2 =
        LastDeclMatcher<FriendDecl>().match(FromTu, friendDecl());

    FriendDecl *ToImportedFriend1 = Import(FromFriend1, Lang_CXX03);
    FriendDecl *ToImportedFriend2 = Import(FromFriend2, Lang_CXX03);

    EXPECT_NE(ToImportedFriend1, ToImportedFriend2);
    EXPECT_EQ(ToFriend1, ToImportedFriend1);
    EXPECT_EQ(ToFriend2, ToImportedFriend2);
  }
};

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
      Lang_CXX03, "input0.cc");

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

  auto *ToClass = cast<CXXRecordDecl>(Import(FromClass, Lang_CXX03));
  auto *ToFriendClass =
      cast<CXXRecordDecl>(Import(FromFriendClass, Lang_CXX03));

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
      Lang_CXX03, "input.cc");

  auto *FromD = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTu, cxxRecordDecl(hasName("declToImport")));
  auto *ToD = Import(FromD, Lang_CXX03);
  auto Pattern = cxxRecordDecl(has(friendDecl()));
  ASSERT_TRUE(MatchVerifier<Decl>{}.match(FromD, Pattern));
  EXPECT_TRUE(MatchVerifier<Decl>{}.match(ToD, Pattern));
}

TEST_P(ImportFriendClasses, UndeclaredFriendClassShouldNotBeVisible) {
  Decl *FromTu =
      getTuDecl("class X { friend class Y; };", Lang_CXX03, "from.cc");
  auto *FromX = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTu, cxxRecordDecl(hasName("X")));
  auto *FromFriend = FirstDeclMatcher<FriendDecl>().match(FromTu, friendDecl());
  RecordDecl *FromRecordOfFriend =
      const_cast<RecordDecl *>(getRecordDeclOfFriend(FromFriend));

  ASSERT_EQ(FromRecordOfFriend->getDeclContext(), cast<DeclContext>(FromTu));
  ASSERT_EQ(FromRecordOfFriend->getLexicalDeclContext(),
            cast<DeclContext>(FromX));
  ASSERT_FALSE(
      FromRecordOfFriend->getDeclContext()->containsDecl(FromRecordOfFriend));
  ASSERT_FALSE(FromRecordOfFriend->getLexicalDeclContext()->containsDecl(
      FromRecordOfFriend));
  ASSERT_FALSE(FromRecordOfFriend->getLookupParent()
                   ->lookup(FromRecordOfFriend->getDeclName())
                   .empty());

  auto *ToX = Import(FromX, Lang_CXX03);
  ASSERT_TRUE(ToX);

  Decl *ToTu = ToX->getTranslationUnitDecl();
  auto *ToFriend = FirstDeclMatcher<FriendDecl>().match(ToTu, friendDecl());
  RecordDecl *ToRecordOfFriend =
      const_cast<RecordDecl *>(getRecordDeclOfFriend(ToFriend));

  ASSERT_EQ(ToRecordOfFriend->getDeclContext(), cast<DeclContext>(ToTu));
  ASSERT_EQ(ToRecordOfFriend->getLexicalDeclContext(), cast<DeclContext>(ToX));
  EXPECT_FALSE(
      ToRecordOfFriend->getDeclContext()->containsDecl(ToRecordOfFriend));
  EXPECT_FALSE(ToRecordOfFriend->getLexicalDeclContext()->containsDecl(
      ToRecordOfFriend));
  EXPECT_FALSE(ToRecordOfFriend->getLookupParent()
                   ->lookup(ToRecordOfFriend->getDeclName())
                   .empty());
}

TEST_P(ImportFriendClasses, ImportOfRecursiveFriendClassTemplate) {
  Decl *FromTu = getTuDecl(
      R"(
      template<class A> class declToImport {
        template<class A1> friend class declToImport;
      };
      )",
      Lang_CXX03, "input.cc");

  testRecursiveFriendClassTemplate(FromTu);
}

TEST_P(ImportFriendClasses,
       ImportOfRecursiveFriendClassTemplateWithNonTypeParm) {
  Decl *FromTu = getTuDecl(
      R"(
      template<class A1, A1 A> class declToImport {
        template<class B1, B1> friend class declToImport;
      };
      )",
      Lang_CXX03, "input.cc");
  testRecursiveFriendClassTemplate(FromTu);
}

TEST_P(ImportFriendClasses, ProperPrevDeclForClassTemplateDecls) {
  auto Pattern = classTemplateSpecializationDecl(hasName("X"));

  ClassTemplateSpecializationDecl *Imported1;
  {
    Decl *FromTU = getTuDecl("template<class T> class X;"
                             "struct Y { friend class X<int>; };",
                             Lang_CXX03, "input0.cc");
    auto *FromD = FirstDeclMatcher<ClassTemplateSpecializationDecl>().match(
        FromTU, Pattern);

    Imported1 =
        cast<ClassTemplateSpecializationDecl>(Import(FromD, Lang_CXX03));
  }
  ClassTemplateSpecializationDecl *Imported2;
  {
    Decl *FromTU = getTuDecl("template<class T> class X;"
                             "template<> class X<int>{};"
                             "struct Z { friend class X<int>; };",
                             Lang_CXX03, "input1.cc");
    auto *FromD = FirstDeclMatcher<ClassTemplateSpecializationDecl>().match(
        FromTU, Pattern);

    Imported2 =
        cast<ClassTemplateSpecializationDecl>(Import(FromD, Lang_CXX03));
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
      Lang_CXX03, "input0.cc");
  auto *Fwd = FirstDeclMatcher<ClassTemplateDecl>().match(
      FromTU0, classTemplateDecl(hasName("F")));
  auto *Imported0 = cast<ClassTemplateDecl>(Import(Fwd, Lang_CXX03));
  Decl *FromTU1 = getTuDecl(
      R"(
      template <typename T>
      class F {};
      )",
      Lang_CXX03, "input1.cc");
  auto *Definition = FirstDeclMatcher<ClassTemplateDecl>().match(
      FromTU1, classTemplateDecl(hasName("F")));
  auto *Imported1 = cast<ClassTemplateDecl>(Import(Definition, Lang_CXX03));
  EXPECT_EQ(Imported0->getTemplatedDecl()->getTypeForDecl(),
            Imported1->getTemplatedDecl()->getTypeForDecl());
}

TEST_P(ImportFriendClasses, DeclsFromFriendsShouldBeInRedeclChains) {
  Decl *From, *To;
  std::tie(From, To) =
      getImportedDecl("class declToImport {};", Lang_CXX03,
                      "class Y { friend class declToImport; };", Lang_CXX03);
  auto *Imported = cast<CXXRecordDecl>(To);

  EXPECT_TRUE(Imported->getPreviousDecl());
}

TEST_P(ImportFriendClasses, SkipComparingFriendTemplateDepth) {
  Decl *ToTU = getToTuDecl(
      R"(
      template <class T, T U>
      class A;

      template <class T, T U>
      class A {
      public:
        template <class P, P Q>
        friend class A;

        A(T x)  :x(x) {}

      private:
        T x;
      };
      )",
      Lang_CXX11);

  auto *Fwd = FirstDeclMatcher<ClassTemplateDecl>().match(
      ToTU,
      classTemplateDecl(has(cxxRecordDecl(hasDefinition(), hasName("A")))));
  Decl *FromTU = getTuDecl(
      R"(
      template <class T, T U>
      class A;

      template <class T, T U>
      class A {
      public:
        template <class P, P Q>
        friend class A;

        A(T x) : x(x) {}

      private:
        T x;
      };

      A<int,3> a1(0);
      )",
      Lang_CXX11, "input1.cc");
  auto *FromA = FirstDeclMatcher<ClassTemplateDecl>().match(
      FromTU,
      classTemplateDecl(has(cxxRecordDecl(hasDefinition(), hasName("A")))));
  auto *ToA = Import(FromA, Lang_CXX11);
  EXPECT_TRUE(ToA);
  EXPECT_EQ(Fwd->getTemplatedDecl()->getTypeForDecl(),
            ToA->getTemplatedDecl()->getTypeForDecl());
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
      Lang_CXX03);
  auto *ToDecl = FirstDeclMatcher<ClassTemplateDecl>().match(
      ToTU, classTemplateDecl(hasName("F")));
  Decl *FromTU = getTuDecl(
      R"(
      template <typename T>
      class F {};
      )",
      Lang_CXX03, "input0.cc");
  auto *Definition = FirstDeclMatcher<ClassTemplateDecl>().match(
      FromTU, classTemplateDecl(hasName("F")));
  auto *ImportedDef = cast<ClassTemplateDecl>(Import(Definition, Lang_CXX03));
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
      Lang_CXX03, "input0.cc");
  auto *Fwd = FirstDeclMatcher<ClassTemplateDecl>().match(
      FromTU0, classTemplateDecl(hasName("F")));
  auto *ImportedFwd = cast<ClassTemplateDecl>(Import(Fwd, Lang_CXX03));
  Decl *FromTU1 = getTuDecl(
      R"(
      template <typename T>
      class F {};
      )",
      Lang_CXX03, "input1.cc");
  auto *Definition = FirstDeclMatcher<ClassTemplateDecl>().match(
      FromTU1, classTemplateDecl(hasName("F")));
  auto *ImportedDef = cast<ClassTemplateDecl>(Import(Definition, Lang_CXX03));
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
      Lang_CXX03, "input0.cc");
  auto *Friend = FirstDeclMatcher<FriendDecl>().match(FromTU0, friendDecl());
  QualType FT = Friend->getFriendType()->getType();
  FT = FromTU0->getASTContext().getCanonicalType(FT);
  auto *Fwd = cast<TagType>(FT)->getDecl();
  auto *ImportedFwd = Import(Fwd, Lang_CXX03);
  Decl *FromTU1 = getTuDecl(
      R"(
      class F {};
      )",
      Lang_CXX03, "input1.cc");
  auto *Definition = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU1, cxxRecordDecl(hasName("F")));
  auto *ImportedDef = Import(Definition, Lang_CXX03);
  EXPECT_TRUE(ImportedDef->getPreviousDecl());
  EXPECT_EQ(ImportedFwd, ImportedDef->getPreviousDecl());
}

TEST_P(ImportFriendClasses, ImportOfRepeatedFriendType) {
  const char *Code =
      R"(
      class Container {
        friend class X;
        friend class X;
      };
      )";
  testRepeatedFriendImport(Code);
}

TEST_P(ImportFriendClasses, ImportOfRepeatedFriendDecl) {
  const char *Code =
      R"(
      class Container {
        friend void f();
        friend void f();
      };
      )";
  testRepeatedFriendImport(Code);
}

TEST_P(ImportFriendClasses, ImportOfRepeatedFriendFunctionTemplateDecl) {
  const char *Code =
      R"(
        template <class T>
        class Container {
          template <class U> friend void m();
          template <class U> friend void m();
        };
      )";
  testRepeatedFriendImport(Code);
}

TEST_P(ImportFriendClasses, ImportOfRepeatedFriendClassTemplateDecl) {
  const char *Code =
      R"(
        template <class T>
        class Container {
          template <class U> friend class X;
          template <class U> friend class X;
        };
      )";
  testRepeatedFriendImport(Code);
}

TEST_P(ASTImporterOptionSpecificTestBase, FriendFunInClassTemplate) {
  auto *Code = R"(
  template <class T>
  struct X {
    friend void foo(){}
  };
      )";
  TranslationUnitDecl *ToTU = getToTuDecl(Code, Lang_CXX03);
  auto *ToFoo = FirstDeclMatcher<FunctionDecl>().match(
      ToTU, functionDecl(hasName("foo")));

  TranslationUnitDecl *FromTU = getTuDecl(Code, Lang_CXX03, "input.cc");
  auto *FromFoo = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("foo")));
  auto *ImportedFoo = Import(FromFoo, Lang_CXX03);
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
  Decl *TU = getTuDecl("extern int A; int A;", Lang_CXX03);
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
  // We do not have a list with one element.
  L.setHasExternalDecls();
  ASSERT_FALSE(L.getAsList());
  auto Results = L.getLookupResult();
  ASSERT_EQ(1u, std::distance(Results.begin(), Results.end()));

  // This asserts in the old implementation.
  DC->removeDecl(A0);
  EXPECT_FALSE(DC->containsDecl(A0));

  // Make sure we do not leave a StoredDeclsList with no entries.
  DC->removeDecl(A1);
  ASSERT_EQ(Map->find(A1->getDeclName()), Map->end());
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
      Lang_CXX03, "input0.cc");

  // Check that the function template instantiation is NOT the child of the TU.
  auto Pattern = translationUnitDecl(
      unless(has(functionDecl(hasName("f"), isTemplateInstantiation()))));
  ASSERT_TRUE(MatchVerifier<Decl>{}.match(FromTU, Pattern));

  auto *Foo = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("foo")));
  ASSERT_TRUE(Import(Foo, Lang_CXX03));

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
      Lang_CXX03, "input0.cc");

  // Check that the function template instantiation is NOT the child of the TU.
  auto Instantiation = functionDecl(hasName("f"), isTemplateInstantiation());
  auto Pattern = translationUnitDecl(unless(has(Instantiation)));
  ASSERT_TRUE(MatchVerifier<Decl>{}.match(FromTU, Pattern));

  ASSERT_TRUE(Import(FirstDeclMatcher<Decl>().match(FromTU, Instantiation),
                     Lang_CXX03));

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
      Lang_CXX03, "input0.cc");

  // Check that the function template specialization is the child of the TU.
  auto Specialization =
      functionDecl(hasName("f"), isExplicitTemplateSpecialization());
  auto Pattern = translationUnitDecl(has(Specialization));
  ASSERT_TRUE(MatchVerifier<Decl>{}.match(FromTU, Pattern));

  ASSERT_TRUE(Import(FirstDeclMatcher<Decl>().match(FromTU, Specialization),
                     Lang_CXX03));

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
      Lang_CXX03, "input0.cc");

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

  ASSERT_TRUE(Import(FromSpecD, Lang_CXX03));

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
        Lang_CXX03, "input0.cc");
    auto *FromD = FirstDeclMatcher<ClassTemplateDecl>().match(
        FromTU, classTemplateDecl(hasName("B")));

    Import(FromD, Lang_CXX03);
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
        Lang_CXX03, "input1.cc");
    FunctionDecl *FromD = FirstDeclMatcher<FunctionDecl>().match(
        FromTU, functionDecl(hasName("f")));
    Import(FromD, Lang_CXX03);
    auto *FromCTD = FirstDeclMatcher<ClassTemplateDecl>().match(
        FromTU, classTemplateDecl(hasName("B")));
    auto *ToCTD = cast<ClassTemplateDecl>(Import(FromCTD, Lang_CXX03));
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

TEST_P(ASTImporterOptionSpecificTestBase, ImportTemplateParameterLists) {
  auto Code =
      R"(
      template<class T>
      int f() { return 0; }
      template <> int f<int>() { return 4; }
      )";

  Decl *FromTU = getTuDecl(Code, Lang_CXX03);
  auto *FromD = FirstDeclMatcher<FunctionDecl>().match(FromTU,
      functionDecl(hasName("f"), isExplicitTemplateSpecialization()));
  ASSERT_EQ(FromD->getNumTemplateParameterLists(), 1u);

  auto *ToD = Import(FromD, Lang_CXX03);
  // The template parameter list should exist.
  EXPECT_EQ(ToD->getNumTemplateParameterLists(), 1u);
}

const internal::VariadicDynCastAllOfMatcher<Decl, VarTemplateDecl>
    varTemplateDecl;

const internal::VariadicDynCastAllOfMatcher<
    Decl, VarTemplatePartialSpecializationDecl>
    varTemplatePartialSpecializationDecl;

TEST_P(ASTImporterOptionSpecificTestBase,
       FunctionTemplateParameterDeclContext) {
  constexpr auto Code =
      R"(
      template<class T>
      void f() {};
      )";

  Decl *FromTU = getTuDecl(Code, Lang_CXX11);

  auto *FromD = FirstDeclMatcher<FunctionTemplateDecl>().match(
      FromTU, functionTemplateDecl(hasName("f")));

  ASSERT_EQ(FromD->getTemplateParameters()->getParam(0)->getDeclContext(),
            FromD->getTemplatedDecl());

  auto *ToD = Import(FromD, Lang_CXX11);
  EXPECT_EQ(ToD->getTemplateParameters()->getParam(0)->getDeclContext(),
            ToD->getTemplatedDecl());
  EXPECT_TRUE(SharedStatePtr->getLookupTable()->contains(
      ToD->getTemplatedDecl(), ToD->getTemplateParameters()->getParam(0)));
}

TEST_P(ASTImporterOptionSpecificTestBase, ClassTemplateParameterDeclContext) {
  constexpr auto Code =
      R"(
      template<class T1, class T2>
      struct S {};
      template<class T2>
      struct S<int, T2> {};
      )";

  Decl *FromTU = getTuDecl(Code, Lang_CXX11);

  auto *FromD = FirstDeclMatcher<ClassTemplateDecl>().match(
      FromTU, classTemplateDecl(hasName("S")));
  auto *FromDPart =
      FirstDeclMatcher<ClassTemplatePartialSpecializationDecl>().match(
          FromTU, classTemplatePartialSpecializationDecl(hasName("S")));

  ASSERT_EQ(FromD->getTemplateParameters()->getParam(0)->getDeclContext(),
            FromD->getTemplatedDecl());
  ASSERT_EQ(FromDPart->getTemplateParameters()->getParam(0)->getDeclContext(),
            FromDPart);

  auto *ToD = Import(FromD, Lang_CXX11);
  auto *ToDPart = Import(FromDPart, Lang_CXX11);

  EXPECT_EQ(ToD->getTemplateParameters()->getParam(0)->getDeclContext(),
            ToD->getTemplatedDecl());
  EXPECT_TRUE(SharedStatePtr->getLookupTable()->contains(
      ToD->getTemplatedDecl(), ToD->getTemplateParameters()->getParam(0)));

  EXPECT_EQ(ToDPart->getTemplateParameters()->getParam(0)->getDeclContext(),
            ToDPart);
  EXPECT_TRUE(SharedStatePtr->getLookupTable()->contains(
      ToDPart, ToDPart->getTemplateParameters()->getParam(0)));
}

TEST_P(ASTImporterOptionSpecificTestBase,
       CXXDeductionGuideTemplateParameterDeclContext) {
  Decl *FromTU = getTuDecl(
      R"(
      template <typename T> struct A {
        A(T);
      };
      A a{(int)0};
      )",
      Lang_CXX17, "input.cc");
// clang-format off
/*
|-ClassTemplateDecl 0x1fe5000 <input.cc:2:7, line:4:7> line:2:36 A
| |-TemplateTypeParmDecl 0x1fe4eb0 <col:17, col:26> col:26 referenced typename depth 0 index 0 T
| |-CXXRecordDecl 0x1fe4f70 <col:29, line:4:7> line:2:36 struct A definition

|-FunctionTemplateDecl 0x1fe5860 <line:2:7, line:3:12> col:9 implicit <deduction guide for A>
| |-TemplateTypeParmDecl 0x1fe4eb0 <line:2:17, col:26> col:26 referenced typename depth 0 index 0 T
| |-CXXDeductionGuideDecl 0x1fe57a8 <line:3:9, col:12> col:9 implicit <deduction guide for A> 'auto (T) -> A<T>'
| | `-ParmVarDecl 0x1fe56b0 <col:11> col:12 'T'
| `-CXXDeductionGuideDecl 0x20515d8 <col:9, col:12> col:9 implicit used <deduction guide for A> 'auto (int) -> A<int>'
|   |-TemplateArgument type 'int'
|   | `-BuiltinType 0x20587e0 'int'
|   `-ParmVarDecl 0x2051388 <col:11> col:12 'int':'int'
`-FunctionTemplateDecl 0x1fe5a78 <line:2:7, col:36> col:36 implicit <deduction guide for A>
  |-TemplateTypeParmDecl 0x1fe4eb0 <col:17, col:26> col:26 referenced typename depth 0 index 0 T
  `-CXXDeductionGuideDecl 0x1fe59c0 <col:36> col:36 implicit <deduction guide for A> 'auto (A<T>) -> A<T>'
    `-ParmVarDecl 0x1fe5958 <col:36> col:36 'A<T>'
*/
// clang-format on
  auto *FromD1 = FirstDeclMatcher<CXXDeductionGuideDecl>().match(
      FromTU, cxxDeductionGuideDecl());
  auto *FromD2 = LastDeclMatcher<CXXDeductionGuideDecl>().match(
      FromTU, cxxDeductionGuideDecl());

  NamedDecl *P1 =
      FromD1->getDescribedFunctionTemplate()->getTemplateParameters()->getParam(
          0);
  NamedDecl *P2 =
      FromD2->getDescribedFunctionTemplate()->getTemplateParameters()->getParam(
          0);
  DeclContext *DC = P1->getDeclContext();

  ASSERT_EQ(P1, P2);
  ASSERT_TRUE(DC == FromD1 || DC == FromD2);

  auto *ToD1 = Import(FromD1, Lang_CXX17);
  auto *ToD2 = Import(FromD2, Lang_CXX17);
  ASSERT_TRUE(ToD1 && ToD2);

  P1 = ToD1->getDescribedFunctionTemplate()->getTemplateParameters()->getParam(
      0);
  P2 = ToD2->getDescribedFunctionTemplate()->getTemplateParameters()->getParam(
      0);
  DC = P1->getDeclContext();

  EXPECT_EQ(P1, P2);
  EXPECT_TRUE(DC == ToD1 || DC == ToD2);

  ASTImporterLookupTable *Tbl = SharedStatePtr->getLookupTable();
  if (Tbl->contains(ToD1, P1)) {
    EXPECT_FALSE(Tbl->contains(ToD2, P1));
  } else {
    EXPECT_TRUE(Tbl->contains(ToD2, P1));
  }
}

TEST_P(ASTImporterOptionSpecificTestBase, VarTemplateParameterDeclContext) {
  constexpr auto Code =
      R"(
      template<class T1, class T2>
      int X1;
      template<class T2>
      int X1<int, T2>;

      namespace Ns {
        template<class T1, class T2>
        int X2;
        template<class T2>
        int X2<int, T2>;
      }
      )";

  Decl *FromTU = getTuDecl(Code, Lang_CXX14);

  auto *FromD1 = FirstDeclMatcher<VarTemplateDecl>().match(
      FromTU, varTemplateDecl(hasName("X1")));
  auto *FromD1Part =
      FirstDeclMatcher<VarTemplatePartialSpecializationDecl>().match(
          FromTU, varTemplatePartialSpecializationDecl(hasName("X1")));
  auto *FromD2 = FirstDeclMatcher<VarTemplateDecl>().match(
      FromTU, varTemplateDecl(hasName("X2")));
  auto *FromD2Part =
      FirstDeclMatcher<VarTemplatePartialSpecializationDecl>().match(
          FromTU, varTemplatePartialSpecializationDecl(hasName("X2")));

  ASSERT_EQ(FromD1->getTemplateParameters()->getParam(0)->getDeclContext(),
            FromD1->getDeclContext());
  ASSERT_EQ(FromD2->getTemplateParameters()->getParam(0)->getDeclContext(),
            FromD2->getDeclContext());

  ASSERT_EQ(FromD1Part->getTemplateParameters()->getParam(0)->getDeclContext(),
            FromD1Part->getDeclContext());
  // FIXME: VarTemplatePartialSpecializationDecl does not update ("adopt")
  // template parameter decl context
  // ASSERT_EQ(FromD2Part->getTemplateParameters()->getParam(0)->getDeclContext(),
  // FromD2Part->getDeclContext());

  auto *ToD1 = Import(FromD1, Lang_CXX14);
  auto *ToD2 = Import(FromD2, Lang_CXX14);

  auto *ToD1Part = Import(FromD1Part, Lang_CXX14);
  auto *ToD2Part = Import(FromD2Part, Lang_CXX14);

  EXPECT_EQ(ToD1->getTemplateParameters()->getParam(0)->getDeclContext(),
            ToD1->getDeclContext());
  EXPECT_TRUE(SharedStatePtr->getLookupTable()->contains(
      ToD1->getDeclContext(), ToD1->getTemplateParameters()->getParam(0)));
  EXPECT_EQ(ToD2->getTemplateParameters()->getParam(0)->getDeclContext(),
            ToD2->getDeclContext());
  EXPECT_TRUE(SharedStatePtr->getLookupTable()->contains(
      ToD2->getDeclContext(), ToD2->getTemplateParameters()->getParam(0)));

  EXPECT_EQ(ToD1Part->getTemplateParameters()->getParam(0)->getDeclContext(),
            ToD1Part->getDeclContext());
  EXPECT_TRUE(SharedStatePtr->getLookupTable()->contains(
      ToD1Part->getDeclContext(),
      ToD1Part->getTemplateParameters()->getParam(0)));
  // EXPECT_EQ(ToD2Part->getTemplateParameters()->getParam(0)->getDeclContext(),
  // ToD2Part->getDeclContext());
  // EXPECT_TRUE(SharedStatePtr->getLookupTable()->contains(
  //     ToD2Part->getDeclContext(),
  //     ToD2Part->getTemplateParameters()->getParam(0)));
  (void)ToD2Part;
}

TEST_P(ASTImporterOptionSpecificTestBase,
       TypeAliasTemplateParameterDeclContext) {
  constexpr auto Code =
      R"(
      template<class T1, class T2>
      struct S {};
      template<class T> using S1 = S<T, int>;
      namespace Ns {
        template<class T> using S2 = S<T, int>;
      }
      )";

  Decl *FromTU = getTuDecl(Code, Lang_CXX11);

  auto *FromD1 = FirstDeclMatcher<TypeAliasTemplateDecl>().match(
      FromTU, typeAliasTemplateDecl(hasName("S1")));
  auto *FromD2 = FirstDeclMatcher<TypeAliasTemplateDecl>().match(
      FromTU, typeAliasTemplateDecl(hasName("S2")));

  ASSERT_EQ(FromD1->getTemplateParameters()->getParam(0)->getDeclContext(),
            FromD1->getDeclContext());
  ASSERT_EQ(FromD2->getTemplateParameters()->getParam(0)->getDeclContext(),
            FromD2->getDeclContext());

  auto *ToD1 = Import(FromD1, Lang_CXX11);
  auto *ToD2 = Import(FromD2, Lang_CXX11);

  EXPECT_EQ(ToD1->getTemplateParameters()->getParam(0)->getDeclContext(),
            ToD1->getDeclContext());
  EXPECT_TRUE(SharedStatePtr->getLookupTable()->contains(
      ToD1->getDeclContext(), ToD1->getTemplateParameters()->getParam(0)));
  EXPECT_EQ(ToD2->getTemplateParameters()->getParam(0)->getDeclContext(),
            ToD2->getDeclContext());
  EXPECT_TRUE(SharedStatePtr->getLookupTable()->contains(
      ToD2->getDeclContext(), ToD2->getTemplateParameters()->getParam(0)));
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportSubstTemplateTypeParmType) {
  constexpr auto Code = R"(
    template <class A1, class... A2> struct A {
      using B = A1(A2...);
    };
    template struct A<void, char, float, int, short>;
    )";
  Decl *FromTU = getTuDecl(Code, Lang_CXX11, "input.cpp");
  auto *FromClass = FirstDeclMatcher<ClassTemplateSpecializationDecl>().match(
      FromTU, classTemplateSpecializationDecl());

  auto testType = [&](ASTContext &Ctx, const char *Name,
                      std::optional<unsigned> PackIndex) {
    const auto *Subst = selectFirst<SubstTemplateTypeParmType>(
        "sttp", match(substTemplateTypeParmType(
                          hasReplacementType(hasCanonicalType(asString(Name))))
                          .bind("sttp"),
                      Ctx));
    const char *ExpectedTemplateParamName = PackIndex ? "A2" : "A1";
    ASSERT_TRUE(Subst);
    ASSERT_EQ(Subst->getReplacedParameter()->getIdentifier()->getName(),
              ExpectedTemplateParamName);
    ASSERT_EQ(Subst->getPackIndex(), PackIndex);
  };
  auto tests = [&](ASTContext &Ctx) {
    testType(Ctx, "void", std::nullopt);
    testType(Ctx, "char", 3);
    testType(Ctx, "float", 2);
    testType(Ctx, "int", 1);
    testType(Ctx, "short", 0);
  };

  tests(FromTU->getASTContext());

  ClassTemplateSpecializationDecl *ToClass = Import(FromClass, Lang_CXX11);
  tests(ToClass->getASTContext());
}

const AstTypeMatcher<SubstTemplateTypeParmPackType>
    substTemplateTypeParmPackType;

TEST_P(ASTImporterOptionSpecificTestBase, ImportSubstTemplateTypeParmPackType) {
  constexpr auto Code = R"(
    template<typename ...T> struct D {
      template<typename... U> using B = int(int (*...p)(T, U));
      template<typename U1, typename U2> D(B<U1, U2>*);
    };
    int f(int(int, int), int(int, int));

    using asd = D<float, double, float>::B<int, long, int>;
    )";
  Decl *FromTU = getTuDecl(Code, Lang_CXX11, "input.cpp");
  auto *FromClass = FirstDeclMatcher<ClassTemplateSpecializationDecl>().match(
      FromTU, classTemplateSpecializationDecl());

  {
    ASTContext &FromCtx = FromTU->getASTContext();
    const auto *FromSubstPack = selectFirst<SubstTemplateTypeParmPackType>(
        "pack", match(substTemplateTypeParmPackType().bind("pack"), FromCtx));

    ASSERT_TRUE(FromSubstPack);
    ASSERT_EQ(FromSubstPack->getIdentifier()->getName(), "T");
    ArrayRef<TemplateArgument> FromArgPack =
        FromSubstPack->getArgumentPack().pack_elements();
    ASSERT_EQ(FromArgPack.size(), 3u);
    ASSERT_EQ(FromArgPack[0].getAsType(), FromCtx.FloatTy);
    ASSERT_EQ(FromArgPack[1].getAsType(), FromCtx.DoubleTy);
    ASSERT_EQ(FromArgPack[2].getAsType(), FromCtx.FloatTy);
  }
  {
    // Let's do the import.
    ClassTemplateSpecializationDecl *ToClass = Import(FromClass, Lang_CXX11);
    ASTContext &ToCtx = ToClass->getASTContext();

    const auto *ToSubstPack = selectFirst<SubstTemplateTypeParmPackType>(
        "pack", match(substTemplateTypeParmPackType().bind("pack"), ToCtx));

    // Check if it meets the requirements.
    ASSERT_TRUE(ToSubstPack);
    ASSERT_EQ(ToSubstPack->getIdentifier()->getName(), "T");
    ArrayRef<TemplateArgument> ToArgPack =
        ToSubstPack->getArgumentPack().pack_elements();
    ASSERT_EQ(ToArgPack.size(), 3u);
    ASSERT_EQ(ToArgPack[0].getAsType(), ToCtx.FloatTy);
    ASSERT_EQ(ToArgPack[1].getAsType(), ToCtx.DoubleTy);
    ASSERT_EQ(ToArgPack[2].getAsType(), ToCtx.FloatTy);
  }
}

struct ASTImporterLookupTableTest : ASTImporterOptionSpecificTestBase {};

TEST_P(ASTImporterLookupTableTest, OneDecl) {
  auto *ToTU = getToTuDecl("int a;", Lang_CXX03);
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
  TranslationUnitDecl *ToTU = getToTuDecl(Code, Lang_CXX03);
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
      getToTuDecl("struct A { struct Foo *p; };", Lang_C99);
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

  // Finds via linear search of its LexicalDC (A).
  FooLexicalDC->getRedeclContext()->localUncachedLookup(FooName, FoundDecls);
  EXPECT_EQ(FoundDecls.size(), 1u);

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
      getToTuDecl("int V; struct A { int V; }; struct B { int V; };", Lang_C99);
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
      Lang_CXX03);

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
      Lang_CXX03);

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
      Lang_CXX03);
  auto *ToPlus = FirstDeclMatcher<FunctionDecl>().match(
      ToTU, functionDecl(hasOverloadedOperatorName("+")));

  Decl *FromTU = getTuDecl(
      R"(
      struct X {};
      void operator+(X, X);
      )",
      Lang_CXX03);
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

TEST_P(ASTImporterLookupTableTest,
       LookupFindsFwdFriendClassDeclWithElaboratedType) {
  TranslationUnitDecl *ToTU = getToTuDecl(
      R"(
      class Y { friend class F; };
      )",
      Lang_CXX03);

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

TEST_P(ASTImporterLookupTableTest,
       LookupFindsFwdFriendClassDeclWithUnelaboratedType) {
  TranslationUnitDecl *ToTU = getToTuDecl(
      R"(
      class F;
      class Y { friend F; };
      )",
      Lang_CXX11);

  // In this case, the CXXRecordDecl is hidden, the FriendDecl is not a parent.
  // So we must dig up the underlying CXXRecordDecl.
  ASTImporterLookupTable LT(*ToTU);
  auto *FriendD = FirstDeclMatcher<FriendDecl>().match(ToTU, friendDecl());
  const RecordDecl *RD = getRecordDeclOfFriend(FriendD);
  auto *Y = FirstDeclMatcher<CXXRecordDecl>().match(ToTU, cxxRecordDecl(hasName("Y")));

  DeclarationName Name = RD->getDeclName();
  auto Res = LT.lookup(ToTU, Name);
  EXPECT_EQ(Res.size(), 1u);
  EXPECT_EQ(*Res.begin(), RD);

  Res = LT.lookup(Y, Name);
  EXPECT_EQ(Res.size(), 0u);
}

TEST_P(ASTImporterLookupTableTest,
       LookupFindsFriendClassDeclWithTypeAliasDoesNotAssert) {
  TranslationUnitDecl *ToTU = getToTuDecl(
      R"(
      class F;
      using alias_of_f = F;
      class Y { friend alias_of_f; };
      )",
      Lang_CXX11);

  // ASTImporterLookupTable constructor handles using declarations correctly,
  // no assert is expected.
  ASTImporterLookupTable LT(*ToTU);

  auto *Alias = FirstDeclMatcher<TypeAliasDecl>().match(
      ToTU, typeAliasDecl(hasName("alias_of_f")));
  DeclarationName Name = Alias->getDeclName();
  auto Res = LT.lookup(ToTU, Name);
  EXPECT_EQ(Res.count(Alias), 1u);
}

TEST_P(ASTImporterLookupTableTest,
       LookupFindsFriendClassDeclWithUsingTypeDoesNotAssert) {
  TranslationUnitDecl *ToTU = getToTuDecl(
      R"(
      namespace a {
        namespace b { class InnerClass; }
        using b::InnerClass;
      }
      class B {
        friend a::InnerClass;
      };
      )",
      Lang_CXX11);

  // ASTImporterLookupTable constructor handles friend with using-type without
  // asserts.
  ASTImporterLookupTable LT(*ToTU);

  auto *Using = FirstDeclMatcher<UsingDecl>().match(
      ToTU, usingDecl(hasName("InnerClass")));
  DeclarationName Name = Using->getDeclName();
  auto Res = LT.lookup(ToTU, Name);
  EXPECT_EQ(Res.size(), 0u);
  auto *NsA = FirstDeclMatcher<NamespaceDecl>().match(
      ToTU, namespaceDecl(hasName("a")));
  auto *RecordB = FirstDeclMatcher<CXXRecordDecl>().match(
      ToTU, cxxRecordDecl(hasName("B")));
  auto Res1 = LT.lookup(NsA, Name);
  EXPECT_EQ(Res1.count(Using), 1u);
  auto Res2 = LT.lookup(RecordB, Name);
  EXPECT_EQ(Res2.size(), 0u);
}

TEST_P(ASTImporterLookupTableTest, LookupFindsFwdFriendClassTemplateDecl) {
  TranslationUnitDecl *ToTU = getToTuDecl(
      R"(
      class Y { template <class T> friend class F; };
      )",
      Lang_CXX03);

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
      Lang_CXX03);

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
      Lang_CXX03);

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
      Lang_CXX03);

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
      Lang_CXX03);

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
      Lang_CXX03);

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
      Lang_CXX03);

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
      Lang_C99);

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
      Lang_CXX03);
  auto *N1 =
      LastDeclMatcher<NamespaceDecl>().match(ToTU, namespaceDecl(hasName("N")));
  auto *A = FirstDeclMatcher<VarDecl>().match(ToTU, varDecl(hasName("A")));
  DeclarationName Name = A->getDeclName();

  ASTImporterLookupTable LT(*ToTU);
  auto Res = LT.lookup(N1, Name);
  ASSERT_EQ(Res.size(), 1u);
  EXPECT_EQ(*Res.begin(), A);
}

TEST_P(ASTImporterOptionSpecificTestBase,
       RedeclChainShouldBeCorrectAmongstNamespaces) {
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
      Lang_CXX03);
  auto *FromFwd = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("X"), unless(isImplicit())));
  auto *FromDef = LastDeclMatcher<CXXRecordDecl>().match(
      FromTU,
      cxxRecordDecl(hasName("X"), isDefinition(), unless(isImplicit())));
  ASSERT_NE(FromFwd, FromDef);
  ASSERT_FALSE(FromFwd->isThisDeclarationADefinition());
  ASSERT_TRUE(FromDef->isThisDeclarationADefinition());
  ASSERT_EQ(FromFwd->getCanonicalDecl(), FromDef->getCanonicalDecl());

  auto *ToDef = cast_or_null<CXXRecordDecl>(Import(FromDef, Lang_CXX03));
  auto *ToFwd = cast_or_null<CXXRecordDecl>(Import(FromFwd, Lang_CXX03));
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
      Lang_CXX03);
  auto *Friend = FirstDeclMatcher<FunctionTemplateDecl>().match(
      ToTU, functionTemplateDecl(hasName("foo")));

  Decl *FromTU = getTuDecl(
      R"(
      template <typename T> void foo();
      )",
      Lang_CXX03);
  auto *FromFoo = FirstDeclMatcher<FunctionTemplateDecl>().match(
      FromTU, functionTemplateDecl(hasName("foo")));
  auto *Imported = Import(FromFoo, Lang_CXX03);

  EXPECT_EQ(Imported->getPreviousDecl(), Friend);
}

TEST_P(ImportFriendFunctionTemplates, ImportFriendFunctionInsideClassTemplate) {
  Decl *From, *To;
  std::tie(From, To) = getImportedDecl(
      R"(
      template <typename T> struct X {
        template <typename U> friend void f();
      };
      )",
      Lang_CXX03, "", Lang_CXX03, "X");

  auto *FromFriend = FirstDeclMatcher<FriendDecl>().match(From, friendDecl());
  auto *ToFriend = FirstDeclMatcher<FriendDecl>().match(To, friendDecl());

  EXPECT_TRUE(FromFriend ==
              LastDeclMatcher<FriendDecl>().match(From, friendDecl()));
  EXPECT_TRUE(ToFriend ==
              LastDeclMatcher<FriendDecl>().match(To, friendDecl()));

  auto *FromDecl = FromFriend->getFriendDecl();
  auto *FromDC = FromFriend->getDeclContext();
  auto *FromLexicalDC = FromFriend->getLexicalDeclContext();

  EXPECT_TRUE(FromDC->containsDecl(FromFriend));
  EXPECT_FALSE(FromDC->containsDecl(FromDecl));
  EXPECT_TRUE(FromLexicalDC->containsDecl(FromFriend));
  EXPECT_FALSE(FromLexicalDC->containsDecl(FromDecl));

  auto *ToDecl = ToFriend->getFriendDecl();
  auto *ToDC = ToFriend->getDeclContext();
  auto *ToLexicalDC = ToFriend->getLexicalDeclContext();

  EXPECT_TRUE(ToDC->containsDecl(ToFriend));
  EXPECT_FALSE(ToDC->containsDecl(ToDecl));
  EXPECT_TRUE(ToLexicalDC->containsDecl(ToFriend));
  EXPECT_FALSE(ToLexicalDC->containsDecl(ToDecl));
}

struct ASTImporterWithFakeErrors : ASTImporter {
  using ASTImporter::ASTImporter;
  bool returnWithErrorInTest() override { return true; }
};

struct ErrorHandlingTest : ASTImporterOptionSpecificTestBase {
  ErrorHandlingTest() {
    Creator = [](ASTContext &ToContext, FileManager &ToFileManager,
                 ASTContext &FromContext, FileManager &FromFileManager,
                 bool MinimalImport,
                 const std::shared_ptr<ASTImporterSharedState> &SharedState) {
      return new ASTImporterWithFakeErrors(ToContext, ToFileManager,
                                           FromContext, FromFileManager,
                                           MinimalImport, SharedState);
    };
  }
  // In this test we purposely report an error (UnsupportedConstruct) when
  // importing the below stmt.
  static constexpr auto* ErroneousStmt = R"( asm(""); )";
};

// Check a case when no new AST node is created in the AST before encountering
// the error.
TEST_P(ErrorHandlingTest, ErrorHappensBeforeCreatingANewNode) {
  TranslationUnitDecl *ToTU = getToTuDecl(
      R"(
      template <typename T>
      class X {};
      template <>
      class X<int> { int a; };
      )",
      Lang_CXX03);
  TranslationUnitDecl *FromTU = getTuDecl(
      R"(
      template <typename T>
      class X {};
      template <>
      class X<int> { double b; };
      )",
      Lang_CXX03);
  auto *FromSpec = FirstDeclMatcher<ClassTemplateSpecializationDecl>().match(
      FromTU, classTemplateSpecializationDecl(hasName("X")));
  ClassTemplateSpecializationDecl *ImportedSpec = Import(FromSpec, Lang_CXX03);
  EXPECT_FALSE(ImportedSpec);

  // The original Decl is kept, no new decl is created.
  EXPECT_EQ(DeclCounter<ClassTemplateSpecializationDecl>().match(
                ToTU, classTemplateSpecializationDecl(hasName("X"))),
            1u);

  // But an error is set to the counterpart in the "from" context.
  ASTImporter *Importer = findFromTU(FromSpec)->Importer.get();
  std::optional<ASTImportError> OptErr =
      Importer->getImportDeclErrorIfAny(FromSpec);
  ASSERT_TRUE(OptErr);
  EXPECT_EQ(OptErr->Error, ASTImportError::NameConflict);
}

// Check a case when a new AST node is created but not linked to the AST before
// encountering the error.
TEST_P(ErrorHandlingTest,
       ErrorHappensAfterCreatingTheNodeButBeforeLinkingThatToTheAST) {
  TranslationUnitDecl *FromTU = getTuDecl(
      std::string("void foo() { ") + ErroneousStmt + " }", Lang_CXX03);
  auto *FromFoo = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("foo")));

  FunctionDecl *ImportedFoo = Import(FromFoo, Lang_CXX03);
  EXPECT_FALSE(ImportedFoo);

  TranslationUnitDecl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  // Created, but not linked.
  EXPECT_EQ(
      DeclCounter<FunctionDecl>().match(ToTU, functionDecl(hasName("foo"))),
      0u);

  ASTImporter *Importer = findFromTU(FromFoo)->Importer.get();
  std::optional<ASTImportError> OptErr =
      Importer->getImportDeclErrorIfAny(FromFoo);
  ASSERT_TRUE(OptErr);
  EXPECT_EQ(OptErr->Error, ASTImportError::UnsupportedConstruct);
}

// Check a case when a new AST node is created and linked to the AST before
// encountering the error. The error is set for the counterpart of the nodes in
// the "from" context.
TEST_P(ErrorHandlingTest, ErrorHappensAfterNodeIsCreatedAndLinked) {
  TranslationUnitDecl *FromTU = getTuDecl(std::string(R"(
      void f();
      void f() { )") + ErroneousStmt + R"( }
      )",
                                          Lang_CXX03);
  auto *FromProto = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("f")));
  auto *FromDef =
      LastDeclMatcher<FunctionDecl>().match(FromTU, functionDecl(hasName("f")));
  FunctionDecl *ImportedProto = Import(FromProto, Lang_CXX03);
  EXPECT_FALSE(ImportedProto); // Could not import.
  // However, we created two nodes in the AST. 1) the fwd decl 2) the
  // definition. The definition is not added to its DC, but the fwd decl is
  // there.
  TranslationUnitDecl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  EXPECT_EQ(DeclCounter<FunctionDecl>().match(ToTU, functionDecl(hasName("f"))),
            1u);
  // Match the fwd decl.
  auto *ToProto =
      FirstDeclMatcher<FunctionDecl>().match(ToTU, functionDecl(hasName("f")));
  EXPECT_TRUE(ToProto);
  // An error is set to the counterpart in the "from" context both for the fwd
  // decl and the definition.
  ASTImporter *Importer = findFromTU(FromProto)->Importer.get();
  std::optional<ASTImportError> OptErr =
      Importer->getImportDeclErrorIfAny(FromProto);
  ASSERT_TRUE(OptErr);
  EXPECT_EQ(OptErr->Error, ASTImportError::UnsupportedConstruct);
  OptErr = Importer->getImportDeclErrorIfAny(FromDef);
  ASSERT_TRUE(OptErr);
  EXPECT_EQ(OptErr->Error, ASTImportError::UnsupportedConstruct);
}

// An error should be set for a class if we cannot import one member.
TEST_P(ErrorHandlingTest, ErrorIsPropagatedFromMemberToClass) {
  TranslationUnitDecl *FromTU = getTuDecl(std::string(R"(
      class X {
        void f() { )") + ErroneousStmt + R"( } // This member has the error
                                               // during import.
        void ok();        // The error should not prevent importing this.
      };                  // An error will be set for X too.
      )",
                                          Lang_CXX03);
  auto *FromX = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("X")));
  CXXRecordDecl *ImportedX = Import(FromX, Lang_CXX03);

  // An error is set for X.
  EXPECT_FALSE(ImportedX);
  ASTImporter *Importer = findFromTU(FromX)->Importer.get();
  std::optional<ASTImportError> OptErr =
      Importer->getImportDeclErrorIfAny(FromX);
  ASSERT_TRUE(OptErr);
  EXPECT_EQ(OptErr->Error, ASTImportError::UnsupportedConstruct);

  // An error is set for f().
  auto *FromF = FirstDeclMatcher<CXXMethodDecl>().match(
      FromTU, cxxMethodDecl(hasName("f")));
  OptErr = Importer->getImportDeclErrorIfAny(FromF);
  ASSERT_TRUE(OptErr);
  EXPECT_EQ(OptErr->Error, ASTImportError::UnsupportedConstruct);
  // And any subsequent import should fail.
  CXXMethodDecl *ImportedF = Import(FromF, Lang_CXX03);
  EXPECT_FALSE(ImportedF);

  // There is an error set for the other member too.
  auto *FromOK = FirstDeclMatcher<CXXMethodDecl>().match(
      FromTU, cxxMethodDecl(hasName("ok")));
  OptErr = Importer->getImportDeclErrorIfAny(FromOK);
  EXPECT_TRUE(OptErr);
  // Cannot import the other member.
  CXXMethodDecl *ImportedOK = Import(FromOK, Lang_CXX03);
  EXPECT_FALSE(ImportedOK);
}

// Check that an error propagates to the dependent AST nodes.
// In the below code it means that an error in X should propagate to A.
// And even to F since the containing A is erroneous.
// And to all AST nodes which we visit during the import process which finally
// ends up in a failure (in the error() function).
TEST_P(ErrorHandlingTest, ErrorPropagatesThroughImportCycles) {
  Decl *FromTU = getTuDecl(std::string(R"(
      namespace NS {
        class A {
          template <int I> class F {};
          class X {
            template <int I> friend class F;
            void error() { )") +
                               ErroneousStmt + R"( }
          };
        };

        class B {};
      } // NS
      )",
                           Lang_CXX03, "input0.cc");

  auto *FromFRD = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("F"), isDefinition()));
  auto *FromA = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("A"), isDefinition()));
  auto *FromB = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("B"), isDefinition()));
  auto *FromNS = FirstDeclMatcher<NamespaceDecl>().match(
      FromTU, namespaceDecl(hasName("NS")));

  // Start by importing the templated CXXRecordDecl of F.
  // Import fails for that.
  EXPECT_FALSE(Import(FromFRD, Lang_CXX03));
  // Import fails for A.
  EXPECT_FALSE(Import(FromA, Lang_CXX03));
  // But we should be able to import the independent B.
  EXPECT_TRUE(Import(FromB, Lang_CXX03));
  // And the namespace.
  EXPECT_TRUE(Import(FromNS, Lang_CXX03));

  // An error is set to the templated CXXRecordDecl of F.
  ASTImporter *Importer = findFromTU(FromFRD)->Importer.get();
  std::optional<ASTImportError> OptErr =
      Importer->getImportDeclErrorIfAny(FromFRD);
  EXPECT_TRUE(OptErr);

  // An error is set to A.
  OptErr = Importer->getImportDeclErrorIfAny(FromA);
  EXPECT_TRUE(OptErr);

  // There is no error set to B.
  OptErr = Importer->getImportDeclErrorIfAny(FromB);
  EXPECT_FALSE(OptErr);

  // There is no error set to NS.
  OptErr = Importer->getImportDeclErrorIfAny(FromNS);
  EXPECT_FALSE(OptErr);

  // Check some of those decls whose ancestor is X, they all should have an
  // error set if we visited them during an import process which finally failed.
  // These decls are part of a cycle in an ImportPath.
  // There would not be any error set for these decls if we hadn't follow the
  // ImportPaths and the cycles.
  OptErr = Importer->getImportDeclErrorIfAny(
      FirstDeclMatcher<ClassTemplateDecl>().match(
          FromTU, classTemplateDecl(hasName("F"))));
  // An error is set to the 'F' ClassTemplateDecl.
  EXPECT_TRUE(OptErr);
  // An error is set to the FriendDecl.
  OptErr = Importer->getImportDeclErrorIfAny(
      FirstDeclMatcher<FriendDecl>().match(
          FromTU, friendDecl()));
  EXPECT_TRUE(OptErr);
  // An error is set to the implicit class of A.
  OptErr =
      Importer->getImportDeclErrorIfAny(FirstDeclMatcher<CXXRecordDecl>().match(
          FromTU, cxxRecordDecl(hasName("A"), isImplicit())));
  EXPECT_TRUE(OptErr);
  // An error is set to the implicit class of X.
  OptErr =
      Importer->getImportDeclErrorIfAny(FirstDeclMatcher<CXXRecordDecl>().match(
          FromTU, cxxRecordDecl(hasName("X"), isImplicit())));
  EXPECT_TRUE(OptErr);
}

TEST_P(ErrorHandlingTest, ErrorIsNotPropagatedFromMemberToNamespace) {
  TranslationUnitDecl *FromTU = getTuDecl(std::string(R"(
      namespace X {
        void f() { )") + ErroneousStmt + R"( } // This member has the error
                                               // during import.
        void ok();        // The error should not prevent importing this.
      };                  // An error will be set for X too.
      )",
                                          Lang_CXX03);
  auto *FromX = FirstDeclMatcher<NamespaceDecl>().match(
      FromTU, namespaceDecl(hasName("X")));
  NamespaceDecl *ImportedX = Import(FromX, Lang_CXX03);

  // There is no error set for X.
  EXPECT_TRUE(ImportedX);
  ASTImporter *Importer = findFromTU(FromX)->Importer.get();
  std::optional<ASTImportError> OptErr =
      Importer->getImportDeclErrorIfAny(FromX);
  ASSERT_FALSE(OptErr);

  // An error is set for f().
  auto *FromF = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("f")));
  OptErr = Importer->getImportDeclErrorIfAny(FromF);
  ASSERT_TRUE(OptErr);
  EXPECT_EQ(OptErr->Error, ASTImportError::UnsupportedConstruct);
  // And any subsequent import should fail.
  FunctionDecl *ImportedF = Import(FromF, Lang_CXX03);
  EXPECT_FALSE(ImportedF);

  // There is no error set for ok().
  auto *FromOK = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("ok")));
  OptErr = Importer->getImportDeclErrorIfAny(FromOK);
  EXPECT_FALSE(OptErr);
  // And we should be able to import.
  FunctionDecl *ImportedOK = Import(FromOK, Lang_CXX03);
  EXPECT_TRUE(ImportedOK);
}

TEST_P(ErrorHandlingTest, ODRViolationWithinTypedefDecls) {
  // Importing `z` should fail - instead of crashing - due to an ODR violation.
  // The `bar::e` typedef sets it's DeclContext after the import is done.
  // However, if the importation fails, it will be left as a nullptr.
  // During the cleanup of the failed import, we should check whether the
  // DeclContext is null or not - instead of dereferencing that unconditionally.
  constexpr auto ToTUCode = R"(
      namespace X {
        struct bar {
          int odr_violation;
        };
      })";
  constexpr auto FromTUCode = R"(
      namespace X {
        enum b {};
        struct bar {
          typedef b e;
          static e d;
        };
      }
      int z = X::bar::d;
      )";
  Decl *ToTU = getToTuDecl(ToTUCode, Lang_CXX11);
  static_cast<void>(ToTU);
  Decl *FromTU = getTuDecl(FromTUCode, Lang_CXX11);
  auto *FromZ =
      FirstDeclMatcher<VarDecl>().match(FromTU, varDecl(hasName("z")));
  ASSERT_TRUE(FromZ);
  ASSERT_TRUE(FromZ->hasInit());

  auto *ImportedZ = Import(FromZ, Lang_CXX11);
  EXPECT_FALSE(ImportedZ);
}

// An error should be set for a class if it had a previous import with an error
// from another TU.
TEST_P(ErrorHandlingTest,
       ImportedDeclWithErrorShouldFailTheImportOfDeclWhichMapToIt) {
  // We already have a fwd decl.
  TranslationUnitDecl *ToTU = getToTuDecl("class X;", Lang_CXX03);
  // Then we import a definition.
  {
    TranslationUnitDecl *FromTU = getTuDecl(std::string(R"(
        class X {
          void f() { )") + ErroneousStmt + R"( }
          void ok();
        };
        )",
                                            Lang_CXX03);
    auto *FromX = FirstDeclMatcher<CXXRecordDecl>().match(
        FromTU, cxxRecordDecl(hasName("X")));
    CXXRecordDecl *ImportedX = Import(FromX, Lang_CXX03);

    // An error is set for X ...
    EXPECT_FALSE(ImportedX);
    ASTImporter *Importer = findFromTU(FromX)->Importer.get();
    std::optional<ASTImportError> OptErr =
        Importer->getImportDeclErrorIfAny(FromX);
    ASSERT_TRUE(OptErr);
    EXPECT_EQ(OptErr->Error, ASTImportError::UnsupportedConstruct);
  }
  // ... but the node had been created.
  auto *ToXDef = FirstDeclMatcher<CXXRecordDecl>().match(
      ToTU, cxxRecordDecl(hasName("X"), isDefinition()));
  // An error is set for "ToXDef" in the shared state.
  std::optional<ASTImportError> OptErr =
      SharedStatePtr->getImportDeclErrorIfAny(ToXDef);
  ASSERT_TRUE(OptErr);
  EXPECT_EQ(OptErr->Error, ASTImportError::UnsupportedConstruct);

  auto *ToXFwd = FirstDeclMatcher<CXXRecordDecl>().match(
      ToTU, cxxRecordDecl(hasName("X"), unless(isDefinition())));
  // An error is NOT set for the fwd Decl of X in the shared state.
  OptErr = SharedStatePtr->getImportDeclErrorIfAny(ToXFwd);
  ASSERT_FALSE(OptErr);

  // Try to import  X again but from another TU.
  {
    TranslationUnitDecl *FromTU = getTuDecl(std::string(R"(
        class X {
          void f() { )") + ErroneousStmt + R"( }
          void ok();
        };
        )",
                                            Lang_CXX03, "input1.cc");

    auto *FromX = FirstDeclMatcher<CXXRecordDecl>().match(
        FromTU, cxxRecordDecl(hasName("X")));
    CXXRecordDecl *ImportedX = Import(FromX, Lang_CXX03);

    // If we did not save the errors for the "to" context then the below checks
    // would fail, because the lookup finds the fwd Decl of the existing
    // definition in the "to" context. We can reach the existing definition via
    // the found fwd Decl. That existing definition is structurally equivalent
    // (we check only the fields) with this one we want to import, so we return
    // with the existing definition, which is erroneous (one method is missing).

    // The import should fail.
    EXPECT_FALSE(ImportedX);
    ASTImporter *Importer = findFromTU(FromX)->Importer.get();
    std::optional<ASTImportError> OptErr =
        Importer->getImportDeclErrorIfAny(FromX);
    // And an error is set for this new X in the "from" ctx.
    ASSERT_TRUE(OptErr);
    EXPECT_EQ(OptErr->Error, ASTImportError::UnsupportedConstruct);
  }
}

TEST_P(ErrorHandlingTest, ImportOfOverriddenMethods) {
  auto MatchFooA =
      functionDecl(hasName("foo"), hasAncestor(cxxRecordDecl(hasName("A"))));
  auto MatchFooB =
      functionDecl(hasName("foo"), hasAncestor(cxxRecordDecl(hasName("B"))));
  auto MatchFooC =
      functionDecl(hasName("foo"), hasAncestor(cxxRecordDecl(hasName("C"))));

  // Provoke import of a method that has overridden methods with import error.
  TranslationUnitDecl *FromTU = getTuDecl(std::string(R"(
        struct C;
        struct A {
          virtual void foo();
          void f1(C *);
        };
        void A::foo() {
          )") + ErroneousStmt + R"(
        }
        struct B : public A {
          void foo() override;
        };
        struct C : public B {
          void foo() override;
        };
        )",
                                          Lang_CXX11);
  auto *FromFooA = FirstDeclMatcher<FunctionDecl>().match(FromTU, MatchFooA);
  auto *FromFooB = FirstDeclMatcher<FunctionDecl>().match(FromTU, MatchFooB);
  auto *FromFooC = FirstDeclMatcher<FunctionDecl>().match(FromTU, MatchFooC);

  EXPECT_FALSE(Import(FromFooA, Lang_CXX11));
  ASTImporter *Importer = findFromTU(FromFooA)->Importer.get();
  auto CheckError = [&Importer](Decl *FromD) {
    std::optional<ASTImportError> OptErr =
        Importer->getImportDeclErrorIfAny(FromD);
    ASSERT_TRUE(OptErr);
    EXPECT_EQ(OptErr->Error, ASTImportError::UnsupportedConstruct);
  };
  CheckError(FromFooA);
  EXPECT_FALSE(Import(FromFooB, Lang_CXX11));
  CheckError(FromFooB);
  EXPECT_FALSE(Import(FromFooC, Lang_CXX11));
  CheckError(FromFooC);
}

TEST_P(ErrorHandlingTest, ODRViolationWithinParmVarDecls) {
  // Importing of 'f' and parameter 'P' should cause an ODR error.
  // The error happens after the ParmVarDecl for 'P' was already created.
  // This is a special case because the ParmVarDecl has a temporary DeclContext.
  // Expected is no crash at error handling of ASTImporter.
  constexpr auto ToTUCode = R"(
      struct X {
        char A;
      };
      )";
  constexpr auto FromTUCode = R"(
      struct X {
        enum Y { Z };
      };
      void f(int P = X::Z);
      )";
  Decl *ToTU = getToTuDecl(ToTUCode, Lang_CXX11);
  static_cast<void>(ToTU);
  Decl *FromTU = getTuDecl(FromTUCode, Lang_CXX11);
  auto *FromF = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("f")));
  ASSERT_TRUE(FromF);

  auto *ImportedF = Import(FromF, Lang_CXX11);
  EXPECT_FALSE(ImportedF);
}

TEST_P(ErrorHandlingTest, DoNotInheritErrorFromNonDependentChild) {
  // Declarations should not inherit an import error from a child object
  // if the declaration has no direct dependence to such a child.
  // For example a namespace should not get import error if one of the
  // declarations inside it fails to import.
  // There was a special case in error handling (when "import path circles" are
  // encountered) when this property was not held. This case is provoked by the
  // following code.
  constexpr auto ToTUCode = R"(
      namespace ns {
        struct Err {
          char A;
        };
      }
      )";
  constexpr auto FromTUCode = R"(
      namespace ns {
        struct A {
          using U = struct Err;
        };
      }
      namespace ns {
        struct Err {}; // ODR violation
        void f(A) {}
      }
      )";

  Decl *ToTU = getToTuDecl(ToTUCode, Lang_CXX11);
  static_cast<void>(ToTU);
  Decl *FromTU = getTuDecl(FromTUCode, Lang_CXX11);
  auto *FromA = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("A"), hasDefinition()));
  ASSERT_TRUE(FromA);
  auto *ImportedA = Import(FromA, Lang_CXX11);
  // 'A' can not be imported: ODR error at 'Err'
  EXPECT_FALSE(ImportedA);
  // When import of 'A' failed there was a "saved import path circle" that
  // contained namespace 'ns' (A - U - Err - ns - f - A). This should not mean
  // that every object in this path fails to import.

  Decl *FromNS = FirstDeclMatcher<NamespaceDecl>().match(
      FromTU, namespaceDecl(hasName("ns")));
  EXPECT_TRUE(FromNS);
  auto *ImportedNS = Import(FromNS, Lang_CXX11);
  EXPECT_TRUE(ImportedNS);
}

TEST_P(ASTImporterOptionSpecificTestBase, LambdaInFunctionBody) {
  Decl *FromTU = getTuDecl(
      R"(
      void f() {
        auto L = [](){};
      }
      )",
      Lang_CXX11, "input0.cc");
  auto Pattern = lambdaExpr();
  CXXRecordDecl *FromL =
      FirstDeclMatcher<LambdaExpr>().match(FromTU, Pattern)->getLambdaClass();

  auto ToL = Import(FromL, Lang_CXX11);
  unsigned ToLSize = std::distance(ToL->decls().begin(), ToL->decls().end());
  unsigned FromLSize =
      std::distance(FromL->decls().begin(), FromL->decls().end());
  EXPECT_NE(ToLSize, 0u);
  EXPECT_EQ(ToLSize, FromLSize);
  EXPECT_FALSE(FromL->isDependentLambda());
}

TEST_P(ASTImporterOptionSpecificTestBase, LambdaInFunctionParam) {
  Decl *FromTU = getTuDecl(
      R"(
      template <typename F>
      void f(F L = [](){}) {}
      )",
      Lang_CXX11, "input0.cc");
  auto Pattern = lambdaExpr();
  CXXRecordDecl *FromL =
      FirstDeclMatcher<LambdaExpr>().match(FromTU, Pattern)->getLambdaClass();

  auto ToL = Import(FromL, Lang_CXX11);
  unsigned ToLSize = std::distance(ToL->decls().begin(), ToL->decls().end());
  unsigned FromLSize =
      std::distance(FromL->decls().begin(), FromL->decls().end());
  EXPECT_NE(ToLSize, 0u);
  EXPECT_EQ(ToLSize, FromLSize);
  EXPECT_TRUE(FromL->isDependentLambda());
}

TEST_P(ASTImporterOptionSpecificTestBase, LambdaInGlobalScope) {
  Decl *FromTU = getTuDecl(
      R"(
      auto l1 = [](unsigned lp) { return 1; };
      auto l2 = [](int lp) { return 2; };
      int f(int p) {
        return l1(p) + l2(p);
      }
      )",
      Lang_CXX11, "input0.cc");
  FunctionDecl *FromF = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("f")));
  FunctionDecl *ToF = Import(FromF, Lang_CXX11);
  EXPECT_TRUE(ToF);
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportExistingFriendClassTemplateDef) {
  auto Code =
      R"(
        template <class T1, class T2>
        struct Base {
          template <class U1, class U2>
          friend struct Class;
        };
        template <class T1, class T2>
        struct Class { };
        )";

  TranslationUnitDecl *ToTU = getToTuDecl(Code, Lang_CXX03);
  TranslationUnitDecl *FromTU = getTuDecl(Code, Lang_CXX03, "input.cc");

  auto *ToClassProto = FirstDeclMatcher<ClassTemplateDecl>().match(
      ToTU, classTemplateDecl(hasName("Class")));
  auto *ToClassDef = LastDeclMatcher<ClassTemplateDecl>().match(
      ToTU, classTemplateDecl(hasName("Class")));
  ASSERT_FALSE(ToClassProto->isThisDeclarationADefinition());
  ASSERT_TRUE(ToClassDef->isThisDeclarationADefinition());
  // Previous friend decl is not linked to it!
  ASSERT_FALSE(ToClassDef->getPreviousDecl());
  ASSERT_EQ(ToClassDef->getMostRecentDecl(), ToClassDef);
  ASSERT_EQ(ToClassProto->getMostRecentDecl(), ToClassProto);

  auto *FromClassProto = FirstDeclMatcher<ClassTemplateDecl>().match(
      FromTU, classTemplateDecl(hasName("Class")));
  auto *FromClassDef = LastDeclMatcher<ClassTemplateDecl>().match(
      FromTU, classTemplateDecl(hasName("Class")));
  ASSERT_FALSE(FromClassProto->isThisDeclarationADefinition());
  ASSERT_TRUE(FromClassDef->isThisDeclarationADefinition());
  ASSERT_FALSE(FromClassDef->getPreviousDecl());
  ASSERT_EQ(FromClassDef->getMostRecentDecl(), FromClassDef);
  ASSERT_EQ(FromClassProto->getMostRecentDecl(), FromClassProto);

  auto *ImportedDef = Import(FromClassDef, Lang_CXX03);
  // At import we should find the definition for 'Class' even if the
  // prototype (inside 'friend') for it comes first in the AST and is not
  // linked to the definition.
  EXPECT_EQ(ImportedDef, ToClassDef);
}

struct LLDBLookupTest : ASTImporterOptionSpecificTestBase {
  LLDBLookupTest() {
    Creator = [](ASTContext &ToContext, FileManager &ToFileManager,
                 ASTContext &FromContext, FileManager &FromFileManager,
                 bool MinimalImport,
                 const std::shared_ptr<ASTImporterSharedState> &SharedState) {
      return new ASTImporter(ToContext, ToFileManager, FromContext,
                             FromFileManager, MinimalImport,
                             // We use the regular lookup.
                             /*SharedState=*/nullptr);
    };
  }
};

TEST_P(LLDBLookupTest, ImporterShouldFindInTransparentContext) {
  TranslationUnitDecl *ToTU = getToTuDecl(
      R"(
      extern "C" {
        class X{};
      };
      )",
      Lang_CXX03);
  auto *ToX = FirstDeclMatcher<CXXRecordDecl>().match(
      ToTU, cxxRecordDecl(hasName("X")));

  // Set up a stub external storage.
  ToTU->setHasExternalLexicalStorage(true);
  // Set up DeclContextBits.HasLazyExternalLexicalLookups to true.
  ToTU->setMustBuildLookupTable();
  struct TestExternalASTSource : ExternalASTSource {};
  ToTU->getASTContext().setExternalSource(new TestExternalASTSource());

  Decl *FromTU = getTuDecl(
      R"(
        class X;
      )",
      Lang_CXX03);
  auto *FromX = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("X")));
  auto *ImportedX = Import(FromX, Lang_CXX03);
  // The lookup must find the existing class definition in the LinkageSpecDecl.
  // Then the importer renders the existing and the new decl into one chain.
  EXPECT_EQ(ImportedX->getCanonicalDecl(), ToX->getCanonicalDecl());
}

struct SVEBuiltins : ASTImporterOptionSpecificTestBase {};

TEST_P(SVEBuiltins, ImportTypes) {
  static const char *const TypeNames[] = {
    "__SVInt8_t",
    "__SVInt16_t",
    "__SVInt32_t",
    "__SVInt64_t",
    "__SVUint8_t",
    "__SVUint16_t",
    "__SVUint32_t",
    "__SVUint64_t",
    "__SVFloat16_t",
    "__SVBFloat16_t",
    "__SVFloat32_t",
    "__SVFloat64_t",
    "__SVBool_t"
  };

  TranslationUnitDecl *ToTU = getToTuDecl("", Lang_CXX03);
  TranslationUnitDecl *FromTU = getTuDecl("", Lang_CXX03, "input.cc");
  for (auto *TypeName : TypeNames) {
    auto *ToTypedef = FirstDeclMatcher<TypedefDecl>().match(
      ToTU, typedefDecl(hasName(TypeName)));
    QualType ToType = ToTypedef->getUnderlyingType();

    auto *FromTypedef = FirstDeclMatcher<TypedefDecl>().match(
      FromTU, typedefDecl(hasName(TypeName)));
    QualType FromType = FromTypedef->getUnderlyingType();

    QualType ImportedType = ImportType(FromType, FromTypedef, Lang_CXX03);
    EXPECT_EQ(ImportedType, ToType);
  }
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportOfDefaultImplicitFunctions) {
  // Test that import of implicit functions works and the functions
  // are merged into one chain.
  auto GetDeclToImport = [this](StringRef File) {
    Decl *FromTU = getTuDecl(
        R"(
        struct X { };
        // Force generating some implicit operator definitions for X.
        void f() { X x1, x2; x1 = x2; X *x3 = new X; delete x3; }
        )",
        Lang_CXX11, File);
    auto *FromD = FirstDeclMatcher<CXXRecordDecl>().match(
        FromTU, cxxRecordDecl(hasName("X"), unless(isImplicit())));
    // Destructor is picked as one example of implicit function.
    return FromD->getDestructor();
  };

  auto *ToD1 = Import(GetDeclToImport("input1.cc"), Lang_CXX11);
  ASSERT_TRUE(ToD1);

  auto *ToD2 = Import(GetDeclToImport("input2.cc"), Lang_CXX11);
  ASSERT_TRUE(ToD2);

  EXPECT_EQ(ToD1->getCanonicalDecl(), ToD2->getCanonicalDecl());
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportOfExplicitlyDefaultedOrDeleted) {
  Decl *FromTU = getTuDecl(
      R"(
        struct X { X() = default; X(const X&) = delete; };
      )",
      Lang_CXX11);
  auto *FromX = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("X")));
  auto *ImportedX = Import(FromX, Lang_CXX11);
  auto *Constr1 = FirstDeclMatcher<CXXConstructorDecl>().match(
      ImportedX, cxxConstructorDecl(hasName("X"), unless(isImplicit())));
  auto *Constr2 = LastDeclMatcher<CXXConstructorDecl>().match(
      ImportedX, cxxConstructorDecl(hasName("X"), unless(isImplicit())));

  ASSERT_TRUE(ImportedX);
  EXPECT_TRUE(Constr1->isDefaulted());
  EXPECT_TRUE(Constr1->isExplicitlyDefaulted());
  EXPECT_TRUE(Constr2->isDeletedAsWritten());
  EXPECT_EQ(ImportedX->isAggregate(), FromX->isAggregate());
}

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, SVEBuiltins,
                         ::testing::Values(std::vector<std::string>{
                             "-target", "aarch64-linux-gnu"}));

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, DeclContextTest,
                         ::testing::Values(std::vector<std::string>()));

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, CanonicalRedeclChain,
                         ::testing::Values(std::vector<std::string>()));

TEST_P(ASTImporterOptionSpecificTestBase, LambdasAreDifferentiated) {
  Decl *FromTU = getTuDecl(
      R"(
      void f() {
        auto L0 = [](){};
        auto L1 = [](){};
      }
      )",
      Lang_CXX11, "input0.cc");
  auto Pattern = lambdaExpr();
  CXXRecordDecl *FromL0 =
      FirstDeclMatcher<LambdaExpr>().match(FromTU, Pattern)->getLambdaClass();
  CXXRecordDecl *FromL1 =
      LastDeclMatcher<LambdaExpr>().match(FromTU, Pattern)->getLambdaClass();
  ASSERT_NE(FromL0, FromL1);

  CXXRecordDecl *ToL0 = Import(FromL0, Lang_CXX11);
  CXXRecordDecl *ToL1 = Import(FromL1, Lang_CXX11);
  EXPECT_NE(ToL0, ToL1);
}

TEST_P(ASTImporterOptionSpecificTestBase,
       LambdasInFunctionParamsAreDifferentiated) {
  Decl *FromTU = getTuDecl(
      R"(
      template <typename F0, typename F1>
      void f(F0 L0 = [](){}, F1 L1 = [](){}) {}
      )",
      Lang_CXX11, "input0.cc");
  auto Pattern = cxxRecordDecl(isLambda());
  CXXRecordDecl *FromL0 =
      FirstDeclMatcher<CXXRecordDecl>().match(FromTU, Pattern);
  CXXRecordDecl *FromL1 =
      LastDeclMatcher<CXXRecordDecl>().match(FromTU, Pattern);
  ASSERT_NE(FromL0, FromL1);

  CXXRecordDecl *ToL0 = Import(FromL0, Lang_CXX11);
  CXXRecordDecl *ToL1 = Import(FromL1, Lang_CXX11);
  ASSERT_NE(ToL0, ToL1);
}

TEST_P(ASTImporterOptionSpecificTestBase,
       LambdasInFunctionParamsAreDifferentiatedWhenMacroIsUsed) {
  Decl *FromTU = getTuDecl(
      R"(
      #define LAMBDA [](){}
      template <typename F0, typename F1>
      void f(F0 L0 = LAMBDA, F1 L1 = LAMBDA) {}
      )",
      Lang_CXX11, "input0.cc");
  auto Pattern = cxxRecordDecl(isLambda());
  CXXRecordDecl *FromL0 =
      FirstDeclMatcher<CXXRecordDecl>().match(FromTU, Pattern);
  CXXRecordDecl *FromL1 =
      LastDeclMatcher<CXXRecordDecl>().match(FromTU, Pattern);
  ASSERT_NE(FromL0, FromL1);

  Import(FromL0, Lang_CXX11);
  Import(FromL1, Lang_CXX11);
  CXXRecordDecl *ToL0 = Import(FromL0, Lang_CXX11);
  CXXRecordDecl *ToL1 = Import(FromL1, Lang_CXX11);
  ASSERT_NE(ToL0, ToL1);
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportAssignedLambda) {
  Decl *FromTU = getTuDecl(
      R"(
      void f() {
        auto x = []{} = {}; auto x2 = x;
      }
      )",
      Lang_CXX20, "input0.cc");
  auto FromF = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("f")));
  // We have only one lambda class.
  ASSERT_EQ(
      DeclCounter<CXXRecordDecl>().match(FromTU, cxxRecordDecl(isLambda())),
      1u);

  FunctionDecl *ToF = Import(FromF, Lang_CXX20);
  EXPECT_TRUE(ToF);
  TranslationUnitDecl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  // We have only one lambda class after the import.
  EXPECT_EQ(DeclCounter<CXXRecordDecl>().match(ToTU, cxxRecordDecl(isLambda())),
            1u);
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportDefaultConstructibleLambdas) {
  Decl *FromTU = getTuDecl(
      R"(
      void f() {
        auto x = []{} = {};
        auto xb = []{} = {};
      }
      )",
      Lang_CXX20, "input0.cc");
  auto FromF = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("f")));
  // We have two lambda classes.
  ASSERT_EQ(
      DeclCounter<CXXRecordDecl>().match(FromTU, cxxRecordDecl(isLambda())),
      2u);

  FunctionDecl *ToF = Import(FromF, Lang_CXX20);
  EXPECT_TRUE(ToF);
  TranslationUnitDecl *ToTU = ToAST->getASTContext().getTranslationUnitDecl();
  // We have two lambda classes after the import.
  EXPECT_EQ(DeclCounter<CXXRecordDecl>().match(ToTU, cxxRecordDecl(isLambda())),
            2u);
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportFunctionDeclWithTypeSourceInfoWithSourceDecl) {
  // This code results in a lambda with implicit constructor.
  // The constructor's TypeSourceInfo points out the function prototype.
  // This prototype has an EST_Unevaluated in its exception information and a
  // SourceDecl that is the function declaration itself.
  // The test verifies that AST import of such AST does not crash.
  // (Here the function's TypeSourceInfo references the function itself.)
  Decl *FromTU = getTuDecl(
      R"(
        template<typename T> void f(T) { auto X = [](){}; }
        void g() { f(10); }
        )",
      Lang_CXX11, "input0.cc");

  // Use LastDeclMatcher to find the LambdaExpr in the template specialization.
  CXXRecordDecl *FromL = LastDeclMatcher<LambdaExpr>()
                             .match(FromTU, lambdaExpr())
                             ->getLambdaClass();

  CXXConstructorDecl *FromCtor = *FromL->ctor_begin();
  ASSERT_TRUE(FromCtor->isCopyConstructor());
  ASSERT_TRUE(FromCtor->getTypeSourceInfo());
  const auto *FromFPT = FromCtor->getType()->getAs<FunctionProtoType>();
  ASSERT_TRUE(FromFPT);
  EXPECT_EQ(FromCtor->getTypeSourceInfo()->getType().getTypePtr(), FromFPT);
  FunctionProtoType::ExtProtoInfo FromEPI = FromFPT->getExtProtoInfo();
  // If type is EST_Unevaluated, SourceDecl should be set to the parent Decl.
  EXPECT_EQ(FromEPI.ExceptionSpec.Type, EST_Unevaluated);
  EXPECT_EQ(FromEPI.ExceptionSpec.SourceDecl, FromCtor);

  auto ToL = Import(FromL, Lang_CXX11);

  // Check if the import was correct.
  CXXConstructorDecl *ToCtor = *ToL->ctor_begin();
  EXPECT_TRUE(ToCtor->getTypeSourceInfo());
  const auto *ToFPT = ToCtor->getType()->getAs<FunctionProtoType>();
  ASSERT_TRUE(ToFPT);
  EXPECT_EQ(ToCtor->getTypeSourceInfo()->getType().getTypePtr(), ToFPT);
  FunctionProtoType::ExtProtoInfo ToEPI = ToFPT->getExtProtoInfo();
  EXPECT_EQ(ToEPI.ExceptionSpec.Type, EST_Unevaluated);
  EXPECT_EQ(ToEPI.ExceptionSpec.SourceDecl, ToCtor);
}

struct ImportAutoFunctions : ASTImporterOptionSpecificTestBase {
  void testImport(llvm::StringRef Code, clang::TestLanguage Lang = Lang_CXX14) {
    Decl *FromTU = getTuDecl(Code, Lang, "input0.cc");
    FunctionDecl *From = FirstDeclMatcher<FunctionDecl>().match(
        FromTU, functionDecl(hasName("foo")));

    FunctionDecl *To = Import(From, Lang);
    EXPECT_TRUE(To);
    // We check here only that the type is auto type.
    // These tests are to verify that no crash happens.
    // The crash possibility is the presence of a reference to a declaration
    // in the function's body from the return type, if the function has auto
    // return type.
    EXPECT_TRUE(isa<AutoType>(To->getReturnType()));
  }
};

TEST_P(ImportAutoFunctions, ReturnWithFunctionTemplate1) {
  testImport(
      R"(
      template<class C>
      C f1() { return C(); }
      auto foo() {
          struct B {};
          return f1<B>();
      }
      )");
}

TEST_P(ImportAutoFunctions, ReturnWithFunctionTemplate2) {
  testImport(
      R"(
      template<class T>
      int f1(T t) { return 1; }
      auto foo() {
          struct B {};
          return f1(B());
      }
      )");
}

TEST_P(ImportAutoFunctions, ReturnWithFunctionTemplate3) {
  testImport(
      R"(
      template<class A> struct S1 {};
      template<class A> struct S2 {};
      template<class C>
      S1<C> f1() { return S1<C>(); }
      auto foo() {
          struct B {};
          return f1<S2<B *>>();
      }
      )");
}

TEST_P(ImportAutoFunctions, ReturnWithFunctionTemplate4) {
  testImport(
      R"(
      template<class... A> struct S1 {};
      template<class... A> struct S2 {};
      template<class... C>
      S1<C...> f1() { return S1<C...>(); }
      auto foo() {
          struct B {};
          return f1<S2<int, B *>, bool>();
      }
      )");
}

TEST_P(ImportAutoFunctions, ReturnWithVarTemplate1) {
  testImport(
      R"(
      template<class T> T X;
      auto foo() {
          struct A {};
          return X<A>;
      }
      )");
}

TEST_P(ImportAutoFunctions, ReturnWithVarTemplate2) {
  testImport(
      R"(
      template<class A> struct S1 {};
      template<class T> S1<T> X;
      auto foo() {
          struct A {};
          return X<S1<A>>;
      }
      )");
}

TEST_P(ImportAutoFunctions, ReturnWithVarTemplate3) {
  testImport(
      R"(
      template<class... A> struct S1 {};
      template<class... T> S1<T...> X;
      auto foo() {
          struct A {};
          return X<bool, S1<A, int>>;
      }
      )");
}

TEST_P(ImportAutoFunctions, ReturnWithAutoUnresolvedArg) {
  testImport(
      R"(
      template<int A>
      auto foo() {
        return 22;
      }
      )");
}

TEST_P(ImportAutoFunctions, ReturnWithTemplateWithTemplateTemplateArg) {
  // FIXME: Is it possible to have the template arg inside the function?
  testImport(
      R"(
      template<int> struct Tmpl {};
      template<template<int> class> struct TmplTmpl {};
      auto foo() {
        return TmplTmpl<Tmpl>();
      }
      )");
}

TEST_P(ImportAutoFunctions, ReturnWithTemplateWithDeclarationTemplateArg) {
  // FIXME: Is it possible to have the template arg inside the function?
  testImport(
      R"(
      template<const int *> struct Tmpl {};
      int A[10];
      auto foo() {
        return Tmpl<A>();
      }
      )");
}

TEST_P(ImportAutoFunctions, ReturnWithTemplateWithNullPtrTemplateArg) {
  testImport(
      R"(
      template<int *> struct Tmpl {};
      auto foo() {
        constexpr int* A = nullptr;
        return Tmpl<A>();
      }
      )");
}

TEST_P(ImportAutoFunctions, ReturnWithTemplateWithIntegralTemplateArg) {
  testImport(
      R"(
      template<int> struct Tmpl {};
      auto foo() {
        using Int = int;
        constexpr Int A = 7;
        return Tmpl<A>();
      }
      )");
}

TEST_P(ImportAutoFunctions, ReturnWithTemplateWithDecltypeTypeDeclaredInside) {
  testImport(
      R"(
      template<class> struct Tmpl {};
      auto foo() {
        struct X {};
        X x;
        return Tmpl<decltype(x)>();
      }
      )");
}

TEST_P(ImportAutoFunctions, ReturnWithTemplateWithUsingTypeDeclaredInside) {
  testImport(
      R"(
      template<class> struct Tmpl {};
      namespace A { struct X {}; }
      auto foo() {
        using A::X;
        return Tmpl<X>();
      }
      )");
}

TEST_P(ImportAutoFunctions, ReturnWithTemplateWithArrayTypeDeclaredInside) {
  testImport(
      R"(
      template<class> struct Tmpl {};
      auto foo() {
        struct X {};
        return Tmpl<X[10]>();
      }
      )");
}

TEST_P(ImportAutoFunctions, ReturnWithTemplateWithArraySizeExprDeclaredInside) {
  testImport(
      R"(
      template<class> struct Tmpl {};
      auto foo() {
        constexpr int S = 10;
        return Tmpl<int[S]>();
      }
      )");
}

TEST_P(ImportAutoFunctions, ReturnWithTemplateWithPackArgDeclaredInside) {
  testImport(
      R"(
      template<class ...> struct Tmpl {};
      auto foo() {
        using X = bool;
        return Tmpl<int, X>();
      }
      )");
}

TEST_P(ImportAutoFunctions, ReturnWithTemplateWithIntegerArgDeclaredInside) {
  testImport(
      R"(
      template<int> struct Tmpl {};
      auto foo() {
        constexpr int X = 1;
        return Tmpl<X>();
      }
      )");
}

TEST_P(ImportAutoFunctions, ReturnWithTemplateWithPtrToStructDeclaredInside) {
  testImport(
      R"(
      template<class> struct Tmpl {};
      auto foo() {
        struct X {};
        return Tmpl<X *>();
      }
      )");
}

TEST_P(ImportAutoFunctions, ReturnWithTemplateWithRefToStructDeclaredInside) {
  testImport(
      R"(
      template<class> struct Tmpl {};
      struct X {};
      auto foo() {
        using Y = X;
        return Tmpl<Y &>();
      }
      )");
}

TEST_P(ImportAutoFunctions, ReturnWithTemplateWithStructDeclaredInside1) {
  testImport(
      R"(
      template<class> struct Tmpl {};
      auto foo() {
        struct X {};
        return Tmpl<X>();
      }
      )");
}

TEST_P(ImportAutoFunctions, ReturnWithTemplateWithStructDeclaredInside2) {
  testImport(
      R"(
      template<class> struct Tmpl {};
      auto foo() {
        struct X {};
        return Tmpl<Tmpl<X>>();
      }
      )");
}

TEST_P(ImportAutoFunctions, ReturnWithTemplateWithTypedefDeclaredInside) {
  testImport(
      R"(
      template<class> struct Tmpl {};
      auto foo() {
        struct X {};
        using x_type = X;
        return Tmpl<x_type>();
      }
      )");
}

TEST_P(ImportAutoFunctions, ReturnWithTypedefDeclaredInside) {
  Decl *FromTU = getTuDecl(
      R"(
      auto X = [](long l) {
        using int_type = long;
        auto dur = 13;
        return static_cast<int_type>(dur);
      };
      )",
      Lang_CXX14, "input0.cc");
  CXXMethodDecl *From =
      FirstDeclMatcher<CXXMethodDecl>().match(FromTU, cxxMethodDecl());

  // Explicitly set the return type of the lambda's operator() to the TypeAlias.
  // Normally the return type would be the built-in 'long' type. However, there
  // are cases when Clang does not use the canonical type and the TypeAlias is
  // used. I could not create such an AST from regular source code, it requires
  // some special state in the preprocessor. I've found such an AST when Clang
  // parsed libcxx/src/filesystem/directory_iterator.cpp, but could not reduce
  // that with creduce, because after preprocessing, the AST no longer
  // contained the TypeAlias as a return type of the lambda.
  ASTContext &Ctx = From->getASTContext();
  TypeAliasDecl *FromTA =
      FirstDeclMatcher<TypeAliasDecl>().match(FromTU, typeAliasDecl());
  QualType TT = Ctx.getTypedefType(FromTA);
  const FunctionProtoType *FPT = cast<FunctionProtoType>(From->getType());
  QualType NewFunType =
      Ctx.getFunctionType(TT, FPT->getParamTypes(), FPT->getExtProtoInfo());
  From->setType(NewFunType);

  CXXMethodDecl *To = Import(From, Lang_CXX14);
  EXPECT_TRUE(To);
  EXPECT_TRUE(isa<TypedefType>(To->getReturnType()));
}

TEST_P(ImportAutoFunctions, ReturnWithStructDeclaredInside) {
  testImport(
      R"(
      auto foo() {
        struct X {};
        return X();
      }
      )");
}

TEST_P(ImportAutoFunctions, ReturnWithStructDeclaredInside2) {
  Decl *FromTU = getTuDecl(
      R"(
      auto foo() {
        struct X {};
        return X();
      }
      )",
      Lang_CXX14, "input0.cc");
  FunctionDecl *From =
      FirstDeclMatcher<FunctionDecl>().match(FromTU, functionDecl());

  // This time import the type directly.
  QualType ToT = ImportType(From->getType(), From, Lang_CXX14);
  const FunctionProtoType *FPT = cast<FunctionProtoType>(ToT);
  EXPECT_TRUE(isa<AutoType>(FPT->getReturnType()));
}

TEST_P(ImportAutoFunctions, ReturnWithStructDeclaredInside3) {
  Decl *FromTU = getTuDecl(
      R"(
      struct S {
        constexpr auto foo();
      };
      constexpr auto S::foo() {
        struct X {};
        return X();
      }
      )",
      Lang_CXX14, "input0.cc");
  FunctionDecl *From = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("foo"), unless(hasBody(stmt()))));
  ASSERT_FALSE(From->isThisDeclarationADefinition());

  FunctionDecl *To = Import(From, Lang_CXX17);
  EXPECT_TRUE(To);
  EXPECT_TRUE(isa<AutoType>(To->getReturnType()));
  EXPECT_FALSE(To->isThisDeclarationADefinition());
}

TEST_P(ImportAutoFunctions, ReturnWithTypedefToStructDeclaredInside) {
  testImport(
      R"(
      auto foo() {
        struct X {};
        using Y = X;
        return Y();
      }
      )");
}

TEST_P(ImportAutoFunctions, ReturnWithStructDeclaredNestedInside) {
  testImport(
      R"(
      auto foo() {
        struct X { struct Y{}; };
        return X::Y();
      }
      )");
}

TEST_P(ImportAutoFunctions, ReturnWithInternalLambdaType) {
  testImport(
      R"(
      auto foo() {
        auto l = []() {
          struct X {};
          return X();
        };
        return l();
      }
      )",
      Lang_CXX17);
}

TEST_P(ImportAutoFunctions, ReturnWithTypeInIf) {
  testImport(
      R"(
      auto foo() {
        if (struct X {} x; true)
          return X();
        else
          return X();
      }
      )",
      Lang_CXX17);
}

TEST_P(ImportAutoFunctions, ReturnWithTypeInFor) {
  testImport(
      R"(
      auto foo() {
        for (struct X {} x;;)
          return X();
      }
      )",
      Lang_CXX17);
}

TEST_P(ImportAutoFunctions, ReturnWithTypeInSwitch) {
  testImport(
      R"(
      auto foo() {
        switch (struct X {} x; 10) {
        case 10:
          return X();
        }
      }
      )",
      Lang_CXX17);
}

struct ImportSourceLocations : ASTImporterOptionSpecificTestBase {};

TEST_P(ImportSourceLocations, PreserveFileIDTreeStructure) {
  // Tests that the FileID tree structure (with the links being the include
  // chains) is preserved while importing other files (which need to be
  // added to this structure with fake include locations.

  SourceLocation Location1;
  {
    auto Pattern = varDecl(hasName("X"));
    Decl *FromTU = getTuDecl("int X;", Lang_C99, "input0.c");
    auto *FromD = FirstDeclMatcher<VarDecl>().match(FromTU, Pattern);

    Location1 = Import(FromD, Lang_C99)->getLocation();
  }
  SourceLocation Location2;
  {
    auto Pattern = varDecl(hasName("Y"));
    Decl *FromTU = getTuDecl("int Y;", Lang_C99, "input1.c");
    auto *FromD = FirstDeclMatcher<VarDecl>().match(FromTU, Pattern);

    Location2 = Import(FromD, Lang_C99)->getLocation();
  }

  SourceManager &ToSM = ToAST->getSourceManager();
  FileID FileID1 = ToSM.getFileID(Location1);
  FileID FileID2 = ToSM.getFileID(Location2);

  // Check that the imported files look like as if they were included from the
  // start of the main file.
  SourceLocation FileStart = ToSM.getLocForStartOfFile(ToSM.getMainFileID());
  EXPECT_NE(FileID1, ToSM.getMainFileID());
  EXPECT_NE(FileID2, ToSM.getMainFileID());
  EXPECT_EQ(ToSM.getIncludeLoc(FileID1), FileStart);
  EXPECT_EQ(ToSM.getIncludeLoc(FileID2), FileStart);

  // Let the SourceManager check the order of the locations. The order should
  // be the order in which the declarations are imported.
  EXPECT_TRUE(ToSM.isBeforeInTranslationUnit(Location1, Location2));
  EXPECT_FALSE(ToSM.isBeforeInTranslationUnit(Location2, Location1));
}

TEST_P(ImportSourceLocations, NormalFileBuffer) {
  // Test importing normal file buffers.

  std::string Path = "input0.c";
  std::string Source = "int X;";
  TranslationUnitDecl *FromTU = getTuDecl(Source, Lang_C99, Path);

  SourceLocation ImportedLoc;
  {
    // Import the VarDecl to trigger the importing of the FileID.
    auto Pattern = varDecl(hasName("X"));
    VarDecl *FromD = FirstDeclMatcher<VarDecl>().match(FromTU, Pattern);
    ImportedLoc = Import(FromD, Lang_C99)->getLocation();
  }

  // Make sure the imported buffer has the original contents.
  SourceManager &ToSM = ToAST->getSourceManager();
  FileID ImportedID = ToSM.getFileID(ImportedLoc);
  EXPECT_EQ(Source,
            ToSM.getBufferOrFake(ImportedID, SourceLocation()).getBuffer());
}

TEST_P(ImportSourceLocations, OverwrittenFileBuffer) {
  // Test importing overwritten file buffers.

  std::string Path = "input0.c";
  TranslationUnitDecl *FromTU = getTuDecl("int X;", Lang_C99, Path);

  // Overwrite the file buffer for our input file with new content.
  const std::string Contents = "overwritten contents";
  SourceLocation ImportedLoc;
  {
    SourceManager &FromSM = FromTU->getASTContext().getSourceManager();
    clang::FileManager &FM = FromSM.getFileManager();
    clang::FileEntryRef FE =
        FM.getVirtualFileRef(Path, static_cast<off_t>(Contents.size()), 0);

    llvm::SmallVector<char, 64> Buffer;
    Buffer.append(Contents.begin(), Contents.end());
    auto FileContents = std::make_unique<llvm::SmallVectorMemoryBuffer>(
        std::move(Buffer), Path, /*RequiresNullTerminator=*/false);
    FromSM.overrideFileContents(FE, std::move(FileContents));

    // Import the VarDecl to trigger the importing of the FileID.
    auto Pattern = varDecl(hasName("X"));
    VarDecl *FromD = FirstDeclMatcher<VarDecl>().match(FromTU, Pattern);
    ImportedLoc = Import(FromD, Lang_C99)->getLocation();
  }

  // Make sure the imported buffer has the overwritten contents.
  SourceManager &ToSM = ToAST->getSourceManager();
  FileID ImportedID = ToSM.getFileID(ImportedLoc);
  EXPECT_EQ(Contents,
            ToSM.getBufferOrFake(ImportedID, SourceLocation()).getBuffer());
}

struct ImportAttributes : public ASTImporterOptionSpecificTestBase {
  void checkAttrImportCommon(const Attr *From, const Attr *To,
                             const Decl *ToD) {

    // Verify that dump does not crash because invalid data.
    ToD->dump(llvm::nulls());

    EXPECT_EQ(From->getParsedKind(), To->getParsedKind());
    EXPECT_EQ(From->getSyntax(), To->getSyntax());
    if (From->getAttrName()) {
      EXPECT_TRUE(To->getAttrName());
      EXPECT_STREQ(From->getAttrName()->getNameStart(),
                   To->getAttrName()->getNameStart());
    } else {
      EXPECT_FALSE(To->getAttrName());
    }
    if (From->getScopeName()) {
      EXPECT_TRUE(To->getScopeName());
      EXPECT_STREQ(From->getScopeName()->getNameStart(),
                   To->getScopeName()->getNameStart());
    } else {
      EXPECT_FALSE(To->getScopeName());
    }
    EXPECT_EQ(From->getSpellingListIndex(), To->getSpellingListIndex());
    EXPECT_STREQ(From->getSpelling(), To->getSpelling());
    EXPECT_EQ(From->isInherited(), To->isInherited());
    EXPECT_EQ(From->isImplicit(), To->isImplicit());
    EXPECT_EQ(From->isPackExpansion(), To->isPackExpansion());
    EXPECT_EQ(From->isLateParsed(), To->isLateParsed());
  }

  template <class DT, class AT>
  void importAttr(const char *Code, AT *&FromAttr, AT *&ToAttr) {
    static_assert(std::is_base_of<Attr, AT>::value, "AT should be an Attr");
    static_assert(std::is_base_of<Decl, DT>::value, "DT should be a Decl");

    Decl *FromTU = getTuDecl(Code, Lang_CXX11, "input.cc");
    DT *FromD =
        FirstDeclMatcher<DT>().match(FromTU, namedDecl(hasName("test")));
    ASSERT_TRUE(FromD);

    DT *ToD = Import(FromD, Lang_CXX11);
    ASSERT_TRUE(ToD);

    FromAttr = FromD->template getAttr<AT>();
    ToAttr = ToD->template getAttr<AT>();
    ASSERT_TRUE(FromAttr);
    EXPECT_TRUE(ToAttr);

    checkAttrImportCommon(FromAttr, ToAttr, ToD);
  }

  template <class T> void checkImported(const T *From, const T *To) {
    EXPECT_TRUE(To);
    EXPECT_NE(From, To);
  }

  template <class T>
  void checkImportVariadicArg(const llvm::iterator_range<T **> &From,
                              const llvm::iterator_range<T **> &To) {
    for (auto FromI = From.begin(), ToI = To.begin(); FromI != From.end();
         ++FromI, ++ToI) {
      ASSERT_NE(ToI, To.end());
      checkImported(*FromI, *ToI);
    }
  }
};

template <>
void ImportAttributes::checkImported<Decl>(const Decl *From, const Decl *To) {
  EXPECT_TRUE(To);
  EXPECT_NE(From, To);
  EXPECT_EQ(To->getTranslationUnitDecl(),
            ToAST->getASTContext().getTranslationUnitDecl());
}

// FIXME: Use ImportAttributes for this test.
TEST_P(ASTImporterOptionSpecificTestBase, ImportExprOfAlignmentAttr) {
  // Test if import of these packed and aligned attributes does not trigger an
  // error situation where source location from 'From' context is referenced in
  // 'To' context through evaluation of the alignof attribute.
  // This happens if the 'alignof(A)' expression is not imported correctly.
  Decl *FromTU = getTuDecl(
      R"(
      struct __attribute__((packed)) A { int __attribute__((aligned(8))) X; };
      struct alignas(alignof(A)) S {};
      )",
      Lang_CXX11, "input.cc");
  auto *FromD = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("S"), unless(isImplicit())));
  ASSERT_TRUE(FromD);

  auto *ToD = Import(FromD, Lang_CXX11);
  ASSERT_TRUE(ToD);

  auto *FromAttr = FromD->getAttr<AlignedAttr>();
  auto *ToAttr = ToD->getAttr<AlignedAttr>();
  EXPECT_EQ(FromAttr->isInherited(), ToAttr->isInherited());
  EXPECT_EQ(FromAttr->isPackExpansion(), ToAttr->isPackExpansion());
  EXPECT_EQ(FromAttr->isImplicit(), ToAttr->isImplicit());
  EXPECT_EQ(FromAttr->getSyntax(), ToAttr->getSyntax());
  EXPECT_EQ(FromAttr->getSemanticSpelling(), ToAttr->getSemanticSpelling());
  EXPECT_TRUE(ToAttr->getAlignmentExpr());

  auto *ToA = FirstDeclMatcher<CXXRecordDecl>().match(
      ToD->getTranslationUnitDecl(),
      cxxRecordDecl(hasName("A"), unless(isImplicit())));
  // Ensure that 'struct A' was imported (through reference from attribute of
  // 'S').
  EXPECT_TRUE(ToA);
}

// FIXME: Use ImportAttributes for this test.
TEST_P(ASTImporterOptionSpecificTestBase, ImportFormatAttr) {
  Decl *FromTU = getTuDecl(
      R"(
      int foo(const char * fmt, ...)
      __attribute__ ((__format__ (__scanf__, 1, 2)));
      )",
      Lang_CXX03, "input.cc");
  auto *FromD = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("foo")));
  ASSERT_TRUE(FromD);

  auto *ToD = Import(FromD, Lang_CXX03);
  ASSERT_TRUE(ToD);
  ToD->dump(); // Should not crash!

  auto *FromAttr = FromD->getAttr<FormatAttr>();
  auto *ToAttr = ToD->getAttr<FormatAttr>();
  EXPECT_EQ(FromAttr->isInherited(), ToAttr->isInherited());
  EXPECT_EQ(FromAttr->isPackExpansion(), ToAttr->isPackExpansion());
  EXPECT_EQ(FromAttr->isImplicit(), ToAttr->isImplicit());
  EXPECT_EQ(FromAttr->getSyntax(), ToAttr->getSyntax());
  EXPECT_EQ(FromAttr->getAttributeSpellingListIndex(),
            ToAttr->getAttributeSpellingListIndex());
  EXPECT_EQ(FromAttr->getType()->getName(), ToAttr->getType()->getName());
}

TEST_P(ImportAttributes, ImportEnableIf) {
  EnableIfAttr *FromAttr, *ToAttr;
  importAttr<FunctionDecl>(
      "void test(int A) __attribute__((enable_if(A == 1, \"message\")));",
      FromAttr, ToAttr);
  checkImported(FromAttr->getCond(), ToAttr->getCond());
  EXPECT_EQ(FromAttr->getMessage(), ToAttr->getMessage());
}

TEST_P(ImportAttributes, ImportGuardedVar) {
  GuardedVarAttr *FromAttr, *ToAttr;
  importAttr<VarDecl>("int test __attribute__((guarded_var));", FromAttr,
                      ToAttr);
}

TEST_P(ImportAttributes, ImportPtGuardedVar) {
  PtGuardedVarAttr *FromAttr, *ToAttr;
  importAttr<VarDecl>("int *test __attribute__((pt_guarded_var));", FromAttr,
                      ToAttr);
}

TEST_P(ImportAttributes, ImportScopedLockable) {
  ScopedLockableAttr *FromAttr, *ToAttr;
  importAttr<CXXRecordDecl>("struct __attribute__((scoped_lockable)) test {};",
                            FromAttr, ToAttr);
}

TEST_P(ImportAttributes, ImportCapability) {
  CapabilityAttr *FromAttr, *ToAttr;
  importAttr<CXXRecordDecl>(
      "struct __attribute__((capability(\"cap\"))) test {};", FromAttr, ToAttr);
  EXPECT_EQ(FromAttr->getName(), ToAttr->getName());
}

TEST_P(ImportAttributes, ImportAssertCapability) {
  AssertCapabilityAttr *FromAttr, *ToAttr;
  importAttr<FunctionDecl>(
      "void test(int A1, int A2) __attribute__((assert_capability(A1, A2)));",
      FromAttr, ToAttr);
  checkImportVariadicArg(FromAttr->args(), ToAttr->args());
}

TEST_P(ImportAttributes, ImportAcquireCapability) {
  AcquireCapabilityAttr *FromAttr, *ToAttr;
  importAttr<FunctionDecl>(
      "void test(int A1, int A2) __attribute__((acquire_capability(A1, A2)));",
      FromAttr, ToAttr);
  checkImportVariadicArg(FromAttr->args(), ToAttr->args());
}

TEST_P(ImportAttributes, ImportTryAcquireCapability) {
  TryAcquireCapabilityAttr *FromAttr, *ToAttr;
  importAttr<FunctionDecl>(
      "void test(int A1, int A2) __attribute__((try_acquire_capability(1, A1, "
      "A2)));",
      FromAttr, ToAttr);
  checkImported(FromAttr->getSuccessValue(), ToAttr->getSuccessValue());
  checkImportVariadicArg(FromAttr->args(), ToAttr->args());
}

TEST_P(ImportAttributes, ImportReleaseCapability) {
  ReleaseCapabilityAttr *FromAttr, *ToAttr;
  importAttr<FunctionDecl>(
      "void test(int A1, int A2) __attribute__((release_capability(A1, A2)));",
      FromAttr, ToAttr);
  checkImportVariadicArg(FromAttr->args(), ToAttr->args());
}

TEST_P(ImportAttributes, ImportRequiresCapability) {
  RequiresCapabilityAttr *FromAttr, *ToAttr;
  importAttr<FunctionDecl>(
      "void test(int A1, int A2) __attribute__((requires_capability(A1, A2)));",
      FromAttr, ToAttr);
  checkImportVariadicArg(FromAttr->args(), ToAttr->args());
}

TEST_P(ImportAttributes, ImportNoThreadSafetyAnalysis) {
  NoThreadSafetyAnalysisAttr *FromAttr, *ToAttr;
  importAttr<FunctionDecl>(
      "void test() __attribute__((no_thread_safety_analysis));", FromAttr,
      ToAttr);
}

TEST_P(ImportAttributes, ImportGuardedBy) {
  GuardedByAttr *FromAttr, *ToAttr;
  importAttr<VarDecl>(
      R"(
      int G;
      int test __attribute__((guarded_by(G)));
      )",
      FromAttr, ToAttr);
  checkImported(FromAttr->getArg(), ToAttr->getArg());
}

TEST_P(ImportAttributes, ImportPtGuardedBy) {
  PtGuardedByAttr *FromAttr, *ToAttr;
  importAttr<VarDecl>(
      R"(
      int G;
      int *test __attribute__((pt_guarded_by(G)));
      )",
      FromAttr, ToAttr);
  checkImported(FromAttr->getArg(), ToAttr->getArg());
}

TEST_P(ImportAttributes, ImportAcquiredAfter) {
  AcquiredAfterAttr *FromAttr, *ToAttr;
  importAttr<VarDecl>(
      R"(
      struct __attribute__((lockable)) L {};
      L A1;
      L A2;
      L test __attribute__((acquired_after(A1, A2)));
      )",
      FromAttr, ToAttr);
  checkImportVariadicArg(FromAttr->args(), ToAttr->args());
}

TEST_P(ImportAttributes, ImportAcquiredBefore) {
  AcquiredBeforeAttr *FromAttr, *ToAttr;
  importAttr<VarDecl>(
      R"(
      struct __attribute__((lockable)) L {};
      L A1;
      L A2;
      L test __attribute__((acquired_before(A1, A2)));
      )",
      FromAttr, ToAttr);
  checkImportVariadicArg(FromAttr->args(), ToAttr->args());
}

TEST_P(ImportAttributes, ImportAssertExclusiveLock) {
  AssertExclusiveLockAttr *FromAttr, *ToAttr;
  importAttr<FunctionDecl>("void test(int A1, int A2) "
                           "__attribute__((assert_exclusive_lock(A1, A2)));",
                           FromAttr, ToAttr);
  checkImportVariadicArg(FromAttr->args(), ToAttr->args());
}

TEST_P(ImportAttributes, ImportAssertSharedLock) {
  AssertSharedLockAttr *FromAttr, *ToAttr;
  importAttr<FunctionDecl>(
      "void test(int A1, int A2) __attribute__((assert_shared_lock(A1, A2)));",
      FromAttr, ToAttr);
  checkImportVariadicArg(FromAttr->args(), ToAttr->args());
}

TEST_P(ImportAttributes, ImportExclusiveTrylockFunction) {
  ExclusiveTrylockFunctionAttr *FromAttr, *ToAttr;
  importAttr<FunctionDecl>(
      "void test(int A1, int A2) __attribute__((exclusive_trylock_function(1, "
      "A1, A2)));",
      FromAttr, ToAttr);
  checkImported(FromAttr->getSuccessValue(), ToAttr->getSuccessValue());
  checkImportVariadicArg(FromAttr->args(), ToAttr->args());
}

TEST_P(ImportAttributes, ImportSharedTrylockFunction) {
  SharedTrylockFunctionAttr *FromAttr, *ToAttr;
  importAttr<FunctionDecl>(
      "void test(int A1, int A2) __attribute__((shared_trylock_function(1, A1, "
      "A2)));",
      FromAttr, ToAttr);
  checkImported(FromAttr->getSuccessValue(), ToAttr->getSuccessValue());
  checkImportVariadicArg(FromAttr->args(), ToAttr->args());
}

TEST_P(ImportAttributes, ImportLockReturned) {
  LockReturnedAttr *FromAttr, *ToAttr;
  importAttr<FunctionDecl>(
      "void test(int A1) __attribute__((lock_returned(A1)));", FromAttr,
      ToAttr);
  checkImported(FromAttr->getArg(), ToAttr->getArg());
}

TEST_P(ImportAttributes, ImportLocksExcluded) {
  LocksExcludedAttr *FromAttr, *ToAttr;
  importAttr<FunctionDecl>(
      "void test(int A1, int A2) __attribute__((locks_excluded(A1, A2)));",
      FromAttr, ToAttr);
  checkImportVariadicArg(FromAttr->args(), ToAttr->args());
}

template <typename T>
auto ExtendWithOptions(const T &Values, const std::vector<std::string> &Args) {
  auto Copy = Values;
  for (std::vector<std::string> &ArgV : Copy) {
    for (const std::string &Arg : Args) {
      ArgV.push_back(Arg);
    }
  }
  return ::testing::ValuesIn(Copy);
}

struct ImportWithExternalSource : ASTImporterOptionSpecificTestBase {
  ImportWithExternalSource() {
    Creator = [](ASTContext &ToContext, FileManager &ToFileManager,
                 ASTContext &FromContext, FileManager &FromFileManager,
                 bool MinimalImport,
                 const std::shared_ptr<ASTImporterSharedState> &SharedState) {
      return new ASTImporter(ToContext, ToFileManager, FromContext,
                             // Use minimal import for these tests.
                             FromFileManager, /*MinimalImport=*/true,
                             // We use the regular lookup.
                             /*SharedState=*/nullptr);
    };
  }
};

/// An ExternalASTSource that keeps track of the tags is completed.
struct SourceWithCompletedTagList : clang::ExternalASTSource {
  std::vector<clang::TagDecl *> &CompletedTags;
  SourceWithCompletedTagList(std::vector<clang::TagDecl *> &CompletedTags)
      : CompletedTags(CompletedTags) {}
  void CompleteType(TagDecl *Tag) override {
    auto *Record = cast<CXXRecordDecl>(Tag);
    Record->startDefinition();
    Record->completeDefinition();
    CompletedTags.push_back(Tag);
  }
  using clang::ExternalASTSource::CompleteType;
};

TEST_P(ImportWithExternalSource, CompleteRecordBeforeImporting) {
  // Create an empty TU.
  TranslationUnitDecl *FromTU = getTuDecl("", Lang_CXX03, "input.cpp");

  // Create and add the test ExternalASTSource.
  std::vector<clang::TagDecl *> CompletedTags;
  IntrusiveRefCntPtr<ExternalASTSource> source =
      new SourceWithCompletedTagList(CompletedTags);
  clang::ASTContext &Context = FromTU->getASTContext();
  Context.setExternalSource(std::move(source));

  // Create a dummy class by hand with external lexical storage.
  IdentifierInfo &Ident = Context.Idents.get("test_class");
  auto *Record = CXXRecordDecl::Create(
      Context, TTK_Class, FromTU, SourceLocation(), SourceLocation(), &Ident);
  Record->setHasExternalLexicalStorage();
  FromTU->addDecl(Record);

  // Do a minimal import of the created class.
  EXPECT_EQ(0U, CompletedTags.size());
  Import(Record, Lang_CXX03);
  EXPECT_EQ(0U, CompletedTags.size());

  // Import the definition of the created class.
  llvm::Error Err = findFromTU(Record)->Importer->ImportDefinition(Record);
  EXPECT_FALSE((bool)Err);
  consumeError(std::move(Err));

  // Make sure the class was completed once.
  EXPECT_EQ(1U, CompletedTags.size());
  EXPECT_EQ(Record, CompletedTags.front());
}

TEST_P(ImportFunctions, CTADImplicit) {
  Decl *FromTU = getTuDecl(
      R"(
      template <typename T> struct A {
        A(T);
      };
      A a{(int)0};
      )",
      Lang_CXX17, "input.cc");
  auto *FromD = FirstDeclMatcher<CXXDeductionGuideDecl>().match(
      FromTU,
      cxxDeductionGuideDecl(hasParameter(0, hasType(asString("A<T>")))));
  auto *ToD = Import(FromD, Lang_CXX17);
  ASSERT_TRUE(ToD);
  EXPECT_EQ(ToD->getDeductionCandidateKind(), DeductionCandidate::Copy);
  // Check that the deduced class template is also imported.
  EXPECT_TRUE(findFromTU(FromD)->Importer->GetAlreadyImportedOrNull(
      FromD->getDeducedTemplate()));
}

TEST_P(ImportFunctions, CTADUserDefinedExplicit) {
  Decl *FromTU = getTuDecl(
      R"(
      template <typename T> struct A {
        A(T);
      };
      template <typename T> explicit A(T) -> A<float>;
      A a{(int)0}; // calls A<float>::A(float)
      )",
      Lang_CXX17, "input.cc");
  auto *FromD = FirstDeclMatcher<CXXDeductionGuideDecl>().match(
      FromTU, cxxDeductionGuideDecl(unless(isImplicit())));
  // Not-implicit: i.e. not compiler-generated, user defined.
  ASSERT_FALSE(FromD->isImplicit());
  ASSERT_TRUE(FromD->isExplicit()); // Has the explicit keyword.
  auto *ToD = Import(FromD, Lang_CXX17);
  ASSERT_TRUE(ToD);
  EXPECT_FALSE(FromD->isImplicit());
  EXPECT_TRUE(ToD->isExplicit());
}

TEST_P(ImportFunctions, CTADWithLocalTypedef) {
  Decl *TU = getTuDecl(
      R"(
      template <typename T> struct A {
        typedef T U;
        A(U);
      };
      A a{(int)0};
      )",
      Lang_CXX17, "input.cc");
  auto *FromD = FirstDeclMatcher<CXXDeductionGuideDecl>().match(
      TU, cxxDeductionGuideDecl());
  auto *ToD = Import(FromD, Lang_CXX17);
  ASSERT_TRUE(ToD);
}

TEST_P(ImportFunctions, ParmVarDeclDeclContext) {
  constexpr auto FromTUCode = R"(
      void f(int P);
      )";
  Decl *FromTU = getTuDecl(FromTUCode, Lang_CXX11);
  auto *FromF = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("f")));
  ASSERT_TRUE(FromF);

  auto *ImportedF = Import(FromF, Lang_CXX11);
  EXPECT_TRUE(ImportedF);
  EXPECT_TRUE(SharedStatePtr->getLookupTable()->contains(
      ImportedF, ImportedF->getParamDecl(0)));
}

// FIXME Move these tests out of ASTImporterTest. For that we need to factor
// out the ASTImporter specific pars from ASTImporterOptionSpecificTestBase
// into a new test Fixture. Then we should lift up this Fixture to its own
// implementation file and only then could we reuse the Fixture in other AST
// unitttests.
struct CTAD : ASTImporterOptionSpecificTestBase {};

TEST_P(CTAD, DeductionGuideShouldReferToANonLocalTypedef) {
  Decl *TU = getTuDecl(
      R"(
      typedef int U;
      template <typename T> struct A {
        A(U, T);
      };
      A a{(int)0, (int)0};
      )",
      Lang_CXX17, "input.cc");
  auto *Guide = FirstDeclMatcher<CXXDeductionGuideDecl>().match(
      TU, cxxDeductionGuideDecl());
  auto *Typedef = FirstDeclMatcher<TypedefNameDecl>().match(
      TU, typedefNameDecl(hasName("U")));
  ParmVarDecl *Param = Guide->getParamDecl(0);
  // The type of the first param (which is a typedef) should match the typedef
  // in the global scope.
  EXPECT_EQ(Param->getType()->getAs<TypedefType>()->getDecl(), Typedef);
}

TEST_P(CTAD, DeductionGuideShouldReferToANonLocalTypedefInParamPtr) {
  Decl *TU = getTuDecl(
      R"(
      typedef int U;
      template <typename T> struct A {
        A(U*, T);
      };
      A a{(int*)0, (int)0};
      )",
      Lang_CXX17, "input.cc");
  auto *Guide = FirstDeclMatcher<CXXDeductionGuideDecl>().match(
      TU, cxxDeductionGuideDecl());
  auto *Typedef = FirstDeclMatcher<TypedefNameDecl>().match(
      TU, typedefNameDecl(hasName("U")));
  ParmVarDecl *Param = Guide->getParamDecl(0);
  EXPECT_EQ(Param->getType()
                ->getAs<PointerType>()
                ->getPointeeType()
                ->getAs<TypedefType>()
                ->getDecl(),
            Typedef);
}

TEST_P(CTAD, DeductionGuideShouldCopyALocalTypedef) {
  Decl *TU = getTuDecl(
      R"(
      template <typename T> struct A {
        typedef T U;
        A(U, T);
      };
      A a{(int)0, (int)0};
      )",
      Lang_CXX17, "input.cc");
  auto *Guide = FirstDeclMatcher<CXXDeductionGuideDecl>().match(
      TU, cxxDeductionGuideDecl());
  auto *Typedef = FirstDeclMatcher<TypedefNameDecl>().match(
      TU, typedefNameDecl(hasName("U")));
  ParmVarDecl *Param = Guide->getParamDecl(0);
  EXPECT_NE(Param->getType()->getAs<TypedefType>()->getDecl(), Typedef);
}

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, CTAD,
                         DefaultTestValuesForRunOptions);

TEST_P(ASTImporterOptionSpecificTestBase, TypedefWithAttribute) {
  Decl *TU = getTuDecl(
      R"(
      namespace N {
        typedef int X __attribute__((annotate("A")));
      }
      )",
      Lang_CXX17, "input.cc");
  auto *FromD =
      FirstDeclMatcher<TypedefDecl>().match(TU, typedefDecl(hasName("X")));
  auto *ToD = Import(FromD, Lang_CXX17);
  ASSERT_TRUE(ToD);
  ASSERT_EQ(ToD->getAttrs().size(), 1U);
  auto *ToAttr = dyn_cast<AnnotateAttr>(ToD->getAttrs()[0]);
  ASSERT_TRUE(ToAttr);
  EXPECT_EQ(ToAttr->getAnnotation(), "A");
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportOfTemplatedDeclWhenPreviousDeclHasNoDescribedTemplateSet) {
  Decl *FromTU = getTuDecl(
      R"(

      namespace std {
        template<typename T>
        class basic_stringbuf;
      }
      namespace std {
        class char_traits;
        template<typename T = char_traits>
        class basic_stringbuf;
      }
      namespace std {
        template<typename T>
        class basic_stringbuf {};
      }

      )",
      Lang_CXX11);

  auto *From1 = FirstDeclMatcher<ClassTemplateDecl>().match(
      FromTU,
      classTemplateDecl(hasName("basic_stringbuf"), unless(isImplicit())));
  auto *To1 = cast_or_null<ClassTemplateDecl>(Import(From1, Lang_CXX11));
  EXPECT_TRUE(To1);

  auto *From2 = LastDeclMatcher<ClassTemplateDecl>().match(
      FromTU,
      classTemplateDecl(hasName("basic_stringbuf"), unless(isImplicit())));
  auto *To2 = cast_or_null<ClassTemplateDecl>(Import(From2, Lang_CXX11));
  EXPECT_TRUE(To2);
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportOfCapturedVLAType) {
  Decl *FromTU = getTuDecl(
      R"(
      void declToImport(int N) {
        int VLA[N];
        [&VLA] {}; // FieldDecl inside the lambda.
      }
      )",
      Lang_CXX14);
  auto *FromFD = FirstDeclMatcher<FieldDecl>().match(FromTU, fieldDecl());
  ASSERT_TRUE(FromFD);
  ASSERT_TRUE(FromFD->hasCapturedVLAType());

  auto *ToFD = Import(FromFD, Lang_CXX14);
  EXPECT_TRUE(ToFD);
  EXPECT_TRUE(ToFD->hasCapturedVLAType());
  EXPECT_NE(FromFD->getCapturedVLAType(), ToFD->getCapturedVLAType());
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportEnumMemberSpecialization) {
  Decl *FromTU = getTuDecl(
      R"(
      template <class T> struct A {
        enum tagname { enumerator };
      };
      template struct A<int>;
      )",
      Lang_CXX03);
  auto *FromD = FirstDeclMatcher<EnumDecl>().match(
      FromTU, enumDecl(hasName("tagname"),
                       hasParent(classTemplateSpecializationDecl())));
  ASSERT_TRUE(FromD);
  ASSERT_TRUE(FromD->getMemberSpecializationInfo());

  auto *ToD = Import(FromD, Lang_CXX03);
  EXPECT_TRUE(ToD);
  EXPECT_TRUE(ToD->getMemberSpecializationInfo());
  EXPECT_EQ(FromD->getTemplateSpecializationKind(),
            ToD->getTemplateSpecializationKind());
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportIsInheritingConstructorBit) {
  Decl *FromTU = getTuDecl(
      R"(
      struct A {
        A(int);
      };
      struct B : A {
        using A::A; // Inherited ctor.
      };
      void f() {
        (B(0));
      }
      )",
      Lang_CXX11);
  auto *FromD = FirstDeclMatcher<CXXConstructorDecl>().match(
      FromTU, cxxConstructorDecl(isInheritingConstructor()));
  ASSERT_TRUE(FromD);
  ASSERT_TRUE(FromD->isInheritingConstructor());

  auto *ToD = Import(FromD, Lang_CXX11);
  ASSERT_TRUE(ToD);
  EXPECT_TRUE(ToD->isInheritingConstructor());
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportConstructorUsingShadow) {
  TranslationUnitDecl *FromTU = getTuDecl(
      R"(
      struct A {
        A(int, int);
      };
      struct B : A {
        using A::A;
      };
      struct C : B {
        using B::B;
      };
      )",
      Lang_CXX11);

  auto CheckAST = [](TranslationUnitDecl *TU, CXXRecordDecl *RecordC) {
    auto *RecordA = FirstDeclMatcher<CXXRecordDecl>().match(
        TU, cxxRecordDecl(hasName("A")));
    auto *RecordB = FirstDeclMatcher<CXXRecordDecl>().match(
        TU, cxxRecordDecl(hasName("B")));
    auto *ConstrA = FirstDeclMatcher<CXXConstructorDecl>().match(
        TU, cxxConstructorDecl(hasParent(equalsNode(RecordA)),
                               parameterCountIs(2)));
    auto *ShadowBA = cast<ConstructorUsingShadowDecl>(
        FirstDeclMatcher<UsingShadowDecl>().match(
            TU, usingShadowDecl(hasParent(equalsNode(RecordB)),
                                hasTargetDecl(equalsNode(ConstrA)))));
    auto *ShadowCA = cast<ConstructorUsingShadowDecl>(
        FirstDeclMatcher<UsingShadowDecl>().match(
            TU, usingShadowDecl(hasParent(equalsNode(RecordC)),
                                hasTargetDecl(equalsNode(ConstrA)))));
    EXPECT_EQ(ShadowBA->getTargetDecl(), ConstrA);
    EXPECT_EQ(ShadowBA->getNominatedBaseClass(), RecordA);
    EXPECT_EQ(ShadowBA->getConstructedBaseClass(), RecordA);
    EXPECT_EQ(ShadowBA->getNominatedBaseClassShadowDecl(), nullptr);
    EXPECT_EQ(ShadowBA->getConstructedBaseClassShadowDecl(), nullptr);
    EXPECT_FALSE(ShadowBA->constructsVirtualBase());
    EXPECT_EQ(ShadowCA->getTargetDecl(), ConstrA);
    EXPECT_EQ(ShadowCA->getNominatedBaseClass(), RecordB);
    EXPECT_EQ(ShadowCA->getConstructedBaseClass(), RecordB);
    EXPECT_EQ(ShadowCA->getNominatedBaseClassShadowDecl(), ShadowBA);
    EXPECT_EQ(ShadowCA->getConstructedBaseClassShadowDecl(), ShadowBA);
    EXPECT_FALSE(ShadowCA->constructsVirtualBase());
  };

  auto *FromC = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("C")));

  auto *ToC = Import(FromC, Lang_CXX11);
  TranslationUnitDecl *ToTU = ToC->getTranslationUnitDecl();

  CheckAST(FromTU, FromC);
  CheckAST(ToTU, ToC);
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportFunctionDeclBitShouldNotOverwriteCtorDeclBits) {
  Decl *From, *To;
  std::tie(From, To) = getImportedDecl(
      R"s(
        struct A {
          A() : m() {}
          int m;
        };

        A foo() { A a; return a; }
        A bar() { return {}; }
      )s",
      Lang_CXX17,
      R"s(
        struct A {
          A() : m() {}
          int m;
        };
        A baz() { return {}; }
      )s",
      Lang_CXX17, "A");

  auto HasCtorInit =
      hasAnyConstructorInitializer(cxxCtorInitializer(isMemberInitializer()));
  auto ImpMoveCtor =
      cxxConstructorDecl(isMoveConstructor(), isImplicit(), HasCtorInit);

  auto *FromImpMoveCtor = FirstDeclMatcher<CXXConstructorDecl>().match(
      From, ImpMoveCtor);
  auto *ToImpMoveCtor = FirstDeclMatcher<CXXConstructorDecl>().match(
      To, ImpMoveCtor);

  EXPECT_TRUE(FromImpMoveCtor->getNumCtorInitializers() == 1);
  EXPECT_FALSE(FromImpMoveCtor->FriendConstraintRefersToEnclosingTemplate());

  EXPECT_TRUE(ToImpMoveCtor->getNumCtorInitializers() == 1);
  EXPECT_FALSE(ToImpMoveCtor->FriendConstraintRefersToEnclosingTemplate());
  EXPECT_TRUE(*ToImpMoveCtor->init_begin());
}

AST_MATCHER_P(UsingShadowDecl, hasIntroducerDecl, internal::Matcher<NamedDecl>,
              InnerMatcher) {
  return InnerMatcher.matches(*Node.getIntroducer(), Finder, Builder);
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportConstructorUsingShadowVirtualBase) {
  TranslationUnitDecl *FromTU = getTuDecl(
      R"(
      struct A { A(int, int); };
      struct B : A { using A::A; };

      struct V1 : virtual B { using B::B; };
      struct V2 : virtual B { using B::B; };

      struct D2 : V1, V2 {
        using V1::V1;
        using V2::V2;
      };
      )",
      Lang_CXX11);

  auto CheckAST = [](TranslationUnitDecl *TU, CXXRecordDecl *RecordD2) {
    auto *RecordA = FirstDeclMatcher<CXXRecordDecl>().match(
        TU, cxxRecordDecl(hasName("A")));
    auto *RecordB = FirstDeclMatcher<CXXRecordDecl>().match(
        TU, cxxRecordDecl(hasName("B")));
    auto *RecordV1 = FirstDeclMatcher<CXXRecordDecl>().match(
        TU, cxxRecordDecl(hasName("V1")));
    auto *RecordV2 = FirstDeclMatcher<CXXRecordDecl>().match(
        TU, cxxRecordDecl(hasName("V2")));
    auto *ConstrA = FirstDeclMatcher<CXXConstructorDecl>().match(
        TU, cxxConstructorDecl(hasParent(equalsNode(RecordA)),
                               parameterCountIs(2)));
    auto *ConstrB = FirstDeclMatcher<CXXConstructorDecl>().match(
        TU, cxxConstructorDecl(hasParent(equalsNode(RecordB)),
                               isCopyConstructor()));
    auto *UsingD2V1 = FirstDeclMatcher<UsingDecl>().match(
        TU, usingDecl(hasParent(equalsNode(RecordD2))));
    auto *UsingD2V2 = LastDeclMatcher<UsingDecl>().match(
        TU, usingDecl(hasParent(equalsNode(RecordD2))));
    auto *ShadowBA = cast<ConstructorUsingShadowDecl>(
        FirstDeclMatcher<UsingShadowDecl>().match(
            TU, usingShadowDecl(hasParent(equalsNode(RecordB)),
                                hasTargetDecl(equalsNode(ConstrA)))));
    auto *ShadowV1A = cast<ConstructorUsingShadowDecl>(
        FirstDeclMatcher<UsingShadowDecl>().match(
            TU, usingShadowDecl(hasParent(equalsNode(RecordV1)),
                                hasTargetDecl(equalsNode(ConstrA)))));
    auto *ShadowV1B = cast<ConstructorUsingShadowDecl>(
        FirstDeclMatcher<UsingShadowDecl>().match(
            TU, usingShadowDecl(hasParent(equalsNode(RecordV1)),
                                hasTargetDecl(equalsNode(ConstrB)))));
    auto *ShadowV2A = cast<ConstructorUsingShadowDecl>(
        FirstDeclMatcher<UsingShadowDecl>().match(
            TU, usingShadowDecl(hasParent(equalsNode(RecordV2)),
                                hasTargetDecl(equalsNode(ConstrA)))));
    auto *ShadowV2B = cast<ConstructorUsingShadowDecl>(
        FirstDeclMatcher<UsingShadowDecl>().match(
            TU, usingShadowDecl(hasParent(equalsNode(RecordV2)),
                                hasTargetDecl(equalsNode(ConstrB)))));
    auto *ShadowD2V1A = cast<ConstructorUsingShadowDecl>(
        FirstDeclMatcher<UsingShadowDecl>().match(
            TU, usingShadowDecl(hasParent(equalsNode(RecordD2)),
                                hasIntroducerDecl(equalsNode(UsingD2V1)),
                                hasTargetDecl(equalsNode(ConstrA)))));
    auto *ShadowD2V1B = cast<ConstructorUsingShadowDecl>(
        FirstDeclMatcher<UsingShadowDecl>().match(
            TU, usingShadowDecl(hasParent(equalsNode(RecordD2)),
                                hasIntroducerDecl(equalsNode(UsingD2V1)),
                                hasTargetDecl(equalsNode(ConstrB)))));
    auto *ShadowD2V2A = cast<ConstructorUsingShadowDecl>(
        FirstDeclMatcher<UsingShadowDecl>().match(
            TU, usingShadowDecl(hasParent(equalsNode(RecordD2)),
                                hasIntroducerDecl(equalsNode(UsingD2V2)),
                                hasTargetDecl(equalsNode(ConstrA)))));
    auto *ShadowD2V2B = cast<ConstructorUsingShadowDecl>(
        FirstDeclMatcher<UsingShadowDecl>().match(
            TU, usingShadowDecl(hasParent(equalsNode(RecordD2)),
                                hasIntroducerDecl(equalsNode(UsingD2V2)),
                                hasTargetDecl(equalsNode(ConstrB)))));

    EXPECT_EQ(ShadowD2V1A->getTargetDecl(), ConstrA);
    EXPECT_EQ(ShadowD2V1A->getNominatedBaseClassShadowDecl(), ShadowV1A);
    EXPECT_EQ(ShadowD2V1A->getNominatedBaseClass(), RecordV1);
    EXPECT_EQ(ShadowD2V1A->getConstructedBaseClassShadowDecl(), ShadowBA);
    EXPECT_EQ(ShadowD2V1A->getConstructedBaseClass(), RecordB);
    EXPECT_TRUE(ShadowD2V1A->constructsVirtualBase());
    EXPECT_EQ(ShadowD2V1B->getTargetDecl(), ConstrB);
    EXPECT_EQ(ShadowD2V1B->getNominatedBaseClassShadowDecl(), ShadowV1B);
    EXPECT_EQ(ShadowD2V1B->getNominatedBaseClass(), RecordV1);
    EXPECT_EQ(ShadowD2V1B->getConstructedBaseClassShadowDecl(), nullptr);
    EXPECT_EQ(ShadowD2V1B->getConstructedBaseClass(), RecordB);
    EXPECT_TRUE(ShadowD2V1B->constructsVirtualBase());
    EXPECT_EQ(ShadowD2V2A->getTargetDecl(), ConstrA);
    EXPECT_EQ(ShadowD2V2A->getNominatedBaseClassShadowDecl(), ShadowV2A);
    EXPECT_EQ(ShadowD2V2A->getNominatedBaseClass(), RecordV2);
    EXPECT_EQ(ShadowD2V2A->getConstructedBaseClassShadowDecl(), ShadowBA);
    EXPECT_EQ(ShadowD2V2A->getConstructedBaseClass(), RecordB);
    EXPECT_TRUE(ShadowD2V2A->constructsVirtualBase());
    EXPECT_EQ(ShadowD2V2B->getTargetDecl(), ConstrB);
    EXPECT_EQ(ShadowD2V2B->getNominatedBaseClassShadowDecl(), ShadowV2B);
    EXPECT_EQ(ShadowD2V2B->getNominatedBaseClass(), RecordV2);
    EXPECT_EQ(ShadowD2V2B->getConstructedBaseClassShadowDecl(), nullptr);
    EXPECT_EQ(ShadowD2V2B->getConstructedBaseClass(), RecordB);
    EXPECT_TRUE(ShadowD2V2B->constructsVirtualBase());

    EXPECT_TRUE(ShadowV1A->constructsVirtualBase());
    EXPECT_TRUE(ShadowV1B->constructsVirtualBase());
    EXPECT_TRUE(ShadowV2A->constructsVirtualBase());
    EXPECT_TRUE(ShadowV2B->constructsVirtualBase());
    EXPECT_FALSE(ShadowBA->constructsVirtualBase());
  };

  auto *FromD2 = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("D2")));

  auto *ToD2 = Import(FromD2, Lang_CXX11);
  TranslationUnitDecl *ToTU = ToD2->getTranslationUnitDecl();

  CheckAST(FromTU, FromD2);
  CheckAST(ToTU, ToD2);
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportUsingShadowList) {
  TranslationUnitDecl *FromTU = getTuDecl(
      R"(
      struct A {
        void f();
        void f(int);
      };
      struct B : A {
        using A::f;
      };
      )",
      Lang_CXX11);

  auto *FromB = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("B")));

  auto *ToB = Import(FromB, Lang_CXX11);
  TranslationUnitDecl *ToTU = ToB->getTranslationUnitDecl();

  auto *ToUsing = FirstDeclMatcher<UsingDecl>().match(
      ToTU, usingDecl(hasParent(equalsNode(ToB))));
  auto *ToUsingShadowF1 = FirstDeclMatcher<UsingShadowDecl>().match(
      ToTU, usingShadowDecl(hasTargetDecl(
                functionDecl(hasName("f"), parameterCountIs(0)))));
  auto *ToUsingShadowF2 = FirstDeclMatcher<UsingShadowDecl>().match(
      ToTU, usingShadowDecl(hasTargetDecl(
                functionDecl(hasName("f"), parameterCountIs(1)))));

  EXPECT_EQ(ToUsing->shadow_size(), 2u);
  auto ShadowI = ToUsing->shadow_begin();
  EXPECT_EQ(*ShadowI, ToUsingShadowF1);
  ++ShadowI;
  EXPECT_EQ(*ShadowI, ToUsingShadowF2);
}

AST_MATCHER_P(FunctionTemplateDecl, templateParameterCountIs, unsigned, Cnt) {
  return Node.getTemplateParameters()->size() == Cnt;
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportDeductionGuide) {
  TranslationUnitDecl *FromTU = getTuDecl(
      R"(
      template<class> class A { };
      template<class T> class B {
          template<class T1, typename = A<T>> B(T1);
      };
      template<class T>
      B(T, T) -> B<int>;
      )",
      Lang_CXX17);

  // Get the implicit deduction guide for (non-default) constructor of 'B'.
  auto *FromDGCtor = FirstDeclMatcher<FunctionTemplateDecl>().match(
      FromTU, functionTemplateDecl(templateParameterCountIs(3)));
  // Implicit deduction guide for copy constructor of 'B'.
  auto *FromDGCopyCtor = FirstDeclMatcher<FunctionTemplateDecl>().match(
      FromTU, functionTemplateDecl(templateParameterCountIs(1), isImplicit()));
  // User defined deduction guide.
  auto *FromDGOther = FirstDeclMatcher<CXXDeductionGuideDecl>().match(
      FromTU, cxxDeductionGuideDecl(unless(isImplicit())));

  TemplateParameterList *FromDGCtorTP = FromDGCtor->getTemplateParameters();
  // Don't know why exactly but this is the DeclContext here.
  EXPECT_EQ(FromDGCtorTP->getParam(0)->getDeclContext(),
            FromDGCopyCtor->getTemplatedDecl());
  EXPECT_EQ(FromDGCtorTP->getParam(1)->getDeclContext(),
            FromDGCtor->getTemplatedDecl());
  EXPECT_EQ(FromDGCtorTP->getParam(2)->getDeclContext(),
            FromDGCtor->getTemplatedDecl());
  EXPECT_EQ(
      FromDGCopyCtor->getTemplateParameters()->getParam(0)->getDeclContext(),
      FromDGCopyCtor->getTemplatedDecl());
  EXPECT_EQ(FromDGOther->getDescribedTemplate()
                ->getTemplateParameters()
                ->getParam(0)
                ->getDeclContext(),
            FromDGOther);

  auto *ToDGCtor = Import(FromDGCtor, Lang_CXX17);
  auto *ToDGCopyCtor = Import(FromDGCopyCtor, Lang_CXX17);
  auto *ToDGOther = Import(FromDGOther, Lang_CXX17);
  ASSERT_TRUE(ToDGCtor);
  ASSERT_TRUE(ToDGCopyCtor);
  ASSERT_TRUE(ToDGOther);

  TemplateParameterList *ToDGCtorTP = ToDGCtor->getTemplateParameters();
  EXPECT_EQ(ToDGCtorTP->getParam(0)->getDeclContext(),
            ToDGCopyCtor->getTemplatedDecl());
  EXPECT_EQ(ToDGCtorTP->getParam(1)->getDeclContext(),
            ToDGCtor->getTemplatedDecl());
  EXPECT_EQ(ToDGCtorTP->getParam(2)->getDeclContext(),
            ToDGCtor->getTemplatedDecl());
  EXPECT_EQ(
      ToDGCopyCtor->getTemplateParameters()->getParam(0)->getDeclContext(),
      ToDGCopyCtor->getTemplatedDecl());
  EXPECT_EQ(ToDGOther->getDescribedTemplate()
                ->getTemplateParameters()
                ->getParam(0)
                ->getDeclContext(),
            ToDGOther);
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportDeductionGuideDifferentOrder) {
  // This test demonstrates that the DeclContext of the imported object is
  // dependent on the order of import. The test is an exact copy of the previous
  // one except at the indicated locations.
  TranslationUnitDecl *FromTU = getTuDecl(
      R"(
      template<class> class A { };
      template<class T> class B {
          template<class T1, typename = A<T>> B(T1);
      };
      template<class T>
      B(T, T) -> B<int>;
      )",
      Lang_CXX17);

  // Get the implicit deduction guide for (non-default) constructor of 'B'.
  auto *FromDGCtor = FirstDeclMatcher<FunctionTemplateDecl>().match(
      FromTU, functionTemplateDecl(templateParameterCountIs(3)));
  // Implicit deduction guide for copy constructor of 'B'.
  auto *FromDGCopyCtor = FirstDeclMatcher<FunctionTemplateDecl>().match(
      FromTU, functionTemplateDecl(templateParameterCountIs(1), isImplicit()));
  // User defined deduction guide.
  auto *FromDGOther = FirstDeclMatcher<CXXDeductionGuideDecl>().match(
      FromTU, cxxDeductionGuideDecl(unless(isImplicit())));

  TemplateParameterList *FromDGCtorTP = FromDGCtor->getTemplateParameters();
  // Don't know why exactly but this is the DeclContext here.
  EXPECT_EQ(FromDGCtorTP->getParam(0)->getDeclContext(),
            FromDGCopyCtor->getTemplatedDecl());
  EXPECT_EQ(FromDGCtorTP->getParam(1)->getDeclContext(),
            FromDGCtor->getTemplatedDecl());
  EXPECT_EQ(FromDGCtorTP->getParam(2)->getDeclContext(),
            FromDGCtor->getTemplatedDecl());
  EXPECT_EQ(
      FromDGCopyCtor->getTemplateParameters()->getParam(0)->getDeclContext(),
      FromDGCopyCtor->getTemplatedDecl());
  EXPECT_EQ(FromDGOther->getDescribedTemplate()
                ->getTemplateParameters()
                ->getParam(0)
                ->getDeclContext(),
            FromDGOther);

  // Here the import of 'ToDGCopyCtor' and 'ToDGCtor' is reversed relative to
  // the previous test.
  auto *ToDGCopyCtor = Import(FromDGCopyCtor, Lang_CXX17);
  auto *ToDGCtor = Import(FromDGCtor, Lang_CXX17);
  auto *ToDGOther = Import(FromDGOther, Lang_CXX17);
  ASSERT_TRUE(ToDGCtor);
  ASSERT_TRUE(ToDGCopyCtor);
  ASSERT_TRUE(ToDGOther);

  TemplateParameterList *ToDGCtorTP = ToDGCtor->getTemplateParameters();
  // Next line: DeclContext is different relative to the previous test.
  EXPECT_EQ(ToDGCtorTP->getParam(0)->getDeclContext(),
            ToDGCtor->getTemplatedDecl());
  EXPECT_EQ(ToDGCtorTP->getParam(1)->getDeclContext(),
            ToDGCtor->getTemplatedDecl());
  EXPECT_EQ(ToDGCtorTP->getParam(2)->getDeclContext(),
            ToDGCtor->getTemplatedDecl());
  // Next line: DeclContext is different relative to the previous test.
  EXPECT_EQ(
      ToDGCopyCtor->getTemplateParameters()->getParam(0)->getDeclContext(),
      ToDGCtor->getTemplatedDecl());
  EXPECT_EQ(ToDGOther->getDescribedTemplate()
                ->getTemplateParameters()
                ->getParam(0)
                ->getDeclContext(),
            ToDGOther);
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportFieldsFirstForCorrectRecordLayout) {
  // UnaryOperator(&) triggers RecordLayout computation, which relies on
  // correctly imported fields.
  auto Code =
      R"(
      class A {
        int m() {
          return &((A *)0)->f1 - &((A *)0)->f2;
        }
        int f1;
        int f2;
      };
      )";
  Decl *FromTU = getTuDecl(Code, Lang_CXX11);

  auto *FromF = FirstDeclMatcher<CXXMethodDecl>().match(
      FromTU, cxxMethodDecl(hasName("A::m")));
  Import(FromF, Lang_CXX11);
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportCirularRefFieldsWithoutCorruptedRecordLayoutCache) {
  // Import sequence: A => A.b => B => B.f() => ... => UnaryOperator(&) => ...
  //
  // UnaryOperator(&) should not introduce invalid RecordLayout since 'A' is
  // still not completely imported.
  auto Code =
      R"(
      class B;
      class A {
        B* b;
        int c;
      };
      class B {
        A *f() { return &((B *)0)->a; }
        A a;
      };
      )";

  auto *FromR = FirstDeclMatcher<CXXRecordDecl>().match(
      getTuDecl(Code, Lang_CXX11), cxxRecordDecl(hasName("A")));
  FromR = FromR->getDefinition();
  auto &FromAST = FromR->getASTContext();
  auto *ToR = Import(FromR, Lang_CXX11);
  auto &ToAST = ToR->getASTContext();

  uint64_t SecondFieldOffset = FromAST.getTypeSize(FromAST.VoidPtrTy);

  EXPECT_TRUE(FromR->isCompleteDefinition());
  const auto &FromLayout = FromAST.getASTRecordLayout(FromR);
  EXPECT_TRUE(FromLayout.getFieldOffset(0) == 0);
  EXPECT_TRUE(FromLayout.getFieldOffset(1) == SecondFieldOffset);

  EXPECT_TRUE(ToR->isCompleteDefinition());
  const auto &ToLayout = ToAST.getASTRecordLayout(ToR);
  EXPECT_TRUE(ToLayout.getFieldOffset(0) == 0);
  EXPECT_TRUE(ToLayout.getFieldOffset(1) == SecondFieldOffset);
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportRecordWithLayoutRequestingExpr) {
  TranslationUnitDecl *FromTU = getTuDecl(
      R"(
      struct A {
        int idx;
        static void foo(A x) {
          (void)&"text"[x.idx];
        }
      };
      )",
      Lang_CXX11);

  auto *FromA = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("A")));

  // Test that during import of 'foo' the record layout can be obtained without
  // crash.
  auto *ToA = Import(FromA, Lang_CXX11);
  EXPECT_TRUE(ToA);
  EXPECT_TRUE(ToA->isCompleteDefinition());
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportRecordWithLayoutRequestingExprDifferentRecord) {
  TranslationUnitDecl *FromTU = getTuDecl(
      R"(
      struct B;
      struct A {
        int idx;
        B *b;
      };
      struct B {
        static void foo(A x) {
          (void)&"text"[x.idx];
        }
      };
      )",
      Lang_CXX11);

  auto *FromA = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("A")));

  // Test that during import of 'foo' the record layout (of 'A') can be obtained
  // without crash. It is not possible to have all of the fields of 'A' imported
  // at that time (without big code changes).
  auto *ToA = Import(FromA, Lang_CXX11);
  EXPECT_TRUE(ToA);
  EXPECT_TRUE(ToA->isCompleteDefinition());
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportInClassInitializerFromField) {
  // Encounter import of a field when the field already exists but has the
  // in-class initializer expression not yet set. Such case can occur in the AST
  // of generated template specializations.
  // The first code forces to create a template specialization of
  // `A<int>` but without implicit constructors.
  // The second ("From") code contains a variable of type `A<int>`, this
  // results in a template specialization that has constructors and
  // CXXDefaultInitExpr nodes.
  Decl *ToTU = getToTuDecl(
      R"(
      void f();
      template<typename> struct A { int X = 1; };
      struct B { A<int> Y; };
      )",
      Lang_CXX11);
  auto *ToX = FirstDeclMatcher<FieldDecl>().match(
      ToTU,
      fieldDecl(hasName("X"), hasParent(classTemplateSpecializationDecl())));
  ASSERT_TRUE(ToX->hasInClassInitializer());
  ASSERT_FALSE(ToX->getInClassInitializer());

  Decl *FromTU = getTuDecl(
      R"(
      void f();
      template<typename> struct A { int X = 1; };
      struct B { A<int> Y; };
      //
      A<int> Z;
      )",
      Lang_CXX11, "input1.cc");
  auto *FromX = FirstDeclMatcher<FieldDecl>().match(
      FromTU,
      fieldDecl(hasName("X"), hasParent(classTemplateSpecializationDecl())));

  auto *ToXImported = Import(FromX, Lang_CXX11);
  EXPECT_EQ(ToXImported, ToX);
  EXPECT_TRUE(ToX->getInClassInitializer());
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportInClassInitializerFromCXXDefaultInitExpr) {
  // Encounter AST import of a CXXDefaultInitExpr where the "to-field"
  // of it exists but has the in-class initializer not set yet.
  Decl *ToTU = getToTuDecl(
      R"(
      namespace N {
        template<typename> int b;
        struct X;
      }
      template<typename> struct A { N::X *X = nullptr; };
      struct B { A<int> Y; };
      )",
      Lang_CXX14);
  auto *ToX = FirstDeclMatcher<FieldDecl>().match(
      ToTU,
      fieldDecl(hasName("X"), hasParent(classTemplateSpecializationDecl())));
  ASSERT_TRUE(ToX->hasInClassInitializer());
  ASSERT_FALSE(ToX->getInClassInitializer());

  Decl *FromTU = getTuDecl(
      R"(
      namespace N {
        template<typename> int b;
        struct X;
      }
      template<typename> struct A { N::X *X = nullptr; };
      struct B { A<int> Y; };
      //
      void f() {
        (void)A<int>{};
      }
      struct C {
        C(): attr(new A<int>{}){}
        A<int> *attr;
        const int value = N::b<C>;
      };
      )",
      Lang_CXX14, "input1.cc");
  auto *FromF = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("f"), isDefinition()));
  auto *ToF = Import(FromF, Lang_CXX11);
  EXPECT_TRUE(ToF);
  EXPECT_TRUE(ToX->getInClassInitializer());
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportRecursiveFieldInitializer) {
  const char *Code =
      R"(
      struct AP_TECS;

      struct AP_Landing {
        AP_TECS *TECS_controller;
      };

      struct AP_TECS {
        AP_Landing landing;
      };

      class Plane {
        AP_TECS TECS_controller{landing};
        AP_Landing landing{&TECS_controller};
      };
      )";
  Decl *FromTU = getTuDecl(Code, Lang_CXX11);

  auto *FromR = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("Plane")));
  for (FieldDecl *F : FromR->fields())
    EXPECT_TRUE(F->getInClassInitializer());
  auto *ToR = Import(FromR, Lang_CXX11);
  for (FieldDecl *F : ToR->fields())
    EXPECT_TRUE(F->getInClassInitializer());
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportFieldInitializerWithItself) {
  const char *Code =
      R"(
      class A {
        int a{a};
      };
      )";
  Decl *FromTU = getTuDecl(Code, Lang_CXX11);
  auto *FromA = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("A")));
  EXPECT_TRUE(FromA->field_begin()->getInClassInitializer());
  auto *ToA = Import(FromA, Lang_CXX11);
  EXPECT_TRUE(ToA->field_begin()->getInClassInitializer());
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportRecursiveFieldInitializer1) {
  // FIXME: This is a example of recursive field initialization that is not
  // supported.
  // The following import chain occurs (not complete):
  // import of A => A.a => in-class initializer of A.a => ref_B() => B => B.b
  // => in-class initializer of B.b => ref_A() => CXXConstructExpr for A =>
  // CXXDefaultInitExpr for A.a => in-class initializer of A.a
  // in-class initializer of A.a is created in two different instances in this
  // case (import of FieldDecl and CXXDefaultInitExpr). Probably not a big
  // problem because it is an Expr (the second construction can be ignored
  // instead of assert). But such recursive init code should not occur in
  // practice.
  const char *Code =
      R"(
      static int ref_A();
      static int ref_B();
      struct A {
        int a = ref_B();
      };
      struct B {
        int b = ref_A();
      };
      int ref_B() { B b; return b.b; }
      int ref_A() { A a; return a.a; }
      )";
  Decl *FromTU = getTuDecl(Code, Lang_CXX11);
  auto *FromA = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("A")));
  EXPECT_TRUE(FromA->field_begin()->getInClassInitializer());
  // auto *ToA = Import(FromA, Lang_CXX11);
  // EXPECT_TRUE(ToA->field_begin()->getInClassInitializer());
}

TEST_P(ASTImporterOptionSpecificTestBase, isNewDecl) {
  Decl *FromTU = getTuDecl(
      R"(
      int bar() {
        return 0;
      }
      void other() {
        bar();
      }
      )",
      Lang_CXX11);
  Decl *ToTU = getToTuDecl(
      R"(
      int bar() {
        return 0;
      }
      )",
      Lang_CXX11);
  auto *FromOther = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("other")));
  ASSERT_TRUE(FromOther);

  auto *ToOther = Import(FromOther, Lang_CXX11);
  ASSERT_TRUE(ToOther);

  auto *ToBar = FirstDeclMatcher<FunctionDecl>().match(
      ToTU, functionDecl(hasName("bar")));

  EXPECT_TRUE(SharedStatePtr->isNewDecl(ToOther));
  EXPECT_FALSE(SharedStatePtr->isNewDecl(ToBar));
}

struct ImportInjectedClassNameType : public ASTImporterOptionSpecificTestBase {
protected:
  const CXXRecordDecl *findInjected(const CXXRecordDecl *Parent) {
    for (Decl *Found : Parent->decls()) {
      const auto *Record = dyn_cast<CXXRecordDecl>(Found);
      if (Record && Record->isInjectedClassName())
        return Record;
    }
    return nullptr;
  }

  void checkInjType(const CXXRecordDecl *D) {
    // The whole redecl chain should have the same InjectedClassNameType
    // instance. The injected record declaration is a separate chain, this
    // should contain the same type too.
    const Type *Ty = nullptr;
    for (const Decl *ReD : D->redecls()) {
      const auto *ReRD = cast<CXXRecordDecl>(ReD);
      EXPECT_TRUE(ReRD->getTypeForDecl());
      EXPECT_TRUE(!Ty || Ty == ReRD->getTypeForDecl());
      Ty = ReRD->getTypeForDecl();
    }
    ASSERT_TRUE(Ty);
    const auto *InjTy = Ty->castAs<InjectedClassNameType>();
    EXPECT_TRUE(InjTy);
    if (CXXRecordDecl *Def = D->getDefinition()) {
      const CXXRecordDecl *InjRD = findInjected(Def);
      EXPECT_TRUE(InjRD);
      EXPECT_EQ(InjRD->getTypeForDecl(), InjTy);
    }
  }

  void testImport(Decl *ToTU, Decl *FromTU, Decl *FromD) {
    checkInjType(cast<CXXRecordDecl>(FromD));
    Decl *ToD = Import(FromD, Lang_CXX11);
    if (auto *ToRD = dyn_cast<CXXRecordDecl>(ToD))
      checkInjType(ToRD);
  }

  const char *ToCodeA =
      R"(
      template <class T>
      struct A;
      )";
  const char *ToCodeADef =
      R"(
      template <class T>
      struct A {
        typedef A T1;
      };
      )";
  const char *ToCodeC =
      R"(
      template <class T>
      struct C;
      )";
  const char *ToCodeCDef =
      R"(
      template <class T>
      struct A {
        typedef A T1;
      };

      template <class T1, class T2>
      struct B {};

      template<class T>
      struct C {
        typedef typename A<T>::T1 T1;
        typedef B<T1, T> T2;
        typedef B<T1, C> T3;
      };
      )";
  const char *FromCode =
      R"(
      template <class T>
      struct A;
      template <class T>
      struct A {
        typedef A T1;
      };
      template <class T>
      struct A;

      template <class T1, class T2>
      struct B {};

      template <class T>
      struct C;
      template <class T>
      struct C {
        typedef typename A<T>::T1 T1;
        typedef B<T1, T> T2;
        typedef B<T1, C> T3;
      };
      template <class T>
      struct C;

      template <class T>
      struct D {
        void f(typename C<T>::T3 *);
      };
      )";
};

TEST_P(ImportInjectedClassNameType, ImportADef) {
  Decl *ToTU = getToTuDecl(ToCodeA, Lang_CXX11);
  Decl *FromTU = getTuDecl(FromCode, Lang_CXX11);
  auto *FromA = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("A"), isDefinition()));
  testImport(ToTU, FromTU, FromA);
}

TEST_P(ImportInjectedClassNameType, ImportAFirst) {
  Decl *ToTU = getToTuDecl(ToCodeA, Lang_CXX11);
  Decl *FromTU = getTuDecl(FromCode, Lang_CXX11);
  auto *FromA = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("A")));
  testImport(ToTU, FromTU, FromA);
}

TEST_P(ImportInjectedClassNameType, ImportALast) {
  Decl *ToTU = getToTuDecl(ToCodeA, Lang_CXX11);
  Decl *FromTU = getTuDecl(FromCode, Lang_CXX11);
  auto *FromA = LastDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("A")));
  testImport(ToTU, FromTU, FromA);
}

TEST_P(ImportInjectedClassNameType, ImportADefToDef) {
  Decl *ToTU = getToTuDecl(ToCodeADef, Lang_CXX11);
  Decl *FromTU = getTuDecl(FromCode, Lang_CXX11);
  auto *FromA = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("A"), isDefinition()));
  testImport(ToTU, FromTU, FromA);
}

TEST_P(ImportInjectedClassNameType, ImportAFirstToDef) {
  Decl *ToTU = getToTuDecl(ToCodeADef, Lang_CXX11);
  Decl *FromTU = getTuDecl(FromCode, Lang_CXX11);
  auto *FromA = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("A")));
  testImport(ToTU, FromTU, FromA);
}

TEST_P(ImportInjectedClassNameType, ImportALastToDef) {
  Decl *ToTU = getToTuDecl(ToCodeADef, Lang_CXX11);
  Decl *FromTU = getTuDecl(FromCode, Lang_CXX11);
  auto *FromA = LastDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("A")));
  testImport(ToTU, FromTU, FromA);
}

TEST_P(ImportInjectedClassNameType, ImportCDef) {
  Decl *ToTU = getToTuDecl(ToCodeC, Lang_CXX11);
  Decl *FromTU = getTuDecl(FromCode, Lang_CXX11);
  auto *FromC = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("C"), isDefinition()));
  testImport(ToTU, FromTU, FromC);
}

TEST_P(ImportInjectedClassNameType, ImportCLast) {
  Decl *ToTU = getToTuDecl(ToCodeC, Lang_CXX11);
  Decl *FromTU = getTuDecl(FromCode, Lang_CXX11);
  auto *FromC = LastDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("C")));
  testImport(ToTU, FromTU, FromC);
}

TEST_P(ImportInjectedClassNameType, ImportCDefToDef) {
  Decl *ToTU = getToTuDecl(ToCodeCDef, Lang_CXX11);
  Decl *FromTU = getTuDecl(FromCode, Lang_CXX11);
  auto *FromC = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("C"), isDefinition()));
  testImport(ToTU, FromTU, FromC);
}

TEST_P(ImportInjectedClassNameType, ImportCLastToDef) {
  Decl *ToTU = getToTuDecl(ToCodeCDef, Lang_CXX11);
  Decl *FromTU = getTuDecl(FromCode, Lang_CXX11);
  auto *FromC = LastDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("C")));
  testImport(ToTU, FromTU, FromC);
}

TEST_P(ImportInjectedClassNameType, ImportD) {
  Decl *ToTU = getToTuDecl("", Lang_CXX11);
  Decl *FromTU = getTuDecl(FromCode, Lang_CXX11);
  auto *FromD = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("D"), isDefinition()));
  testImport(ToTU, FromTU, FromD);
}

TEST_P(ImportInjectedClassNameType, ImportDToDef) {
  Decl *ToTU = getToTuDecl(ToCodeCDef, Lang_CXX11);
  Decl *FromTU = getTuDecl(FromCode, Lang_CXX11);
  auto *FromD = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("D"), isDefinition()));
  testImport(ToTU, FromTU, FromD);
}

TEST_P(ImportInjectedClassNameType, ImportTypedefType) {
  Decl *ToTU = getToTuDecl(
      R"(
      template <class T>
      struct A {
        typedef A A1;
        void f(A1 *);
      };
      )",
      Lang_CXX11);
  Decl *FromTU = getTuDecl(
      R"(
      template <class T>
      struct A {
        typedef A A1;
        void f(A1 *);
      };
      template<class T>
      void A<T>::f(A::A1 *) {}
      )",
      Lang_CXX11);

  auto *FromF = FirstDeclMatcher<FunctionDecl>().match(
      FromTU, functionDecl(hasName("f"), isDefinition()));
  auto *ToF = Import(FromF, Lang_CXX11);
  EXPECT_TRUE(ToF);
  ASTContext &ToCtx = ToF->getDeclContext()->getParentASTContext();

  auto *ToA1 =
      FirstDeclMatcher<TypedefDecl>().match(ToTU, typedefDecl(hasName("A1")));
  QualType ToInjTypedef = ToA1->getUnderlyingType().getCanonicalType();
  QualType ToInjParmVar =
      ToF->parameters()[0]->getType().getDesugaredType(ToCtx);
  ToInjParmVar =
      ToInjParmVar->getAs<PointerType>()->getPointeeType().getCanonicalType();
  EXPECT_TRUE(isa<InjectedClassNameType>(ToInjTypedef));
  EXPECT_TRUE(isa<InjectedClassNameType>(ToInjParmVar));
  EXPECT_TRUE(ToCtx.hasSameType(ToInjTypedef, ToInjParmVar));
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportMacroQualifiedType) {
  Decl *From, *To;
  std::tie(From, To) = getImportedDecl(
      R"(
        #define CDECL __attribute__((cdecl))
        typedef void (CDECL *X)();
      )",
      Lang_CXX03, "", Lang_CXX03, "X");

  auto *FromTy =
      FirstDeclMatcher<MacroQualifiedType>().match(From, macroQualifiedType());
  auto *ToTy =
      FirstDeclMatcher<MacroQualifiedType>().match(To, macroQualifiedType());

  EXPECT_TRUE(isa<AttributedType>(FromTy->getUnderlyingType()));
  EXPECT_TRUE(isa<AttributedType>(ToTy->getUnderlyingType()));
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportCorrectTemplateName) {
  constexpr auto TestCode = R"(
      template <class T>
      struct A;
      template <class T>
      struct A {};
      template <template<class> class T = A>
      struct B {};
      using C = B<>;
      )";
  Decl *ToTU = getToTuDecl(TestCode, Lang_CXX11);
  Decl *FromTU = getTuDecl(TestCode, Lang_CXX11);

  auto *ToUsingFirst = FirstDeclMatcher<TypeAliasDecl>().match(
      ToTU, typeAliasDecl(hasName("C")));

  auto *FromUsing = FirstDeclMatcher<TypeAliasDecl>().match(
      FromTU, typeAliasDecl(hasName("C")));
  auto *ToUsing = Import(FromUsing, Lang_CXX11);
  EXPECT_TRUE(ToUsing);

  auto *ToB = FirstDeclMatcher<ClassTemplateDecl>().match(
      ToTU, classTemplateDecl(hasName("B")));
  auto *ToB1 = LastDeclMatcher<ClassTemplateDecl>().match(
      ToTU, classTemplateDecl(hasName("B")));
  // One template definition of 'B' should exist.
  EXPECT_EQ(ToB, ToB1);

  // These declarations are imported separately.
  EXPECT_NE(ToUsingFirst, ToUsing);

  auto SpB = ToB->spec_begin();
  auto SpE = ToB->spec_end();
  EXPECT_TRUE(SpB != SpE);
  ClassTemplateSpecializationDecl *Spec1 = *SpB;
  ++SpB;
  // The template 'B' should have one specialization (with default argument).
  EXPECT_TRUE(SpB == SpE);

  // Even if 'B' has one specialization with the default arguments, the AST
  // contains after the import two specializations that are linked in the
  // declaration chain. The 'spec_begin' iteration does not find these because
  // the template arguments are the same. But the imported type alias has the
  // link to the second specialization. The template name object in these
  // specializations must point to the same (and one) instance of definition of
  // 'B'.
  auto *Spec2 = cast<ClassTemplateSpecializationDecl>(
      ToUsing->getUnderlyingType()
          ->getAs<TemplateSpecializationType>()
          ->getAsRecordDecl());
  EXPECT_NE(Spec1, Spec2);
  EXPECT_TRUE(Spec1->getPreviousDecl() == Spec2 ||
              Spec2->getPreviousDecl() == Spec1);
  TemplateDecl *Templ1 =
      Spec1->getTemplateArgs()[0].getAsTemplate().getAsTemplateDecl();
  TemplateDecl *Templ2 =
      Spec2->getTemplateArgs()[0].getAsTemplate().getAsTemplateDecl();
  EXPECT_EQ(Templ1, Templ2);
}

TEST_P(ASTImporterOptionSpecificTestBase, VaListC) {
  Decl *FromTU = getTuDecl(R"(typedef __builtin_va_list va_list;)", Lang_C99);

  auto *FromVaList = FirstDeclMatcher<TypedefDecl>().match(
      FromTU, typedefDecl(hasName("va_list")));
  ASSERT_TRUE(FromVaList);

  auto *ToVaList = Import(FromVaList, Lang_C99);
  ASSERT_TRUE(ToVaList);

  auto *ToBuiltinVaList = FirstDeclMatcher<TypedefDecl>().match(
      ToAST->getASTContext().getTranslationUnitDecl(),
      typedefDecl(hasName("__builtin_va_list")));

  ASSERT_TRUE(ToAST->getASTContext().hasSameType(
      ToVaList->getUnderlyingType(), ToBuiltinVaList->getUnderlyingType()));
}

TEST_P(ASTImporterOptionSpecificTestBase, VaListCpp) {
  Decl *FromTU = getTuDecl(R"(typedef __builtin_va_list va_list;)", Lang_CXX03);

  auto *FromVaList = FirstDeclMatcher<TypedefDecl>().match(
      FromTU, typedefDecl(hasName("va_list")));
  ASSERT_TRUE(FromVaList);

  auto *ToVaList = Import(FromVaList, Lang_CXX03);
  ASSERT_TRUE(ToVaList);

  auto *ToBuiltinVaList = FirstDeclMatcher<TypedefDecl>().match(
      ToAST->getASTContext().getTranslationUnitDecl(),
      typedefDecl(hasName("__builtin_va_list")));

  ASSERT_TRUE(ToAST->getASTContext().hasSameType(
      ToVaList->getUnderlyingType(), ToBuiltinVaList->getUnderlyingType()));
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportDefinitionOfEmptyClassWithNoUniqueAddressField) {
  Decl *FromTU = getTuDecl(
      R"(
      struct B {};
      struct A { B b; };
      )",
      Lang_CXX20);

  CXXRecordDecl *FromD = FirstDeclMatcher<CXXRecordDecl>().match(
      FromTU, cxxRecordDecl(hasName("A")));

  for (auto *FD : FromD->fields())
    FD->addAttr(clang::NoUniqueAddressAttr::Create(FromD->getASTContext(),
                                                   clang::SourceRange()));
  FromD->markEmpty();

  CXXRecordDecl *ToD = Import(FromD, Lang_CXX20);
  EXPECT_TRUE(ToD->isEmpty());
  for (auto *FD : ToD->fields())
    EXPECT_EQ(true, FD->hasAttr<NoUniqueAddressAttr>());
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportExistingTypedefToRecord) {
  const char *Code =
      R"(
      struct S { int i; };
      typedef struct S T;
      extern T x;
      )";
  Decl *ToTU = getToTuDecl(Code, Lang_C99);
  Decl *FromTU = getTuDecl(Code, Lang_C99);

  auto *FromX =
      FirstDeclMatcher<VarDecl>().match(FromTU, varDecl(hasName("x")));
  auto *ToX = Import(FromX, Lang_C99);
  EXPECT_TRUE(ToX);

  auto *Typedef1 =
      FirstDeclMatcher<TypedefDecl>().match(ToTU, typedefDecl(hasName("T")));
  auto *Typedef2 =
      LastDeclMatcher<TypedefDecl>().match(ToTU, typedefDecl(hasName("T")));
  EXPECT_EQ(Typedef1, Typedef2);
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportExistingTypedefToUnnamedRecord) {
  const char *Code =
      R"(
      typedef const struct { int f; } T;
      extern T x;
      )";
  Decl *ToTU = getToTuDecl(Code, Lang_C99);
  Decl *FromTU = getTuDecl(Code, Lang_C99);

  auto *FromX =
      FirstDeclMatcher<VarDecl>().match(FromTU, varDecl(hasName("x")));
  auto *ToX = Import(FromX, Lang_C99);
  EXPECT_TRUE(ToX);

  auto *Typedef1 =
      FirstDeclMatcher<TypedefDecl>().match(ToTU, typedefDecl(hasName("T")));
  auto *Typedef2 =
      LastDeclMatcher<TypedefDecl>().match(ToTU, typedefDecl(hasName("T")));
  EXPECT_NE(Typedef1, Typedef2);
  EXPECT_NE(Typedef1->getUnderlyingType().getTypePtr(),
            Typedef2->getUnderlyingType().getTypePtr());
  EXPECT_EQ(ToX->getType()->getAs<TypedefType>()->getDecl(), Typedef2);
}

TEST_P(ASTImporterOptionSpecificTestBase, ImportTwoTypedefsToUnnamedRecord) {
  const char *Code =
      R"(
      typedef struct { int f; } T1;
      typedef struct { int f; } T2;
      extern T1 x1;
      extern T2 x2;
      )";
  Decl *ToTU = getToTuDecl("", Lang_C99);
  Decl *FromTU = getTuDecl(Code, Lang_C99);

  auto *FromX1 =
      FirstDeclMatcher<VarDecl>().match(FromTU, varDecl(hasName("x1")));
  auto *FromX2 =
      FirstDeclMatcher<VarDecl>().match(FromTU, varDecl(hasName("x2")));
  auto *ToX1 = Import(FromX1, Lang_C99);
  EXPECT_TRUE(ToX1);
  auto *ToX2 = Import(FromX2, Lang_C99);
  EXPECT_TRUE(ToX2);

  auto *Typedef1 =
      FirstDeclMatcher<TypedefDecl>().match(ToTU, typedefDecl(hasName("T1")));
  auto *Typedef2 =
      FirstDeclMatcher<TypedefDecl>().match(ToTU, typedefDecl(hasName("T2")));
  EXPECT_NE(Typedef1->getUnderlyingType().getTypePtr(),
            Typedef2->getUnderlyingType().getTypePtr());
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportExistingTypedefToUnnamedRecordPtr) {
  const char *Code =
      R"(
      typedef const struct { int fff; } * const T;
      extern T x;
      )";
  Decl *ToTU = getToTuDecl(Code, Lang_C99);
  Decl *FromTU = getTuDecl(Code, Lang_C99);

  auto *FromX =
      FirstDeclMatcher<VarDecl>().match(FromTU, varDecl(hasName("x")));
  auto *ToX = Import(FromX, Lang_C99);
  EXPECT_TRUE(ToX);

  auto *Typedef1 =
      FirstDeclMatcher<TypedefDecl>().match(ToTU, typedefDecl(hasName("T")));
  auto *Typedef2 =
      LastDeclMatcher<TypedefDecl>().match(ToTU, typedefDecl(hasName("T")));
  // FIXME: These should be imported separately, like in the test above.
  // Or: In the test above these should be merged too.
  EXPECT_EQ(Typedef1, Typedef2);

  auto *FromR = FirstDeclMatcher<RecordDecl>().match(
      FromTU, recordDecl(hasDescendant(fieldDecl(hasName("fff")))));
  auto *ToRExisting = FirstDeclMatcher<RecordDecl>().match(
      ToTU, recordDecl(hasDescendant(fieldDecl(hasName("fff")))));
  ASSERT_TRUE(FromR);
  auto *ToRImported = Import(FromR, Lang_C99);
  // FIXME: If typedefs are not imported separately, do not import ToRImported
  // separately.
  EXPECT_NE(ToRExisting, ToRImported);
}

TEST_P(ASTImporterOptionSpecificTestBase,
       ImportTypedefWithDifferentUnderlyingType) {
  const char *Code =
      R"(
      using X1 = int;
      using Y1 = int;

      using RPB1 = X1*;
      typedef RPB1 RPX1;
      using RPB1 = Y1*; // redeclared
      typedef RPB1 RPY1;

      auto X = 0 ? (RPX1){} : (RPY1){};
      )";
  Decl *FromTU = getTuDecl(Code, Lang_CXX11);

  auto *FromX =
      FirstDeclMatcher<VarDecl>().match(FromTU, varDecl(hasName("X")));

  auto *FromXType = FromX->getType()->getAs<TypedefType>();
  EXPECT_FALSE(FromXType->typeMatchesDecl());

  auto *ToX = Import(FromX, Lang_CXX11);
  auto *ToXType = ToX->getType()->getAs<TypedefType>();
  // FIXME: This should be false.
  EXPECT_TRUE(ToXType->typeMatchesDecl());
}

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, ASTImporterLookupTableTest,
                         DefaultTestValuesForRunOptions);

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, ImportPath,
                         ::testing::Values(std::vector<std::string>()));

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, ImportExpr,
                         DefaultTestValuesForRunOptions);

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, ImportFixedPointExpr,
                         ExtendWithOptions(DefaultTestArrayForRunOptions,
                                           std::vector<std::string>{
                                               "-ffixed-point"}));

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, ImportBlock,
                         ExtendWithOptions(DefaultTestArrayForRunOptions,
                                           std::vector<std::string>{
                                               "-fblocks"}));

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, ImportType,
                         DefaultTestValuesForRunOptions);

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, ImportDecl,
                         DefaultTestValuesForRunOptions);

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, ASTImporterOptionSpecificTestBase,
                         DefaultTestValuesForRunOptions);

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, ErrorHandlingTest,
                         DefaultTestValuesForRunOptions);

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, RedirectingImporterTest,
                         DefaultTestValuesForRunOptions);

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, ImportFunctions,
                         DefaultTestValuesForRunOptions);

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, ImportAutoFunctions,
                         DefaultTestValuesForRunOptions);

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, ImportFunctionTemplates,
                         DefaultTestValuesForRunOptions);

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, ImportFriendFunctionTemplates,
                         DefaultTestValuesForRunOptions);

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, ImportClasses,
                         DefaultTestValuesForRunOptions);

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, ImportFriendFunctions,
                         DefaultTestValuesForRunOptions);

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, ImportFriendClasses,
                         DefaultTestValuesForRunOptions);

INSTANTIATE_TEST_SUITE_P(ParameterizedTests,
                         ImportFunctionTemplateSpecializations,
                         DefaultTestValuesForRunOptions);

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, ImportImplicitMethods,
                         DefaultTestValuesForRunOptions);

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, ImportVariables,
                         DefaultTestValuesForRunOptions);

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, LLDBLookupTest,
                         DefaultTestValuesForRunOptions);

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, ImportSourceLocations,
                         DefaultTestValuesForRunOptions);

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, ImportWithExternalSource,
                         DefaultTestValuesForRunOptions);

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, ImportAttributes,
                         DefaultTestValuesForRunOptions);

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, ImportInjectedClassNameType,
                         DefaultTestValuesForRunOptions);

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, ImportMatrixType,
                         DefaultTestValuesForRunOptions);

// FIXME: Make ImportOpenCLPipe test work.
// INSTANTIATE_TEST_SUITE_P(ParameterizedTests, ImportOpenCLPipe,
//                          DefaultTestValuesForRunOptions);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ImportOpenCLPipe);

} // end namespace ast_matchers
} // end namespace clang
