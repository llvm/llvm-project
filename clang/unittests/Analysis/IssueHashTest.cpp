//===- IssueHashTest.cpp - IssueHash unit tests --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/IssueHash.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"
#include <memory>
#include <string>

namespace clang {
namespace {

using namespace ast_matchers;

std::unique_ptr<ASTUnit> buildAST(llvm::StringRef Code,
                                  std::vector<std::string> Args = {
                                      "-fsyntax-only", "-std=c++20"}) {
  return tooling::buildASTFromCodeWithArgs(Code, Args);
}

// getIssueString() joins several '$'-delimited fields together; the
// enclosing-decl signature (computed by the internal, file-local
// GetEnclosingDeclContextSignature()) is always the second field. Pull it
// out in isolation so these tests don't have to hardcode the unrelated
// column number and source-line fields.
std::string getEnclosingDeclSignature(ASTContext &Ctx, const Decl *IssueDecl) {
  FullSourceLoc Loc(Ctx.getSourceManager().getLocForStartOfFile(
                        Ctx.getSourceManager().getMainFileID()),
                    Ctx.getSourceManager());
  std::string Full =
      getIssueString(Loc, "checker", "message", IssueDecl, Ctx.getLangOpts());
  size_t FirstDollar = Full.find('$');
  size_t SecondDollar = Full.find('$', FirstDollar + 1);
  return Full.substr(FirstDollar + 1, SecondDollar - FirstDollar - 1);
}

TEST(IssueHashTest, EnclosingVarDeclUsesQualifiedName) {
  auto AST = buildAST(R"cpp(
    namespace ns {
      int global_var;
    }
  )cpp");
  ASTContext &Ctx = AST->getASTContext();
  const auto *VD = selectFirst<VarDecl>(
      "v", match(varDecl(hasName("global_var")).bind("v"), Ctx));
  ASSERT_NE(VD, nullptr);

  EXPECT_EQ(getEnclosingDeclSignature(Ctx, VD), "ns::global_var");
}

TEST(IssueHashTest, EnclosingFieldDeclUsesQualifiedName) {
  auto AST = buildAST(R"cpp(
    struct S {
      int field;
    };
  )cpp");
  ASTContext &Ctx = AST->getASTContext();
  const auto *FD = selectFirst<FieldDecl>(
      "f", match(fieldDecl(hasName("field")).bind("f"), Ctx));
  ASSERT_NE(FD, nullptr);

  EXPECT_EQ(getEnclosingDeclSignature(Ctx, FD), "S::field");
}

TEST(IssueHashTest, EnclosingEnumConstantDeclUsesQualifiedName) {
  auto AST = buildAST(R"cpp(
    enum class Color { Red };
  )cpp");
  ASTContext &Ctx = AST->getASTContext();
  const auto *ECD = selectFirst<EnumConstantDecl>(
      "e", match(enumConstantDecl(hasName("Red")).bind("e"), Ctx));
  ASSERT_NE(ECD, nullptr);

  EXPECT_EQ(getEnclosingDeclSignature(Ctx, ECD), "Color::Red");
}

TEST(IssueHashTest, EnclosingFunctionDeclUsesSignatureNotQualifiedName) {
  auto AST = buildAST(R"cpp(
    void foo(int);
  )cpp");
  ASTContext &Ctx = AST->getASTContext();
  const auto *Fn = selectFirst<FunctionDecl>(
      "fn", match(functionDecl(hasName("foo")).bind("fn"), Ctx));
  ASSERT_NE(Fn, nullptr);

  // Functions (and methods/constructors/destructors) still get the full
  // signature, not just the qualified name, so overloads don't collide.
  EXPECT_EQ(getEnclosingDeclSignature(Ctx, Fn), "void foo(int)");
}

TEST(IssueHashTest, EnclosingCXXRecordDeclUsesQualifiedName) {
  auto AST = buildAST(R"cpp(
    namespace ns {
      struct Widget {};
    }
  )cpp");
  ASTContext &Ctx = AST->getASTContext();
  const auto *RD = selectFirst<CXXRecordDecl>(
      "r", match(cxxRecordDecl(hasName("Widget")).bind("r"), Ctx));
  ASSERT_NE(RD, nullptr);

  EXPECT_EQ(getEnclosingDeclSignature(Ctx, RD), "ns::Widget");
}

// The remaining tests below each cover one of the case labels that used to
// be explicitly listed in GetEnclosingDeclContextSignature()'s switch,
// before it was simplified to a single dyn_cast<FunctionDecl> check plus a
// fallback to getQualifiedNameAsString() for everything else.

TEST(IssueHashTest, EnclosingNamespaceDeclUsesQualifiedName) {
  auto AST = buildAST(R"cpp(
    namespace outer {
      namespace inner {}
    }
  )cpp");
  ASTContext &Ctx = AST->getASTContext();
  const auto *NS = selectFirst<NamespaceDecl>(
      "n", match(namespaceDecl(hasName("inner")).bind("n"), Ctx));
  ASSERT_NE(NS, nullptr);

  EXPECT_EQ(getEnclosingDeclSignature(Ctx, NS), "outer::inner");
}

TEST(IssueHashTest, EnclosingRecordDeclUsesQualifiedName) {
  // A plain (non-C++) 'struct' is a RecordDecl, not a CXXRecordDecl -- that
  // distinction only exists when parsing as C++, where every struct/class
  // is upgraded to a CXXRecordDecl. So this specifically needs C, not C++,
  // to exercise the Decl::Record case label rather than Decl::CXXRecord.
  auto AST = buildAST("struct S { int x; };",
                      {"-fsyntax-only", "-std=c17", "-x", "c"});
  ASTContext &Ctx = AST->getASTContext();
  const auto *RD = selectFirst<RecordDecl>(
      "r", match(recordDecl(hasName("S")).bind("r"), Ctx));
  ASSERT_NE(RD, nullptr);
  ASSERT_FALSE(isa<CXXRecordDecl>(RD));

  EXPECT_EQ(getEnclosingDeclSignature(Ctx, RD), "S");
}

TEST(IssueHashTest, EnclosingEnumDeclUsesQualifiedName) {
  auto AST = buildAST(R"cpp(
    enum class Color { Red };
  )cpp");
  ASTContext &Ctx = AST->getASTContext();
  const auto *ED = selectFirst<EnumDecl>(
      "e", match(enumDecl(hasName("Color")).bind("e"), Ctx));
  ASSERT_NE(ED, nullptr);

  EXPECT_EQ(getEnclosingDeclSignature(Ctx, ED), "Color");
}

TEST(IssueHashTest, EnclosingObjCInterfaceDeclUsesQualifiedName) {
  auto AST = buildAST(R"objc(
    __attribute__((objc_root_class))
    @interface Foo
    @end
  )objc",
                      {"-fsyntax-only", "-x", "objective-c++", "-std=c++17"});
  ASTContext &Ctx = AST->getASTContext();
  const auto *ID = selectFirst<ObjCInterfaceDecl>(
      "i", match(objcInterfaceDecl(hasName("Foo")).bind("i"), Ctx));
  ASSERT_NE(ID, nullptr);

  EXPECT_EQ(getEnclosingDeclSignature(Ctx, ID), "Foo");
}

TEST(IssueHashTest, EnclosingObjCImplementationDeclUsesQualifiedName) {
  auto AST = buildAST(R"objc(
    __attribute__((objc_root_class))
    @interface Foo
    @end
    @implementation Foo
    @end
  )objc",
                      {"-fsyntax-only", "-x", "objective-c++", "-std=c++17"});
  ASTContext &Ctx = AST->getASTContext();
  const auto *ImplD = selectFirst<ObjCImplementationDecl>(
      "i", match(objcImplementationDecl(hasName("Foo")).bind("i"), Ctx));
  ASSERT_NE(ImplD, nullptr);

  EXPECT_EQ(getEnclosingDeclSignature(Ctx, ImplD), "Foo");
}

TEST(IssueHashTest, EnclosingObjCCategoryDeclUsesQualifiedName) {
  auto AST = buildAST(R"objc(
    __attribute__((objc_root_class))
    @interface Foo
    @end
    @interface Foo (Cat)
    @end
  )objc",
                      {"-fsyntax-only", "-x", "objective-c++", "-std=c++17"});
  ASTContext &Ctx = AST->getASTContext();
  const auto *CatD = selectFirst<ObjCCategoryDecl>(
      "c", match(objcCategoryDecl(hasName("Cat")).bind("c"), Ctx));
  ASSERT_NE(CatD, nullptr);

  EXPECT_EQ(getEnclosingDeclSignature(Ctx, CatD), "Cat");
}

TEST(IssueHashTest, EnclosingObjCCategoryImplDeclUsesQualifiedName) {
  auto AST = buildAST(R"objc(
    __attribute__((objc_root_class))
    @interface Foo
    @end
    @interface Foo (Cat)
    @end
    @implementation Foo (Cat)
    @end
  )objc",
                      {"-fsyntax-only", "-x", "objective-c++", "-std=c++17"});
  ASTContext &Ctx = AST->getASTContext();
  const auto *CatImplD = selectFirst<ObjCCategoryImplDecl>(
      "c", match(objcCategoryImplDecl(hasName("Cat")).bind("c"), Ctx));
  ASSERT_NE(CatImplD, nullptr);

  EXPECT_EQ(getEnclosingDeclSignature(Ctx, CatImplD), "Cat");
}

TEST(IssueHashTest, EnclosingObjCProtocolDeclUsesQualifiedName) {
  auto AST = buildAST(R"objc(
    @protocol Proto
    @end
  )objc",
                      {"-fsyntax-only", "-x", "objective-c++", "-std=c++17"});
  ASTContext &Ctx = AST->getASTContext();
  const auto *PD = selectFirst<ObjCProtocolDecl>(
      "p", match(objcProtocolDecl(hasName("Proto")).bind("p"), Ctx));
  ASSERT_NE(PD, nullptr);

  EXPECT_EQ(getEnclosingDeclSignature(Ctx, PD), "Proto");
}

TEST(IssueHashTest, EnclosingCXXConstructorDeclUsesSignature) {
  auto AST = buildAST(R"cpp(
    struct S { S(int); };
  )cpp");
  ASTContext &Ctx = AST->getASTContext();
  const auto *Ctor = selectFirst<CXXConstructorDecl>(
      "c", match(cxxConstructorDecl(ofClass(hasName("S"))).bind("c"), Ctx));
  ASSERT_NE(Ctor, nullptr);

  EXPECT_EQ(getEnclosingDeclSignature(Ctx, Ctor), "S::S(int)");
}

TEST(IssueHashTest, EnclosingCXXDestructorDeclUsesSignature) {
  auto AST = buildAST(R"cpp(
    struct S { ~S(); };
  )cpp");
  ASTContext &Ctx = AST->getASTContext();
  const auto *Dtor = selectFirst<CXXDestructorDecl>(
      "d", match(cxxDestructorDecl(ofClass(hasName("S"))).bind("d"), Ctx));
  ASSERT_NE(Dtor, nullptr);

  EXPECT_EQ(getEnclosingDeclSignature(Ctx, Dtor), "S::~S()");
}

TEST(IssueHashTest, EnclosingCXXConversionDeclUsesSignature) {
  auto AST = buildAST(R"cpp(
    struct S { operator int(); };
  )cpp");
  ASTContext &Ctx = AST->getASTContext();
  const auto *Conv = selectFirst<CXXConversionDecl>(
      "cv", match(cxxConversionDecl().bind("cv"), Ctx));
  ASSERT_NE(Conv, nullptr);

  EXPECT_EQ(getEnclosingDeclSignature(Ctx, Conv), "S::operator int()");
}

TEST(IssueHashTest, EnclosingCXXMethodDeclUsesSignature) {
  auto AST = buildAST(R"cpp(
    struct S { void method(int); };
  )cpp");
  ASTContext &Ctx = AST->getASTContext();
  const auto *M = selectFirst<CXXMethodDecl>(
      "m", match(cxxMethodDecl(hasName("method")).bind("m"), Ctx));
  ASSERT_NE(M, nullptr);

  EXPECT_EQ(getEnclosingDeclSignature(Ctx, M), "void S::method(int)");
}

TEST(IssueHashTest, EnclosingObjCMethodDeclUsesQualifiedName) {
  auto AST = buildAST(R"objc(
    @interface Foo
    - (void)method;
    @end
  )objc",
                      {"-fsyntax-only", "-x", "objective-c++", "-std=c++17"});
  ASTContext &Ctx = AST->getASTContext();
  const auto *MD = selectFirst<ObjCMethodDecl>(
      "m", match(objcMethodDecl(hasName("method")).bind("m"), Ctx));
  ASSERT_NE(MD, nullptr);

  EXPECT_EQ(getEnclosingDeclSignature(Ctx, MD), "Foo::method");
}

TEST(IssueHashTest, NullDeclProducesEmptySignature) {
  auto AST = buildAST("");
  ASTContext &Ctx = AST->getASTContext();

  EXPECT_EQ(getEnclosingDeclSignature(Ctx, nullptr), "");
}

} // namespace
} // namespace clang
