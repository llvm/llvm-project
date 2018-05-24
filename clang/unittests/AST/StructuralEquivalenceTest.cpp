#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/AST/ASTStructuralEquivalence.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"

#include "Language.h"
#include "DeclMatcher.h"

#include "gtest/gtest.h"

namespace clang {
namespace ast_matchers {

using std::get;

struct StructuralEquivalenceTest : ::testing::Test {
  std::unique_ptr<ASTUnit> AST0, AST1;
  std::string Code0, Code1; // Buffers for SourceManager

  // Get a pair of Decl pointers to the synthetised declarations from the given
  // code snipets. By default we search for the unique Decl with name 'foo' in
  // both snippets.
  std::tuple<NamedDecl *, NamedDecl *>
  makeNamedDecls(const std::string &SrcCode0, const std::string &SrcCode1,
                 Language Lang, const char *const Identifier = "foo") {

    this->Code0 = SrcCode0;
    this->Code1 = SrcCode1;
    ArgVector Args = getBasicRunOptionsForLanguage(Lang);

    const char *const InputFileName = "input.cc";

    AST0 = tooling::buildASTFromCodeWithArgs(Code0, Args, InputFileName);
    AST1 = tooling::buildASTFromCodeWithArgs(Code1, Args, InputFileName);

    ASTContext &Ctx0 = AST0->getASTContext(), &Ctx1 = AST1->getASTContext();

    auto getDecl = [](ASTContext &Ctx, const std::string &Name) -> NamedDecl * {
      IdentifierInfo *SearchedII = &Ctx.Idents.get(Name);
      assert(SearchedII && "Declaration with the identifier "
                           "should be specified in test!");
      DeclarationName SearchDeclName(SearchedII);
      SmallVector<NamedDecl *, 4> FoundDecls;
      Ctx.getTranslationUnitDecl()->localUncachedLookup(SearchDeclName,
                                                        FoundDecls);

      // We should find one Decl but one only.
      assert(FoundDecls.size() == 1);

      return FoundDecls[0];
    };

    NamedDecl *D0 = getDecl(Ctx0, Identifier);
    NamedDecl *D1 = getDecl(Ctx1, Identifier);
    assert(D0);
    assert(D1);
    return std::make_tuple(D0, D1);
  }

  bool testStructuralMatch(NamedDecl *D0, NamedDecl *D1) {
    llvm::DenseSet<std::pair<Decl *, Decl *>> NonEquivalentDecls;
    StructuralEquivalenceContext Ctx(D0->getASTContext(), D1->getASTContext(),
                                     NonEquivalentDecls, false, false);
    return Ctx.IsStructurallyEquivalent(D0, D1);
  }

  bool testStructuralMatch(std::tuple<NamedDecl *, NamedDecl *> t) {
    return testStructuralMatch(get<0>(t), get<1>(t));
  }
};

TEST_F(StructuralEquivalenceTest, Int) {
  auto Decls = makeNamedDecls("int foo;", "int foo;", Lang_CXX);
  EXPECT_TRUE(testStructuralMatch(Decls));
}

TEST_F(StructuralEquivalenceTest, IntVsSignedInt) {
  auto Decls = makeNamedDecls("int foo;", "signed int foo;", Lang_CXX);
  EXPECT_TRUE(testStructuralMatch(Decls));
}

TEST_F(StructuralEquivalenceTest, Char) {
  auto Decls = makeNamedDecls("char foo;", "char foo;", Lang_CXX);
  EXPECT_TRUE(testStructuralMatch(Decls));
}

// This test is disabled for now.
// FIXME Whether this is equivalent is dependendant on the target.
TEST_F(StructuralEquivalenceTest, DISABLED_CharVsSignedChar) {
  auto Decls = makeNamedDecls("char foo;", "signed char foo;", Lang_CXX);
  EXPECT_FALSE(testStructuralMatch(Decls));
}

TEST_F(StructuralEquivalenceTest, ForwardRecordDecl) {
  auto Decls = makeNamedDecls("struct foo;", "struct foo;", Lang_CXX);
  EXPECT_TRUE(testStructuralMatch(Decls));
}

TEST_F(StructuralEquivalenceTest, IntVsSignedIntInStruct) {
  auto Decls = makeNamedDecls("struct foo { int x; };",
                              "struct foo { signed int x; };", Lang_CXX);
  EXPECT_TRUE(testStructuralMatch(Decls));
}

TEST_F(StructuralEquivalenceTest, CharVsSignedCharInStruct) {
  auto Decls = makeNamedDecls("struct foo { char x; };",
                              "struct foo { signed char x; };", Lang_CXX);
  EXPECT_FALSE(testStructuralMatch(Decls));
}

TEST_F(StructuralEquivalenceTest, IntVsSignedIntTemplateSpec) {
  auto Decls = makeNamedDecls(
      "template <class T> struct foo; template<> struct foo<int>{};",
      "template <class T> struct foo; template<> struct foo<signed int>{};",
      Lang_CXX);
  ClassTemplateSpecializationDecl *Spec0 =
      *cast<ClassTemplateDecl>(get<0>(Decls))->spec_begin();
  ClassTemplateSpecializationDecl *Spec1 =
      *cast<ClassTemplateDecl>(get<1>(Decls))->spec_begin();
  ASSERT_TRUE(Spec0 != nullptr);
  ASSERT_TRUE(Spec1 != nullptr);
  EXPECT_TRUE(testStructuralMatch(Spec0, Spec1));
}

TEST_F(StructuralEquivalenceTest, CharVsSignedCharTemplateSpec) {
  auto Decls = makeNamedDecls(
      "template <class T> struct foo; template<> struct foo<char>{};",
      "template <class T> struct foo; template<> struct foo<signed char>{};",
      Lang_CXX);
  ClassTemplateSpecializationDecl *Spec0 =
      *cast<ClassTemplateDecl>(get<0>(Decls))->spec_begin();
  ClassTemplateSpecializationDecl *Spec1 =
      *cast<ClassTemplateDecl>(get<1>(Decls))->spec_begin();
  ASSERT_TRUE(Spec0 != nullptr);
  ASSERT_TRUE(Spec1 != nullptr);
  EXPECT_FALSE(testStructuralMatch(Spec0, Spec1));
}

TEST_F(StructuralEquivalenceTest, CharVsSignedCharTemplateSpecWithInheritance) {
  auto Decls = makeNamedDecls(
      R"(
      struct true_type{};
      template <class T> struct foo;
      template<> struct foo<char> : true_type {};
      )",
      R"(
      struct true_type{};
      template <class T> struct foo;
      template<> struct foo<signed char> : true_type {};
      )",
      Lang_CXX);
  ClassTemplateSpecializationDecl *Spec0 =
      *cast<ClassTemplateDecl>(get<0>(Decls))->spec_begin();
  ClassTemplateSpecializationDecl *Spec1 =
      *cast<ClassTemplateDecl>(get<1>(Decls))->spec_begin();
  ASSERT_TRUE(Spec0 != nullptr);
  ASSERT_TRUE(Spec1 != nullptr);
  EXPECT_FALSE(testStructuralMatch(Spec0, Spec1));
}

// This test is disabled for now.
// FIXME Enable it, once the check is implemented.
TEST_F(StructuralEquivalenceTest, DISABLED_WrongOrderInNamespace) {
  auto Code =
      R"(
      namespace NS {
      template <class T> class Base {
          int a;
      };
      class Derived : Base<Derived> {
      };
      }
      void foo(NS::Derived &);
      )";
  auto Decls = makeNamedDecls(Code, Code, Lang_CXX);

  NamespaceDecl *NS =
      LastDeclMatcher<NamespaceDecl>().match(get<1>(Decls), namespaceDecl());
  ClassTemplateDecl *TD = LastDeclMatcher<ClassTemplateDecl>().match(
      get<1>(Decls), classTemplateDecl(hasName("Base")));

  // Reorder the decls, move the TD to the last place in the DC.
  NS->removeDecl(TD);
  NS->addDeclInternal(TD);

  EXPECT_FALSE(testStructuralMatch(Decls));
}

TEST_F(StructuralEquivalenceTest, WrongOrderOfFieldsInClass) {
  auto Code = "class X { int a; int b; };";
  auto Decls = makeNamedDecls(Code, Code, Lang_CXX, "X");

  CXXRecordDecl *RD = FirstDeclMatcher<CXXRecordDecl>().match(
      get<1>(Decls), cxxRecordDecl(hasName("X")));
  FieldDecl *FD =
      FirstDeclMatcher<FieldDecl>().match(get<1>(Decls), fieldDecl(hasName("a")));

  // Reorder the FieldDecls
  RD->removeDecl(FD);
  RD->addDeclInternal(FD);

  EXPECT_FALSE(testStructuralMatch(Decls));
}

} // end namespace ast_matchers
} // end namespace clang
