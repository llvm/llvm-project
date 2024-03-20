//===- unittest/AST/ASTImporterTest.cpp - AST node import test ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Type-parameterized tests for the correct import of redecl chains.
//
//===----------------------------------------------------------------------===//

#include "ASTImporterFixtures.h"

namespace clang {
namespace ast_matchers {

using internal::BindableMatcher;

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

struct EnumClass {
  using DeclTy = EnumDecl;
  static constexpr auto *Prototype = "enum class X;";
  static constexpr auto *Definition = "enum class X {};";
  BindableMatcher<Decl> getPattern() {
    return enumDecl(hasName("X"), unless(isImplicit()));
  }
};

struct Variable {
  using DeclTy = VarDecl;
  static constexpr auto *Prototype = "extern int X;";
  static constexpr auto *Definition = "int X;";
  BindableMatcher<Decl> getPattern() { return varDecl(hasName("X")); }
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

struct VariableTemplate {
  using DeclTy = VarTemplateDecl;
  static constexpr auto *Prototype = "template <class T> extern T X;";
  static constexpr auto *Definition =
      R"(
      template <class T> T X;
      )";
  // There is no matcher for varTemplateDecl so use a work-around.
  BindableMatcher<Decl> getPattern() {
    return namedDecl(hasName("X"), unless(isImplicit()),
                     has(templateTypeParmDecl()));
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

struct ClassTemplateSpec {
  using DeclTy = ClassTemplateSpecializationDecl;
  static constexpr auto *Prototype =
      R"(
      template <class T> class X;
      template <> class X<int>;
      )";
  static constexpr auto *Definition =
      R"(
      template <class T> class X;
      template <> class X<int> {};
      )";
  BindableMatcher<Decl> getPattern() {
    return classTemplateSpecializationDecl(hasName("X"), unless(isImplicit()));
  }
};

const internal::VariadicDynCastAllOfMatcher<Decl, VarTemplateDecl>
    varTemplateDecl;
const internal::VariadicDynCastAllOfMatcher<Decl, VarTemplateSpecializationDecl>
    varTemplateSpecializationDecl;

struct VariableTemplateSpec {
  using DeclTy = VarTemplateSpecializationDecl;
  static constexpr auto *Prototype =
      R"(
      template <class T> extern T X;
      template <> extern int X<int>;
      )";
  static constexpr auto *Definition =
      R"(
      template <class T> T X;
      template <> int X<int>;
      )";
  BindableMatcher<Decl> getPattern() {
    return varTemplateSpecializationDecl(hasName("X"), unless(isImplicit()));
  }
};

template <typename TypeParam>
struct RedeclChain : ASTImporterOptionSpecificTestBase {

  using DeclTy = typename TypeParam::DeclTy;
  std::string getPrototype() { return TypeParam::Prototype; }
  std::string getDefinition() { return TypeParam::Definition; }
  BindableMatcher<Decl> getPattern() const { return TypeParam().getPattern(); }

  void CheckPreviousDecl(Decl *Prev, Decl *Current) {
    ASSERT_NE(Prev, Current);
    ASSERT_EQ(&Prev->getASTContext(), &Current->getASTContext());
    EXPECT_EQ(Prev->getCanonicalDecl(), Current->getCanonicalDecl());

    // Templates.
    if (auto *PrevT = dyn_cast<TemplateDecl>(Prev)) {
      EXPECT_EQ(Current->getPreviousDecl(), Prev);
      auto *CurrentT = cast<TemplateDecl>(Current);
      ASSERT_TRUE(PrevT->getTemplatedDecl());
      ASSERT_TRUE(CurrentT->getTemplatedDecl());
      EXPECT_EQ(CurrentT->getTemplatedDecl()->getPreviousDecl(),
                PrevT->getTemplatedDecl());
      return;
    }

    // Specializations.
    if (auto *PrevF = dyn_cast<FunctionDecl>(Prev)) {
      if (PrevF->getTemplatedKind() ==
          FunctionDecl::TK_FunctionTemplateSpecialization) {
        // There may be a hidden fwd spec decl before a spec decl.
        // In that case the previous visible decl can be reached through that
        // invisible one.
        EXPECT_THAT(Prev, testing::AnyOf(
                              Current->getPreviousDecl(),
                              Current->getPreviousDecl()->getPreviousDecl()));
        auto *ToTU = Prev->getTranslationUnitDecl();
        auto *TemplateD = FirstDeclMatcher<FunctionTemplateDecl>().match(
            ToTU, functionTemplateDecl());
        auto *FirstSpecD = *(TemplateD->spec_begin());
        EXPECT_EQ(FirstSpecD->getCanonicalDecl(), PrevF->getCanonicalDecl());
        return;
      }
    }

    // The rest: Classes, Functions, etc.
    EXPECT_EQ(Current->getPreviousDecl(), Prev);
  }

  void
  TypedTest_PrototypeShouldBeImportedAsAPrototypeWhenThereIsNoDefinition() {
    Decl *FromTU = getTuDecl(getPrototype(), Lang_CXX03);
    auto *FromD = FirstDeclMatcher<DeclTy>().match(FromTU, getPattern());
    ASSERT_FALSE(FromD->isThisDeclarationADefinition());

    Decl *ImportedD = Import(FromD, Lang_CXX03);
    Decl *ToTU = ImportedD->getTranslationUnitDecl();

    EXPECT_EQ(DeclCounter<DeclTy>().match(ToTU, getPattern()), 1u);
    auto *ToD = LastDeclMatcher<DeclTy>().match(ToTU, getPattern());
    EXPECT_TRUE(ImportedD == ToD);
    EXPECT_FALSE(ToD->isThisDeclarationADefinition());
    if (auto *ToT = dyn_cast<TemplateDecl>(ToD)) {
      EXPECT_TRUE(ToT->getTemplatedDecl());
    }
  }

  void TypedTest_DefinitionShouldBeImportedAsADefinition() {
    Decl *FromTU = getTuDecl(getDefinition(), Lang_CXX03);
    auto *FromD = FirstDeclMatcher<DeclTy>().match(FromTU, getPattern());
    ASSERT_TRUE(FromD->isThisDeclarationADefinition());

    Decl *ImportedD = Import(FromD, Lang_CXX03);
    Decl *ToTU = ImportedD->getTranslationUnitDecl();

    EXPECT_EQ(DeclCounter<DeclTy>().match(ToTU, getPattern()), 1u);
    auto *ToD = LastDeclMatcher<DeclTy>().match(ToTU, getPattern());
    EXPECT_TRUE(ToD->isThisDeclarationADefinition());
    if (auto *ToT = dyn_cast<TemplateDecl>(ToD)) {
      EXPECT_TRUE(ToT->getTemplatedDecl());
    }
  }

  void TypedTest_ImportPrototypeAfterImportedPrototype() {
    Decl *FromTU = getTuDecl(getPrototype() + getPrototype(), Lang_CXX03);
    auto *From0 = FirstDeclMatcher<DeclTy>().match(FromTU, getPattern());
    auto *From1 = LastDeclMatcher<DeclTy>().match(FromTU, getPattern());
    ASSERT_FALSE(From0->isThisDeclarationADefinition());
    ASSERT_FALSE(From1->isThisDeclarationADefinition());

    Decl *Imported0 = Import(From0, Lang_CXX03);
    Decl *Imported1 = Import(From1, Lang_CXX03);
    Decl *ToTU = Imported0->getTranslationUnitDecl();

    EXPECT_EQ(DeclCounter<DeclTy>().match(ToTU, getPattern()), 2u);
    auto *To0 = FirstDeclMatcher<DeclTy>().match(ToTU, getPattern());
    auto *To1 = LastDeclMatcher<DeclTy>().match(ToTU, getPattern());
    EXPECT_TRUE(Imported0 == To0);
    EXPECT_TRUE(Imported1 == To1);
    EXPECT_FALSE(To0->isThisDeclarationADefinition());
    EXPECT_FALSE(To1->isThisDeclarationADefinition());

    CheckPreviousDecl(To0, To1);
  }

  void TypedTest_ImportDefinitionAfterImportedPrototype() {
    Decl *FromTU = getTuDecl(getPrototype() + getDefinition(), Lang_CXX03);
    auto *FromProto = FirstDeclMatcher<DeclTy>().match(FromTU, getPattern());
    auto *FromDef = LastDeclMatcher<DeclTy>().match(FromTU, getPattern());
    ASSERT_FALSE(FromProto->isThisDeclarationADefinition());
    ASSERT_TRUE(FromDef->isThisDeclarationADefinition());

    Decl *ImportedProto = Import(FromProto, Lang_CXX03);
    Decl *ImportedDef = Import(FromDef, Lang_CXX03);
    Decl *ToTU = ImportedProto->getTranslationUnitDecl();

    EXPECT_EQ(DeclCounter<DeclTy>().match(ToTU, getPattern()), 2u);
    auto *ToProto = FirstDeclMatcher<DeclTy>().match(ToTU, getPattern());
    auto *ToDef = LastDeclMatcher<DeclTy>().match(ToTU, getPattern());
    EXPECT_TRUE(ImportedProto == ToProto);
    EXPECT_TRUE(ImportedDef == ToDef);
    EXPECT_FALSE(ToProto->isThisDeclarationADefinition());
    EXPECT_TRUE(ToDef->isThisDeclarationADefinition());

    CheckPreviousDecl(ToProto, ToDef);
  }

  void TypedTest_ImportPrototypeAfterImportedDefinition() {
    Decl *FromTU = getTuDecl(getDefinition() + getPrototype(), Lang_CXX03);
    auto *FromDef = FirstDeclMatcher<DeclTy>().match(FromTU, getPattern());
    auto *FromProto = LastDeclMatcher<DeclTy>().match(FromTU, getPattern());
    ASSERT_TRUE(FromDef->isThisDeclarationADefinition());
    ASSERT_FALSE(FromProto->isThisDeclarationADefinition());

    Decl *ImportedDef = Import(FromDef, Lang_CXX03);
    Decl *ImportedProto = Import(FromProto, Lang_CXX03);
    Decl *ToTU = ImportedDef->getTranslationUnitDecl();

    EXPECT_EQ(DeclCounter<DeclTy>().match(ToTU, getPattern()), 2u);
    auto *ToDef = FirstDeclMatcher<DeclTy>().match(ToTU, getPattern());
    auto *ToProto = LastDeclMatcher<DeclTy>().match(ToTU, getPattern());
    EXPECT_TRUE(ImportedDef == ToDef);
    EXPECT_TRUE(ImportedProto == ToProto);
    EXPECT_TRUE(ToDef->isThisDeclarationADefinition());
    EXPECT_FALSE(ToProto->isThisDeclarationADefinition());

    CheckPreviousDecl(ToDef, ToProto);
  }

  void TypedTest_ImportPrototypes() {
    Decl *FromTU0 = getTuDecl(getPrototype(), Lang_CXX03, "input0.cc");
    Decl *FromTU1 = getTuDecl(getPrototype(), Lang_CXX03, "input1.cc");
    auto *From0 = FirstDeclMatcher<DeclTy>().match(FromTU0, getPattern());
    auto *From1 = FirstDeclMatcher<DeclTy>().match(FromTU1, getPattern());
    ASSERT_FALSE(From0->isThisDeclarationADefinition());
    ASSERT_FALSE(From1->isThisDeclarationADefinition());

    Decl *Imported0 = Import(From0, Lang_CXX03);
    Decl *Imported1 = Import(From1, Lang_CXX03);
    Decl *ToTU = Imported0->getTranslationUnitDecl();

    EXPECT_EQ(DeclCounter<DeclTy>().match(ToTU, getPattern()), 2u);
    auto *To0 = FirstDeclMatcher<DeclTy>().match(ToTU, getPattern());
    auto *To1 = LastDeclMatcher<DeclTy>().match(ToTU, getPattern());
    EXPECT_TRUE(Imported0 == To0);
    EXPECT_TRUE(Imported1 == To1);
    EXPECT_FALSE(To0->isThisDeclarationADefinition());
    EXPECT_FALSE(To1->isThisDeclarationADefinition());

    CheckPreviousDecl(To0, To1);
  }

  void TypedTest_ImportDefinitions() {
    Decl *FromTU0 = getTuDecl(getDefinition(), Lang_CXX03, "input0.cc");
    Decl *FromTU1 = getTuDecl(getDefinition(), Lang_CXX03, "input1.cc");
    auto *From0 = FirstDeclMatcher<DeclTy>().match(FromTU0, getPattern());
    auto *From1 = FirstDeclMatcher<DeclTy>().match(FromTU1, getPattern());
    ASSERT_TRUE(From0->isThisDeclarationADefinition());
    ASSERT_TRUE(From1->isThisDeclarationADefinition());

    Decl *Imported0 = Import(From0, Lang_CXX03);
    Decl *Imported1 = Import(From1, Lang_CXX03);
    Decl *ToTU = Imported0->getTranslationUnitDecl();

    EXPECT_EQ(Imported0, Imported1);
    EXPECT_EQ(DeclCounter<DeclTy>().match(ToTU, getPattern()), 1u);
    auto *To0 = FirstDeclMatcher<DeclTy>().match(ToTU, getPattern());
    EXPECT_TRUE(Imported0 == To0);
    EXPECT_TRUE(To0->isThisDeclarationADefinition());
    if (auto *ToT0 = dyn_cast<TemplateDecl>(To0)) {
      EXPECT_TRUE(ToT0->getTemplatedDecl());
    }
  }

  void TypedTest_ImportDefinitionThenPrototype() {
    Decl *FromTUDef = getTuDecl(getDefinition(), Lang_CXX03, "input0.cc");
    Decl *FromTUProto = getTuDecl(getPrototype(), Lang_CXX03, "input1.cc");
    auto *FromDef = FirstDeclMatcher<DeclTy>().match(FromTUDef, getPattern());
    auto *FromProto =
        FirstDeclMatcher<DeclTy>().match(FromTUProto, getPattern());
    ASSERT_TRUE(FromDef->isThisDeclarationADefinition());
    ASSERT_FALSE(FromProto->isThisDeclarationADefinition());

    Decl *ImportedDef = Import(FromDef, Lang_CXX03);
    Decl *ImportedProto = Import(FromProto, Lang_CXX03);
    Decl *ToTU = ImportedDef->getTranslationUnitDecl();

    EXPECT_NE(ImportedDef, ImportedProto);
    EXPECT_EQ(DeclCounter<DeclTy>().match(ToTU, getPattern()), 2u);
    auto *ToDef = FirstDeclMatcher<DeclTy>().match(ToTU, getPattern());
    auto *ToProto = LastDeclMatcher<DeclTy>().match(ToTU, getPattern());
    EXPECT_TRUE(ImportedDef == ToDef);
    EXPECT_TRUE(ImportedProto == ToProto);
    EXPECT_TRUE(ToDef->isThisDeclarationADefinition());
    EXPECT_FALSE(ToProto->isThisDeclarationADefinition());

    CheckPreviousDecl(ToDef, ToProto);
  }

  void TypedTest_ImportPrototypeThenDefinition() {
    Decl *FromTUProto = getTuDecl(getPrototype(), Lang_CXX03, "input0.cc");
    Decl *FromTUDef = getTuDecl(getDefinition(), Lang_CXX03, "input1.cc");
    auto *FromProto =
        FirstDeclMatcher<DeclTy>().match(FromTUProto, getPattern());
    auto *FromDef = FirstDeclMatcher<DeclTy>().match(FromTUDef, getPattern());
    ASSERT_TRUE(FromDef->isThisDeclarationADefinition());
    ASSERT_FALSE(FromProto->isThisDeclarationADefinition());

    Decl *ImportedProto = Import(FromProto, Lang_CXX03);
    Decl *ImportedDef = Import(FromDef, Lang_CXX03);
    Decl *ToTU = ImportedDef->getTranslationUnitDecl();

    EXPECT_NE(ImportedDef, ImportedProto);
    EXPECT_EQ(DeclCounter<DeclTy>().match(ToTU, getPattern()), 2u);
    auto *ToProto = FirstDeclMatcher<DeclTy>().match(ToTU, getPattern());
    auto *ToDef = LastDeclMatcher<DeclTy>().match(ToTU, getPattern());
    EXPECT_TRUE(ImportedDef == ToDef);
    EXPECT_TRUE(ImportedProto == ToProto);
    EXPECT_TRUE(ToDef->isThisDeclarationADefinition());
    EXPECT_FALSE(ToProto->isThisDeclarationADefinition());

    CheckPreviousDecl(ToProto, ToDef);
  }

  void TypedTest_WholeRedeclChainIsImportedAtOnce() {
    Decl *FromTU = getTuDecl(getPrototype() + getDefinition(), Lang_CXX03);
    auto *FromD = // Definition
        LastDeclMatcher<DeclTy>().match(FromTU, getPattern());
    ASSERT_TRUE(FromD->isThisDeclarationADefinition());

    Decl *ImportedD = Import(FromD, Lang_CXX03);
    Decl *ToTU = ImportedD->getTranslationUnitDecl();

    // The whole redecl chain is imported at once.
    EXPECT_EQ(DeclCounter<DeclTy>().match(ToTU, getPattern()), 2u);
    EXPECT_TRUE(cast<DeclTy>(ImportedD)->isThisDeclarationADefinition());
  }

  void TypedTest_ImportPrototypeThenProtoAndDefinition() {
    {
      Decl *FromTU = getTuDecl(getPrototype(), Lang_CXX03, "input0.cc");
      auto *FromD = FirstDeclMatcher<DeclTy>().match(FromTU, getPattern());
      Import(FromD, Lang_CXX03);
    }
    {
      Decl *FromTU =
          getTuDecl(getPrototype() + getDefinition(), Lang_CXX03, "input1.cc");
      auto *FromD = FirstDeclMatcher<DeclTy>().match(FromTU, getPattern());
      Import(FromD, Lang_CXX03);
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

    CheckPreviousDecl(ProtoD, DefinitionD->getPreviousDecl());
  }
};

#define ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(BaseTemplate, TypeParam,       \
                                                NamePrefix, TestCase)          \
  using BaseTemplate##TypeParam = BaseTemplate<TypeParam>;                     \
  TEST_P(BaseTemplate##TypeParam, NamePrefix##TestCase) {                      \
    TypedTest_##TestCase();                                                    \
  }

ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(
    RedeclChain, Function, ,
    PrototypeShouldBeImportedAsAPrototypeWhenThereIsNoDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(
    RedeclChain, Class, ,
    PrototypeShouldBeImportedAsAPrototypeWhenThereIsNoDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(
    RedeclChain, EnumClass, ,
    PrototypeShouldBeImportedAsAPrototypeWhenThereIsNoDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(
    RedeclChain, Variable, ,
    PrototypeShouldBeImportedAsAPrototypeWhenThereIsNoDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(
    RedeclChain, FunctionTemplate, ,
    PrototypeShouldBeImportedAsAPrototypeWhenThereIsNoDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(
    RedeclChain, ClassTemplate, ,
    PrototypeShouldBeImportedAsAPrototypeWhenThereIsNoDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(
    RedeclChain, VariableTemplate, ,
    PrototypeShouldBeImportedAsAPrototypeWhenThereIsNoDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(
    RedeclChain, FunctionTemplateSpec, ,
    PrototypeShouldBeImportedAsAPrototypeWhenThereIsNoDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(
    RedeclChain, ClassTemplateSpec, ,
    PrototypeShouldBeImportedAsAPrototypeWhenThereIsNoDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(
    RedeclChain, VariableTemplateSpec, ,
    PrototypeShouldBeImportedAsAPrototypeWhenThereIsNoDefinition)

ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Function, ,
                                        DefinitionShouldBeImportedAsADefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Class, ,
                                        DefinitionShouldBeImportedAsADefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, EnumClass, ,
                                        DefinitionShouldBeImportedAsADefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Variable, ,
                                        DefinitionShouldBeImportedAsADefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, FunctionTemplate, ,
                                        DefinitionShouldBeImportedAsADefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, ClassTemplate, ,
                                        DefinitionShouldBeImportedAsADefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, VariableTemplate, ,
                                        DefinitionShouldBeImportedAsADefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, FunctionTemplateSpec, ,
                                        DefinitionShouldBeImportedAsADefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, ClassTemplateSpec, ,
                                        DefinitionShouldBeImportedAsADefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(
    RedeclChain, VariableTemplateSpec, ,
    DefinitionShouldBeImportedAsADefinition)

ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Function, ,
                                        ImportPrototypeAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Class, ,
                                        ImportPrototypeAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, EnumClass, ,
                                        ImportPrototypeAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Variable, ,
                                        ImportPrototypeAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, FunctionTemplate, ,
                                        ImportPrototypeAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, ClassTemplate, ,
                                        ImportPrototypeAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, VariableTemplate, ,
                                        ImportPrototypeAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, FunctionTemplateSpec, ,
                                        ImportPrototypeAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, ClassTemplateSpec, ,
                                        ImportPrototypeAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, VariableTemplateSpec, ,
                                         ImportPrototypeAfterImportedPrototype)

ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Function, ,
                                        ImportDefinitionAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Class, ,
                                        ImportDefinitionAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, EnumClass, ,
                                        ImportDefinitionAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Variable, ,
                                        ImportDefinitionAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, FunctionTemplate, ,
                                        ImportDefinitionAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, ClassTemplate, ,
                                        ImportDefinitionAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, VariableTemplate, ,
                                        ImportDefinitionAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, FunctionTemplateSpec, ,
                                        ImportDefinitionAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, ClassTemplateSpec, ,
                                        ImportDefinitionAfterImportedPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, VariableTemplateSpec, ,
                                         ImportDefinitionAfterImportedPrototype)

ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Function, ,
                                        ImportPrototypeAfterImportedDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Class, ,
                                        ImportPrototypeAfterImportedDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, EnumClass, ,
                                        ImportPrototypeAfterImportedDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Variable, ,
                                        ImportPrototypeAfterImportedDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, FunctionTemplate, ,
                                        ImportPrototypeAfterImportedDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, ClassTemplate, ,
                                        ImportPrototypeAfterImportedDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, VariableTemplate, ,
                                        ImportPrototypeAfterImportedDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, FunctionTemplateSpec, ,
                                        ImportPrototypeAfterImportedDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, ClassTemplateSpec, ,
                                        ImportPrototypeAfterImportedDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, VariableTemplateSpec, ,
                                         ImportPrototypeAfterImportedDefinition)

ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Function, ,
                                        ImportPrototypes)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Class, , ImportPrototypes)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, EnumClass, ,
                                        ImportPrototypes)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Variable, ,
                                        ImportPrototypes)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, FunctionTemplate, ,
                                        ImportPrototypes)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, ClassTemplate, ,
                                        ImportPrototypes)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, VariableTemplate, ,
                                        ImportPrototypes)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, ClassTemplateSpec, ,
                                        ImportPrototypes)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, FunctionTemplateSpec, ,
                                        ImportPrototypes)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, VariableTemplateSpec, ,
                                         ImportPrototypes)

ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Function, ,
                                        ImportDefinitions)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Class, , ImportDefinitions)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, EnumClass, ,
                                        ImportDefinitions)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Variable, ,
                                        ImportDefinitions)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, FunctionTemplate, ,
                                        ImportDefinitions)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, ClassTemplate, ,
                                        ImportDefinitions)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, VariableTemplate, ,
                                        ImportDefinitions)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, ClassTemplateSpec, ,
                                        ImportDefinitions)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, FunctionTemplateSpec, ,
                                        ImportDefinitions)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, VariableTemplateSpec, ,
                                         ImportDefinitions)

ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Function, ,
                                        ImportDefinitionThenPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Class, ,
                                        ImportDefinitionThenPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, EnumClass, ,
                                        ImportDefinitionThenPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Variable, ,
                                        ImportDefinitionThenPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, FunctionTemplate, ,
                                        ImportDefinitionThenPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, ClassTemplate, ,
                                        ImportDefinitionThenPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, VariableTemplate, ,
                                        ImportDefinitionThenPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, FunctionTemplateSpec, ,
                                        ImportDefinitionThenPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, ClassTemplateSpec, ,
                                        ImportDefinitionThenPrototype)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, VariableTemplateSpec, ,
                                         ImportDefinitionThenPrototype)

ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Function, ,
                                        ImportPrototypeThenDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Class, ,
                                        ImportPrototypeThenDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, EnumClass, ,
                                        ImportPrototypeThenDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Variable, ,
                                        ImportPrototypeThenDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, FunctionTemplate, ,
                                        ImportPrototypeThenDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, ClassTemplate, ,
                                        ImportPrototypeThenDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, VariableTemplate, ,
                                        ImportPrototypeThenDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, FunctionTemplateSpec, ,
                                        ImportPrototypeThenDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, ClassTemplateSpec, ,
                                        ImportPrototypeThenDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, VariableTemplateSpec, ,
                                         ImportPrototypeThenDefinition)

ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Function, ,
                                        WholeRedeclChainIsImportedAtOnce)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Variable, ,
                                        WholeRedeclChainIsImportedAtOnce)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, FunctionTemplate, ,
                                        WholeRedeclChainIsImportedAtOnce)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, VariableTemplate, ,
                                        WholeRedeclChainIsImportedAtOnce)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, FunctionTemplateSpec, ,
                                        WholeRedeclChainIsImportedAtOnce)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, VariableTemplateSpec, ,
                                         WholeRedeclChainIsImportedAtOnce)

ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Function, ,
                                        ImportPrototypeThenProtoAndDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, Variable, ,
                                        ImportPrototypeThenProtoAndDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, FunctionTemplate, ,
                                        ImportPrototypeThenProtoAndDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, VariableTemplate, ,
                                        ImportPrototypeThenProtoAndDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, FunctionTemplateSpec, ,
                                        ImportPrototypeThenProtoAndDefinition)
ASTIMPORTER_INSTANTIATE_TYPED_TEST_SUITE(RedeclChain, VariableTemplateSpec, ,
                                         ImportPrototypeThenProtoAndDefinition)

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, RedeclChainFunction,
                        DefaultTestValuesForRunOptions);
INSTANTIATE_TEST_SUITE_P(ParameterizedTests, RedeclChainClass,
                        DefaultTestValuesForRunOptions);
INSTANTIATE_TEST_SUITE_P(ParameterizedTests, RedeclChainEnumClass,
                        DefaultTestValuesForRunOptions);
INSTANTIATE_TEST_SUITE_P(ParameterizedTests, RedeclChainVariable,
                        DefaultTestValuesForRunOptions);
INSTANTIATE_TEST_SUITE_P(ParameterizedTests, RedeclChainFunctionTemplate,
                        DefaultTestValuesForRunOptions);
INSTANTIATE_TEST_SUITE_P(ParameterizedTests, RedeclChainClassTemplate,
                        DefaultTestValuesForRunOptions);
INSTANTIATE_TEST_SUITE_P(ParameterizedTests, RedeclChainVariableTemplate,
                        DefaultTestValuesForRunOptions);
INSTANTIATE_TEST_SUITE_P(ParameterizedTests, RedeclChainFunctionTemplateSpec,
                        DefaultTestValuesForRunOptions);
INSTANTIATE_TEST_SUITE_P(ParameterizedTests, RedeclChainClassTemplateSpec,
                        DefaultTestValuesForRunOptions);
INSTANTIATE_TEST_SUITE_P(ParameterizedTests, RedeclChainVariableTemplateSpec,
                         DefaultTestValuesForRunOptions);

} // end namespace ast_matchers
} // end namespace clang
