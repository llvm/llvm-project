//===- unittest/AST/ASTImporterODRStrategiesTest.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Type-parameterized tests to verify the import behaviour in case of ODR
// violation.
//
//===----------------------------------------------------------------------===//

#include "ASTImporterFixtures.h"

namespace clang {
namespace ast_matchers {

using internal::BindableMatcher;

// DeclTy: Type of the Decl to check.
// Prototype: "Prototype" (forward declaration) of the Decl.
// Definition: A definition for the Prototype.
// ConflictingPrototype: A prototype with the same name but different
// declaration.
// ConflictingDefinition: A different definition for Prototype.
// ConflictingProtoDef: A definition for ConflictingPrototype.
// getPattern: Return a matcher that matches any of Prototype, Definition,
// ConflictingPrototype, ConflictingDefinition, ConflictingProtoDef.

struct Function {
  using DeclTy = FunctionDecl;
  static constexpr auto *Prototype = "void X(int);";
  static constexpr auto *ConflictingPrototype = "void X(double);";
  static constexpr auto *Definition = "void X(int a) {}";
  static constexpr auto *ConflictingDefinition = "void X(double a) {}";
  BindableMatcher<Decl> getPattern() {
    return functionDecl(hasName("X"), unless(isImplicit()));
  }
  Language getLang() { return Lang_C; }
};

struct Typedef {
  using DeclTy = TypedefNameDecl;
  static constexpr auto *Definition = "typedef int X;";
  static constexpr auto *ConflictingDefinition = "typedef double X;";
  BindableMatcher<Decl> getPattern() { return typedefNameDecl(hasName("X")); }
  Language getLang() { return Lang_CXX; }
};

struct TypedefAlias {
  using DeclTy = TypedefNameDecl;
  static constexpr auto *Definition = "using X = int;";
  static constexpr auto *ConflictingDefinition = "using X = double;";
  BindableMatcher<Decl> getPattern() { return typedefNameDecl(hasName("X")); }
  Language getLang() { return Lang_CXX11; }
};

struct Enum {
  using DeclTy = EnumDecl;
  static constexpr auto *Definition = "enum X { a, b };";
  static constexpr auto *ConflictingDefinition = "enum X { a, b, c };";
  BindableMatcher<Decl> getPattern() { return enumDecl(hasName("X")); }
  Language getLang() { return Lang_CXX; }
};

struct EnumConstant {
  using DeclTy = EnumConstantDecl;
  static constexpr auto *Definition = "enum E { X = 0 };";
  static constexpr auto *ConflictingDefinition = "enum E { X = 1 };";
  BindableMatcher<Decl> getPattern() { return enumConstantDecl(hasName("X")); }
  Language getLang() { return Lang_CXX; }
};

struct Class {
  using DeclTy = CXXRecordDecl;
  static constexpr auto *Prototype = "class X;";
  static constexpr auto *Definition = "class X {};";
  static constexpr auto *ConflictingDefinition = "class X { int A; };";
  BindableMatcher<Decl> getPattern() {
    return cxxRecordDecl(hasName("X"), unless(isImplicit()));
  }
  Language getLang() { return Lang_CXX; }
};

struct Variable {
  using DeclTy = VarDecl;
  static constexpr auto *Prototype = "extern int X;";
  static constexpr auto *ConflictingPrototype = "extern float X;";
  static constexpr auto *Definition = "int X;";
  static constexpr auto *ConflictingDefinition = "float X;";
  BindableMatcher<Decl> getPattern() { return varDecl(hasName("X")); }
  Language getLang() { return Lang_CXX; }
};

struct ClassTemplate {
  using DeclTy = ClassTemplateDecl;
  static constexpr auto *Prototype = "template <class> class X;";
  static constexpr auto *ConflictingPrototype = "template <int> class X;";
  static constexpr auto *Definition = "template <class> class X {};";
  static constexpr auto *ConflictingDefinition =
      "template <class> class X { int A; };";
  static constexpr auto *ConflictingProtoDef = "template <int> class X { };";
  BindableMatcher<Decl> getPattern() {
    return classTemplateDecl(hasName("X"), unless(isImplicit()));
  }
  Language getLang() { return Lang_CXX; }
};

struct FunctionTemplate {
  using DeclTy = FunctionTemplateDecl;
  static constexpr auto *Definition0 =
      R"(
      template <class T>
      void X(T a) {};
      )";
  // This is actually not a conflicting definition, but another primary template.
  static constexpr auto *Definition1 =
      R"(
      template <class T>
      void X(T* a) {};
      )";
  BindableMatcher<Decl> getPattern() {
    return functionTemplateDecl(hasName("X"), unless(isImplicit()));
  }
  static std::string getDef0() { return Definition0; }
  static std::string getDef1() { return Definition1; }
  Language getLang() { return Lang_CXX; }
};

static const internal::VariadicDynCastAllOfMatcher<Decl, VarTemplateDecl>
    varTemplateDecl;

struct VarTemplate {
  using DeclTy = VarTemplateDecl;
  static constexpr auto *Definition =
      R"(
      template <class T>
      constexpr T X = 0;
      )";
  static constexpr auto *ConflictingDefinition =
      R"(
      template <int>
      constexpr int X = 0;
      )";
  BindableMatcher<Decl> getPattern() { return varTemplateDecl(hasName("X")); }
  Language getLang() { return Lang_CXX14; }
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
  static constexpr auto *ConflictingDefinition =
      R"(
      template <class T> class X;
      template <> class X<int> { int A; };
      )";
  BindableMatcher<Decl> getPattern() {
    return classTemplateSpecializationDecl(hasName("X"), unless(isImplicit()));
  }
  Language getLang() { return Lang_CXX; }
};

// Function template specializations are all "full" specializations.
// Structural equivalency does not check the body of functions, so we cannot
// create conflicting function template specializations.
struct FunctionTemplateSpec {
  using DeclTy = FunctionDecl;

  static constexpr auto *Definition0 =
      R"(
      template <class T>
      void X(T a);
      template <> void X(int a) {};
      )";

  // This is actually not a conflicting definition, but another full
  // specialization.
  // Thus, during the import we would create a new specialization with a
  // different type argument.
  static constexpr auto *Definition1 =
      R"(
      template <class T>
      void X(T a);
      template <> void X(double a) {};
      )";

  BindableMatcher<Decl> getPattern() {
    return functionDecl(hasName("X"), isExplicitTemplateSpecialization(),
                        unless(isImplicit()));
  }
  static std::string getDef0() { return Definition0; }
  static std::string getDef1() { return Definition1; }
  Language getLang() { return Lang_CXX; }
};

static const internal::VariadicDynCastAllOfMatcher<
    Decl, VarTemplateSpecializationDecl>
    varTemplateSpecializationDecl;

struct VarTemplateSpec {
  using DeclTy = VarTemplateSpecializationDecl;
  static constexpr auto *Definition =
      R"(
      template <class T> T X = 0;
      template <> int X<int> = 0;
      )";
  static constexpr auto *ConflictingDefinition =
      R"(
      template <class T> T X = 0;
      template <> float X<int> = 1.0;
      )";
  BindableMatcher<Decl> getPattern() {
    return varTemplateSpecializationDecl(hasName("X"), unless(isImplicit()));
  }
  Language getLang() { return Lang_CXX14; }
};

template <typename TypeParam, ASTImporter::ODRHandlingType ODRHandlingParam>
struct ODRViolation : ASTImporterOptionSpecificTestBase {

  using DeclTy = typename TypeParam::DeclTy;

  ODRViolation() { ODRHandling = ODRHandlingParam; }

  static std::string getPrototype() { return TypeParam::Prototype; }
  static std::string getConflictingPrototype() {
    return TypeParam::ConflictingPrototype;
  }
  static std::string getDefinition() { return TypeParam::Definition; }
  static std::string getConflictingDefinition() {
    return TypeParam::ConflictingDefinition;
  }
  static std::string getConflictingProtoDef() {
    return TypeParam::ConflictingProtoDef;
  }
  static BindableMatcher<Decl> getPattern() { return TypeParam().getPattern(); }
  static Language getLang() { return TypeParam().getLang(); }

  template <std::string (*ToTUContent)(), std::string (*FromTUContent)(),
            void (*ResultChecker)(llvm::Expected<Decl *> &, Decl *, Decl *)>
  void TypedTest_ImportAfter() {
    Decl *ToTU = getToTuDecl(ToTUContent(), getLang());
    auto *ToD = FirstDeclMatcher<DeclTy>().match(ToTU, getPattern());

    Decl *FromTU = getTuDecl(FromTUContent(), getLang());
    auto *FromD = FirstDeclMatcher<DeclTy>().match(FromTU, getPattern());

    auto Result = importOrError(FromD, getLang());

    ResultChecker(Result, ToTU, ToD);
  }

  // Check that a Decl has been successfully imported into a standalone redecl
  // chain.
  static void CheckImportedAsNew(llvm::Expected<Decl *> &Result, Decl *ToTU,
                                 Decl *ToD) {
    ASSERT_TRUE(isSuccess(Result));
    Decl *ImportedD = *Result;
    ASSERT_TRUE(ImportedD);
    EXPECT_NE(ImportedD, ToD);
    EXPECT_EQ(DeclCounter<DeclTy>().match(ToTU, getPattern()), 2u);

    // There may be a hidden fwd spec decl before a function spec decl.
    if (auto *ImportedF = dyn_cast<FunctionDecl>(ImportedD))
      if (ImportedF->getTemplatedKind() ==
          FunctionDecl::TK_FunctionTemplateSpecialization)
        return;

    EXPECT_FALSE(ImportedD->getPreviousDecl());
  }

  // Check that a Decl was not imported because of NameConflict.
  static void CheckImportNameConflict(llvm::Expected<Decl *> &Result,
                                      Decl *ToTU, Decl *ToD) {
    EXPECT_TRUE(isImportError(Result, ImportError::NameConflict));
    EXPECT_EQ(DeclCounter<DeclTy>().match(ToTU, getPattern()), 1u);
  }

  // Check that a Decl was not imported because lookup found the same decl.
  static void CheckImportFoundExisting(llvm::Expected<Decl *> &Result,
                                      Decl *ToTU, Decl *ToD) {
    ASSERT_TRUE(isSuccess(Result));
    EXPECT_EQ(DeclCounter<DeclTy>().match(ToTU, getPattern()), 1u);
  }

  void TypedTest_ImportConflictingDefAfterDef() {
    TypedTest_ImportAfter<getDefinition, getConflictingDefinition,
                          CheckImportedAsNew>();
  }
  void TypedTest_ImportConflictingProtoAfterProto() {
    TypedTest_ImportAfter<getPrototype, getConflictingPrototype,
                          CheckImportedAsNew>();
  }
  void TypedTest_ImportConflictingProtoAfterDef() {
    TypedTest_ImportAfter<getDefinition, getConflictingPrototype,
                          CheckImportedAsNew>();
  }
  void TypedTest_ImportConflictingDefAfterProto() {
    TypedTest_ImportAfter<getConflictingPrototype, getDefinition,
                          CheckImportedAsNew>();
  }
  void TypedTest_ImportConflictingProtoDefAfterProto() {
    TypedTest_ImportAfter<getPrototype, getConflictingProtoDef,
                          CheckImportedAsNew>();
  }
  void TypedTest_ImportConflictingProtoAfterProtoDef() {
    TypedTest_ImportAfter<getConflictingProtoDef, getPrototype,
                          CheckImportedAsNew>();
  }
  void TypedTest_ImportConflictingProtoDefAfterDef() {
    TypedTest_ImportAfter<getDefinition, getConflictingProtoDef,
                          CheckImportedAsNew>();
  }
  void TypedTest_ImportConflictingDefAfterProtoDef() {
    TypedTest_ImportAfter<getConflictingProtoDef, getDefinition,
                          CheckImportedAsNew>();
  }

  void TypedTest_DontImportConflictingProtoAfterProto() {
    TypedTest_ImportAfter<getPrototype, getConflictingPrototype,
                          CheckImportNameConflict>();
  }
  void TypedTest_DontImportConflictingDefAfterDef() {
    TypedTest_ImportAfter<getDefinition, getConflictingDefinition,
                          CheckImportNameConflict>();
  }
  void TypedTest_DontImportConflictingProtoAfterDef() {
    TypedTest_ImportAfter<getDefinition, getConflictingPrototype,
                          CheckImportNameConflict>();
  }
  void TypedTest_DontImportConflictingDefAfterProto() {
    TypedTest_ImportAfter<getConflictingPrototype, getDefinition,
                          CheckImportNameConflict>();
  }
  void TypedTest_DontImportConflictingProtoDefAfterProto() {
    TypedTest_ImportAfter<getPrototype, getConflictingProtoDef,
                          CheckImportNameConflict>();
  }
  void TypedTest_DontImportConflictingProtoAfterProtoDef() {
    TypedTest_ImportAfter<getConflictingProtoDef, getPrototype,
                          CheckImportNameConflict>();
  }
  void TypedTest_DontImportConflictingProtoDefAfterDef() {
    TypedTest_ImportAfter<getDefinition, getConflictingProtoDef,
                          CheckImportNameConflict>();
  }
  void TypedTest_DontImportConflictingDefAfterProtoDef() {
    TypedTest_ImportAfter<getConflictingProtoDef, getDefinition,
                          CheckImportNameConflict>();
  }

  // Used for function templates and function template specializations.
  void TypedTest_ImportDifferentDefAfterDef() {
    TypedTest_ImportAfter<TypeParam::getDef0, TypeParam::getDef1,
                          CheckImportedAsNew>();
  }
  void TypedTest_DontImportSameDefAfterDef() {
    TypedTest_ImportAfter<TypeParam::getDef0, TypeParam::getDef0,
                          CheckImportFoundExisting>();
  }
};

// ==============================
// Define the parametrized tests.
// ==============================

#define ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(                           \
    TypeParam, ODRHandlingParam, NamePrefix, TestCase)                         \
  using TypeParam##ODRHandlingParam =                                          \
      ODRViolation<TypeParam, ASTImporter::ODRHandlingType::ODRHandlingParam>; \
  TEST_P(TypeParam##ODRHandlingParam, NamePrefix##TestCase) {                  \
    TypedTest_##TestCase();                                                    \
  }

// clang-format off

ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    Function, Liberal, ,
    ImportConflictingDefAfterDef)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    Typedef, Liberal, ,
    ImportConflictingDefAfterDef)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    TypedefAlias, Liberal, ,
    ImportConflictingDefAfterDef)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    Enum, Liberal, ,
    ImportConflictingDefAfterDef)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    EnumConstant, Liberal, ,
    ImportConflictingDefAfterDef)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    Class, Liberal, ,
    ImportConflictingDefAfterDef)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    Variable, Liberal, ,
    ImportConflictingDefAfterDef)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    ClassTemplate, Liberal, ,
    ImportConflictingDefAfterDef)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    VarTemplate, Liberal, ,
    ImportConflictingDefAfterDef)
// Class and variable template specializations/instantiatons are always
// imported conservatively, because the AST holds the specializations in a set,
// and the key within the set is a hash calculated from the arguments of the
// specialization.
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    ClassTemplateSpec, Liberal, ,
    DontImportConflictingDefAfterDef) // Don't import !!!
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    VarTemplateSpec, Liberal, ,
    DontImportConflictingDefAfterDef) // Don't import !!!

ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    Function, Conservative, ,
    DontImportConflictingDefAfterDef)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    Typedef, Conservative, ,
    DontImportConflictingDefAfterDef)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    TypedefAlias, Conservative, ,
    DontImportConflictingDefAfterDef)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    Enum, Conservative, ,
    DontImportConflictingDefAfterDef)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    EnumConstant, Conservative, ,
    DontImportConflictingDefAfterDef)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    Class, Conservative, ,
    DontImportConflictingDefAfterDef)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    Variable, Conservative, ,
    DontImportConflictingDefAfterDef)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    ClassTemplate, Conservative, ,
    DontImportConflictingDefAfterDef)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    VarTemplate, Conservative, ,
    DontImportConflictingDefAfterDef)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    ClassTemplateSpec, Conservative, ,
    DontImportConflictingDefAfterDef)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    VarTemplateSpec, Conservative, ,
    DontImportConflictingDefAfterDef)

ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    Function, Liberal, ,
    ImportConflictingProtoAfterProto)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    Variable, Liberal, ,
    ImportConflictingProtoAfterProto)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    ClassTemplate, Liberal, ,
    ImportConflictingProtoAfterProto)

ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    Function, Conservative, ,
    DontImportConflictingProtoAfterProto)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    Variable, Conservative, ,
    DontImportConflictingProtoAfterProto)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    ClassTemplate, Conservative, ,
    DontImportConflictingProtoAfterProto)

ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    Variable, Liberal, ,
    ImportConflictingProtoAfterDef)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    ClassTemplate, Liberal, ,
    ImportConflictingProtoAfterDef)

ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    Variable, Conservative, ,
    DontImportConflictingProtoAfterDef)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    ClassTemplate, Conservative, ,
    DontImportConflictingProtoAfterDef)

ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    Function, Liberal, ,
    ImportConflictingDefAfterProto)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    Variable, Liberal, ,
    ImportConflictingDefAfterProto)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    ClassTemplate, Liberal, ,
    ImportConflictingDefAfterProto)

ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    Function, Conservative, ,
    DontImportConflictingDefAfterProto)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    Variable, Conservative, ,
    DontImportConflictingDefAfterProto)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    ClassTemplate, Conservative, ,
    DontImportConflictingDefAfterProto)

ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    ClassTemplate, Liberal, ,
    ImportConflictingProtoDefAfterProto)

ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    ClassTemplate, Conservative, ,
    DontImportConflictingProtoDefAfterProto)

ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    ClassTemplate, Liberal, ,
    ImportConflictingProtoAfterProtoDef)

ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    ClassTemplate, Conservative, ,
    DontImportConflictingProtoAfterProtoDef)

ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    ClassTemplate, Liberal, ,
    ImportConflictingProtoDefAfterDef)

ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    ClassTemplate, Conservative, ,
    DontImportConflictingProtoDefAfterDef)

ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    ClassTemplate, Liberal, ,
    ImportConflictingDefAfterProtoDef)

ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    ClassTemplate, Conservative, ,
    DontImportConflictingDefAfterProtoDef)

// FunctionTemplate decls overload with each other. Thus, they are imported
// always as a new node, independently from any ODRHandling strategy.
//
// Function template specializations are "full" specializations. Structural
// equivalency does not check the body of functions, so we cannot create
// conflicting function template specializations. Thus, ODR handling strategies
// has nothing to do with function template specializations. Fully specialized
// function templates are imported as new nodes if their template arguments are
// different.
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    FunctionTemplate, Liberal, ,
    ImportDifferentDefAfterDef)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    FunctionTemplateSpec, Liberal, ,
    ImportDifferentDefAfterDef)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    FunctionTemplate, Conservative, ,
    ImportDifferentDefAfterDef)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    FunctionTemplateSpec, Conservative, ,
    ImportDifferentDefAfterDef)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    FunctionTemplate, Liberal, ,
    DontImportSameDefAfterDef)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    FunctionTemplateSpec, Liberal, ,
    DontImportSameDefAfterDef)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    FunctionTemplate, Conservative, ,
    DontImportSameDefAfterDef)
ASTIMPORTER_ODR_INSTANTIATE_TYPED_TEST_CASE(
    FunctionTemplateSpec, Conservative, ,
    DontImportSameDefAfterDef)

// ======================
// Instantiate the tests.
// ======================

// FIXME: These fail on Windows.
#if !defined(_WIN32)
INSTANTIATE_TEST_CASE_P(
    ODRViolationTests, FunctionConservative,
    DefaultTestValuesForRunOptions, );
#endif
INSTANTIATE_TEST_CASE_P(
    ODRViolationTests, TypedefConservative,
    DefaultTestValuesForRunOptions, );
INSTANTIATE_TEST_CASE_P(
    ODRViolationTests, TypedefAliasConservative,
    DefaultTestValuesForRunOptions, );
INSTANTIATE_TEST_CASE_P(
    ODRViolationTests, EnumConservative,
    DefaultTestValuesForRunOptions, );
INSTANTIATE_TEST_CASE_P(
    ODRViolationTests, EnumConstantConservative,
    DefaultTestValuesForRunOptions, );
INSTANTIATE_TEST_CASE_P(
    ODRViolationTests, ClassConservative,
    DefaultTestValuesForRunOptions, );
INSTANTIATE_TEST_CASE_P(
    ODRViolationTests, VariableConservative,
    DefaultTestValuesForRunOptions, );
INSTANTIATE_TEST_CASE_P(
    ODRViolationTests, ClassTemplateConservative,
    DefaultTestValuesForRunOptions, );
INSTANTIATE_TEST_CASE_P(
    ODRViolationTests, FunctionTemplateConservative,
    DefaultTestValuesForRunOptions, );
// FIXME: Make VarTemplate tests work.
//INSTANTIATE_TEST_CASE_P(
    //ODRViolationTests, VarTemplateConservative,
    //DefaultTestValuesForRunOptions, );
INSTANTIATE_TEST_CASE_P(
    ODRViolationTests, FunctionTemplateSpecConservative,
    DefaultTestValuesForRunOptions, );
INSTANTIATE_TEST_CASE_P(
    ODRViolationTests, ClassTemplateSpecConservative,
    DefaultTestValuesForRunOptions, );
// FIXME: Make VarTemplateSpec tests work.
//INSTANTIATE_TEST_CASE_P(
    //ODRViolationTests, VarTemplateSpecConservative,
    //DefaultTestValuesForRunOptions, );

// FIXME: These fail on Windows.
#if !defined(_WIN32)
INSTANTIATE_TEST_CASE_P(
    ODRViolationTests, FunctionLiberal,
    DefaultTestValuesForRunOptions, );
#endif
INSTANTIATE_TEST_CASE_P(
    ODRViolationTests, TypedefLiberal,
    DefaultTestValuesForRunOptions, );
INSTANTIATE_TEST_CASE_P(
    ODRViolationTests, TypedefAliasLiberal,
    DefaultTestValuesForRunOptions, );
INSTANTIATE_TEST_CASE_P(
    ODRViolationTests, EnumLiberal,
    DefaultTestValuesForRunOptions, );
INSTANTIATE_TEST_CASE_P(
    ODRViolationTests, EnumConstantLiberal,
    DefaultTestValuesForRunOptions, );
INSTANTIATE_TEST_CASE_P(
    ODRViolationTests, ClassLiberal,
    DefaultTestValuesForRunOptions, );
INSTANTIATE_TEST_CASE_P(
    ODRViolationTests, VariableLiberal,
    DefaultTestValuesForRunOptions, );
INSTANTIATE_TEST_CASE_P(
    ODRViolationTests, ClassTemplateLiberal,
    DefaultTestValuesForRunOptions, );
INSTANTIATE_TEST_CASE_P(
    ODRViolationTests, FunctionTemplateLiberal,
    DefaultTestValuesForRunOptions, );
// FIXME: Make VarTemplate tests work.
// INSTANTIATE_TEST_CASE_P(
//     ODRViolationTests, VarTemplateLiberal,
//     DefaultTestValuesForRunOptions, );
INSTANTIATE_TEST_CASE_P(
    ODRViolationTests, ClassTemplateSpecLiberal,
    DefaultTestValuesForRunOptions, );
INSTANTIATE_TEST_CASE_P(
    ODRViolationTests, FunctionTemplateSpecLiberal,
    DefaultTestValuesForRunOptions, );
// FIXME: Make VarTemplateSpec tests work.
//INSTANTIATE_TEST_CASE_P(
    //ODRViolationTests, VarTemplateSpecLiberal,
    //DefaultTestValuesForRunOptions, );

// clang-format on

} // end namespace ast_matchers
} // end namespace clang
