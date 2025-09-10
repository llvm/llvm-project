//===- unittest/Tooling/QualTypeNameTest.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/QualTypeNames.h"
#include "TestVisitor.h"
using namespace clang;

namespace {
struct TypeNameVisitor : TestVisitor {
  llvm::StringMap<std::string> ExpectedQualTypeNames;
  bool WithGlobalNsPrefix = false;

  // ValueDecls are the least-derived decl with both a qualtype and a name.
  bool VisitValueDecl(ValueDecl *VD) override {
    std::string ExpectedName =
        ExpectedQualTypeNames.lookup(VD->getNameAsString());
    if (ExpectedName != "") {
      PrintingPolicy Policy(Context->getPrintingPolicy());
      Policy.SuppressScope = false;
      Policy.AnonymousTagLocations = true;
      Policy.PolishForDeclaration = true;
      Policy.SuppressUnwrittenScope = true;
      std::string ActualName = TypeName::getFullyQualifiedName(
          VD->getType(), *Context, Policy, WithGlobalNsPrefix);
      if (ExpectedName != ActualName) {
        // A custom message makes it much easier to see what declaration
        // failed compared to EXPECT_EQ.
        ADD_FAILURE() << "Typename::getFullyQualifiedName failed for "
                      << VD->getQualifiedNameAsString() << std::endl
                      << "   Actual: " << ActualName << std::endl
                      << " Expected: " << ExpectedName;
      }
    }
    return true;
  }
};

// named namespaces inside anonymous namespaces

TEST(QualTypeNameTest, Simple) {
  TypeNameVisitor Visitor;
  // Simple case to test the test framework itself.
  Visitor.ExpectedQualTypeNames["CheckInt"] = "int";

  // Keeping the names of the variables whose types we check unique
  // within the entire test--regardless of their own scope--makes it
  // easier to diagnose test failures.

  // Simple namespace qualifier
  Visitor.ExpectedQualTypeNames["CheckA"] = "A::B::Class0";
  // Lookup up the enclosing scopes, then down another one. (These
  // appear as elaborated type in the AST. In that case--even if
  // policy.SuppressScope = 0--qual_type.getAsString(policy) only
  // gives the name as it appears in the source, not the full name.
  Visitor.ExpectedQualTypeNames["CheckB"] = "A::B::C::Class1";
  // Template parameter expansion.
  Visitor.ExpectedQualTypeNames["CheckC"] =
      "A::B::Template0<A::B::C::MyInt, A::B::AnotherClass>";
  // Recursive template parameter expansion.
  Visitor.ExpectedQualTypeNames["CheckD"] =
      "A::B::Template0<A::B::Template1<A::B::C::MyInt, A::B::AnotherClass>, "
      "A::B::Template0<int, long>>";
  // Variadic Template expansion.
  Visitor.ExpectedQualTypeNames["CheckE"] =
      "A::Variadic<int, A::B::Template0<int, char>, "
      "A::B::Template1<int, long>, A::B::C::MyInt>";
  // Using declarations should be fully expanded.
  Visitor.ExpectedQualTypeNames["CheckF"] = "A::B::Class0";
  // Elements found within "using namespace foo;" should be fully
  // expanded.
  Visitor.ExpectedQualTypeNames["CheckG"] = "A::B::C::MyInt";
  // Type inside function
  Visitor.ExpectedQualTypeNames["CheckH"] = "struct X";
  // Anonymous Namespaces
  Visitor.ExpectedQualTypeNames["CheckI"] = "aClass";
  // Keyword inclusion with namespaces
  Visitor.ExpectedQualTypeNames["CheckJ"] = "struct A::aStruct";
  // Anonymous Namespaces nested in named namespaces and vice-versa.
  Visitor.ExpectedQualTypeNames["CheckK"] = "D::aStruct";
  // Namespace alias
  Visitor.ExpectedQualTypeNames["CheckL"] = "A::B::C::MyInt";
  Visitor.ExpectedQualTypeNames["non_dependent_type_var"] =
      "Foo<X>::non_dependent_type";
  Visitor.ExpectedQualTypeNames["AnEnumVar"] = "EnumScopeClass::AnEnum";
  Visitor.ExpectedQualTypeNames["AliasTypeVal"] = "A::B::C::InnerAlias<int>";
  Visitor.ExpectedQualTypeNames["AliasInnerTypeVal"] =
      "OuterTemplateClass<A::B::Class0>::Inner";
  Visitor.ExpectedQualTypeNames["CheckM"] = "const A::B::Class0 *";
  Visitor.ExpectedQualTypeNames["CheckN"] = "const X *";
  Visitor.ExpectedQualTypeNames["ttp_using"] =
      "OuterTemplateClass<A::B::Class0>";
  Visitor.ExpectedQualTypeNames["alias_of_template"] = "ABTemplate0IntInt";
  Visitor.runOver(
      "int CheckInt;\n"
      "template <typename T>\n"
      "class OuterTemplateClass { public: struct Inner {}; };\n"
      "namespace A {\n"
      " namespace B {\n"
      "   class Class0 { };\n"
      "   namespace C {\n"
      "     typedef int MyInt;"
      "     template <typename T>\n"
      "     using InnerAlias = OuterTemplateClass<T>;\n"
      "     InnerAlias<int> AliasTypeVal;\n"
      "     InnerAlias<Class0>::Inner AliasInnerTypeVal;\n"
      "   }\n"
      "   template<class X, class Y> class Template0;"
      "   template<class X, class Y> class Template1;"
      "   typedef B::Class0 AnotherClass;\n"
      "   void Function1(Template0<C::MyInt,\n"
      "                  AnotherClass> CheckC);\n"
      "   void Function2(Template0<Template1<C::MyInt, AnotherClass>,\n"
      "                            Template0<int, long> > CheckD);\n"
      "   void Function3(const B::Class0* CheckM);\n"
      "  }\n"
      "template<typename... Values> class Variadic {};\n"
      "Variadic<int, B::Template0<int, char>, "
      "         B::Template1<int, long>, "
      "         B::C::MyInt > CheckE;\n"
      " namespace BC = B::C;\n"
      " BC::MyInt CheckL;\n"
      "}\n"
      "using A::B::Class0;\n"
      "void Function(Class0 CheckF);\n"
      "OuterTemplateClass<Class0> ttp_using;\n"
      "using ABTemplate0IntInt = A::B::Template0<int, int>;\n"
      "void Function(ABTemplate0IntInt alias_of_template);\n"
      "using namespace A::B::C;\n"
      "void Function(MyInt CheckG);\n"
      "void f() {\n"
      "  struct X {} CheckH;\n"
      "}\n"
      "struct X;\n"
      "void f(const ::X* CheckN) {}\n"
      "namespace {\n"
      "  class aClass {};\n"
      "   aClass CheckI;\n"
      "}\n"
      "namespace A {\n"
      "  struct aStruct {} CheckJ;\n"
      "}\n"
      "namespace {\n"
      "  namespace D {\n"
      "    namespace {\n"
      "      class aStruct {};\n"
      "      aStruct CheckK;\n"
      "    }\n"
      "  }\n"
      "}\n"
      "template<class T> struct Foo {\n"
      "  typedef typename T::A dependent_type;\n"
      "  typedef int non_dependent_type;\n"
      "  dependent_type dependent_type_var;\n"
      "  non_dependent_type non_dependent_type_var;\n"
      "};\n"
      "struct X { typedef int A; };"
      "Foo<X> var;"
      "void F() {\n"
      "  var.dependent_type_var = 0;\n"
      "var.non_dependent_type_var = 0;\n"
      "}\n"
      "class EnumScopeClass {\n"
      "public:\n"
      "  enum AnEnum { ZERO, ONE };\n"
      "};\n"
      "EnumScopeClass::AnEnum AnEnumVar;\n",
      TypeNameVisitor::Lang_CXX11);
}

TEST(QualTypeNameTest, Complex) {
  TypeNameVisitor Complex;
  Complex.ExpectedQualTypeNames["CheckTX"] = "B::TX";
  Complex.runOver(
      "namespace A {"
      "  struct X {};"
      "}"
      "using A::X;"
      "namespace fake_std {"
      "  template<class... Types > class tuple {};"
      "}"
      "namespace B {"
      "  using fake_std::tuple;"
      "  typedef tuple<X> TX;"
      "  TX CheckTX;"
      "  struct A { typedef int X; };"
      "}");
}

TEST(QualTypeNameTest, DoubleUsing) {
  TypeNameVisitor DoubleUsing;
  DoubleUsing.ExpectedQualTypeNames["direct"] = "a::A<0>";
  DoubleUsing.ExpectedQualTypeNames["indirect"] = "b::B";
  DoubleUsing.ExpectedQualTypeNames["double_indirect"] = "b::B";
  DoubleUsing.runOver(R"cpp(
    namespace a {
      template<int> class A {};
      A<0> direct;
    }
    namespace b {
      using B = ::a::A<0>;
      B indirect;
    }
    namespace b {
      using ::b::B;
      B double_indirect;
    }
  )cpp");
}

TEST(QualTypeNameTest, GlobalNsPrefix) {
  TypeNameVisitor GlobalNsPrefix;
  GlobalNsPrefix.WithGlobalNsPrefix = true;
  GlobalNsPrefix.ExpectedQualTypeNames["IntVal"] = "int";
  GlobalNsPrefix.ExpectedQualTypeNames["BoolVal"] = "bool";
  GlobalNsPrefix.ExpectedQualTypeNames["XVal"] = "::A::B::X";
  GlobalNsPrefix.ExpectedQualTypeNames["IntAliasVal"] = "::A::B::Alias<int>";
  GlobalNsPrefix.ExpectedQualTypeNames["ZVal"] = "::A::B::Y::Z";
  GlobalNsPrefix.ExpectedQualTypeNames["GlobalZVal"] = "::Z";
  GlobalNsPrefix.ExpectedQualTypeNames["CheckK"] = "D::aStruct";
  GlobalNsPrefix.ExpectedQualTypeNames["YZMPtr"] = "::A::B::X ::A::B::Y::Z::*";
  GlobalNsPrefix.runOver(
      "namespace A {\n"
      "  namespace B {\n"
      "    int IntVal;\n"
      "    bool BoolVal;\n"
      "    struct X {};\n"
      "    X XVal;\n"
      "    template <typename T> class CCC { };\n"
      "    template <typename T>\n"
      "    using Alias = CCC<T>;\n"
      "    Alias<int> IntAliasVal;\n"
      "    struct Y { struct Z { X YZIPtr; }; };\n"
      "    Y::Z ZVal;\n"
      "    X Y::Z::*YZMPtr;\n"
      "  }\n"
      "}\n"
      "struct Z {};\n"
      "Z GlobalZVal;\n"
      "namespace {\n"
      "  namespace D {\n"
      "    namespace {\n"
      "      class aStruct {};\n"
      "      aStruct CheckK;\n"
      "    }\n"
      "  }\n"
      "}\n"
  );
}

TEST(QualTypeNameTest, InlineNamespace) {
  TypeNameVisitor InlineNamespace;
  InlineNamespace.ExpectedQualTypeNames["c"] = "B::C";
  InlineNamespace.runOver("inline namespace A {\n"
                          "  namespace B {\n"
                          "    class C {};\n"
                          "  }\n"
                          "}\n"
                          "using namespace A::B;\n"
                          "C c;\n",
                          TypeNameVisitor::Lang_CXX11);
}

TEST(QualTypeNameTest, TemplatedClass) {
  std::unique_ptr<ASTUnit> AST =
      tooling::buildASTFromCode("template <unsigned U1> struct A {\n"
                                "  template <unsigned U2> struct B {};\n"
                                "};\n"
                                "template struct A<1>;\n"
                                "template struct A<2u>;\n"
                                "template struct A<1>::B<3>;\n"
                                "template struct A<2u>::B<4u>;\n");

  auto &Context = AST->getASTContext();
  auto &Policy = Context.getPrintingPolicy();
  auto getFullyQualifiedName = [&](QualType QT) {
    return TypeName::getFullyQualifiedName(QT, Context, Policy);
  };

  auto *A = Context.getTranslationUnitDecl()
                ->lookup(&Context.Idents.get("A"))
                .find_first<ClassTemplateDecl>();
  ASSERT_NE(A, nullptr);

  // A has two explicit instantiations: A<1> and A<2u>
  auto ASpec = A->spec_begin();
  ASSERT_NE(ASpec, A->spec_end());
  auto *A1 = *ASpec;
  ASpec++;
  ASSERT_NE(ASpec, A->spec_end());
  auto *A2 = *ASpec;

  // Their type names follow the records.
  CanQualType A1RecordTy = Context.getCanonicalTagType(A1);
  EXPECT_EQ(getFullyQualifiedName(A1RecordTy), "A<1>");
  CanQualType A2RecordTy = Context.getCanonicalTagType(A2);
  EXPECT_EQ(getFullyQualifiedName(A2RecordTy), "A<2U>");

  // getTemplateSpecializationType() gives types that print the integral
  // argument directly.
  TemplateArgument Args1[] = {
      {Context, llvm::APSInt::getUnsigned(1u), Context.UnsignedIntTy}};
  QualType A1TemplateSpecTy = Context.getTemplateSpecializationType(
      ElaboratedTypeKeyword::None, TemplateName(A), Args1, Args1, A1RecordTy);
  EXPECT_EQ(A1TemplateSpecTy.getAsString(), "A<1>");

  TemplateArgument Args2[] = {
      {Context, llvm::APSInt::getUnsigned(2u), Context.UnsignedIntTy}};
  QualType A2TemplateSpecTy = Context.getTemplateSpecializationType(
      ElaboratedTypeKeyword::None, TemplateName(A), Args2, Args2, A2RecordTy);
  EXPECT_EQ(A2TemplateSpecTy.getAsString(), "A<2>");

  // Find A<1>::B and its specialization B<3>.
  auto *A1B =
      A1->lookup(&Context.Idents.get("B")).find_first<ClassTemplateDecl>();
  ASSERT_NE(A1B, nullptr);
  auto A1BSpec = A1B->spec_begin();
  ASSERT_NE(A1BSpec, A1B->spec_end());
  auto *A1B3 = *A1BSpec;
  CanQualType A1B3RecordTy = Context.getCanonicalTagType(A1B3);
  EXPECT_EQ(getFullyQualifiedName(A1B3RecordTy), "A<1>::B<3>");

  // Construct A<1>::B<3> and check name.
  NestedNameSpecifier A1Nested(A1TemplateSpecTy.getTypePtr());
  TemplateName A1B3Name = Context.getQualifiedTemplateName(
      A1Nested, /*TemplateKeyword=*/false, TemplateName(A1B));

  TemplateArgument Args3[] = {
      {Context, llvm::APSInt::getUnsigned(3u), Context.UnsignedIntTy}};
  QualType A1B3TemplateSpecTy = Context.getTemplateSpecializationType(
      ElaboratedTypeKeyword::None, A1B3Name, Args3, Args3, A1B3RecordTy);
  EXPECT_EQ(A1B3TemplateSpecTy.getAsString(), "A<1>::B<3>");

  // Find A<2u>::B and its specialization B<4u>.
  auto *A2B =
      A2->lookup(&Context.Idents.get("B")).find_first<ClassTemplateDecl>();
  ASSERT_NE(A2B, nullptr);
  auto A2BSpec = A2B->spec_begin();
  ASSERT_NE(A2BSpec, A2B->spec_end());
  auto *A2B4 = *A2BSpec;
  CanQualType A2B4RecordTy = Context.getCanonicalTagType(A2B4);
  EXPECT_EQ(getFullyQualifiedName(A2B4RecordTy), "A<2U>::B<4U>");

  // Construct A<2>::B<4> and check name.
  NestedNameSpecifier A2Nested(A2TemplateSpecTy.getTypePtr());
  TemplateName A2B4Name = Context.getQualifiedTemplateName(
      A2Nested, /*TemplateKeyword=*/false, TemplateName(A2B));

  TemplateArgument Args4[] = {
      {Context, llvm::APSInt::getUnsigned(4u), Context.UnsignedIntTy}};
  QualType A2B4TemplateSpecTy = Context.getTemplateSpecializationType(
      ElaboratedTypeKeyword::None, A2B4Name, Args4, Args4, A2B4RecordTy);
  EXPECT_EQ(A2B4TemplateSpecTy.getAsString(), "A<2>::B<4>");
}

TEST(QualTypeNameTest, AnonStrucs) {
  TypeNameVisitor AnonStrucs;
  AnonStrucs.ExpectedQualTypeNames["a"] = "short";
  AnonStrucs.ExpectedQualTypeNames["un_in_st_1"] =
      "union (unnamed struct at input.cc:1:1)::(unnamed union at "
      "input.cc:2:27)";
  AnonStrucs.ExpectedQualTypeNames["b"] = "short";
  AnonStrucs.ExpectedQualTypeNames["un_in_st_2"] =
      "union (unnamed struct at input.cc:1:1)::(unnamed union at "
      "input.cc:5:27)";
  AnonStrucs.ExpectedQualTypeNames["anon_st"] =
      "struct (unnamed struct at input.cc:1:1)";
  AnonStrucs.runOver(R"(struct {
                          union {
                            short a;
                          } un_in_st_1;
                          union {
                            short b;
                          } un_in_st_2;
                        } anon_st;)");
}

TEST(QualTypeNameTest, ConstUsing) {
  TypeNameVisitor ConstUsing;
  ConstUsing.ExpectedQualTypeNames["param1"] = "const A::S &";
  ConstUsing.ExpectedQualTypeNames["param2"] = "const A::S";
  ConstUsing.runOver(R"(namespace A {
                          class S {};
                        }
                        using ::A::S;
                        void foo(const S& param1, const S param2);)");
}

TEST(QualTypeNameTest, NullableAttributesWithGlobalNs) {
  TypeNameVisitor Visitor;
  Visitor.WithGlobalNsPrefix = true;
  Visitor.ExpectedQualTypeNames["param1"] = "::std::unique_ptr<int> _Nullable";
  Visitor.ExpectedQualTypeNames["param2"] = "::std::unique_ptr<int> _Nonnull";
  Visitor.ExpectedQualTypeNames["param3"] =
      "::std::unique_ptr< ::std::unique_ptr<int> _Nullable> _Nonnull";
  Visitor.ExpectedQualTypeNames["param4"] =
      "::std::unique_ptr<int>  _Nullable const *";
  Visitor.ExpectedQualTypeNames["param5"] =
      "::std::unique_ptr<int>  _Nullable const *";
  Visitor.ExpectedQualTypeNames["param6"] =
      "::std::unique_ptr<int>  _Nullable const *";
  Visitor.runOver(R"(namespace std {
                        template<class T> class unique_ptr {};
                     }
                     void foo(
                      std::unique_ptr<int> _Nullable param1,
                      _Nonnull std::unique_ptr<int> param2,
                      std::unique_ptr<std::unique_ptr<int> _Nullable> _Nonnull param3,
                      const std::unique_ptr<int> _Nullable *param4,
                      _Nullable std::unique_ptr<int> const *param5,
                      std::unique_ptr<int> _Nullable const *param6
                      );
                     )");
}
}  // end anonymous namespace
