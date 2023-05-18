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
struct TypeNameVisitor : TestVisitor<TypeNameVisitor> {
  llvm::StringMap<std::string> ExpectedQualTypeNames;
  bool WithGlobalNsPrefix = false;

  // ValueDecls are the least-derived decl with both a qualtype and a name.
  bool VisitValueDecl(const ValueDecl *VD) {
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
}  // end anonymous namespace
