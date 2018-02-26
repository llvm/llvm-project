//===- unittest/Tooling/QualTypeNameTest.cpp ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/QualTypeNames.h"
#include "TestVisitor.h"
using namespace clang;

namespace {
struct TypeNameVisitor : TestVisitor<TypeNameVisitor> {
  llvm::StringMap<std::string> ExpectedQualTypeNames;
  bool WithGlobalNsPrefix = false;

  // ValueDecls are the least-derived decl with both a qualtype and a
  // name.
  bool traverseDecl(Decl *D) {
    return true;  // Always continue
  }

  bool VisitValueDecl(const ValueDecl *VD) {
    std::string ExpectedName =
        ExpectedQualTypeNames.lookup(VD->getNameAsString());
    if (ExpectedName != "") {
      std::string ActualName =
          TypeName::getFullyQualifiedName(VD->getType(), *Context,
                                          WithGlobalNsPrefix);
      if (ExpectedName != ActualName) {
        // A custom message makes it much easier to see what declaration
        // failed compared to EXPECT_EQ.
        EXPECT_TRUE(false) << "Typename::getFullyQualifiedName failed for "
                           << VD->getQualifiedNameAsString() << std::endl
                           << "   Actual: " << ActualName << std::endl
                           << " Exepcted: " << ExpectedName;
      }
    }
    return true;
  }
};

// named namespaces inside anonymous namespaces

TEST(QualTypeNameTest, getFullyQualifiedName) {
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
      "A::B::Template0<int, long> >";
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
  Visitor.ExpectedQualTypeNames["CheckM"] = "const A::B::Class0 *";
  Visitor.ExpectedQualTypeNames["CheckN"] = "const X *";
  Visitor.runOver(
      "int CheckInt;\n"
      "template <typename T>\n"
      "class OuterTemplateClass { };\n"
      "namespace A {\n"
      " namespace B {\n"
      "   class Class0 { };\n"
      "   namespace C {\n"
      "     typedef int MyInt;"
      "     template <typename T>\n"
      "     using InnerAlias = OuterTemplateClass<T>;\n"
      "     InnerAlias<int> AliasTypeVal;\n"
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
      TypeNameVisitor::Lang_CXX11
);

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

  TypeNameVisitor GlobalNsPrefix;
  GlobalNsPrefix.WithGlobalNsPrefix = true;
  GlobalNsPrefix.ExpectedQualTypeNames["IntVal"] = "int";
  GlobalNsPrefix.ExpectedQualTypeNames["BoolVal"] = "bool";
  GlobalNsPrefix.ExpectedQualTypeNames["XVal"] = "::A::B::X";
  GlobalNsPrefix.ExpectedQualTypeNames["IntAliasVal"] = "::A::B::Alias<int>";
  GlobalNsPrefix.ExpectedQualTypeNames["ZVal"] = "::A::B::Y::Z";
  GlobalNsPrefix.ExpectedQualTypeNames["GlobalZVal"] = "::Z";
  GlobalNsPrefix.ExpectedQualTypeNames["CheckK"] = "D::aStruct";
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
      "    struct Y { struct Z {}; };\n"
      "    Y::Z ZVal;\n"
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

}  // end anonymous namespace
