//===-- FindTargetTests.cpp --------------------------*- C++ -*------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "FindTarget.h"

#include "Selection.h"
#include "TestTU.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Annotations.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <initializer_list>

namespace clang {
namespace clangd {
namespace {

// A referenced Decl together with its DeclRelationSet, for assertions.
//
// There's no great way to assert on the "content" of a Decl in the general case
// that's both expressive and unambiguous (e.g. clearly distinguishes between
// templated decls and their specializations).
//
// We use the result of pretty-printing the decl, with the {body} truncated.
struct PrintedDecl {
  PrintedDecl(const char *Name, DeclRelationSet Relations = {})
      : Name(Name), Relations(Relations) {}
  PrintedDecl(const NamedDecl *D, DeclRelationSet Relations = {})
      : Relations(Relations) {
    std::string S;
    llvm::raw_string_ostream OS(S);
    D->print(OS);
    llvm::StringRef FirstLine =
        llvm::StringRef(OS.str()).take_until([](char C) { return C == '\n'; });
    FirstLine = FirstLine.rtrim(" {");
    Name = std::string(FirstLine.rtrim(" {"));
  }

  std::string Name;
  DeclRelationSet Relations;
};
bool operator==(const PrintedDecl &L, const PrintedDecl &R) {
  return std::tie(L.Name, L.Relations) == std::tie(R.Name, R.Relations);
}
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const PrintedDecl &D) {
  return OS << D.Name << " Rel=" << D.Relations;
}

// The test cases in for targetDecl() take the form
//  - a piece of code (Code = "...")
//  - Code should have a single AST node marked as a [[range]]
//  - an EXPECT_DECLS() assertion that verify the type of node selected, and
//    all the decls that targetDecl() considers it to reference
// Despite the name, these cases actually test allTargetDecls() for brevity.
class TargetDeclTest : public ::testing::Test {
protected:
  using Rel = DeclRelation;
  std::string Code;
  std::vector<std::string> Flags;

  // Asserts that `Code` has a marked selection of a node `NodeType`,
  // and returns allTargetDecls() as PrintedDecl structs.
  // Use via EXPECT_DECLS().
  std::vector<PrintedDecl> assertNodeAndPrintDecls(const char *NodeType) {
    llvm::Annotations A(Code);
    auto TU = TestTU::withCode(A.code());
    TU.ExtraArgs = Flags;
    auto AST = TU.build();
    llvm::Annotations::Range R = A.range();
    auto Selection = SelectionTree::createRight(
        AST.getASTContext(), AST.getTokens(), R.Begin, R.End);
    const SelectionTree::Node *N = Selection.commonAncestor();
    if (!N) {
      ADD_FAILURE() << "No node selected!\n" << Code;
      return {};
    }
    EXPECT_EQ(N->kind(), NodeType) << Selection;

    std::vector<PrintedDecl> ActualDecls;
    for (const auto &Entry : allTargetDecls(N->ASTNode))
      ActualDecls.emplace_back(Entry.first, Entry.second);
    return ActualDecls;
  }
};

// This is a macro to preserve line numbers in assertion failures.
// It takes the expected decls as varargs to work around comma-in-macro issues.
#define EXPECT_DECLS(NodeType, ...)                                            \
  EXPECT_THAT(assertNodeAndPrintDecls(NodeType),                               \
              ::testing::UnorderedElementsAreArray(                            \
                  std::vector<PrintedDecl>({__VA_ARGS__})))                    \
      << Code
using ExpectedDecls = std::vector<PrintedDecl>;

TEST_F(TargetDeclTest, Exprs) {
  Code = R"cpp(
    int f();
    int x = [[f]]();
  )cpp";
  EXPECT_DECLS("DeclRefExpr", "int f()");

  Code = R"cpp(
    struct S { S operator+(S) const; };
    auto X = S() [[+]] S();
  )cpp";
  EXPECT_DECLS("DeclRefExpr", "S operator+(S) const");

  Code = R"cpp(
    int foo();
    int s = foo[[()]];
  )cpp";
  EXPECT_DECLS("CallExpr", "int foo()");

  Code = R"cpp(
    struct X {
    void operator()(int n);
    };
    void test() {
      X x;
      x[[(123)]];
    }
  )cpp";
  EXPECT_DECLS("CXXOperatorCallExpr", "void operator()(int n)");

  Code = R"cpp(
    void test() {
      goto [[label]];
    label:
      return;
    }
  )cpp";
  EXPECT_DECLS("GotoStmt", "label:");
  Code = R"cpp(
    void test() {
    [[label]]:
      return;
    }
  )cpp";
  EXPECT_DECLS("LabelStmt", "label:");
}

TEST_F(TargetDeclTest, Recovery) {
  Code = R"cpp(
    // error-ok: testing behavior on broken code
    int f();
    int f(int, int);
    int x = [[f]](42);
  )cpp";
  EXPECT_DECLS("UnresolvedLookupExpr", "int f()", "int f(int, int)");
}

TEST_F(TargetDeclTest, RecoveryType) {
  Code = R"cpp(
    // error-ok: testing behavior on broken code
    struct S { int member; };
    S overloaded(int);
    void foo() {
      // No overload matches, but we have recovery-expr with the correct type.
      overloaded().[[member]];
    }
  )cpp";
  EXPECT_DECLS("MemberExpr", "int member");
}

TEST_F(TargetDeclTest, UsingDecl) {
  Code = R"cpp(
    namespace foo {
      int f(int);
      int f(char);
    }
    using foo::f;
    int x = [[f]](42);
  )cpp";
  // f(char) is not referenced!
  EXPECT_DECLS("DeclRefExpr", {"using foo::f", Rel::Alias},
               {"int f(int)", Rel::Underlying});

  Code = R"cpp(
    namespace foo {
      int f(int);
      int f(char);
    }
    [[using foo::f]];
  )cpp";
  // All overloads are referenced.
  EXPECT_DECLS("UsingDecl", {"using foo::f", Rel::Alias},
               {"int f(int)", Rel::Underlying},
               {"int f(char)", Rel::Underlying});

  Code = R"cpp(
    struct X {
      int foo();
    };
    struct Y : X {
      using X::foo;
    };
    int x = Y().[[foo]]();
  )cpp";
  EXPECT_DECLS("MemberExpr", {"using X::foo", Rel::Alias},
               {"int foo()", Rel::Underlying});

  Code = R"cpp(
      template <typename T>
      struct Base {
        void waldo() {}
      };
      template <typename T>
      struct Derived : Base<T> {
        using Base<T>::[[waldo]];
      };
    )cpp";
  EXPECT_DECLS("UnresolvedUsingValueDecl", {"using Base<T>::waldo", Rel::Alias},
               {"void waldo()", Rel::Underlying});
}

TEST_F(TargetDeclTest, ConstructorInitList) {
  Code = R"cpp(
    struct X {
      int a;
      X() : [[a]](42) {}
    };
  )cpp";
  EXPECT_DECLS("CXXCtorInitializer", "int a");

  Code = R"cpp(
    struct X {
      X() : [[X]](1) {}
      X(int);
    };
  )cpp";
  EXPECT_DECLS("RecordTypeLoc", "struct X");
}

TEST_F(TargetDeclTest, DesignatedInit) {
  Flags = {"-xc"}; // array designators are a C99 extension.
  Code = R"c(
    struct X { int a; };
    struct Y { int b; struct X c[2]; };
    struct Y y = { .c[0].[[a]] = 1 };
  )c";
  EXPECT_DECLS("DesignatedInitExpr", "int a");
}

TEST_F(TargetDeclTest, NestedNameSpecifier) {
  Code = R"cpp(
    namespace a { namespace b { int c; } }
    int x = a::[[b::]]c;
  )cpp";
  EXPECT_DECLS("NestedNameSpecifierLoc", "namespace b");

  Code = R"cpp(
    namespace a { struct X { enum { y }; }; }
    int x = a::[[X::]]y;
  )cpp";
  EXPECT_DECLS("NestedNameSpecifierLoc", "struct X");

  Code = R"cpp(
    template <typename T>
    int x = [[T::]]y;
  )cpp";
  EXPECT_DECLS("NestedNameSpecifierLoc", "typename T");

  Code = R"cpp(
    namespace a { int x; }
    namespace b = a;
    int y = [[b]]::x;
  )cpp";
  EXPECT_DECLS("NestedNameSpecifierLoc", {"namespace b = a", Rel::Alias},
               {"namespace a", Rel::Underlying});
}

TEST_F(TargetDeclTest, Types) {
  Code = R"cpp(
    struct X{};
    [[X]] x;
  )cpp";
  EXPECT_DECLS("RecordTypeLoc", "struct X");

  Code = R"cpp(
    struct S{};
    typedef S X;
    [[X]] x;
  )cpp";
  EXPECT_DECLS("TypedefTypeLoc", {"typedef S X", Rel::Alias},
               {"struct S", Rel::Underlying});
  Code = R"cpp(
    namespace ns { struct S{}; }
    typedef ns::S X;
    [[X]] x;
  )cpp";
  EXPECT_DECLS("TypedefTypeLoc", {"typedef ns::S X", Rel::Alias},
               {"struct S", Rel::Underlying});

  // FIXME: Auto-completion in a template requires disabling delayed template
  // parsing.
  Flags = {"-fno-delayed-template-parsing"};
  Code = R"cpp(
    template<class T>
    void foo() { [[T]] x; }
  )cpp";
  EXPECT_DECLS("TemplateTypeParmTypeLoc", "class T");
  Flags.clear();

  // FIXME: Auto-completion in a template requires disabling delayed template
  // parsing.
  Flags = {"-fno-delayed-template-parsing"};
  Code = R"cpp(
    template<template<typename> class T>
    void foo() { [[T<int>]] x; }
  )cpp";
  EXPECT_DECLS("TemplateSpecializationTypeLoc", "template <typename> class T");
  Flags.clear();

  Code = R"cpp(
    struct S{};
    S X;
    [[decltype]](X) Y;
  )cpp";
  EXPECT_DECLS("DecltypeTypeLoc", {"struct S", Rel::Underlying});

  Code = R"cpp(
    struct S{};
    [[auto]] X = S{};
  )cpp";
  // FIXME: deduced type missing in AST. https://llvm.org/PR42914
  EXPECT_DECLS("AutoTypeLoc");

  Code = R"cpp(
    template <typename... E>
    struct S {
      static const int size = sizeof...([[E]]);
    };
  )cpp";
  EXPECT_DECLS("SizeOfPackExpr", "typename ...E");

  Code = R"cpp(
    template <typename T>
    class Foo {
      void f([[Foo]] x);
    };
  )cpp";
  EXPECT_DECLS("InjectedClassNameTypeLoc", "class Foo");
}

TEST_F(TargetDeclTest, ClassTemplate) {
  Code = R"cpp(
    // Implicit specialization.
    template<int x> class Foo{};
    [[Foo<42>]] B;
  )cpp";
  EXPECT_DECLS("TemplateSpecializationTypeLoc",
               {"template<> class Foo<42>", Rel::TemplateInstantiation},
               {"class Foo", Rel::TemplatePattern});

  Code = R"cpp(
    template<typename T> class Foo {};
    // The "Foo<int>" SpecializationDecl is incomplete, there is no
    // instantiation happening.
    void func([[Foo<int>]] *);
  )cpp";
  EXPECT_DECLS("TemplateSpecializationTypeLoc",
               {"class Foo", Rel::TemplatePattern},
               {"template<> class Foo<int>", Rel::TemplateInstantiation});

  Code = R"cpp(
    // Explicit specialization.
    template<int x> class Foo{};
    template<> class Foo<42>{};
    [[Foo<42>]] B;
  )cpp";
  EXPECT_DECLS("TemplateSpecializationTypeLoc", "template<> class Foo<42>");

  Code = R"cpp(
    // Partial specialization.
    template<typename T> class Foo{};
    template<typename T> class Foo<T*>{};
    [[Foo<int*>]] B;
  )cpp";
  EXPECT_DECLS("TemplateSpecializationTypeLoc",
               {"template<> class Foo<int *>", Rel::TemplateInstantiation},
               {"template <typename T> class Foo<T *>", Rel::TemplatePattern});

  Code = R"cpp(
    // Template template argument.
    template<typename T> struct Vector {};
    template <template <typename> class Container>
    struct A {};
    A<[[Vector]]> a;
  )cpp";
  EXPECT_DECLS("TemplateArgumentLoc", {"template <typename T> struct Vector"});

  Flags.push_back("-std=c++17"); // for CTAD tests

  Code = R"cpp(
    // Class template argument deduction
    template <typename T>
    struct Test {
      Test(T);
    };
    void foo() {
      [[Test]] a(5);
    }
  )cpp";
  EXPECT_DECLS("DeducedTemplateSpecializationTypeLoc",
               {"struct Test", Rel::TemplatePattern});

  Code = R"cpp(
    // Deduction guide
    template <typename T>
    struct Test {
      template <typename I>
      Test(I, I);
    };
    template <typename I>
    [[Test]](I, I) -> Test<typename I::type>;
  )cpp";
  EXPECT_DECLS("CXXDeductionGuideDecl", {"template <typename T> struct Test"});
}

TEST_F(TargetDeclTest, Concept) {
  Flags.push_back("-std=c++20");

  // FIXME: Should we truncate the pretty-printed form of a concept decl
  // somewhere?

  Code = R"cpp(
    template <typename T>
    concept Fooable = requires (T t) { t.foo(); };

    template <typename T> requires [[Fooable]]<T>
    void bar(T t) {
      t.foo();
    }
  )cpp";
  EXPECT_DECLS(
      "ConceptSpecializationExpr",
      {"template <typename T> concept Fooable = requires (T t) { t.foo(); };"});

  // trailing requires clause
  Code = R"cpp(
      template <typename T>
      concept Fooable = true;

      template <typename T>
      void foo() requires [[Fooable]]<T>;
  )cpp";
  EXPECT_DECLS("ConceptSpecializationExpr",
               {"template <typename T> concept Fooable = true;"});

  // constrained-parameter
  Code = R"cpp(
    template <typename T>
    concept Fooable = true;

    template <[[Fooable]] T>
    void bar(T t);
  )cpp";
  EXPECT_DECLS("ConceptSpecializationExpr",
               {"template <typename T> concept Fooable = true;"});

  // partial-concept-id
  Code = R"cpp(
    template <typename T, typename U>
    concept Fooable = true;

    template <[[Fooable]]<int> T>
    void bar(T t);
  )cpp";
  EXPECT_DECLS("ConceptSpecializationExpr",
               {"template <typename T, typename U> concept Fooable = true;"});
}

TEST_F(TargetDeclTest, FunctionTemplate) {
  Code = R"cpp(
    // Implicit specialization.
    template<typename T> bool foo(T) { return false; };
    bool x = [[foo]](42);
  )cpp";
  EXPECT_DECLS("DeclRefExpr",
               {"template<> bool foo<int>(int)", Rel::TemplateInstantiation},
               {"bool foo(T)", Rel::TemplatePattern});

  Code = R"cpp(
    // Explicit specialization.
    template<typename T> bool foo(T) { return false; };
    template<> bool foo<int>(int) { return false; };
    bool x = [[foo]](42);
  )cpp";
  EXPECT_DECLS("DeclRefExpr", "template<> bool foo<int>(int)");
}

TEST_F(TargetDeclTest, VariableTemplate) {
  // Pretty-printer doesn't do a very good job of variable templates :-(
  Code = R"cpp(
    // Implicit specialization.
    template<typename T> int foo;
    int x = [[foo]]<char>;
  )cpp";
  EXPECT_DECLS("DeclRefExpr", {"int foo", Rel::TemplateInstantiation},
               {"int foo", Rel::TemplatePattern});

  Code = R"cpp(
    // Explicit specialization.
    template<typename T> int foo;
    template <> bool foo<char>;
    int x = [[foo]]<char>;
  )cpp";
  EXPECT_DECLS("DeclRefExpr", "bool foo");

  Code = R"cpp(
    // Partial specialization.
    template<typename T> int foo;
    template<typename T> bool foo<T*>;
    bool x = [[foo]]<char*>;
  )cpp";
  EXPECT_DECLS("DeclRefExpr", {"bool foo", Rel::TemplateInstantiation},
               {"bool foo", Rel::TemplatePattern});
}

TEST_F(TargetDeclTest, TypeAliasTemplate) {
  Code = R"cpp(
    template<typename T, int X> class SmallVector {};
    template<typename U> using TinyVector = SmallVector<U, 1>;
    [[TinyVector<int>]] X;
  )cpp";
  EXPECT_DECLS("TemplateSpecializationTypeLoc",
               {"template<> class SmallVector<int, 1>",
                Rel::TemplateInstantiation | Rel::Underlying},
               {"class SmallVector", Rel::TemplatePattern | Rel::Underlying},
               {"using TinyVector = SmallVector<U, 1>",
                Rel::Alias | Rel::TemplatePattern});
}

TEST_F(TargetDeclTest, MemberOfTemplate) {
  Code = R"cpp(
    template <typename T> struct Foo {
      int x(T);
    };
    int y = Foo<int>().[[x]](42);
  )cpp";
  EXPECT_DECLS("MemberExpr", {"int x(int)", Rel::TemplateInstantiation},
               {"int x(T)", Rel::TemplatePattern});

  Code = R"cpp(
    template <typename T> struct Foo {
      template <typename U>
      int x(T, U);
    };
    int y = Foo<char>().[[x]]('c', 42);
  )cpp";
  EXPECT_DECLS("MemberExpr",
               {"template<> int x<int>(char, int)", Rel::TemplateInstantiation},
               {"int x(T, U)", Rel::TemplatePattern});
}

TEST_F(TargetDeclTest, Lambda) {
  Code = R"cpp(
    void foo(int x = 42) {
      auto l = [ [[x]] ]{ return x + 1; };
    };
  )cpp";
  EXPECT_DECLS("DeclRefExpr", "int x = 42");

  // It seems like this should refer to another var, with the outer param being
  // an underlying decl. But it doesn't seem to exist.
  Code = R"cpp(
    void foo(int x = 42) {
      auto l = [x]{ return [[x]] + 1; };
    };
  )cpp";
  EXPECT_DECLS("DeclRefExpr", "int x = 42");

  Code = R"cpp(
    void foo() {
      auto l = [x = 1]{ return [[x]] + 1; };
    };
  )cpp";
  // FIXME: why both auto and int?
  EXPECT_DECLS("DeclRefExpr", "auto int x = 1");
}

TEST_F(TargetDeclTest, OverloadExpr) {
  // FIXME: Auto-completion in a template requires disabling delayed template
  // parsing.
  Flags = {"-fno-delayed-template-parsing"};
  Flags.push_back("--target=x86_64-pc-linux-gnu");

  Code = R"cpp(
    void func(int*);
    void func(char*);

    template <class T>
    void foo(T t) {
      [[func]](t);
    };
  )cpp";
  EXPECT_DECLS("UnresolvedLookupExpr", "void func(int *)", "void func(char *)");

  Code = R"cpp(
    struct X {
      void func(int*);
      void func(char*);
    };

    template <class T>
    void foo(X x, T t) {
      x.[[func]](t);
    };
  )cpp";
  EXPECT_DECLS("UnresolvedMemberExpr", "void func(int *)", "void func(char *)");

  Code = R"cpp(
    struct X {
      static void *operator new(unsigned long);
    };
    auto* k = [[new]] X();
  )cpp";
  EXPECT_DECLS("CXXNewExpr", "static void *operator new(unsigned long)");
  Code = R"cpp(
    void *operator new(unsigned long);
    auto* k = [[new]] int();
  )cpp";
  EXPECT_DECLS("CXXNewExpr", "void *operator new(unsigned long)");

  Code = R"cpp(
    struct X {
      static void operator delete(void *) noexcept;
    };
    void k(X* x) {
      [[delete]] x;
    }
  )cpp";
  EXPECT_DECLS("CXXDeleteExpr", "static void operator delete(void *) noexcept");
  Code = R"cpp(
    void operator delete(void *) noexcept;
    void k(int* x) {
      [[delete]] x;
    }
  )cpp";
  EXPECT_DECLS("CXXDeleteExpr", "void operator delete(void *) noexcept");
}

TEST_F(TargetDeclTest, DependentExprs) {
  Flags = {"-fno-delayed-template-parsing"};

  // Heuristic resolution of method of dependent field
  Code = R"cpp(
        struct A { void foo() {} };
        template <typename T>
        struct B {
          A a;
          void bar() {
            this->a.[[foo]]();
          }
        };
      )cpp";
  EXPECT_DECLS("CXXDependentScopeMemberExpr", "void foo()");

  // Similar to above but base expression involves a function call.
  Code = R"cpp(
        struct A {
          void foo() {}
        };
        struct B {
          A getA();
        };
        template <typename T>
        struct C {
          B c;
          void bar() {
            this->c.getA().[[foo]]();
          }
        };
      )cpp";
  EXPECT_DECLS("CXXDependentScopeMemberExpr", "void foo()");

  // Similar to above but uses a function pointer.
  Code = R"cpp(
        struct A {
          void foo() {}
        };
        struct B {
          using FPtr = A(*)();
          FPtr fptr;
        };
        template <typename T>
        struct C {
          B c;
          void bar() {
            this->c.fptr().[[foo]]();
          }
        };
      )cpp";
  EXPECT_DECLS("CXXDependentScopeMemberExpr", "void foo()");

  // Base expression involves a member access into this.
  Code = R"cpp(
        struct Bar {
          int aaaa;
        };
        template <typename T> struct Foo {
          Bar func(int);
          void test() {
            func(1).[[aaaa]];
          }
        };
      )cpp";
  EXPECT_DECLS("CXXDependentScopeMemberExpr", "int aaaa");

  Code = R"cpp(
        class Foo {
        public:
          static Foo k(int);
          template <typename T> T convert() const;
        };
        template <typename T>
        void test() {
          Foo::k(T()).template [[convert]]<T>();
        }
      )cpp";
  EXPECT_DECLS("CXXDependentScopeMemberExpr",
               "template <typename T> T convert() const");
}

TEST_F(TargetDeclTest, ObjC) {
  Flags = {"-xobjective-c"};
  Code = R"cpp(
    @interface Foo {}
    -(void)bar;
    @end
    void test(Foo *f) {
      [f [[bar]] ];
    }
  )cpp";
  EXPECT_DECLS("ObjCMessageExpr", "- (void)bar");

  Code = R"cpp(
    @interface Foo { @public int bar; }
    @end
    int test(Foo *f) {
      return [[f->bar]];
    }
  )cpp";
  EXPECT_DECLS("ObjCIvarRefExpr", "int bar");

  Code = R"cpp(
    @interface Foo {}
    -(int) x;
    -(void) setX:(int)x;
    @end
    void test(Foo *f) {
      [[f.x]] = 42;
    }
  )cpp";
  EXPECT_DECLS("ObjCPropertyRefExpr", "- (void)setX:(int)x");

  Code = R"cpp(
    @interface I {}
    @property(retain) I* x;
    @property(retain) I* y;
    @end
    void test(I *f) {
      [[f.x]].y = 0;
    }
  )cpp";
  EXPECT_DECLS("ObjCPropertyRefExpr",
               "@property(atomic, retain, readwrite) I *x");

  Code = R"cpp(
    @protocol Foo
    @end
    id test() {
      return [[@protocol(Foo)]];
    }
  )cpp";
  EXPECT_DECLS("ObjCProtocolExpr", "@protocol Foo");

  Code = R"cpp(
    @interface Foo
    @end
    void test([[Foo]] *p);
  )cpp";
  EXPECT_DECLS("ObjCInterfaceTypeLoc", "@interface Foo");

  Code = R"cpp(// Don't consider implicit interface as the target.
    @implementation [[Implicit]]
    @end
  )cpp";
  EXPECT_DECLS("ObjCImplementationDecl", "@implementation Implicit");

  Code = R"cpp(
    @interface Foo
    @end
    @implementation [[Foo]]
    @end
  )cpp";
  EXPECT_DECLS("ObjCImplementationDecl", "@interface Foo");

  Code = R"cpp(
    @interface Foo
    @end
    @interface Foo (Ext)
    @end
    @implementation [[Foo]] (Ext)
    @end
  )cpp";
  EXPECT_DECLS("ObjCCategoryImplDecl", "@interface Foo(Ext)");

  Code = R"cpp(
    @protocol Foo
    @end
    void test([[id<Foo>]] p);
  )cpp";
  EXPECT_DECLS("ObjCObjectTypeLoc", "@protocol Foo");

  Code = R"cpp(
    @class C;
    @protocol Foo
    @end
    void test(C<[[Foo]]> *p);
  )cpp";
  // FIXME: there's no AST node corresponding to 'Foo', so we're stuck.
  EXPECT_DECLS("ObjCObjectTypeLoc");
}

class FindExplicitReferencesTest : public ::testing::Test {
protected:
  struct AllRefs {
    std::string AnnotatedCode;
    std::string DumpedReferences;
  };

  /// Parses \p Code, finds function or namespace '::foo' and annotates its body
  /// with results of findExplicitReferences.
  /// See actual tests for examples of annotation format.
  AllRefs annotateReferencesInFoo(llvm::StringRef Code) {
    TestTU TU;
    TU.Code = std::string(Code);

    // FIXME: Auto-completion in a template requires disabling delayed template
    // parsing.
    TU.ExtraArgs.push_back("-fno-delayed-template-parsing");
    TU.ExtraArgs.push_back("-std=c++20");
    TU.ExtraArgs.push_back("-xobjective-c++");

    auto AST = TU.build();
    auto *TestDecl = &findDecl(AST, "foo");
    if (auto *T = llvm::dyn_cast<FunctionTemplateDecl>(TestDecl))
      TestDecl = T->getTemplatedDecl();

    std::vector<ReferenceLoc> Refs;
    if (const auto *Func = llvm::dyn_cast<FunctionDecl>(TestDecl))
      findExplicitReferences(Func->getBody(), [&Refs](ReferenceLoc R) {
        Refs.push_back(std::move(R));
      });
    else if (const auto *NS = llvm::dyn_cast<NamespaceDecl>(TestDecl))
      findExplicitReferences(NS, [&Refs, &NS](ReferenceLoc R) {
        // Avoid adding the namespace foo decl to the results.
        if (R.Targets.size() == 1 && R.Targets.front() == NS)
          return;
        Refs.push_back(std::move(R));
      });
    else
      ADD_FAILURE() << "Failed to find ::foo decl for test";

    auto &SM = AST.getSourceManager();
    llvm::sort(Refs, [&](const ReferenceLoc &L, const ReferenceLoc &R) {
      return SM.isBeforeInTranslationUnit(L.NameLoc, R.NameLoc);
    });

    std::string AnnotatedCode;
    unsigned NextCodeChar = 0;
    for (unsigned I = 0; I < Refs.size(); ++I) {
      auto &R = Refs[I];

      SourceLocation Pos = R.NameLoc;
      assert(Pos.isValid());
      if (Pos.isMacroID()) // FIXME: figure out how to show macro locations.
        Pos = SM.getExpansionLoc(Pos);
      assert(Pos.isFileID());

      FileID File;
      unsigned Offset;
      std::tie(File, Offset) = SM.getDecomposedLoc(Pos);
      if (File == SM.getMainFileID()) {
        // Print the reference in a source code.
        assert(NextCodeChar <= Offset);
        AnnotatedCode += Code.substr(NextCodeChar, Offset - NextCodeChar);
        AnnotatedCode += "$" + std::to_string(I) + "^";

        NextCodeChar = Offset;
      }
    }
    AnnotatedCode += Code.substr(NextCodeChar);

    std::string DumpedReferences;
    for (unsigned I = 0; I < Refs.size(); ++I)
      DumpedReferences += std::string(llvm::formatv("{0}: {1}\n", I, Refs[I]));

    return AllRefs{std::move(AnnotatedCode), std::move(DumpedReferences)};
  }
};

TEST_F(FindExplicitReferencesTest, All) {
  std::pair</*Code*/ llvm::StringRef, /*References*/ llvm::StringRef> Cases[] =
      {
          // Simple expressions.
          {R"cpp(
        int global;
        int func();
        void foo(int param) {
          $0^global = $1^param + $2^func();
        }
        )cpp",
           "0: targets = {global}\n"
           "1: targets = {param}\n"
           "2: targets = {func}\n"},
          {R"cpp(
        struct X { int a; };
        void foo(X x) {
          $0^x.$1^a = 10;
        }
        )cpp",
           "0: targets = {x}\n"
           "1: targets = {X::a}\n"},
          {R"cpp(
        // error-ok: testing with broken code
        int bar();
        int foo() {
          return $0^bar() + $1^bar(42);
        }
        )cpp",
           "0: targets = {bar}\n"
           "1: targets = {bar}\n"},
          // Namespaces and aliases.
          {R"cpp(
          namespace ns {}
          namespace alias = ns;
          void foo() {
            using namespace $0^ns;
            using namespace $1^alias;
          }
        )cpp",
           "0: targets = {ns}\n"
           "1: targets = {alias}\n"},
          // Using declarations.
          {R"cpp(
          namespace ns { int global; }
          void foo() {
            using $0^ns::$1^global;
          }
        )cpp",
           "0: targets = {ns}\n"
           "1: targets = {ns::global}, qualifier = 'ns::'\n"},
          // Simple types.
          {R"cpp(
         struct Struct { int a; };
         using Typedef = int;
         void foo() {
           $0^Struct $1^x;
           $2^Typedef $3^y;
           static_cast<$4^Struct*>(0);
         }
       )cpp",
           "0: targets = {Struct}\n"
           "1: targets = {x}, decl\n"
           "2: targets = {Typedef}\n"
           "3: targets = {y}, decl\n"
           "4: targets = {Struct}\n"},
          // Name qualifiers.
          {R"cpp(
         namespace a { namespace b { struct S { typedef int type; }; } }
         void foo() {
           $0^a::$1^b::$2^S $3^x;
           using namespace $4^a::$5^b;
           $6^S::$7^type $8^y;
         }
        )cpp",
           "0: targets = {a}\n"
           "1: targets = {a::b}, qualifier = 'a::'\n"
           "2: targets = {a::b::S}, qualifier = 'a::b::'\n"
           "3: targets = {x}, decl\n"
           "4: targets = {a}\n"
           "5: targets = {a::b}, qualifier = 'a::'\n"
           "6: targets = {a::b::S}\n"
           "7: targets = {a::b::S::type}, qualifier = 'struct S::'\n"
           "8: targets = {y}, decl\n"},
          {R"cpp(
         void foo() {
           $0^ten: // PRINT "HELLO WORLD!"
           goto $1^ten;
         }
       )cpp",
        "0: targets = {ten}, decl\n"
        "1: targets = {ten}\n"},
       // Simple templates.
       {R"cpp(
          template <class T> struct vector { using value_type = T; };
          template <> struct vector<bool> { using value_type = bool; };
          void foo() {
            $0^vector<int> $1^vi;
            $2^vector<bool> $3^vb;
          }
        )cpp",
           "0: targets = {vector<int>}\n"
           "1: targets = {vi}, decl\n"
           "2: targets = {vector<bool>}\n"
           "3: targets = {vb}, decl\n"},
          // Template type aliases.
          {R"cpp(
            template <class T> struct vector { using value_type = T; };
            template <> struct vector<bool> { using value_type = bool; };
            template <class T> using valias = vector<T>;
            void foo() {
              $0^valias<int> $1^vi;
              $2^valias<bool> $3^vb;
            }
          )cpp",
           "0: targets = {valias}\n"
           "1: targets = {vi}, decl\n"
           "2: targets = {valias}\n"
           "3: targets = {vb}, decl\n"},
          // Injected class name.
          {R"cpp(
            namespace foo {
              template <typename $0^T>
              class $1^Bar {
                ~$2^Bar();
                void $3^f($4^Bar);
              };
            }
          )cpp",
           "0: targets = {foo::Bar::T}, decl\n"
           "1: targets = {foo::Bar}, decl\n"
           "2: targets = {foo::Bar}\n"
           "3: targets = {foo::Bar::f}, decl\n"
           "4: targets = {foo::Bar}\n"},
          // MemberExpr should know their using declaration.
          {R"cpp(
            struct X { void func(int); };
            struct Y : X {
              using X::func;
            };
            void foo(Y y) {
              $0^y.$1^func(1);
            }
        )cpp",
           "0: targets = {y}\n"
           "1: targets = {Y::func}\n"},
          // DeclRefExpr should know their using declaration.
          {R"cpp(
            namespace ns { void bar(int); }
            using ns::bar;

            void foo() {
              $0^bar(10);
            }
        )cpp",
           "0: targets = {bar}\n"},
          // References from a macro.
          {R"cpp(
            #define FOO a
            #define BAR b

            void foo(int a, int b) {
              $0^FOO+$1^BAR;
            }
        )cpp",
           "0: targets = {a}\n"
           "1: targets = {b}\n"},
          // No references from implicit nodes.
          {R"cpp(
            struct vector {
              int *begin();
              int *end();
            };

            void foo() {
              for (int $0^x : $1^vector()) {
                $2^x = 10;
              }
            }
        )cpp",
           "0: targets = {x}, decl\n"
           "1: targets = {vector}\n"
           "2: targets = {x}\n"},
// Handle UnresolvedLookupExpr.
// FIXME
// This case fails when expensive checks are enabled.
// Seems like the order of ns1::func and ns2::func isn't defined.
#ifndef EXPENSIVE_CHECKS
          {R"cpp(
            namespace ns1 { void func(char*); }
            namespace ns2 { void func(int*); }
            using namespace ns1;
            using namespace ns2;

            template <class T>
            void foo(T t) {
              $0^func($1^t);
            }
        )cpp",
           "0: targets = {ns1::func, ns2::func}\n"
           "1: targets = {t}\n"},
#endif
          // Handle UnresolvedMemberExpr.
          {R"cpp(
            struct X {
              void func(char*);
              void func(int*);
            };

            template <class T>
            void foo(X x, T t) {
              $0^x.$1^func($2^t);
            }
        )cpp",
           "0: targets = {x}\n"
           "1: targets = {X::func, X::func}\n"
           "2: targets = {t}\n"},
          // Handle DependentScopeDeclRefExpr.
          {R"cpp(
            template <class T>
            struct S {
              static int value;
            };

            template <class T>
            void foo() {
              $0^S<$1^T>::$2^value;
            }
       )cpp",
           "0: targets = {S}\n"
           "1: targets = {T}\n"
           "2: targets = {S::value}, qualifier = 'S<T>::'\n"},
          // Handle CXXDependentScopeMemberExpr.
          {R"cpp(
            template <class T>
            struct S {
              int value;
            };

            template <class T>
            void foo(S<T> t) {
              $0^t.$1^value;
            }
       )cpp",
           "0: targets = {t}\n"
           "1: targets = {S::value}\n"},
          // Type template parameters.
          {R"cpp(
            template <class T>
            void foo() {
              static_cast<$0^T>(0);
              $1^T();
              $2^T $3^t;
            }
        )cpp",
           "0: targets = {T}\n"
           "1: targets = {T}\n"
           "2: targets = {T}\n"
           "3: targets = {t}, decl\n"},
          // Non-type template parameters.
          {R"cpp(
            template <int I>
            void foo() {
              int $0^x = $1^I;
            }
        )cpp",
           "0: targets = {x}, decl\n"
           "1: targets = {I}\n"},
          // Template template parameters.
          {R"cpp(
            template <class T> struct vector {};

            template <template<class> class TT, template<class> class ...TP>
            void foo() {
              $0^TT<int> $1^x;
              $2^foo<$3^TT>();
              $4^foo<$5^vector>();
              $6^foo<$7^TP...>();
            }
        )cpp",
           "0: targets = {TT}\n"
           "1: targets = {x}, decl\n"
           "2: targets = {foo}\n"
           "3: targets = {TT}\n"
           "4: targets = {foo}\n"
           "5: targets = {vector}\n"
           "6: targets = {foo}\n"
           "7: targets = {TP}\n"},
          // Non-type template parameters with declarations.
          {R"cpp(
            int func();
            template <int(*)()> struct wrapper {};

            template <int(*FuncParam)()>
            void foo() {
              $0^wrapper<$1^func> $2^w;
              $3^FuncParam();
            }
        )cpp",
           "0: targets = {wrapper<&func>}\n"
           "1: targets = {func}\n"
           "2: targets = {w}, decl\n"
           "3: targets = {FuncParam}\n"},
          // declaration references.
          {R"cpp(
             namespace ns {}
             class S {};
             void foo() {
               class $0^Foo { $1^Foo(); ~$2^Foo(); int $3^field; };
               int $4^Var;
               enum $5^E { $6^ABC };
               typedef int $7^INT;
               using $8^INT2 = int;
               namespace $9^NS = $10^ns;
             }
           )cpp",
           "0: targets = {Foo}, decl\n"
           "1: targets = {foo()::Foo::Foo}, decl\n"
           "2: targets = {Foo}\n"
           "3: targets = {foo()::Foo::field}, decl\n"
           "4: targets = {Var}, decl\n"
           "5: targets = {E}, decl\n"
           "6: targets = {foo()::ABC}, decl\n"
           "7: targets = {INT}, decl\n"
           "8: targets = {INT2}, decl\n"
           "9: targets = {NS}, decl\n"
           "10: targets = {ns}\n"},
          // User-defined conversion operator.
          {R"cpp(
            void foo() {
               class $0^Bar {};
               class $1^Foo {
               public:
                 // FIXME: This should have only one reference to Bar.
                 $2^operator $3^$4^Bar();
               };

               $5^Foo $6^f;
               $7^f.$8^operator $9^Bar();
            }
        )cpp",
           "0: targets = {Bar}, decl\n"
           "1: targets = {Foo}, decl\n"
           "2: targets = {foo()::Foo::operator Bar}, decl\n"
           "3: targets = {Bar}\n"
           "4: targets = {Bar}\n"
           "5: targets = {Foo}\n"
           "6: targets = {f}, decl\n"
           "7: targets = {f}\n"
           "8: targets = {foo()::Foo::operator Bar}\n"
           "9: targets = {Bar}\n"},
          // Destructor.
          {R"cpp(
             void foo() {
               class $0^Foo {
               public:
                 ~$1^Foo() {}

                 void $2^destructMe() {
                   this->~$3^Foo();
                 }
               };

               $4^Foo $5^f;
               $6^f.~ /*...*/ $7^Foo();
             }
           )cpp",
           "0: targets = {Foo}, decl\n"
           // FIXME: It's better to target destructor's FunctionDecl instead of
           // the type itself (similar to constructor).
           "1: targets = {Foo}\n"
           "2: targets = {foo()::Foo::destructMe}, decl\n"
           "3: targets = {Foo}\n"
           "4: targets = {Foo}\n"
           "5: targets = {f}, decl\n"
           "6: targets = {f}\n"
           "7: targets = {Foo}\n"},
          // cxx constructor initializer.
          {R"cpp(
             class Base {};
             void foo() {
               // member initializer
               class $0^X {
                 int $1^abc;
                 $2^X(): $3^abc() {}
               };
               // base initializer
               class $4^Derived : public $5^Base {
                 $6^Base $7^B;
                 $8^Derived() : $9^Base() {}
               };
               // delegating initializer
               class $10^Foo {
                 $11^Foo(int);
                 $12^Foo(): $13^Foo(111) {}
               };
             }
           )cpp",
           "0: targets = {X}, decl\n"
           "1: targets = {foo()::X::abc}, decl\n"
           "2: targets = {foo()::X::X}, decl\n"
           "3: targets = {foo()::X::abc}\n"
           "4: targets = {Derived}, decl\n"
           "5: targets = {Base}\n"
           "6: targets = {Base}\n"
           "7: targets = {foo()::Derived::B}, decl\n"
           "8: targets = {foo()::Derived::Derived}, decl\n"
           "9: targets = {Base}\n"
           "10: targets = {Foo}, decl\n"
           "11: targets = {foo()::Foo::Foo}, decl\n"
           "12: targets = {foo()::Foo::Foo}, decl\n"
           "13: targets = {Foo}\n"},
          // Anonymous entities should not be reported.
          {
              R"cpp(
             void foo() {
              class {} $0^x;
              int (*$1^fptr)(int $2^a, int) = nullptr;
             }
           )cpp",
              "0: targets = {x}, decl\n"
              "1: targets = {fptr}, decl\n"
              "2: targets = {a}, decl\n"},
          // Namespace aliases should be handled properly.
          {
              R"cpp(
                namespace ns { struct Type {}; }
                namespace alias = ns;
                namespace rec_alias = alias;

                void foo() {
                  $0^ns::$1^Type $2^a;
                  $3^alias::$4^Type $5^b;
                  $6^rec_alias::$7^Type $8^c;
                }
           )cpp",
              "0: targets = {ns}\n"
              "1: targets = {ns::Type}, qualifier = 'ns::'\n"
              "2: targets = {a}, decl\n"
              "3: targets = {alias}\n"
              "4: targets = {ns::Type}, qualifier = 'alias::'\n"
              "5: targets = {b}, decl\n"
              "6: targets = {rec_alias}\n"
              "7: targets = {ns::Type}, qualifier = 'rec_alias::'\n"
              "8: targets = {c}, decl\n"},
          // Handle SizeOfPackExpr.
          {
              R"cpp(
                template <typename... E>
                void foo() {
                  constexpr int $0^size = sizeof...($1^E);
                };
            )cpp",
              "0: targets = {size}, decl\n"
              "1: targets = {E}\n"},
          // Class template argument deduction
          {
              R"cpp(
                template <typename T>
                struct Test {
                Test(T);
              };
              void foo() {
                $0^Test $1^a(5);
              }
            )cpp",
              "0: targets = {Test}\n"
              "1: targets = {a}, decl\n"},
          // Templates
          {R"cpp(
            namespace foo {
              template <typename $0^T>
              class $1^Bar {};
            }
          )cpp",
           "0: targets = {foo::Bar::T}, decl\n"
           "1: targets = {foo::Bar}, decl\n"},
          // Templates
          {R"cpp(
            namespace foo {
              template <typename $0^T>
              void $1^func();
            }
          )cpp",
           "0: targets = {T}, decl\n"
           "1: targets = {foo::func}, decl\n"},
          // Templates
          {R"cpp(
            namespace foo {
              template <typename $0^T>
              $1^T $2^x;
            }
          )cpp",
           "0: targets = {foo::T}, decl\n"
           "1: targets = {foo::T}\n"
           "2: targets = {foo::x}, decl\n"},
          // Templates
          {R"cpp(
            template<typename T> class vector {};
            namespace foo {
              template <typename $0^T>
              using $1^V = $2^vector<$3^T>;
            }
          )cpp",
           "0: targets = {foo::T}, decl\n"
           "1: targets = {foo::V}, decl\n"
           "2: targets = {vector}\n"
           "3: targets = {foo::T}\n"},
          // Concept
          {
              R"cpp(
              template <typename T>
              concept Drawable = requires (T t) { t.draw(); };

              namespace foo {
                template <typename $0^T> requires $1^Drawable<$2^T>
                void $3^bar($4^T $5^t) {
                  $6^t.$7^draw();
                }
              }
          )cpp",
              "0: targets = {T}, decl\n"
              "1: targets = {Drawable}\n"
              "2: targets = {T}\n"
              "3: targets = {foo::bar}, decl\n"
              "4: targets = {T}\n"
              "5: targets = {t}, decl\n"
              "6: targets = {t}\n"
              "7: targets = {}\n"},
          // Objective-C: properties
          {
              R"cpp(
            @interface I {}
            @property(retain) I* x;
            @property(retain) I* y;
            @end
            I *f;
            void foo() {
              $0^f.$1^x.$2^y = 0;
            }
          )cpp",
              "0: targets = {f}\n"
              "1: targets = {I::x}\n"
              "2: targets = {I::y}\n"},
          // Objective-C: implicit properties
          {
              R"cpp(
            @interface I {}
            -(I*)x;
            -(void)setY:(I*)y;
            @end
            I *f;
            void foo() {
              $0^f.$1^x.$2^y = 0;
            }
          )cpp",
              "0: targets = {f}\n"
              "1: targets = {I::x}\n"
              "2: targets = {I::setY:}\n"},
          // Designated initializers.
          {R"cpp(
            void foo() {
              struct $0^Foo {
                int $1^Bar;
              };
              $2^Foo $3^f { .$4^Bar = 42 };
            }
        )cpp",
           "0: targets = {Foo}, decl\n"
           "1: targets = {foo()::Foo::Bar}, decl\n"
           "2: targets = {Foo}\n"
           "3: targets = {f}, decl\n"
           "4: targets = {foo()::Foo::Bar}\n"},
          {R"cpp(
            void foo() {
              struct $0^Baz {
                int $1^Field;
              };
              struct $2^Bar {
                $3^Baz $4^Foo;
              };
              $5^Bar $6^bar { .$7^Foo.$8^Field = 42 };
            }
        )cpp",
        "0: targets = {Baz}, decl\n"
        "1: targets = {foo()::Baz::Field}, decl\n"
        "2: targets = {Bar}, decl\n"
        "3: targets = {Baz}\n"
        "4: targets = {foo()::Bar::Foo}, decl\n"
        "5: targets = {Bar}\n"
        "6: targets = {bar}, decl\n"
        "7: targets = {foo()::Bar::Foo}\n"
        "8: targets = {foo()::Baz::Field}\n"},
       {R"cpp(
           template<typename T>
           void crash(T);
           template<typename T>
           void foo() {
             $0^crash({.$1^x = $2^T()});
           }
        )cpp",
        "0: targets = {crash}\n"
        "1: targets = {}\n"
        "2: targets = {T}\n"},
       // unknown template name should not crash.
       {R"cpp(
        template <template <typename> typename T>
        struct Base {};
        namespace foo {
        template <typename $0^T>
        struct $1^Derive : $2^Base<$3^T::template $4^Unknown> {};
        }
      )cpp",
        "0: targets = {foo::Derive::T}, decl\n"
        "1: targets = {foo::Derive}, decl\n"
        "2: targets = {Base}\n"
        "3: targets = {foo::Derive::T}\n"
        "4: targets = {}, qualifier = 'T::'\n"},
       // deduction guide
       {R"cpp(
          namespace foo {
            template <typename $0^T>
            struct $1^Test {
              template <typename $2^I>
              $3^Test($4^I);
            };
            template <typename $5^I>
            $6^Test($7^I) -> $8^Test<typename $9^I::$10^type>;
          }
        )cpp",
        "0: targets = {T}, decl\n"
        "1: targets = {foo::Test}, decl\n"
        "2: targets = {I}, decl\n"
        "3: targets = {foo::Test::Test<T>}, decl\n"
        "4: targets = {I}\n"
        "5: targets = {I}, decl\n"
        "6: targets = {foo::Test}\n"
        "7: targets = {I}\n"
        "8: targets = {foo::Test}\n"
        "9: targets = {I}\n"
        "10: targets = {}, qualifier = 'I::'\n"}};

  for (const auto &C : Cases) {
    llvm::StringRef ExpectedCode = C.first;
    llvm::StringRef ExpectedRefs = C.second;

    auto Actual =
        annotateReferencesInFoo(llvm::Annotations(ExpectedCode).code());
    EXPECT_EQ(ExpectedCode, Actual.AnnotatedCode);
    EXPECT_EQ(ExpectedRefs, Actual.DumpedReferences) << ExpectedCode;
  }
}

} // namespace
} // namespace clangd
} // namespace clang
