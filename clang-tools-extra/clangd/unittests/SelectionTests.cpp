//===-- SelectionTests.cpp - ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Annotations.h"
#include "Selection.h"
#include "SourceCode.h"
#include "TestTU.h"
#include "support/TestTracer.h"
#include "clang/AST/Decl.h"
#include "llvm/Support/Casting.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {
using ::testing::ElementsAreArray;
using ::testing::UnorderedElementsAreArray;

// Create a selection tree corresponding to a point or pair of points.
// This uses the precisely-defined createRight semantics. The fuzzier
// createEach is tested separately.
SelectionTree makeSelectionTree(const StringRef MarkedCode, ParsedAST &AST) {
  Annotations Test(MarkedCode);
  switch (Test.points().size()) {
  case 1: { // Point selection.
    unsigned Offset = cantFail(positionToOffset(Test.code(), Test.point()));
    return SelectionTree::createRight(AST.getASTContext(), AST.getTokens(),
                                      Offset, Offset);
  }
  case 2: // Range selection.
    return SelectionTree::createRight(
        AST.getASTContext(), AST.getTokens(),
        cantFail(positionToOffset(Test.code(), Test.points()[0])),
        cantFail(positionToOffset(Test.code(), Test.points()[1])));
  default:
    ADD_FAILURE() << "Expected 1-2 points for selection.\n" << MarkedCode;
    return SelectionTree::createRight(AST.getASTContext(), AST.getTokens(), 0u,
                                      0u);
  }
}

Range nodeRange(const SelectionTree::Node *N, ParsedAST &AST) {
  if (!N)
    return Range{};
  const SourceManager &SM = AST.getSourceManager();
  const LangOptions &LangOpts = AST.getLangOpts();
  StringRef Buffer = SM.getBufferData(SM.getMainFileID());
  if (llvm::isa_and_nonnull<TranslationUnitDecl>(N->ASTNode.get<Decl>()))
    return Range{Position{}, offsetToPosition(Buffer, Buffer.size())};
  auto FileRange =
      toHalfOpenFileRange(SM, LangOpts, N->ASTNode.getSourceRange());
  assert(FileRange && "We should be able to get the File Range");
  return Range{
      offsetToPosition(Buffer, SM.getFileOffset(FileRange->getBegin())),
      offsetToPosition(Buffer, SM.getFileOffset(FileRange->getEnd()))};
}

std::string nodeKind(const SelectionTree::Node *N) {
  return N ? N->kind() : "<null>";
}

std::vector<const SelectionTree::Node *> allNodes(const SelectionTree &T) {
  std::vector<const SelectionTree::Node *> Result = {&T.root()};
  for (unsigned I = 0; I < Result.size(); ++I) {
    const SelectionTree::Node *N = Result[I];
    Result.insert(Result.end(), N->Children.begin(), N->Children.end());
  }
  return Result;
}

// Returns true if Common is a descendent of Root.
// Verifies nothing is selected above Common.
bool verifyCommonAncestor(const SelectionTree::Node &Root,
                          const SelectionTree::Node *Common,
                          StringRef MarkedCode) {
  if (&Root == Common)
    return true;
  if (Root.Selected)
    ADD_FAILURE() << "Selected nodes outside common ancestor\n" << MarkedCode;
  bool Seen = false;
  for (const SelectionTree::Node *Child : Root.Children)
    if (verifyCommonAncestor(*Child, Common, MarkedCode)) {
      if (Seen)
        ADD_FAILURE() << "Saw common ancestor twice\n" << MarkedCode;
      Seen = true;
    }
  return Seen;
}

TEST(SelectionTest, CommonAncestor) {
  struct Case {
    // Selection is between ^marks^.
    // common ancestor marked with a [[range]].
    const char *Code;
    const char *CommonAncestorKind;
  };
  Case Cases[] = {
      {
          R"cpp(
            template <typename T>
            int x = [[T::^U::]]ccc();
          )cpp",
          "NestedNameSpecifierLoc",
      },
      {
          R"cpp(
            struct AAA { struct BBB { static int ccc(); };};
            int x = AAA::[[B^B^B]]::ccc();
          )cpp",
          "RecordTypeLoc",
      },
      {
          R"cpp(
            struct AAA { struct BBB { static int ccc(); };};
            int x = AAA::[[B^BB^]]::ccc();
          )cpp",
          "RecordTypeLoc",
      },
      {
          R"cpp(
            struct AAA { struct BBB { static int ccc(); };};
            int x = [[AAA::BBB::c^c^c]]();
          )cpp",
          "DeclRefExpr",
      },
      {
          R"cpp(
            struct AAA { struct BBB { static int ccc(); };};
            int x = [[AAA::BBB::cc^c(^)]];
          )cpp",
          "CallExpr",
      },

      {
          R"cpp(
            void foo() { [[if (1^11) { return; } else {^ }]] }
          )cpp",
          "IfStmt",
      },
      {
          R"cpp(
            int x(int);
            #define M(foo) x(foo)
            int a = 42;
            int b = M([[^a]]);
          )cpp",
          "DeclRefExpr",
      },
      {
          R"cpp(
            void foo();
            #define CALL_FUNCTION(X) X()
            void bar() { CALL_FUNCTION([[f^o^o]]); }
          )cpp",
          "DeclRefExpr",
      },
      {
          R"cpp(
            void foo();
            #define CALL_FUNCTION(X) X()
            void bar() { [[CALL_FUNC^TION(fo^o)]]; }
          )cpp",
          "CallExpr",
      },
      {
          R"cpp(
            void foo();
            #define CALL_FUNCTION(X) X()
            void bar() { [[C^ALL_FUNC^TION(foo)]]; }
          )cpp",
          "CallExpr",
      },
      {
          R"cpp(
            void foo();
            #^define CALL_FUNCTION(X) X(^)
            void bar() { CALL_FUNCTION(foo); }
          )cpp",
          nullptr,
      },
      {
          R"cpp(
            void foo();
            #define CALL_FUNCTION(X) X()
            void bar() { CALL_FUNCTION(foo^)^; }
          )cpp",
          nullptr,
      },
      {
          R"cpp(
            namespace ns {
            #if 0
            void fo^o() {}
            #endif
            }
          )cpp",
          nullptr,
      },
      {
          R"cpp(
            #define TARGET void foo()
            [[TAR^GET{ return; }]]
          )cpp",
          "FunctionDecl",
      },
      {
          R"cpp(
            struct S { S(const char*); };
            [[S s ^= "foo"]];
          )cpp",
          // The AST says a CXXConstructExpr covers the = sign in C++14.
          // But we consider CXXConstructExpr to only own brackets.
          // (It's not the interesting constructor anyway, just S(&&)).
          "VarDecl",
      },
      {
          R"cpp(
            struct S { S(const char*); };
            [[S ^s = "foo"]];
          )cpp",
          "VarDecl",
      },
      {
          R"cpp(
            [[^void]] (*S)(int) = nullptr;
          )cpp",
          "BuiltinTypeLoc",
      },
      {
          R"cpp(
            [[void (*S)^(int)]] = nullptr;
          )cpp",
          "FunctionProtoTypeLoc",
      },
      {
          R"cpp(
            [[void (^*S)(int)]] = nullptr;
          )cpp",
          "PointerTypeLoc",
      },
      {
          R"cpp(
            [[void (*^S)(int) = nullptr]];
          )cpp",
          "VarDecl",
      },
      {
          R"cpp(
            [[void ^(*S)(int)]] = nullptr;
          )cpp",
          "ParenTypeLoc",
      },
      {
          R"cpp(
            struct S {
              int foo() const;
              int bar() { return [[f^oo]](); }
            };
          )cpp",
          "MemberExpr", // Not implicit CXXThisExpr, or its implicit cast!
      },
      {
          R"cpp(
            auto lambda = [](const char*){ return 0; };
            int x = lambda([["y^"]]);
          )cpp",
          "StringLiteral", // Not DeclRefExpr to operator()!
      },
      {
          R"cpp(
            struct Foo {};
            struct Bar : [[v^ir^tual private Foo]] {};
          )cpp",
          "CXXBaseSpecifier",
      },
      {
          R"cpp(
            struct Foo {};
            struct Bar : private [[Fo^o]] {};
          )cpp",
          "RecordTypeLoc",
      },
      {
          R"cpp(
            struct Foo {};
            struct Bar : [[Fo^o]] {};
          )cpp",
          "RecordTypeLoc",
      },

      // Point selections.
      {"void foo() { [[^foo]](); }", "DeclRefExpr"},
      {"void foo() { [[f^oo]](); }", "DeclRefExpr"},
      {"void foo() { [[fo^o]](); }", "DeclRefExpr"},
      {"void foo() { [[foo^()]]; }", "CallExpr"},
      {"void foo() { [[foo^]] (); }", "DeclRefExpr"},
      {"int bar; void foo() [[{ foo (); }]]^", "CompoundStmt"},
      {"int x = [[42]]^;", "IntegerLiteral"},

      // Ignores whitespace, comments, and semicolons in the selection.
      {"void foo() { [[foo^()]]; /*comment*/^}", "CallExpr"},

      // Tricky case: FunctionTypeLoc in FunctionDecl has a hole in it.
      {"[[^void]] foo();", "BuiltinTypeLoc"},
      {"[[void foo^()]];", "FunctionProtoTypeLoc"},
      {"[[^void foo^()]];", "FunctionDecl"},
      {"[[void ^foo()]];", "FunctionDecl"},
      // Tricky case: two VarDecls share a specifier.
      {"[[int ^a]], b;", "VarDecl"},
      {"[[int a, ^b]];", "VarDecl"},
      // Tricky case: CXXConstructExpr wants to claim the whole init range.
      {
          R"cpp(
            struct X { X(int); };
            class Y {
              X x;
              Y() : [[^x(4)]] {}
            };
          )cpp",
          "CXXCtorInitializer", // Not the CXXConstructExpr!
      },
      // Tricky case: anonymous struct is a sibling of the VarDecl.
      {"[[st^ruct {int x;}]] y;", "CXXRecordDecl"},
      {"[[struct {int x;} ^y]];", "VarDecl"},
      {"struct {[[int ^x]];} y;", "FieldDecl"},

      // Tricky case: nested ArrayTypeLocs have the same token range.
      {"const int x = 1, y = 2; int array[^[[x]]][10][y];", "DeclRefExpr"},
      {"const int x = 1, y = 2; int array[x][10][^[[y]]];", "DeclRefExpr"},
      {"const int x = 1, y = 2; int array[x][^[[10]]][y];", "IntegerLiteral"},
      {"const int x = 1, y = 2; [[i^nt]] array[x][10][y];", "BuiltinTypeLoc"},
      {"void func(int x) { int v_array[^[[x]]][10]; }", "DeclRefExpr"},

      {"int (*getFunc([[do^uble]]))(int);", "BuiltinTypeLoc"},

      // Member pointers and pack expansion use declarator syntax, but are
      // restricted so they don't need special casing.
      {"class X{}; [[int X::^*]]y[10];", "MemberPointerTypeLoc"},
      {"template<typename ...T> void foo([[T*^...]]x);",
       "PackExpansionTypeLoc"},
      {"template<typename ...T> void foo([[^T]]*...x);",
       "TemplateTypeParmTypeLoc"},

      // FIXME: the AST has no location info for qualifiers.
      {"const [[a^uto]] x = 42;", "AutoTypeLoc"},
      {"co^nst auto x = 42;", nullptr},

      {"^", nullptr},
      {"void foo() { [[foo^^]] (); }", "DeclRefExpr"},

      // FIXME: Ideally we'd get a declstmt or the VarDecl itself here.
      // This doesn't happen now; the RAV doesn't traverse a node containing ;.
      {"int x = 42;^", nullptr},

      // Common ancestor is logically TUDecl, but we never return that.
      {"^int x; int y;^", nullptr},

      // Node types that have caused problems in the past.
      {"template <typename T> void foo() { [[^T]] t; }",
       "TemplateTypeParmTypeLoc"},

      // No crash
      {
          R"cpp(
            template <class T> struct Foo {};
            template <[[template<class> class /*cursor here*/^U]]>
             struct Foo<U<int>*> {};
          )cpp",
          "TemplateTemplateParmDecl"},

      // Foreach has a weird AST, ensure we can select parts of the range init.
      // This used to fail, because the DeclStmt for C claimed the whole range.
      {
          R"cpp(
            struct Str {
              const char *begin();
              const char *end();
            };
            Str makeStr(const char*);
            void loop() {
              for (const char C : [[mak^eStr("foo"^)]])
                ;
            }
          )cpp",
          "CallExpr"},

      // User-defined literals are tricky: is 12_i one token or two?
      // For now we treat it as one, and the UserDefinedLiteral as a leaf.
      {
          R"cpp(
            struct Foo{};
            Foo operator""_ud(unsigned long long);
            Foo x = [[^12_ud]];
          )cpp",
          "UserDefinedLiteral"},

      {
          R"cpp(
        int a;
        decltype([[^a]] + a) b;
        )cpp",
          "DeclRefExpr"},
      {"[[decltype^(1)]] b;", "DecltypeTypeLoc"}, // Not the VarDecl.
      // decltype(auto) is an AutoTypeLoc!
      {"[[de^cltype(a^uto)]] a = 1;", "AutoTypeLoc"},

      // Objective-C nullability attributes.
      {
          R"cpp(
            @interface I{}
            @property(nullable) [[^I]] *x;
            @end
          )cpp",
          "ObjCInterfaceTypeLoc"},
      {
          R"cpp(
            @interface I{}
            - (void)doSomething:(nonnull [[i^d]])argument;
            @end
          )cpp",
          "TypedefTypeLoc"},

      // Objective-C OpaqueValueExpr/PseudoObjectExpr has weird ASTs.
      // Need to traverse the contents of the OpaqueValueExpr to the POE,
      // and ensure we traverse only the syntactic form of the PseudoObjectExpr.
      {
          R"cpp(
            @interface I{}
            @property(retain) I*x;
            @property(retain) I*y;
            @end
            void test(I *f) { [[^f]].x.y = 0; }
          )cpp",
          "DeclRefExpr"},
      {
          R"cpp(
            @interface I{}
            @property(retain) I*x;
            @property(retain) I*y;
            @end
            void test(I *f) { [[f.^x]].y = 0; }
          )cpp",
          "ObjCPropertyRefExpr"},
      // Examples with implicit properties.
      {
          R"cpp(
            @interface I{}
            -(int)foo;
            @end
            int test(I *f) { return 42 + [[^f]].foo; }
          )cpp",
          "DeclRefExpr"},
      {
          R"cpp(
            @interface I{}
            -(int)foo;
            @end
            int test(I *f) { return 42 + [[f.^foo]]; }
          )cpp",
          "ObjCPropertyRefExpr"},
      {"struct foo { [[int has^h<:32:>]]; };", "FieldDecl"},
      {"struct foo { [[op^erator int()]]; };", "CXXConversionDecl"},
      {"struct foo { [[^~foo()]]; };", "CXXDestructorDecl"},
      // FIXME: The following to should be class itself instead.
      {"struct foo { [[fo^o(){}]] };", "CXXConstructorDecl"},

      {R"cpp(
        struct S1 { void f(); };
        struct S2 { S1 * operator->(); };
        void test(S2 s2) {
          s2[[-^>]]f();
        }
      )cpp",
       "DeclRefExpr"}, // DeclRefExpr to the "operator->" method.

      // Template template argument.
      {R"cpp(
        template <typename> class Vector {};
        template <template <typename> class Container> class A {};
        A<[[V^ector]]> a;
      )cpp",
       "TemplateArgumentLoc"},

      // Attributes
      {R"cpp(
        void f(int * __attribute__(([[no^nnull]])) );
      )cpp",
       "NonNullAttr"},

      {R"cpp(
        // Digraph syntax for attributes to avoid accidental annotations.
        class <:[gsl::Owner([[in^t]])]:> X{};
      )cpp",
       "BuiltinTypeLoc"},

      // This case used to crash - AST has a null Attr
      {R"cpp(
        @interface I
        [[@property(retain, nonnull) <:[My^Object2]:> *x]]; // error-ok
        @end
      )cpp",
       "ObjCPropertyDecl"},

      {R"cpp(
        typedef int Foo;
        enum Bar : [[Fo^o]] {};
      )cpp",
       "TypedefTypeLoc"},
      {R"cpp(
        typedef int Foo;
        enum Bar : [[Fo^o]];
      )cpp",
       "TypedefTypeLoc"},

      // lambda captured var-decl
      {R"cpp(
        void test(int bar) {
          auto l = [^[[foo = bar]]] { };
        })cpp",
       "VarDecl"},
      {R"cpp(
        /*error-ok*/
        void func() [[{^]])cpp",
       "CompoundStmt"},
      {R"cpp(
        void func() { [[__^func__]]; }
        )cpp",
       "PredefinedExpr"},
  };

  for (const Case &C : Cases) {
    trace::TestTracer Tracer;
    Annotations Test(C.Code);

    TestTU TU;
    TU.Code = std::string(Test.code());

    TU.ExtraArgs.push_back("-xobjective-c++");

    auto AST = TU.build();
    auto T = makeSelectionTree(C.Code, AST);
    EXPECT_EQ("TranslationUnitDecl", nodeKind(&T.root())) << C.Code;

    if (Test.ranges().empty()) {
      // If no [[range]] is marked in the example, there should be no selection.
      EXPECT_FALSE(T.commonAncestor()) << C.Code << "\n" << T;
      EXPECT_THAT(Tracer.takeMetric("selection_recovery", "C++"),
                  testing::IsEmpty());
    } else {
      // If there is an expected selection, common ancestor should exist
      // with the appropriate node type.
      EXPECT_EQ(C.CommonAncestorKind, nodeKind(T.commonAncestor()))
          << C.Code << "\n"
          << T;
      // Convert the reported common ancestor to a range and verify it.
      EXPECT_EQ(nodeRange(T.commonAncestor(), AST), Test.range())
          << C.Code << "\n"
          << T;

      // Check that common ancestor is reachable on exactly one path from root,
      // and no nodes outside it are selected.
      EXPECT_TRUE(verifyCommonAncestor(T.root(), T.commonAncestor(), C.Code))
          << C.Code;
      EXPECT_THAT(Tracer.takeMetric("selection_recovery", "C++"),
                  ElementsAreArray({0}));
    }
  }
}

// Regression test: this used to match the injected X, not the outer X.
TEST(SelectionTest, InjectedClassName) {
  const char *Code = "struct ^X { int x; };";
  auto AST = TestTU::withCode(Annotations(Code).code()).build();
  auto T = makeSelectionTree(Code, AST);
  ASSERT_EQ("CXXRecordDecl", nodeKind(T.commonAncestor())) << T;
  auto *D = dyn_cast<CXXRecordDecl>(T.commonAncestor()->ASTNode.get<Decl>());
  EXPECT_FALSE(D->isInjectedClassName());
}

TEST(SelectionTree, Metrics) {
  const char *Code = R"cpp(
    // error-ok: testing behavior on recovery expression
    int foo();
    int foo(int, int);
    int x = fo^o(42);
  )cpp";
  auto AST = TestTU::withCode(Annotations(Code).code()).build();
  trace::TestTracer Tracer;
  auto T = makeSelectionTree(Code, AST);
  EXPECT_THAT(Tracer.takeMetric("selection_recovery", "C++"),
              ElementsAreArray({1}));
  EXPECT_THAT(Tracer.takeMetric("selection_recovery_type", "C++"),
              ElementsAreArray({1}));
}

// FIXME: Doesn't select the binary operator node in
//          #define FOO(X) X + 1
//          int a, b = [[FOO(a)]];
TEST(SelectionTest, Selected) {
  // Selection with ^marks^.
  // Partially selected nodes marked with a [[range]].
  // Completely selected nodes marked with a $C[[range]].
  const char *Cases[] = {
      R"cpp( int abc, xyz = [[^ab^c]]; )cpp",
      R"cpp( int abc, xyz = [[a^bc^]]; )cpp",
      R"cpp( int abc, xyz = $C[[^abc^]]; )cpp",
      R"cpp(
        void foo() {
          [[if ([[1^11]]) $C[[{
            $C[[return]];
          }]] else [[{^
          }]]]]
          char z;
        }
      )cpp",
      R"cpp(
          template <class T>
          struct unique_ptr {};
          void foo(^$C[[unique_ptr<$C[[unique_ptr<$C[[int]]>]]>]]^ a) {}
      )cpp",
      R"cpp(int a = [[5 >^> 1]];)cpp",
      R"cpp(
        #define ECHO(X) X
        ECHO(EC^HO($C[[int]]) EC^HO(a));
      )cpp",
      R"cpp( $C[[^$C[[int]] a^]]; )cpp",
      R"cpp( $C[[^$C[[int]] a = $C[[5]]^]]; )cpp",
  };
  for (const char *C : Cases) {
    Annotations Test(C);
    auto AST = TestTU::withCode(Test.code()).build();
    auto T = makeSelectionTree(C, AST);

    std::vector<Range> Complete, Partial;
    for (const SelectionTree::Node *N : allNodes(T))
      if (N->Selected == SelectionTree::Complete)
        Complete.push_back(nodeRange(N, AST));
      else if (N->Selected == SelectionTree::Partial)
        Partial.push_back(nodeRange(N, AST));
    EXPECT_THAT(Complete, UnorderedElementsAreArray(Test.ranges("C"))) << C;
    EXPECT_THAT(Partial, UnorderedElementsAreArray(Test.ranges())) << C;
  }
}

TEST(SelectionTest, PathologicalPreprocessor) {
  const char *Case = R"cpp(
#define MACRO while(1)
    void test() {
#include "Expand.inc"
        br^eak;
    }
  )cpp";
  Annotations Test(Case);
  auto TU = TestTU::withCode(Test.code());
  TU.AdditionalFiles["Expand.inc"] = "MACRO\n";
  auto AST = TU.build();
  EXPECT_THAT(*AST.getDiagnostics(), ::testing::IsEmpty());
  auto T = makeSelectionTree(Case, AST);

  EXPECT_EQ("BreakStmt", T.commonAncestor()->kind());
  EXPECT_EQ("WhileStmt", T.commonAncestor()->Parent->kind());
}

TEST(SelectionTest, IncludedFile) {
  const char *Case = R"cpp(
    void test() {
#include "Exp^and.inc"
        break;
    }
  )cpp";
  Annotations Test(Case);
  auto TU = TestTU::withCode(Test.code());
  TU.AdditionalFiles["Expand.inc"] = "while(1)\n";
  auto AST = TU.build();
  auto T = makeSelectionTree(Case, AST);

  EXPECT_EQ(nullptr, T.commonAncestor());
}

TEST(SelectionTest, MacroArgExpansion) {
  // If a macro arg is expanded several times, we only consider the first one
  // selected.
  const char *Case = R"cpp(
    int mul(int, int);
    #define SQUARE(X) mul(X, X);
    int nine = SQUARE(^3);
  )cpp";
  Annotations Test(Case);
  auto AST = TestTU::withCode(Test.code()).build();
  auto T = makeSelectionTree(Case, AST);
  EXPECT_EQ("IntegerLiteral", T.commonAncestor()->kind());
  EXPECT_TRUE(T.commonAncestor()->Selected);

  // Verify that the common assert() macro doesn't suffer from this.
  // (This is because we don't associate the stringified token with the arg).
  Case = R"cpp(
    void die(const char*);
    #define assert(x) (x ? (void)0 : die(#x))
    void foo() { assert(^42); }
  )cpp";
  Test = Annotations(Case);
  AST = TestTU::withCode(Test.code()).build();
  T = makeSelectionTree(Case, AST);

  EXPECT_EQ("IntegerLiteral", T.commonAncestor()->kind());
}

TEST(SelectionTest, Implicit) {
  const char *Test = R"cpp(
    struct S { S(const char*); };
    int f(S);
    int x = f("^");
  )cpp";
  auto TU = TestTU::withCode(Annotations(Test).code());
  // C++14 AST contains some temporaries that C++17 elides.
  TU.ExtraArgs.push_back("-std=c++17");
  auto AST = TU.build();
  auto T = makeSelectionTree(Test, AST);

  const SelectionTree::Node *Str = T.commonAncestor();
  EXPECT_EQ("StringLiteral", nodeKind(Str)) << "Implicit selected?";
  EXPECT_EQ("ImplicitCastExpr", nodeKind(Str->Parent));
  EXPECT_EQ("CXXConstructExpr", nodeKind(Str->Parent->Parent));
  const SelectionTree::Node *ICE = Str->Parent->Parent->Parent;
  EXPECT_EQ("ImplicitCastExpr", nodeKind(ICE));
  EXPECT_EQ("CallExpr", nodeKind(ICE->Parent));
  EXPECT_EQ(Str, &ICE->ignoreImplicit())
      << "Didn't unwrap " << nodeKind(&ICE->ignoreImplicit());

  EXPECT_EQ(ICE, &Str->outerImplicit());
}

TEST(SelectionTest, CreateAll) {
  llvm::Annotations Test("int$unique^ a=1$ambiguous^+1; $empty^");
  auto AST = TestTU::withCode(Test.code()).build();
  unsigned Seen = 0;
  SelectionTree::createEach(
      AST.getASTContext(), AST.getTokens(), Test.point("ambiguous"),
      Test.point("ambiguous"), [&](SelectionTree T) {
        // Expect to see the right-biased tree first.
        if (Seen == 0) {
          EXPECT_EQ("BinaryOperator", nodeKind(T.commonAncestor()));
        } else if (Seen == 1) {
          EXPECT_EQ("IntegerLiteral", nodeKind(T.commonAncestor()));
        }
        ++Seen;
        return false;
      });
  EXPECT_EQ(2u, Seen);

  Seen = 0;
  SelectionTree::createEach(AST.getASTContext(), AST.getTokens(),
                            Test.point("ambiguous"), Test.point("ambiguous"),
                            [&](SelectionTree T) {
                              ++Seen;
                              return true;
                            });
  EXPECT_EQ(1u, Seen) << "Return true --> stop iterating";

  Seen = 0;
  SelectionTree::createEach(AST.getASTContext(), AST.getTokens(),
                            Test.point("unique"), Test.point("unique"),
                            [&](SelectionTree T) {
                              ++Seen;
                              return false;
                            });
  EXPECT_EQ(1u, Seen) << "no ambiguity --> only one tree";

  Seen = 0;
  SelectionTree::createEach(AST.getASTContext(), AST.getTokens(),
                            Test.point("empty"), Test.point("empty"),
                            [&](SelectionTree T) {
                              EXPECT_FALSE(T.commonAncestor());
                              ++Seen;
                              return false;
                            });
  EXPECT_EQ(1u, Seen) << "empty tree still created";

  Seen = 0;
  SelectionTree::createEach(AST.getASTContext(), AST.getTokens(),
                            Test.point("unique"), Test.point("ambiguous"),
                            [&](SelectionTree T) {
                              ++Seen;
                              return false;
                            });
  EXPECT_EQ(1u, Seen) << "one tree for nontrivial selection";
}

TEST(SelectionTest, DeclContextIsLexical) {
  llvm::Annotations Test("namespace a { void $1^foo(); } void a::$2^foo();");
  auto AST = TestTU::withCode(Test.code()).build();
  {
    auto ST = SelectionTree::createRight(AST.getASTContext(), AST.getTokens(),
                                         Test.point("1"), Test.point("1"));
    EXPECT_FALSE(ST.commonAncestor()->getDeclContext().isTranslationUnit());
  }
  {
    auto ST = SelectionTree::createRight(AST.getASTContext(), AST.getTokens(),
                                         Test.point("2"), Test.point("2"));
    EXPECT_TRUE(ST.commonAncestor()->getDeclContext().isTranslationUnit());
  }
}

} // namespace
} // namespace clangd
} // namespace clang
