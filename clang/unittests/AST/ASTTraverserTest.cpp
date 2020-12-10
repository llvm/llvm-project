//===- unittests/AST/ASTTraverserTest.h------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTNodeTraverser.h"
#include "clang/AST/TextNodeDumper.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang::tooling;
using namespace clang::ast_matchers;

namespace clang {

class NodeTreePrinter : public TextTreeStructure {
  llvm::raw_ostream &OS;

public:
  NodeTreePrinter(llvm::raw_ostream &OS)
      : TextTreeStructure(OS, /* showColors */ false), OS(OS) {}

  void Visit(const Decl *D) {
    OS << D->getDeclKindName() << "Decl";
    if (auto *ND = dyn_cast<NamedDecl>(D)) {
      OS << " '" << ND->getDeclName() << "'";
    }
  }

  void Visit(const Stmt *S) {
    if (!S) {
      OS << "<<<NULL>>>";
      return;
    }
    OS << S->getStmtClassName();
    if (auto *E = dyn_cast<DeclRefExpr>(S)) {
      OS << " '" << E->getDecl()->getDeclName() << "'";
    }
  }

  void Visit(QualType QT) {
    OS << "QualType " << QT.split().Quals.getAsString();
  }

  void Visit(const Type *T) { OS << T->getTypeClassName() << "Type"; }

  void Visit(const comments::Comment *C, const comments::FullComment *FC) {
    OS << C->getCommentKindName();
  }

  void Visit(const CXXCtorInitializer *Init) {
    OS << "CXXCtorInitializer";
    if (const auto *F = Init->getAnyMember()) {
      OS << " '" << F->getNameAsString() << "'";
    } else if (auto const *TSI = Init->getTypeSourceInfo()) {
      OS << " '" << TSI->getType().getAsString() << "'";
    }
  }

  void Visit(const Attr *A) {
    switch (A->getKind()) {
#define ATTR(X)                                                                \
  case attr::X:                                                                \
    OS << #X;                                                                  \
    break;
#include "clang/Basic/AttrList.inc"
    }
    OS << "Attr";
  }

  void Visit(const OMPClause *C) { OS << "OMPClause"; }
  void Visit(const TemplateArgument &A, SourceRange R = {},
             const Decl *From = nullptr, const char *Label = nullptr) {
    OS << "TemplateArgument";
    switch (A.getKind()) {
    case TemplateArgument::Type: {
      OS << " type " << A.getAsType().getAsString();
      break;
    }
    default:
      break;
    }
  }

  template <typename... T> void Visit(T...) {}
};

class TestASTDumper : public ASTNodeTraverser<TestASTDumper, NodeTreePrinter> {

  NodeTreePrinter MyNodeRecorder;

public:
  TestASTDumper(llvm::raw_ostream &OS) : MyNodeRecorder(OS) {}
  NodeTreePrinter &doGetNodeDelegate() { return MyNodeRecorder; }
};

template <typename... NodeType> std::string dumpASTString(NodeType &&... N) {
  std::string Buffer;
  llvm::raw_string_ostream OS(Buffer);

  TestASTDumper Dumper(OS);

  OS << "\n";

  Dumper.Visit(std::forward<NodeType &&>(N)...);

  return OS.str();
}

template <typename... NodeType>
std::string dumpASTString(TraversalKind TK, NodeType &&... N) {
  std::string Buffer;
  llvm::raw_string_ostream OS(Buffer);

  TestASTDumper Dumper(OS);
  Dumper.SetTraversalKind(TK);

  OS << "\n";

  Dumper.Visit(std::forward<NodeType &&>(N)...);

  return OS.str();
}

const FunctionDecl *getFunctionNode(clang::ASTUnit *AST,
                                    const std::string &Name) {
  auto Result = ast_matchers::match(functionDecl(hasName(Name)).bind("fn"),
                                    AST->getASTContext());
  EXPECT_EQ(Result.size(), 1u);
  return Result[0].getNodeAs<FunctionDecl>("fn");
}

template <typename T> struct Verifier {
  static void withDynNode(T Node, const std::string &DumpString) {
    EXPECT_EQ(dumpASTString(DynTypedNode::create(Node)), DumpString);
  }
};

template <typename T> struct Verifier<T *> {
  static void withDynNode(T *Node, const std::string &DumpString) {
    EXPECT_EQ(dumpASTString(DynTypedNode::create(*Node)), DumpString);
  }
};

template <typename T>
void verifyWithDynNode(T Node, const std::string &DumpString) {
  EXPECT_EQ(dumpASTString(Node), DumpString);

  Verifier<T>::withDynNode(Node, DumpString);
}

TEST(Traverse, Dump) {

  auto AST = buildASTFromCode(R"cpp(
struct A {
  int m_number;

  /// CTor
  A() : m_number(42) {}

  [[nodiscard]] const int func() {
    return 42;
  }

};

template<typename T>
struct templ
{
};

template<>
struct templ<int>
{
};

void parmvardecl_attr(struct A __attribute__((address_space(19)))*);

)cpp");

  const FunctionDecl *Func = getFunctionNode(AST.get(), "func");

  verifyWithDynNode(Func,
                    R"cpp(
CXXMethodDecl 'func'
|-CompoundStmt
| `-ReturnStmt
|   `-IntegerLiteral
`-WarnUnusedResultAttr
)cpp");

  Stmt *Body = Func->getBody();

  verifyWithDynNode(Body,
                    R"cpp(
CompoundStmt
`-ReturnStmt
  `-IntegerLiteral
)cpp");

  QualType QT = Func->getType();

  verifyWithDynNode(QT,
                    R"cpp(
FunctionProtoType
`-QualType const
  `-BuiltinType
)cpp");

  const FunctionDecl *CTorFunc = getFunctionNode(AST.get(), "A");

  verifyWithDynNode(CTorFunc->getType(),
                    R"cpp(
FunctionProtoType
`-BuiltinType
)cpp");

  Attr *A = *Func->attr_begin();

  {
    std::string expectedString = R"cpp(
WarnUnusedResultAttr
)cpp";

    EXPECT_EQ(dumpASTString(A), expectedString);
  }

  auto *CTor = dyn_cast<CXXConstructorDecl>(CTorFunc);
  const CXXCtorInitializer *Init = *CTor->init_begin();

  verifyWithDynNode(Init,
                    R"cpp(
CXXCtorInitializer 'm_number'
`-IntegerLiteral
)cpp");

  const comments::FullComment *Comment =
      AST->getASTContext().getLocalCommentForDeclUncached(CTorFunc);
  {
    std::string expectedString = R"cpp(
FullComment
`-ParagraphComment
  `-TextComment
)cpp";
    EXPECT_EQ(dumpASTString(Comment, Comment), expectedString);
  }

  auto Result = ast_matchers::match(
      classTemplateSpecializationDecl(hasName("templ")).bind("fn"),
      AST->getASTContext());
  EXPECT_EQ(Result.size(), 1u);
  auto Templ = Result[0].getNodeAs<ClassTemplateSpecializationDecl>("fn");

  TemplateArgument TA = Templ->getTemplateArgs()[0];

  verifyWithDynNode(TA,
                    R"cpp(
TemplateArgument type int
`-BuiltinType
)cpp");

  Func = getFunctionNode(AST.get(), "parmvardecl_attr");

  const auto *Parm = Func->getParamDecl(0);
  const auto TL = Parm->getTypeSourceInfo()->getTypeLoc();
  ASSERT_TRUE(TL.getType()->isPointerType());

  const auto ATL = TL.getNextTypeLoc().getAs<AttributedTypeLoc>();
  const auto *AS = cast<AddressSpaceAttr>(ATL.getAttr());
  EXPECT_EQ(toTargetAddressSpace(static_cast<LangAS>(AS->getAddressSpace())),
            19u);
}

TEST(Traverse, IgnoreUnlessSpelledInSourceVars) {

  auto AST = buildASTFromCode(R"cpp(

struct String
{
    String(const char*, int = -1) {}

    int overloaded() const;
    int& overloaded();
};

void stringConstruct()
{
    String s = "foo";
    s = "bar";
}

void overloadCall()
{
   String s = "foo";
   (s).overloaded();
}

struct C1 {};
struct C2 { operator C1(); };

void conversionOperator()
{
    C2* c2;
    C1 c1 = (*c2);
}

template <unsigned alignment>
void template_test() {
  static_assert(alignment, "");
}
void actual_template_test() {
  template_test<4>();
}

struct OneParamCtor {
  explicit OneParamCtor(int);
};
struct TwoParamCtor {
  explicit TwoParamCtor(int, int);
};

void varDeclCtors() {
  {
  auto var1 = OneParamCtor(5);
  auto var2 = TwoParamCtor(6, 7);
  }
  {
  OneParamCtor var3(5);
  TwoParamCtor var4(6, 7);
  }
  int i = 0;
  {
  auto var5 = OneParamCtor(i);
  auto var6 = TwoParamCtor(i, 7);
  }
  {
  OneParamCtor var7(i);
  TwoParamCtor var8(i, 7);
  }
}

)cpp");

  {
    auto FN =
        ast_matchers::match(functionDecl(hasName("stringConstruct")).bind("fn"),
                            AST->getASTContext());
    EXPECT_EQ(FN.size(), 1u);

    EXPECT_EQ(dumpASTString(TK_AsIs, FN[0].getNodeAs<Decl>("fn")),
              R"cpp(
FunctionDecl 'stringConstruct'
`-CompoundStmt
  |-DeclStmt
  | `-VarDecl 's'
  |   `-ExprWithCleanups
  |     `-CXXConstructExpr
  |       `-MaterializeTemporaryExpr
  |         `-ImplicitCastExpr
  |           `-CXXConstructExpr
  |             |-ImplicitCastExpr
  |             | `-StringLiteral
  |             `-CXXDefaultArgExpr
  `-ExprWithCleanups
    `-CXXOperatorCallExpr
      |-ImplicitCastExpr
      | `-DeclRefExpr 'operator='
      |-DeclRefExpr 's'
      `-MaterializeTemporaryExpr
        `-CXXConstructExpr
          |-ImplicitCastExpr
          | `-StringLiteral
          `-CXXDefaultArgExpr
)cpp");

    EXPECT_EQ(dumpASTString(TK_IgnoreUnlessSpelledInSource,
                            FN[0].getNodeAs<Decl>("fn")),
              R"cpp(
FunctionDecl 'stringConstruct'
`-CompoundStmt
  |-DeclStmt
  | `-VarDecl 's'
  |   `-StringLiteral
  `-CXXOperatorCallExpr
    |-DeclRefExpr 'operator='
    |-DeclRefExpr 's'
    `-StringLiteral
)cpp");
  }

  {
    auto FN =
        ast_matchers::match(functionDecl(hasName("overloadCall")).bind("fn"),
                            AST->getASTContext());
    EXPECT_EQ(FN.size(), 1u);

    EXPECT_EQ(dumpASTString(TK_AsIs, FN[0].getNodeAs<Decl>("fn")),
              R"cpp(
FunctionDecl 'overloadCall'
`-CompoundStmt
  |-DeclStmt
  | `-VarDecl 's'
  |   `-ExprWithCleanups
  |     `-CXXConstructExpr
  |       `-MaterializeTemporaryExpr
  |         `-ImplicitCastExpr
  |           `-CXXConstructExpr
  |             |-ImplicitCastExpr
  |             | `-StringLiteral
  |             `-CXXDefaultArgExpr
  `-CXXMemberCallExpr
    `-MemberExpr
      `-ParenExpr
        `-DeclRefExpr 's'
)cpp");

    EXPECT_EQ(dumpASTString(TK_IgnoreUnlessSpelledInSource,
                            FN[0].getNodeAs<Decl>("fn")),
              R"cpp(
FunctionDecl 'overloadCall'
`-CompoundStmt
  |-DeclStmt
  | `-VarDecl 's'
  |   `-StringLiteral
  `-CXXMemberCallExpr
    `-MemberExpr
      `-DeclRefExpr 's'
)cpp");
  }

  {
    auto FN = ast_matchers::match(
        functionDecl(hasName("conversionOperator"),
                     hasDescendant(varDecl(hasName("c1")).bind("var"))),
        AST->getASTContext());
    EXPECT_EQ(FN.size(), 1u);

    EXPECT_EQ(dumpASTString(TK_AsIs, FN[0].getNodeAs<Decl>("var")),
              R"cpp(
VarDecl 'c1'
`-ExprWithCleanups
  `-CXXConstructExpr
    `-MaterializeTemporaryExpr
      `-ImplicitCastExpr
        `-CXXMemberCallExpr
          `-MemberExpr
            `-ParenExpr
              `-UnaryOperator
                `-ImplicitCastExpr
                  `-DeclRefExpr 'c2'
)cpp");

    EXPECT_EQ(dumpASTString(TK_IgnoreUnlessSpelledInSource,
                            FN[0].getNodeAs<Decl>("var")),
              R"cpp(
VarDecl 'c1'
`-UnaryOperator
  `-DeclRefExpr 'c2'
)cpp");
  }

  {
    auto FN = ast_matchers::match(
        functionDecl(hasName("template_test"),
                     hasDescendant(staticAssertDecl().bind("staticAssert"))),
        AST->getASTContext());
    EXPECT_EQ(FN.size(), 2u);

    EXPECT_EQ(dumpASTString(TK_AsIs, FN[1].getNodeAs<Decl>("staticAssert")),
              R"cpp(
StaticAssertDecl
|-ImplicitCastExpr
| `-SubstNonTypeTemplateParmExpr
|   |-NonTypeTemplateParmDecl 'alignment'
|   `-IntegerLiteral
`-StringLiteral
)cpp");

    EXPECT_EQ(dumpASTString(TK_IgnoreUnlessSpelledInSource,
                            FN[1].getNodeAs<Decl>("staticAssert")),
              R"cpp(
StaticAssertDecl
|-IntegerLiteral
`-StringLiteral
)cpp");
  }

  auto varChecker = [&AST](StringRef varName, StringRef SemanticDump,
                           StringRef SyntacticDump) {
    auto FN = ast_matchers::match(
        functionDecl(
            hasName("varDeclCtors"),
            forEachDescendant(varDecl(hasName(varName)).bind("varDeclCtor"))),
        AST->getASTContext());
    EXPECT_EQ(FN.size(), 1u);

    EXPECT_EQ(dumpASTString(TK_AsIs, FN[0].getNodeAs<Decl>("varDeclCtor")),
              SemanticDump);

    EXPECT_EQ(dumpASTString(TK_IgnoreUnlessSpelledInSource,
                            FN[0].getNodeAs<Decl>("varDeclCtor")),
              SyntacticDump);
  };

  varChecker("var1",
             R"cpp(
VarDecl 'var1'
`-ExprWithCleanups
  `-CXXConstructExpr
    `-MaterializeTemporaryExpr
      `-CXXFunctionalCastExpr
        `-CXXConstructExpr
          `-IntegerLiteral
)cpp",
             R"cpp(
VarDecl 'var1'
`-CXXConstructExpr
  `-IntegerLiteral
)cpp");

  varChecker("var2",
             R"cpp(
VarDecl 'var2'
`-ExprWithCleanups
  `-CXXConstructExpr
    `-MaterializeTemporaryExpr
      `-CXXTemporaryObjectExpr
        |-IntegerLiteral
        `-IntegerLiteral
)cpp",
             R"cpp(
VarDecl 'var2'
`-CXXTemporaryObjectExpr
  |-IntegerLiteral
  `-IntegerLiteral
)cpp");

  varChecker("var3",
             R"cpp(
VarDecl 'var3'
`-CXXConstructExpr
  `-IntegerLiteral
)cpp",
             R"cpp(
VarDecl 'var3'
`-CXXConstructExpr
  `-IntegerLiteral
)cpp");

  varChecker("var4",
             R"cpp(
VarDecl 'var4'
`-CXXConstructExpr
  |-IntegerLiteral
  `-IntegerLiteral
)cpp",
             R"cpp(
VarDecl 'var4'
`-CXXConstructExpr
  |-IntegerLiteral
  `-IntegerLiteral
)cpp");

  varChecker("var5",
             R"cpp(
VarDecl 'var5'
`-ExprWithCleanups
  `-CXXConstructExpr
    `-MaterializeTemporaryExpr
      `-CXXFunctionalCastExpr
        `-CXXConstructExpr
          `-ImplicitCastExpr
            `-DeclRefExpr 'i'
)cpp",
             R"cpp(
VarDecl 'var5'
`-CXXConstructExpr
  `-DeclRefExpr 'i'
)cpp");

  varChecker("var6",
             R"cpp(
VarDecl 'var6'
`-ExprWithCleanups
  `-CXXConstructExpr
    `-MaterializeTemporaryExpr
      `-CXXTemporaryObjectExpr
        |-ImplicitCastExpr
        | `-DeclRefExpr 'i'
        `-IntegerLiteral
)cpp",
             R"cpp(
VarDecl 'var6'
`-CXXTemporaryObjectExpr
  |-DeclRefExpr 'i'
  `-IntegerLiteral
)cpp");

  varChecker("var7",
             R"cpp(
VarDecl 'var7'
`-CXXConstructExpr
  `-ImplicitCastExpr
    `-DeclRefExpr 'i'
)cpp",
             R"cpp(
VarDecl 'var7'
`-CXXConstructExpr
  `-DeclRefExpr 'i'
)cpp");

  varChecker("var8",
             R"cpp(
VarDecl 'var8'
`-CXXConstructExpr
  |-ImplicitCastExpr
  | `-DeclRefExpr 'i'
  `-IntegerLiteral
)cpp",
             R"cpp(
VarDecl 'var8'
`-CXXConstructExpr
  |-DeclRefExpr 'i'
  `-IntegerLiteral
)cpp");
}

TEST(Traverse, IgnoreUnlessSpelledInSourceStructs) {
  auto AST = buildASTFromCode(R"cpp(

struct MyStruct {
  MyStruct();
  MyStruct(int i) {
    MyStruct();
  }
  ~MyStruct();
};

)cpp");

  auto BN = ast_matchers::match(
      cxxConstructorDecl(hasName("MyStruct"),
                         hasParameter(0, parmVarDecl(hasType(isInteger()))))
          .bind("ctor"),
      AST->getASTContext());
  EXPECT_EQ(BN.size(), 1u);

  EXPECT_EQ(dumpASTString(TK_IgnoreUnlessSpelledInSource,
                          BN[0].getNodeAs<Decl>("ctor")),
            R"cpp(
CXXConstructorDecl 'MyStruct'
|-ParmVarDecl 'i'
`-CompoundStmt
  `-CXXTemporaryObjectExpr
)cpp");

  EXPECT_EQ(dumpASTString(TK_AsIs, BN[0].getNodeAs<Decl>("ctor")),
            R"cpp(
CXXConstructorDecl 'MyStruct'
|-ParmVarDecl 'i'
`-CompoundStmt
  `-ExprWithCleanups
    `-CXXBindTemporaryExpr
      `-CXXTemporaryObjectExpr
)cpp");
}

TEST(Traverse, IgnoreUnlessSpelledInSourceReturnStruct) {

  auto AST = buildASTFromCode(R"cpp(
struct Retval {
  Retval() {}
  ~Retval() {}
};

Retval someFun();

void foo()
{
    someFun();
}
)cpp");

  auto BN = ast_matchers::match(functionDecl(hasName("foo")).bind("fn"),
                                AST->getASTContext());
  EXPECT_EQ(BN.size(), 1u);

  EXPECT_EQ(dumpASTString(TK_IgnoreUnlessSpelledInSource,
                          BN[0].getNodeAs<Decl>("fn")),
            R"cpp(
FunctionDecl 'foo'
`-CompoundStmt
  `-CallExpr
    `-DeclRefExpr 'someFun'
)cpp");

  EXPECT_EQ(dumpASTString(TK_AsIs, BN[0].getNodeAs<Decl>("fn")),
            R"cpp(
FunctionDecl 'foo'
`-CompoundStmt
  `-ExprWithCleanups
    `-CXXBindTemporaryExpr
      `-CallExpr
        `-ImplicitCastExpr
          `-DeclRefExpr 'someFun'
)cpp");
}

TEST(Traverse, IgnoreUnlessSpelledInSourceReturns) {

  auto AST = buildASTFromCode(R"cpp(

struct A
{
};

struct B
{
  B(int);
  B(A const& a);
  B();
};

struct C
{
  operator B();
};

B func1() {
  return 42;
}

B func2() {
  return B{42};
}

B func3() {
  return B(42);
}

B func4() {
  return B();
}

B func5() {
  return B{};
}

B func6() {
  return C();
}

B func7() {
  return A();
}

B func8() {
  return C{};
}

B func9() {
  return A{};
}

B func10() {
  A a;
  return a;
}

B func11() {
  B b;
  return b;
}

B func12() {
  C c;
  return c;
}

)cpp");

  auto getFunctionNode = [&AST](const std::string &name) {
    auto BN = ast_matchers::match(functionDecl(hasName(name)).bind("fn"),
                                  AST->getASTContext());
    EXPECT_EQ(BN.size(), 1u);
    return BN[0].getNodeAs<Decl>("fn");
  };

  {
    auto FN = getFunctionNode("func1");
    llvm::StringRef Expected = R"cpp(
FunctionDecl 'func1'
`-CompoundStmt
  `-ReturnStmt
    `-ExprWithCleanups
      `-CXXConstructExpr
        `-MaterializeTemporaryExpr
          `-ImplicitCastExpr
            `-CXXConstructExpr
              `-IntegerLiteral
)cpp";

    EXPECT_EQ(dumpASTString(TK_AsIs, FN), Expected);

    Expected = R"cpp(
FunctionDecl 'func1'
`-CompoundStmt
  `-ReturnStmt
    `-IntegerLiteral
)cpp";
    EXPECT_EQ(dumpASTString(TK_IgnoreUnlessSpelledInSource, FN), Expected);
  }

  llvm::StringRef Expected = R"cpp(
FunctionDecl 'func2'
`-CompoundStmt
  `-ReturnStmt
    `-CXXTemporaryObjectExpr
      `-IntegerLiteral
)cpp";
  EXPECT_EQ(
      dumpASTString(TK_IgnoreUnlessSpelledInSource, getFunctionNode("func2")),
      Expected);

  Expected = R"cpp(
FunctionDecl 'func3'
`-CompoundStmt
  `-ReturnStmt
    `-CXXConstructExpr
      `-IntegerLiteral
)cpp";
  EXPECT_EQ(
      dumpASTString(TK_IgnoreUnlessSpelledInSource, getFunctionNode("func3")),
      Expected);

  Expected = R"cpp(
FunctionDecl 'func4'
`-CompoundStmt
  `-ReturnStmt
    `-CXXTemporaryObjectExpr
)cpp";
  EXPECT_EQ(
      dumpASTString(TK_IgnoreUnlessSpelledInSource, getFunctionNode("func4")),
      Expected);

  Expected = R"cpp(
FunctionDecl 'func5'
`-CompoundStmt
  `-ReturnStmt
    `-CXXTemporaryObjectExpr
)cpp";
  EXPECT_EQ(
      dumpASTString(TK_IgnoreUnlessSpelledInSource, getFunctionNode("func5")),
      Expected);

  Expected = R"cpp(
FunctionDecl 'func6'
`-CompoundStmt
  `-ReturnStmt
    `-CXXTemporaryObjectExpr
)cpp";
  EXPECT_EQ(
      dumpASTString(TK_IgnoreUnlessSpelledInSource, getFunctionNode("func6")),
      Expected);

  Expected = R"cpp(
FunctionDecl 'func7'
`-CompoundStmt
  `-ReturnStmt
    `-CXXTemporaryObjectExpr
)cpp";
  EXPECT_EQ(
      dumpASTString(TK_IgnoreUnlessSpelledInSource, getFunctionNode("func7")),
      Expected);

  Expected = R"cpp(
FunctionDecl 'func8'
`-CompoundStmt
  `-ReturnStmt
    `-CXXFunctionalCastExpr
      `-InitListExpr
)cpp";
  EXPECT_EQ(
      dumpASTString(TK_IgnoreUnlessSpelledInSource, getFunctionNode("func8")),
      Expected);

  Expected = R"cpp(
FunctionDecl 'func9'
`-CompoundStmt
  `-ReturnStmt
    `-CXXFunctionalCastExpr
      `-InitListExpr
)cpp";
  EXPECT_EQ(
      dumpASTString(TK_IgnoreUnlessSpelledInSource, getFunctionNode("func9")),
      Expected);

  Expected = R"cpp(
FunctionDecl 'func10'
`-CompoundStmt
  |-DeclStmt
  | `-VarDecl 'a'
  |   `-CXXConstructExpr
  `-ReturnStmt
    `-DeclRefExpr 'a'
)cpp";
  EXPECT_EQ(
      dumpASTString(TK_IgnoreUnlessSpelledInSource, getFunctionNode("func10")),
      Expected);

  Expected = R"cpp(
FunctionDecl 'func11'
`-CompoundStmt
  |-DeclStmt
  | `-VarDecl 'b'
  |   `-CXXConstructExpr
  `-ReturnStmt
    `-DeclRefExpr 'b'
)cpp";
  EXPECT_EQ(
      dumpASTString(TK_IgnoreUnlessSpelledInSource, getFunctionNode("func11")),
      Expected);

  Expected = R"cpp(
FunctionDecl 'func12'
`-CompoundStmt
  |-DeclStmt
  | `-VarDecl 'c'
  |   `-CXXConstructExpr
  `-ReturnStmt
    `-DeclRefExpr 'c'
)cpp";
  EXPECT_EQ(
      dumpASTString(TK_IgnoreUnlessSpelledInSource, getFunctionNode("func12")),
      Expected);
}

TEST(Traverse, LambdaUnlessSpelledInSource) {

  auto AST =
      buildASTFromCodeWithArgs(R"cpp(

void captures() {
  int a = 0;
  int b = 0;
  int d = 0;
  int f = 0;

  [a, &b, c = d, &e = f](int g, int h = 42) {};
}

void templated() {
  int a = 0;
  [a]<typename T>(T t) {};
}

struct SomeStruct {
    int a = 0;
    void capture_this() {
        [this]() {};
    }
    void capture_this_copy() {
        [self = *this]() {};
    }
};
)cpp",
                               {"-Wno-unused-value", "-Wno-c++2a-extensions"});

  auto getLambdaNode = [&AST](const std::string &name) {
    auto BN = ast_matchers::match(
        lambdaExpr(hasAncestor(functionDecl(hasName(name)))).bind("lambda"),
        AST->getASTContext());
    EXPECT_EQ(BN.size(), 1u);
    return BN[0].getNodeAs<LambdaExpr>("lambda");
  };

  {
    auto L = getLambdaNode("captures");

    llvm::StringRef Expected = R"cpp(
LambdaExpr
|-DeclRefExpr 'a'
|-DeclRefExpr 'b'
|-VarDecl 'c'
| `-DeclRefExpr 'd'
|-VarDecl 'e'
| `-DeclRefExpr 'f'
|-ParmVarDecl 'g'
|-ParmVarDecl 'h'
| `-IntegerLiteral
`-CompoundStmt
)cpp";
    EXPECT_EQ(dumpASTString(TK_IgnoreUnlessSpelledInSource, L), Expected);

    Expected = R"cpp(
LambdaExpr
|-CXXRecordDecl ''
| |-CXXMethodDecl 'operator()'
| | |-ParmVarDecl 'g'
| | |-ParmVarDecl 'h'
| | | `-IntegerLiteral
| | `-CompoundStmt
| |-FieldDecl ''
| |-FieldDecl ''
| |-FieldDecl ''
| |-FieldDecl ''
| `-CXXDestructorDecl '~'
|-ImplicitCastExpr
| `-DeclRefExpr 'a'
|-DeclRefExpr 'b'
|-ImplicitCastExpr
| `-DeclRefExpr 'd'
|-DeclRefExpr 'f'
`-CompoundStmt
)cpp";
    EXPECT_EQ(dumpASTString(TK_AsIs, L), Expected);
  }

  {
    auto L = getLambdaNode("templated");

    llvm::StringRef Expected = R"cpp(
LambdaExpr
|-DeclRefExpr 'a'
|-TemplateTypeParmDecl 'T'
|-ParmVarDecl 't'
`-CompoundStmt
)cpp";
    EXPECT_EQ(dumpASTString(TK_IgnoreUnlessSpelledInSource, L), Expected);
  }

  {
    auto L = getLambdaNode("capture_this");

    llvm::StringRef Expected = R"cpp(
LambdaExpr
|-CXXThisExpr
`-CompoundStmt
)cpp";
    EXPECT_EQ(dumpASTString(TK_IgnoreUnlessSpelledInSource, L), Expected);
  }

  {
    auto L = getLambdaNode("capture_this_copy");

    llvm::StringRef Expected = R"cpp(
LambdaExpr
|-VarDecl 'self'
| `-UnaryOperator
|   `-CXXThisExpr
`-CompoundStmt
)cpp";
    EXPECT_EQ(dumpASTString(TK_IgnoreUnlessSpelledInSource, L), Expected);
  }
}

TEST(Traverse, IgnoreUnlessSpelledInSourceImplicit) {
  {
    auto AST = buildASTFromCode(R"cpp(
int i = 0;
)cpp");
    const auto *TUDecl = AST->getASTContext().getTranslationUnitDecl();

    EXPECT_EQ(dumpASTString(TK_IgnoreUnlessSpelledInSource, TUDecl),
              R"cpp(
TranslationUnitDecl
`-VarDecl 'i'
  `-IntegerLiteral
)cpp");
  }

  auto AST2 = buildASTFromCodeWithArgs(R"cpp(
struct Simple {
};
struct Other {
};

struct Record : Simple, Other {
  Record() : Simple(), m_i(42) {}
private:
  int m_i;
  int m_i2 = 42;
  Simple m_s;
};

struct NonTrivial {
    NonTrivial() {}
    NonTrivial(NonTrivial&) {}
    NonTrivial& operator=(NonTrivial&) { return *this; }

    ~NonTrivial() {}
};

struct ContainsArray {
    NonTrivial arr[2];
    int irr[2];
    ContainsArray& operator=(ContainsArray &) = default;
};

void copyIt()
{
    ContainsArray ca;
    ContainsArray ca2;
    ca2 = ca;
}

void forLoop()
{
    int arr[2];
    for (auto i : arr)
    {

    }
    for (auto& a = arr; auto i : a)
    {

    }
}

struct DefaultedAndDeleted {
  NonTrivial nt;
  DefaultedAndDeleted() = default;
  ~DefaultedAndDeleted() = default;
  DefaultedAndDeleted(DefaultedAndDeleted &) = default;
  DefaultedAndDeleted& operator=(DefaultedAndDeleted &) = default;
  DefaultedAndDeleted(DefaultedAndDeleted &&) = delete;
  DefaultedAndDeleted& operator=(DefaultedAndDeleted &&) = delete;
};

void copyIt2()
{
    DefaultedAndDeleted ca;
    DefaultedAndDeleted ca2;
    ca2 = ca;
}

void hasDefaultArg(int i, int j = 0)
{
}
void callDefaultArg()
{
  hasDefaultArg(42);
}
)cpp",
                                       {"-std=c++20"});

  {
    auto BN = ast_matchers::match(
        cxxRecordDecl(hasName("Record"), unless(isImplicit())).bind("rec"),
        AST2->getASTContext());
    EXPECT_EQ(BN.size(), 1u);

    EXPECT_EQ(dumpASTString(TK_AsIs, BN[0].getNodeAs<Decl>("rec")),
              R"cpp(
CXXRecordDecl 'Record'
|-CXXRecordDecl 'Record'
|-CXXConstructorDecl 'Record'
| |-CXXCtorInitializer 'struct Simple'
| | `-CXXConstructExpr
| |-CXXCtorInitializer 'struct Other'
| | `-CXXConstructExpr
| |-CXXCtorInitializer 'm_i'
| | `-IntegerLiteral
| |-CXXCtorInitializer 'm_i2'
| | `-CXXDefaultInitExpr
| |-CXXCtorInitializer 'm_s'
| | `-CXXConstructExpr
| `-CompoundStmt
|-AccessSpecDecl
|-FieldDecl 'm_i'
|-FieldDecl 'm_i2'
| `-IntegerLiteral
`-FieldDecl 'm_s'
)cpp");

    EXPECT_EQ(dumpASTString(TK_IgnoreUnlessSpelledInSource,
                            BN[0].getNodeAs<Decl>("rec")),
              R"cpp(
CXXRecordDecl 'Record'
|-CXXConstructorDecl 'Record'
| |-CXXCtorInitializer 'struct Simple'
| | `-CXXConstructExpr
| |-CXXCtorInitializer 'm_i'
| | `-IntegerLiteral
| `-CompoundStmt
|-AccessSpecDecl
|-FieldDecl 'm_i'
|-FieldDecl 'm_i2'
| `-IntegerLiteral
`-FieldDecl 'm_s'
)cpp");
  }
  {
    auto BN = ast_matchers::match(
        cxxRecordDecl(hasName("ContainsArray"), unless(isImplicit()))
            .bind("rec"),
        AST2->getASTContext());
    EXPECT_EQ(BN.size(), 1u);

    EXPECT_EQ(dumpASTString(TK_AsIs, BN[0].getNodeAs<Decl>("rec")),
              R"cpp(
CXXRecordDecl 'ContainsArray'
|-CXXRecordDecl 'ContainsArray'
|-FieldDecl 'arr'
|-FieldDecl 'irr'
|-CXXMethodDecl 'operator='
| |-ParmVarDecl ''
| `-CompoundStmt
|   |-ForStmt
|   | |-DeclStmt
|   | | `-VarDecl '__i0'
|   | |   `-IntegerLiteral
|   | |-<<<NULL>>>
|   | |-BinaryOperator
|   | | |-ImplicitCastExpr
|   | | | `-DeclRefExpr '__i0'
|   | | `-IntegerLiteral
|   | |-UnaryOperator
|   | | `-DeclRefExpr '__i0'
|   | `-CXXMemberCallExpr
|   |   |-MemberExpr
|   |   | `-ArraySubscriptExpr
|   |   |   |-ImplicitCastExpr
|   |   |   | `-MemberExpr
|   |   |   |   `-CXXThisExpr
|   |   |   `-ImplicitCastExpr
|   |   |     `-DeclRefExpr '__i0'
|   |   `-ArraySubscriptExpr
|   |     |-ImplicitCastExpr
|   |     | `-MemberExpr
|   |     |   `-DeclRefExpr ''
|   |     `-ImplicitCastExpr
|   |       `-DeclRefExpr '__i0'
|   |-CallExpr
|   | |-ImplicitCastExpr
|   | | `-DeclRefExpr '__builtin_memcpy'
|   | |-ImplicitCastExpr
|   | | `-UnaryOperator
|   | |   `-MemberExpr
|   | |     `-CXXThisExpr
|   | |-ImplicitCastExpr
|   | | `-UnaryOperator
|   | |   `-MemberExpr
|   | |     `-DeclRefExpr ''
|   | `-IntegerLiteral
|   `-ReturnStmt
|     `-UnaryOperator
|       `-CXXThisExpr
|-CXXConstructorDecl 'ContainsArray'
| `-ParmVarDecl ''
|-CXXDestructorDecl '~ContainsArray'
| `-CompoundStmt
`-CXXConstructorDecl 'ContainsArray'
  |-CXXCtorInitializer 'arr'
  | `-CXXConstructExpr
  `-CompoundStmt
)cpp");

    EXPECT_EQ(dumpASTString(TK_IgnoreUnlessSpelledInSource,
                            BN[0].getNodeAs<Decl>("rec")),
              R"cpp(
CXXRecordDecl 'ContainsArray'
|-FieldDecl 'arr'
|-FieldDecl 'irr'
`-CXXMethodDecl 'operator='
  `-ParmVarDecl ''
)cpp");
  }
  {
    auto BN = ast_matchers::match(functionDecl(hasName("forLoop")).bind("func"),
                                  AST2->getASTContext());
    EXPECT_EQ(BN.size(), 1u);

    EXPECT_EQ(dumpASTString(TK_AsIs, BN[0].getNodeAs<Decl>("func")),
              R"cpp(
FunctionDecl 'forLoop'
`-CompoundStmt
  |-DeclStmt
  | `-VarDecl 'arr'
  |-CXXForRangeStmt
  | |-<<<NULL>>>
  | |-DeclStmt
  | | `-VarDecl '__range1'
  | |   `-DeclRefExpr 'arr'
  | |-DeclStmt
  | | `-VarDecl '__begin1'
  | |   `-ImplicitCastExpr
  | |     `-DeclRefExpr '__range1'
  | |-DeclStmt
  | | `-VarDecl '__end1'
  | |   `-BinaryOperator
  | |     |-ImplicitCastExpr
  | |     | `-DeclRefExpr '__range1'
  | |     `-IntegerLiteral
  | |-BinaryOperator
  | | |-ImplicitCastExpr
  | | | `-DeclRefExpr '__begin1'
  | | `-ImplicitCastExpr
  | |   `-DeclRefExpr '__end1'
  | |-UnaryOperator
  | | `-DeclRefExpr '__begin1'
  | |-DeclStmt
  | | `-VarDecl 'i'
  | |   `-ImplicitCastExpr
  | |     `-UnaryOperator
  | |       `-ImplicitCastExpr
  | |         `-DeclRefExpr '__begin1'
  | `-CompoundStmt
  `-CXXForRangeStmt
    |-DeclStmt
    | `-VarDecl 'a'
    |   `-DeclRefExpr 'arr'
    |-DeclStmt
    | `-VarDecl '__range1'
    |   `-DeclRefExpr 'a'
    |-DeclStmt
    | `-VarDecl '__begin1'
    |   `-ImplicitCastExpr
    |     `-DeclRefExpr '__range1'
    |-DeclStmt
    | `-VarDecl '__end1'
    |   `-BinaryOperator
    |     |-ImplicitCastExpr
    |     | `-DeclRefExpr '__range1'
    |     `-IntegerLiteral
    |-BinaryOperator
    | |-ImplicitCastExpr
    | | `-DeclRefExpr '__begin1'
    | `-ImplicitCastExpr
    |   `-DeclRefExpr '__end1'
    |-UnaryOperator
    | `-DeclRefExpr '__begin1'
    |-DeclStmt
    | `-VarDecl 'i'
    |   `-ImplicitCastExpr
    |     `-UnaryOperator
    |       `-ImplicitCastExpr
    |         `-DeclRefExpr '__begin1'
    `-CompoundStmt
)cpp");

    EXPECT_EQ(dumpASTString(TK_IgnoreUnlessSpelledInSource,
                            BN[0].getNodeAs<Decl>("func")),
              R"cpp(
FunctionDecl 'forLoop'
`-CompoundStmt
  |-DeclStmt
  | `-VarDecl 'arr'
  |-CXXForRangeStmt
  | |-<<<NULL>>>
  | |-VarDecl 'i'
  | |-DeclRefExpr 'arr'
  | `-CompoundStmt
  `-CXXForRangeStmt
    |-DeclStmt
    | `-VarDecl 'a'
    |   `-DeclRefExpr 'arr'
    |-VarDecl 'i'
    |-DeclRefExpr 'a'
    `-CompoundStmt
)cpp");
  }
  {
    auto BN = ast_matchers::match(
        cxxRecordDecl(hasName("DefaultedAndDeleted"), unless(isImplicit()))
            .bind("rec"),
        AST2->getASTContext());
    EXPECT_EQ(BN.size(), 1u);

    EXPECT_EQ(dumpASTString(TK_AsIs, BN[0].getNodeAs<Decl>("rec")),
              R"cpp(
CXXRecordDecl 'DefaultedAndDeleted'
|-CXXRecordDecl 'DefaultedAndDeleted'
|-FieldDecl 'nt'
|-CXXConstructorDecl 'DefaultedAndDeleted'
| |-CXXCtorInitializer 'nt'
| | `-CXXConstructExpr
| `-CompoundStmt
|-CXXDestructorDecl '~DefaultedAndDeleted'
| `-CompoundStmt
|-CXXConstructorDecl 'DefaultedAndDeleted'
| `-ParmVarDecl ''
|-CXXMethodDecl 'operator='
| |-ParmVarDecl ''
| `-CompoundStmt
|   |-CXXMemberCallExpr
|   | |-MemberExpr
|   | | `-MemberExpr
|   | |   `-CXXThisExpr
|   | `-MemberExpr
|   |   `-DeclRefExpr ''
|   `-ReturnStmt
|     `-UnaryOperator
|       `-CXXThisExpr
|-CXXConstructorDecl 'DefaultedAndDeleted'
| `-ParmVarDecl ''
`-CXXMethodDecl 'operator='
  `-ParmVarDecl ''
)cpp");

    EXPECT_EQ(dumpASTString(TK_IgnoreUnlessSpelledInSource,
                            BN[0].getNodeAs<Decl>("rec")),
              R"cpp(
CXXRecordDecl 'DefaultedAndDeleted'
|-FieldDecl 'nt'
|-CXXConstructorDecl 'DefaultedAndDeleted'
|-CXXDestructorDecl '~DefaultedAndDeleted'
|-CXXConstructorDecl 'DefaultedAndDeleted'
| `-ParmVarDecl ''
|-CXXMethodDecl 'operator='
| `-ParmVarDecl ''
|-CXXConstructorDecl 'DefaultedAndDeleted'
| `-ParmVarDecl ''
`-CXXMethodDecl 'operator='
  `-ParmVarDecl ''
)cpp");
  }
  {
    auto BN = ast_matchers::match(
        callExpr(callee(functionDecl(hasName("hasDefaultArg"))))
            .bind("funcCall"),
        AST2->getASTContext());
    EXPECT_EQ(BN.size(), 1u);

    EXPECT_EQ(dumpASTString(TK_AsIs, BN[0].getNodeAs<CallExpr>("funcCall")),
              R"cpp(
CallExpr
|-ImplicitCastExpr
| `-DeclRefExpr 'hasDefaultArg'
|-IntegerLiteral
`-CXXDefaultArgExpr
)cpp");
    EXPECT_EQ(dumpASTString(TK_IgnoreUnlessSpelledInSource,
                            BN[0].getNodeAs<CallExpr>("funcCall")),
              R"cpp(
CallExpr
|-DeclRefExpr 'hasDefaultArg'
`-IntegerLiteral
)cpp");
  }
}

TEST(Traverse, IgnoreUnlessSpelledInSourceTemplateInstantiations) {

  auto AST = buildASTFromCode(R"cpp(
template<typename T>
struct TemplStruct {
  TemplStruct() {}
  ~TemplStruct() {}

private:
  T m_t;
};

template<typename T>
T timesTwo(T input)
{
  return input * 2;
}

void instantiate()
{
  TemplStruct<int> ti;
  TemplStruct<double> td;
  (void)timesTwo<int>(2);
  (void)timesTwo<double>(2);
}

template class TemplStruct<float>;

extern template class TemplStruct<long>;

template<> class TemplStruct<bool> {
  TemplStruct() {}
  ~TemplStruct() {}

  void foo() {}
private:
  bool m_t;
};

// Explicit instantiation of template functions do not appear in the AST
template float timesTwo(float);

template<> bool timesTwo<bool>(bool) {
  return true;
}
)cpp");
  {
    auto BN = ast_matchers::match(
        classTemplateDecl(hasName("TemplStruct")).bind("rec"),
        AST->getASTContext());
    EXPECT_EQ(BN.size(), 1u);

    EXPECT_EQ(dumpASTString(TK_IgnoreUnlessSpelledInSource,
                            BN[0].getNodeAs<Decl>("rec")),
              R"cpp(
ClassTemplateDecl 'TemplStruct'
|-TemplateTypeParmDecl 'T'
`-CXXRecordDecl 'TemplStruct'
  |-CXXConstructorDecl 'TemplStruct<T>'
  | `-CompoundStmt
  |-CXXDestructorDecl '~TemplStruct<T>'
  | `-CompoundStmt
  |-AccessSpecDecl
  `-FieldDecl 'm_t'
)cpp");

    EXPECT_EQ(dumpASTString(TK_AsIs, BN[0].getNodeAs<Decl>("rec")),
              R"cpp(
ClassTemplateDecl 'TemplStruct'
|-TemplateTypeParmDecl 'T'
|-CXXRecordDecl 'TemplStruct'
| |-CXXRecordDecl 'TemplStruct'
| |-CXXConstructorDecl 'TemplStruct<T>'
| | `-CompoundStmt
| |-CXXDestructorDecl '~TemplStruct<T>'
| | `-CompoundStmt
| |-AccessSpecDecl
| `-FieldDecl 'm_t'
|-ClassTemplateSpecializationDecl 'TemplStruct'
| |-TemplateArgument type int
| | `-BuiltinType
| |-CXXRecordDecl 'TemplStruct'
| |-CXXConstructorDecl 'TemplStruct'
| | `-CompoundStmt
| |-CXXDestructorDecl '~TemplStruct'
| | `-CompoundStmt
| |-AccessSpecDecl
| |-FieldDecl 'm_t'
| `-CXXConstructorDecl 'TemplStruct'
|   `-ParmVarDecl ''
|-ClassTemplateSpecializationDecl 'TemplStruct'
| |-TemplateArgument type double
| | `-BuiltinType
| |-CXXRecordDecl 'TemplStruct'
| |-CXXConstructorDecl 'TemplStruct'
| | `-CompoundStmt
| |-CXXDestructorDecl '~TemplStruct'
| | `-CompoundStmt
| |-AccessSpecDecl
| |-FieldDecl 'm_t'
| `-CXXConstructorDecl 'TemplStruct'
|   `-ParmVarDecl ''
|-ClassTemplateSpecializationDecl 'TemplStruct'
| |-TemplateArgument type float
| | `-BuiltinType
| |-CXXRecordDecl 'TemplStruct'
| |-CXXConstructorDecl 'TemplStruct'
| | `-CompoundStmt
| |-CXXDestructorDecl '~TemplStruct'
| | `-CompoundStmt
| |-AccessSpecDecl
| `-FieldDecl 'm_t'
|-ClassTemplateSpecializationDecl 'TemplStruct'
| |-TemplateArgument type long
| | `-BuiltinType
| |-CXXRecordDecl 'TemplStruct'
| |-CXXConstructorDecl 'TemplStruct'
| |-CXXDestructorDecl '~TemplStruct'
| |-AccessSpecDecl
| `-FieldDecl 'm_t'
`-ClassTemplateSpecializationDecl 'TemplStruct'
  |-TemplateArgument type _Bool
  | `-BuiltinType
  |-CXXRecordDecl 'TemplStruct'
  |-CXXConstructorDecl 'TemplStruct'
  | `-CompoundStmt
  |-CXXDestructorDecl '~TemplStruct'
  | `-CompoundStmt
  |-CXXMethodDecl 'foo'
  | `-CompoundStmt
  |-AccessSpecDecl
  `-FieldDecl 'm_t'
)cpp");
  }
  {
    auto BN = ast_matchers::match(
        classTemplateSpecializationDecl(
            hasTemplateArgument(
                0, templateArgument(refersToType(asString("_Bool")))))
            .bind("templSpec"),
        AST->getASTContext());
    EXPECT_EQ(BN.size(), 1u);

    EXPECT_EQ(dumpASTString(TK_AsIs, BN[0].getNodeAs<Decl>("templSpec")),
              R"cpp(
ClassTemplateSpecializationDecl 'TemplStruct'
|-TemplateArgument type _Bool
| `-BuiltinType
|-CXXRecordDecl 'TemplStruct'
|-CXXConstructorDecl 'TemplStruct'
| `-CompoundStmt
|-CXXDestructorDecl '~TemplStruct'
| `-CompoundStmt
|-CXXMethodDecl 'foo'
| `-CompoundStmt
|-AccessSpecDecl
`-FieldDecl 'm_t'
)cpp");

    EXPECT_EQ(dumpASTString(TK_IgnoreUnlessSpelledInSource,
                            BN[0].getNodeAs<Decl>("templSpec")),
              R"cpp(
ClassTemplateSpecializationDecl 'TemplStruct'
|-TemplateArgument type _Bool
| `-BuiltinType
|-CXXConstructorDecl 'TemplStruct'
| `-CompoundStmt
|-CXXDestructorDecl '~TemplStruct'
| `-CompoundStmt
|-CXXMethodDecl 'foo'
| `-CompoundStmt
|-AccessSpecDecl
`-FieldDecl 'm_t'
)cpp");
  }
  {
    auto BN = ast_matchers::match(
        functionTemplateDecl(hasName("timesTwo")).bind("fn"),
        AST->getASTContext());
    EXPECT_EQ(BN.size(), 1u);

    EXPECT_EQ(dumpASTString(TK_IgnoreUnlessSpelledInSource,
                            BN[0].getNodeAs<Decl>("fn")),
              R"cpp(
FunctionTemplateDecl 'timesTwo'
|-TemplateTypeParmDecl 'T'
`-FunctionDecl 'timesTwo'
  |-ParmVarDecl 'input'
  `-CompoundStmt
    `-ReturnStmt
      `-BinaryOperator
        |-DeclRefExpr 'input'
        `-IntegerLiteral
)cpp");

    EXPECT_EQ(dumpASTString(TK_AsIs, BN[0].getNodeAs<Decl>("fn")),
              R"cpp(
FunctionTemplateDecl 'timesTwo'
|-TemplateTypeParmDecl 'T'
|-FunctionDecl 'timesTwo'
| |-ParmVarDecl 'input'
| `-CompoundStmt
|   `-ReturnStmt
|     `-BinaryOperator
|       |-DeclRefExpr 'input'
|       `-IntegerLiteral
|-FunctionDecl 'timesTwo'
| |-TemplateArgument type int
| | `-BuiltinType
| |-ParmVarDecl 'input'
| `-CompoundStmt
|   `-ReturnStmt
|     `-BinaryOperator
|       |-ImplicitCastExpr
|       | `-DeclRefExpr 'input'
|       `-IntegerLiteral
|-FunctionDecl 'timesTwo'
| |-TemplateArgument type double
| | `-BuiltinType
| |-ParmVarDecl 'input'
| `-CompoundStmt
|   `-ReturnStmt
|     `-BinaryOperator
|       |-ImplicitCastExpr
|       | `-DeclRefExpr 'input'
|       `-ImplicitCastExpr
|         `-IntegerLiteral
|-FunctionDecl 'timesTwo'
| |-TemplateArgument type float
| | `-BuiltinType
| |-ParmVarDecl 'input'
| `-CompoundStmt
|   `-ReturnStmt
|     `-BinaryOperator
|       |-ImplicitCastExpr
|       | `-DeclRefExpr 'input'
|       `-ImplicitCastExpr
|         `-IntegerLiteral
|-FunctionDecl 'timesTwo'
| |-TemplateArgument type _Bool
| | `-BuiltinType
| |-ParmVarDecl ''
| `-CompoundStmt
|   `-ReturnStmt
|     `-CXXBoolLiteralExpr
`-FunctionDecl 'timesTwo'
  |-TemplateArgument type _Bool
  | `-BuiltinType
  `-ParmVarDecl 'input'
)cpp");
  }
  {
    auto BN = ast_matchers::match(
        classTemplateSpecializationDecl(
            hasName("TemplStruct"),
            hasTemplateArgument(
                0, templateArgument(refersToType(asString("float")))),
            hasParent(translationUnitDecl()))
            .bind("rec"),
        AST->getASTContext());
    EXPECT_EQ(BN.size(), 1u);

    EXPECT_EQ(dumpASTString(TK_IgnoreUnlessSpelledInSource,
                            BN[0].getNodeAs<Decl>("rec")),
              R"cpp(
ClassTemplateSpecializationDecl 'TemplStruct'
`-TemplateArgument type float
  `-BuiltinType
)cpp");

    EXPECT_EQ(dumpASTString(TK_AsIs, BN[0].getNodeAs<Decl>("rec")),
              R"cpp(
ClassTemplateSpecializationDecl 'TemplStruct'
|-TemplateArgument type float
| `-BuiltinType
|-CXXRecordDecl 'TemplStruct'
|-CXXConstructorDecl 'TemplStruct'
| `-CompoundStmt
|-CXXDestructorDecl '~TemplStruct'
| `-CompoundStmt
|-AccessSpecDecl
`-FieldDecl 'm_t'
)cpp");
  }
}

} // namespace clang
