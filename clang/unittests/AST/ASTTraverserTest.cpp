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

  void Visit(const Decl *D) { OS << D->getDeclKindName() << "Decl"; }

  void Visit(const Stmt *S) { OS << S->getStmtClassName(); }

  void Visit(QualType QT) {
    OS << "QualType " << QT.split().Quals.getAsString();
  }

  void Visit(const Type *T) { OS << T->getTypeClassName() << "Type"; }

  void Visit(const comments::Comment *C, const comments::FullComment *FC) {
    OS << C->getCommentKindName();
  }

  void Visit(const CXXCtorInitializer *Init) { OS << "CXXCtorInitializer"; }

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

const FunctionDecl *getFunctionNode(clang::ASTUnit *AST,
                                    const std::string &Name) {
  auto Result = ast_matchers::match(functionDecl(hasName(Name)).bind("fn"),
                                    AST->getASTContext());
  EXPECT_EQ(Result.size(), 1u);
  return Result[0].getNodeAs<FunctionDecl>("fn");
}

template <typename T> struct Verifier {
  static void withDynNode(T Node, const std::string &DumpString) {
    EXPECT_EQ(dumpASTString(ast_type_traits::DynTypedNode::create(Node)),
              DumpString);
  }
};

template <typename T> struct Verifier<T *> {
  static void withDynNode(T *Node, const std::string &DumpString) {
    EXPECT_EQ(dumpASTString(ast_type_traits::DynTypedNode::create(*Node)),
              DumpString);
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

)cpp");

  const FunctionDecl *Func = getFunctionNode(AST.get(), "func");

  verifyWithDynNode(Func,
                    R"cpp(
CXXMethodDecl
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
CXXCtorInitializer
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
TemplateArgument
)cpp");
}
} // namespace clang
