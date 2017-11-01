//===- unittest/Tooling/ASTSelectionTest.cpp ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Refactoring/ASTSelection.h"

using namespace clang;
using namespace tooling;

namespace {

struct FileLocation {
  unsigned Line, Column;

  SourceLocation translate(const SourceManager &SM) {
    return SM.translateLineCol(SM.getMainFileID(), Line, Column);
  }
};

using FileRange = std::pair<FileLocation, FileLocation>;

class SelectionFinderVisitor : public TestVisitor<SelectionFinderVisitor> {
  FileLocation Location;
  Optional<FileRange> SelectionRange;
  llvm::function_ref<void(SourceRange SelectionRange,
                          Optional<SelectedASTNode>)>
      Consumer;

public:
  SelectionFinderVisitor(FileLocation Location,
                         Optional<FileRange> SelectionRange,
                         llvm::function_ref<void(SourceRange SelectionRange,
                                                 Optional<SelectedASTNode>)>
                             Consumer)
      : Location(Location), SelectionRange(SelectionRange), Consumer(Consumer) {
  }

  bool VisitTranslationUnitDecl(const TranslationUnitDecl *TU) {
    const ASTContext &Context = TU->getASTContext();
    const SourceManager &SM = Context.getSourceManager();

    SourceRange SelRange;
    if (SelectionRange) {
      SelRange = SourceRange(SelectionRange->first.translate(SM),
                             SelectionRange->second.translate(SM));
    } else {
      SourceLocation Loc = Location.translate(SM);
      SelRange = SourceRange(Loc, Loc);
    }
    Consumer(SelRange, findSelectedASTNodes(Context, SelRange));
    return false;
  }
};

/// This is a test utility function that computes the AST selection at the
/// given location with an optional selection range.
///
/// A location roughly corresponds to a cursor location in an editor, while
/// the optional range corresponds to the selection range in an editor.
void findSelectedASTNodesWithRange(
    StringRef Source, FileLocation Location, Optional<FileRange> SelectionRange,
    llvm::function_ref<void(SourceRange SelectionRange,
                            Optional<SelectedASTNode>)>
        Consumer,
    SelectionFinderVisitor::Language Language =
        SelectionFinderVisitor::Lang_CXX11) {
  SelectionFinderVisitor Visitor(Location, SelectionRange, Consumer);
  EXPECT_TRUE(Visitor.runOver(Source, Language));
}

void findSelectedASTNodes(
    StringRef Source, FileLocation Location, Optional<FileRange> SelectionRange,
    llvm::function_ref<void(Optional<SelectedASTNode>)> Consumer,
    SelectionFinderVisitor::Language Language =
        SelectionFinderVisitor::Lang_CXX11) {
  findSelectedASTNodesWithRange(
      Source, Location, SelectionRange,
      [&](SourceRange, Optional<SelectedASTNode> Selection) {
        Consumer(std::move(Selection));
      },
      Language);
}

void checkNodeImpl(bool IsTypeMatched, const SelectedASTNode &Node,
                   SourceSelectionKind SelectionKind, unsigned NumChildren) {
  ASSERT_TRUE(IsTypeMatched);
  EXPECT_EQ(Node.Children.size(), NumChildren);
  ASSERT_EQ(Node.SelectionKind, SelectionKind);
}

void checkDeclName(const SelectedASTNode &Node, StringRef Name) {
  const auto *ND = Node.Node.get<NamedDecl>();
  EXPECT_TRUE(!!ND);
  ASSERT_EQ(ND->getName(), Name);
}

template <typename T>
const SelectedASTNode &
checkNode(const SelectedASTNode &StmtNode, SourceSelectionKind SelectionKind,
          unsigned NumChildren = 0,
          typename std::enable_if<std::is_base_of<Stmt, T>::value, T>::type
              *StmtOverloadChecker = nullptr) {
  checkNodeImpl(isa<T>(StmtNode.Node.get<Stmt>()), StmtNode, SelectionKind,
                NumChildren);
  return StmtNode;
}

template <typename T>
const SelectedASTNode &
checkNode(const SelectedASTNode &DeclNode, SourceSelectionKind SelectionKind,
          unsigned NumChildren = 0, StringRef Name = "",
          typename std::enable_if<std::is_base_of<Decl, T>::value, T>::type
              *DeclOverloadChecker = nullptr) {
  checkNodeImpl(isa<T>(DeclNode.Node.get<Decl>()), DeclNode, SelectionKind,
                NumChildren);
  if (!Name.empty())
    checkDeclName(DeclNode, Name);
  return DeclNode;
}

struct ForAllChildrenOf {
  const SelectedASTNode &Node;

  static void childKindVerifier(const SelectedASTNode &Node,
                                SourceSelectionKind SelectionKind) {
    for (const SelectedASTNode &Child : Node.Children) {
      ASSERT_EQ(Node.SelectionKind, SelectionKind);
      childKindVerifier(Child, SelectionKind);
    }
  }

public:
  ForAllChildrenOf(const SelectedASTNode &Node) : Node(Node) {}

  void shouldHaveSelectionKind(SourceSelectionKind Kind) {
    childKindVerifier(Node, Kind);
  }
};

ForAllChildrenOf allChildrenOf(const SelectedASTNode &Node) {
  return ForAllChildrenOf(Node);
}

TEST(ASTSelectionFinder, CursorNoSelection) {
  findSelectedASTNodes(
      " void f() { }", {1, 1}, None,
      [](Optional<SelectedASTNode> Node) { EXPECT_FALSE(Node); });
}

TEST(ASTSelectionFinder, CursorAtStartOfFunction) {
  findSelectedASTNodes(
      "void f() { }", {1, 1}, None, [](Optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        checkNode<TranslationUnitDecl>(*Node, SourceSelectionKind::None,
                                       /*NumChildren=*/1);
        checkNode<FunctionDecl>(Node->Children[0],
                                SourceSelectionKind::ContainsSelection,
                                /*NumChildren=*/0, /*Name=*/"f");

        // Check that the dumping works.
        std::string DumpValue;
        llvm::raw_string_ostream OS(DumpValue);
        Node->Children[0].dump(OS);
        ASSERT_EQ(OS.str(), "FunctionDecl \"f\" contains-selection\n");
      });
}

TEST(ASTSelectionFinder, RangeNoSelection) {
  findSelectedASTNodes(
      " void f() { }", {1, 1}, FileRange{{1, 1}, {1, 1}},
      [](Optional<SelectedASTNode> Node) { EXPECT_FALSE(Node); });
  findSelectedASTNodes(
      "  void f() { }", {1, 1}, FileRange{{1, 1}, {1, 2}},
      [](Optional<SelectedASTNode> Node) { EXPECT_FALSE(Node); });
}

TEST(ASTSelectionFinder, EmptyRangeFallbackToCursor) {
  findSelectedASTNodes("void f() { }", {1, 1}, FileRange{{1, 1}, {1, 1}},
                       [](Optional<SelectedASTNode> Node) {
                         EXPECT_TRUE(Node);
                         checkNode<FunctionDecl>(
                             Node->Children[0],
                             SourceSelectionKind::ContainsSelection,
                             /*NumChildren=*/0, /*Name=*/"f");
                       });
}

TEST(ASTSelectionFinder, WholeFunctionSelection) {
  StringRef Source = "int f(int x) { return x;\n}\nvoid f2() { }";
  // From 'int' until just after '}':

  findSelectedASTNodes(
      Source, {1, 1}, FileRange{{1, 1}, {2, 2}},
      [](Optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        EXPECT_EQ(Node->Children.size(), 1u);
        const auto &Fn = checkNode<FunctionDecl>(
            Node->Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/2, /*Name=*/"f");
        checkNode<ParmVarDecl>(Fn.Children[0],
                               SourceSelectionKind::InsideSelection);
        const auto &Body = checkNode<CompoundStmt>(
            Fn.Children[1], SourceSelectionKind::InsideSelection,
            /*NumChildren=*/1);
        const auto &Return = checkNode<ReturnStmt>(
            Body.Children[0], SourceSelectionKind::InsideSelection,
            /*NumChildren=*/1);
        checkNode<ImplicitCastExpr>(Return.Children[0],
                                    SourceSelectionKind::InsideSelection,
                                    /*NumChildren=*/1);
        checkNode<DeclRefExpr>(Return.Children[0].Children[0],
                               SourceSelectionKind::InsideSelection);
      });

  // From 'int' until just before '}':
  findSelectedASTNodes(
      Source, {2, 1}, FileRange{{1, 1}, {2, 1}},
      [](Optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        EXPECT_EQ(Node->Children.size(), 1u);
        const auto &Fn = checkNode<FunctionDecl>(
            Node->Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/2, /*Name=*/"f");
        const auto &Body = checkNode<CompoundStmt>(
            Fn.Children[1], SourceSelectionKind::ContainsSelectionEnd,
            /*NumChildren=*/1);
        checkNode<ReturnStmt>(Body.Children[0],
                              SourceSelectionKind::InsideSelection,
                              /*NumChildren=*/1);
      });
  // From '{' until just after '}':
  findSelectedASTNodes(
      Source, {1, 14}, FileRange{{1, 14}, {2, 2}},
      [](Optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        EXPECT_EQ(Node->Children.size(), 1u);
        const auto &Fn = checkNode<FunctionDecl>(
            Node->Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/1, /*Name=*/"f");
        const auto &Body = checkNode<CompoundStmt>(
            Fn.Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/1);
        checkNode<ReturnStmt>(Body.Children[0],
                              SourceSelectionKind::InsideSelection,
                              /*NumChildren=*/1);
      });
  // From 'x' until just after '}':
  findSelectedASTNodes(
      Source, {2, 2}, FileRange{{1, 11}, {2, 2}},
      [](Optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        EXPECT_EQ(Node->Children.size(), 1u);
        const auto &Fn = checkNode<FunctionDecl>(
            Node->Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/2, /*Name=*/"f");
        checkNode<ParmVarDecl>(Fn.Children[0],
                               SourceSelectionKind::ContainsSelectionStart);
        const auto &Body = checkNode<CompoundStmt>(
            Fn.Children[1], SourceSelectionKind::InsideSelection,
            /*NumChildren=*/1);
        checkNode<ReturnStmt>(Body.Children[0],
                              SourceSelectionKind::InsideSelection,
                              /*NumChildren=*/1);
      });
}

TEST(ASTSelectionFinder, MultipleFunctionSelection) {
  StringRef Source = R"(void f0() {
}
void f1() { }
void f2() { }
void f3() { }
)";
  auto SelectedF1F2 = [](Optional<SelectedASTNode> Node) {
    EXPECT_TRUE(Node);
    EXPECT_EQ(Node->Children.size(), 2u);
    checkNode<FunctionDecl>(Node->Children[0],
                            SourceSelectionKind::InsideSelection,
                            /*NumChildren=*/1, /*Name=*/"f1");
    checkNode<FunctionDecl>(Node->Children[1],
                            SourceSelectionKind::InsideSelection,
                            /*NumChildren=*/1, /*Name=*/"f2");
  };
  // Just after '}' of f0 and just before 'void' of f3:
  findSelectedASTNodes(Source, {2, 2}, FileRange{{2, 2}, {5, 1}}, SelectedF1F2);
  // Just before 'void' of f1 and just after '}' of f2:
  findSelectedASTNodes(Source, {3, 1}, FileRange{{3, 1}, {4, 14}},
                       SelectedF1F2);
}

TEST(ASTSelectionFinder, MultipleStatementSelection) {
  StringRef Source = R"(void f(int x, int y) {
  int z = x;
  f(2, 3);
  if (x == 0) {
    return;
  }
  x = 1;
  return;
})";
  // From 'f(2,3)' until just before 'x = 1;':
  findSelectedASTNodes(
      Source, {3, 2}, FileRange{{3, 2}, {7, 1}},
      [](Optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        EXPECT_EQ(Node->Children.size(), 1u);
        const auto &Fn = checkNode<FunctionDecl>(
            Node->Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/1, /*Name=*/"f");
        const auto &Body = checkNode<CompoundStmt>(
            Fn.Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/2);
        allChildrenOf(checkNode<CallExpr>(Body.Children[0],
                                          SourceSelectionKind::InsideSelection,
                                          /*NumChildren=*/3))
            .shouldHaveSelectionKind(SourceSelectionKind::InsideSelection);
        allChildrenOf(checkNode<IfStmt>(Body.Children[1],
                                        SourceSelectionKind::InsideSelection,
                                        /*NumChildren=*/2))
            .shouldHaveSelectionKind(SourceSelectionKind::InsideSelection);
      });
  // From 'f(2,3)' until just before ';' in 'x = 1;':
  findSelectedASTNodes(
      Source, {3, 2}, FileRange{{3, 2}, {7, 8}},
      [](Optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        EXPECT_EQ(Node->Children.size(), 1u);
        const auto &Fn = checkNode<FunctionDecl>(
            Node->Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/1, /*Name=*/"f");
        const auto &Body = checkNode<CompoundStmt>(
            Fn.Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/3);
        checkNode<CallExpr>(Body.Children[0],
                            SourceSelectionKind::InsideSelection,
                            /*NumChildren=*/3);
        checkNode<IfStmt>(Body.Children[1],
                          SourceSelectionKind::InsideSelection,
                          /*NumChildren=*/2);
        checkNode<BinaryOperator>(Body.Children[2],
                                  SourceSelectionKind::InsideSelection,
                                  /*NumChildren=*/2);
      });
  // From the middle of 'int z = 3' until the middle of 'x = 1;':
  findSelectedASTNodes(
      Source, {2, 10}, FileRange{{2, 10}, {7, 5}},
      [](Optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        EXPECT_EQ(Node->Children.size(), 1u);
        const auto &Fn = checkNode<FunctionDecl>(
            Node->Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/1, /*Name=*/"f");
        const auto &Body = checkNode<CompoundStmt>(
            Fn.Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/4);
        checkNode<DeclStmt>(Body.Children[0],
                            SourceSelectionKind::ContainsSelectionStart,
                            /*NumChildren=*/1);
        checkNode<CallExpr>(Body.Children[1],
                            SourceSelectionKind::InsideSelection,
                            /*NumChildren=*/3);
        checkNode<IfStmt>(Body.Children[2],
                          SourceSelectionKind::InsideSelection,
                          /*NumChildren=*/2);
        checkNode<BinaryOperator>(Body.Children[3],
                                  SourceSelectionKind::ContainsSelectionEnd,
                                  /*NumChildren=*/1);
      });
}

TEST(ASTSelectionFinder, SelectionInFunctionInObjCImplementation) {
  StringRef Source = R"(
@interface I
@end
@implementation I

int notSelected() { }

int selected(int x) {
  return x;
}

@end
@implementation I(Cat)

void catF() { }

@end

void outerFunction() { }
)";
  // Just the 'x' expression in 'selected':
  findSelectedASTNodes(
      Source, {9, 10}, FileRange{{9, 10}, {9, 11}},
      [](Optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        EXPECT_EQ(Node->Children.size(), 1u);
        const auto &Impl = checkNode<ObjCImplementationDecl>(
            Node->Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/1, /*Name=*/"I");
        const auto &Fn = checkNode<FunctionDecl>(
            Impl.Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/1, /*Name=*/"selected");
        allChildrenOf(Fn).shouldHaveSelectionKind(
            SourceSelectionKind::ContainsSelection);
      },
      SelectionFinderVisitor::Lang_OBJC);
  // The entire 'catF':
  findSelectedASTNodes(
      Source, {15, 1}, FileRange{{15, 1}, {15, 16}},
      [](Optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        EXPECT_EQ(Node->Children.size(), 1u);
        const auto &Impl = checkNode<ObjCCategoryImplDecl>(
            Node->Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/1, /*Name=*/"Cat");
        const auto &Fn = checkNode<FunctionDecl>(
            Impl.Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/1, /*Name=*/"catF");
        allChildrenOf(Fn).shouldHaveSelectionKind(
            SourceSelectionKind::ContainsSelection);
      },
      SelectionFinderVisitor::Lang_OBJC);
  // From the line before 'selected' to the line after 'catF':
  findSelectedASTNodes(
      Source, {16, 1}, FileRange{{7, 1}, {16, 1}},
      [](Optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        EXPECT_EQ(Node->Children.size(), 2u);
        const auto &Impl = checkNode<ObjCImplementationDecl>(
            Node->Children[0], SourceSelectionKind::ContainsSelectionStart,
            /*NumChildren=*/1, /*Name=*/"I");
        const auto &Selected = checkNode<FunctionDecl>(
            Impl.Children[0], SourceSelectionKind::InsideSelection,
            /*NumChildren=*/2, /*Name=*/"selected");
        allChildrenOf(Selected).shouldHaveSelectionKind(
            SourceSelectionKind::InsideSelection);
        const auto &Cat = checkNode<ObjCCategoryImplDecl>(
            Node->Children[1], SourceSelectionKind::ContainsSelectionEnd,
            /*NumChildren=*/1, /*Name=*/"Cat");
        const auto &CatF = checkNode<FunctionDecl>(
            Cat.Children[0], SourceSelectionKind::InsideSelection,
            /*NumChildren=*/1, /*Name=*/"catF");
        allChildrenOf(CatF).shouldHaveSelectionKind(
            SourceSelectionKind::InsideSelection);
      },
      SelectionFinderVisitor::Lang_OBJC);
  // Just the 'outer' function:
  findSelectedASTNodes(Source, {19, 1}, FileRange{{19, 1}, {19, 25}},
                       [](Optional<SelectedASTNode> Node) {
                         EXPECT_TRUE(Node);
                         EXPECT_EQ(Node->Children.size(), 1u);
                         checkNode<FunctionDecl>(
                             Node->Children[0],
                             SourceSelectionKind::ContainsSelection,
                             /*NumChildren=*/1, /*Name=*/"outerFunction");
                       },
                       SelectionFinderVisitor::Lang_OBJC);
}

TEST(ASTSelectionFinder, FunctionInObjCImplementationCarefulWithEarlyExit) {
  StringRef Source = R"(
@interface I
@end
@implementation I

void selected() {
}

- (void) method { }

@end
)";
  // Just 'selected'
  findSelectedASTNodes(
      Source, {6, 1}, FileRange{{6, 1}, {7, 2}},
      [](Optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        EXPECT_EQ(Node->Children.size(), 1u);
        const auto &Impl = checkNode<ObjCImplementationDecl>(
            Node->Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/1, /*Name=*/"I");
        checkNode<FunctionDecl>(Impl.Children[0],
                                SourceSelectionKind::ContainsSelection,
                                /*NumChildren=*/1, /*Name=*/"selected");
      },
      SelectionFinderVisitor::Lang_OBJC);
}

TEST(ASTSelectionFinder, AvoidImplicitDeclarations) {
  StringRef Source = R"(
struct Copy {
  int x;
};
void foo() {
  Copy x;
  Copy y = x;
}
)";
  // The entire struct 'Copy':
  findSelectedASTNodes(
      Source, {2, 1}, FileRange{{2, 1}, {4, 3}},
      [](Optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        EXPECT_EQ(Node->Children.size(), 1u);
        const auto &Record = checkNode<CXXRecordDecl>(
            Node->Children[0], SourceSelectionKind::InsideSelection,
            /*NumChildren=*/1, /*Name=*/"Copy");
        checkNode<FieldDecl>(Record.Children[0],
                             SourceSelectionKind::InsideSelection);
      });
}

TEST(ASTSelectionFinder, CorrectEndForObjectiveCImplementation) {
  StringRef Source = R"(
@interface I
@end
@implementation I
@ end
)";
  // Just after '@ end'
  findSelectedASTNodes(Source, {5, 6}, None,
                       [](Optional<SelectedASTNode> Node) {
                         EXPECT_TRUE(Node);
                         EXPECT_EQ(Node->Children.size(), 1u);
                         checkNode<ObjCImplementationDecl>(
                             Node->Children[0],
                             SourceSelectionKind::ContainsSelection);
                       },
                       SelectionFinderVisitor::Lang_OBJC);
}

const SelectedASTNode &checkFnBody(const Optional<SelectedASTNode> &Node,
                                   StringRef Name) {
  EXPECT_TRUE(Node);
  EXPECT_EQ(Node->Children.size(), 1u);
  const auto &Fn = checkNode<FunctionDecl>(
      Node->Children[0], SourceSelectionKind::ContainsSelection,
      /*NumChildren=*/1, Name);
  return checkNode<CompoundStmt>(Fn.Children[0],
                                 SourceSelectionKind::ContainsSelection,
                                 /*NumChildren=*/1);
}

TEST(ASTSelectionFinder, SelectObjectiveCPseudoObjectExprs) {
  StringRef Source = R"(
@interface I
@property(readwrite) int prop;
@end
void selectProp(I *i) {
(void)i.prop;
i.prop = 21;
}


@interface NSMutableArray
- (id)objectAtIndexedSubscript:(unsigned int)index;
- (void)setObject:(id)object atIndexedSubscript:(unsigned int)index;
@end

void selectSubscript(NSMutableArray *array, I *i) {
  (void)array[10];
  array[i.prop] = i;
}
)";
  // Just 'i.prop'.
  findSelectedASTNodes(
      Source, {6, 7}, FileRange{{6, 7}, {6, 13}},
      [](Optional<SelectedASTNode> Node) {
        const auto &CS = checkFnBody(Node, /*Name=*/"selectProp");
        const auto &CCast = checkNode<CStyleCastExpr>(
            CS.Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/1);
        const auto &POE = checkNode<PseudoObjectExpr>(
            CCast.Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/1);
        const auto &PRE = checkNode<ObjCPropertyRefExpr>(
            POE.Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/1);
        const auto &Cast = checkNode<ImplicitCastExpr>(
            PRE.Children[0], SourceSelectionKind::InsideSelection,
            /*NumChildren=*/1);
        checkNode<DeclRefExpr>(Cast.Children[0],
                               SourceSelectionKind::InsideSelection);
      },
      SelectionFinderVisitor::Lang_OBJC);
  // Just 'i.prop = 21'
  findSelectedASTNodes(
      Source, {7, 1}, FileRange{{7, 1}, {7, 12}},
      [](Optional<SelectedASTNode> Node) {
        const auto &CS = checkFnBody(Node, /*Name=*/"selectProp");
        const auto &POE = checkNode<PseudoObjectExpr>(
            CS.Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/1);
        const auto &BinOp = checkNode<BinaryOperator>(
            POE.Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/2);
        const auto &PRE = checkNode<ObjCPropertyRefExpr>(
            BinOp.Children[0], SourceSelectionKind::InsideSelection,
            /*NumChildren=*/1);
        const auto &Cast = checkNode<ImplicitCastExpr>(
            PRE.Children[0], SourceSelectionKind::InsideSelection,
            /*NumChildren=*/1);
        checkNode<DeclRefExpr>(Cast.Children[0],
                               SourceSelectionKind::InsideSelection);
        checkNode<IntegerLiteral>(BinOp.Children[1],
                                  SourceSelectionKind::InsideSelection);
      },
      SelectionFinderVisitor::Lang_OBJC);
  // Just 'array[10]'
  findSelectedASTNodes(
      Source, {17, 9}, FileRange{{17, 9}, {17, 18}},
      [](Optional<SelectedASTNode> Node) {
        const auto &CS = checkFnBody(Node, /*Name=*/"selectSubscript");
        const auto &CCast = checkNode<CStyleCastExpr>(
            CS.Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/1);
        const auto &POE = checkNode<PseudoObjectExpr>(
            CCast.Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/1);
        const auto &SRE = checkNode<ObjCSubscriptRefExpr>(
            POE.Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/2);
        const auto &Cast = checkNode<ImplicitCastExpr>(
            SRE.Children[0], SourceSelectionKind::InsideSelection,
            /*NumChildren=*/1);
        checkNode<DeclRefExpr>(Cast.Children[0],
                               SourceSelectionKind::InsideSelection);
        checkNode<IntegerLiteral>(SRE.Children[1],
                                  SourceSelectionKind::InsideSelection);
      },
      SelectionFinderVisitor::Lang_OBJC);
  // Just 'array[i.prop] = array'
  findSelectedASTNodes(
      Source, {18, 3}, FileRange{{18, 3}, {18, 20}},
      [](Optional<SelectedASTNode> Node) {
        const auto &CS = checkFnBody(Node, /*Name=*/"selectSubscript");
        const auto &POE = checkNode<PseudoObjectExpr>(
            CS.Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/1);
        const auto &BinOp = checkNode<BinaryOperator>(
            POE.Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/2);
        const auto &SRE = checkNode<ObjCSubscriptRefExpr>(
            BinOp.Children[0], SourceSelectionKind::InsideSelection,
            /*NumChildren=*/2);
        const auto &Cast = checkNode<ImplicitCastExpr>(
            SRE.Children[0], SourceSelectionKind::InsideSelection,
            /*NumChildren=*/1);
        checkNode<DeclRefExpr>(Cast.Children[0],
                               SourceSelectionKind::InsideSelection);
        const auto &POE2 = checkNode<PseudoObjectExpr>(
            SRE.Children[1], SourceSelectionKind::InsideSelection,
            /*NumChildren=*/1);
        const auto &PRE = checkNode<ObjCPropertyRefExpr>(
            POE2.Children[0], SourceSelectionKind::InsideSelection,
            /*NumChildren=*/1);
        const auto &Cast2 = checkNode<ImplicitCastExpr>(
            PRE.Children[0], SourceSelectionKind::InsideSelection,
            /*NumChildren=*/1);
        checkNode<DeclRefExpr>(Cast2.Children[0],
                               SourceSelectionKind::InsideSelection);
        checkNode<DeclRefExpr>(BinOp.Children[1],
                               SourceSelectionKind::InsideSelection);
      },
      SelectionFinderVisitor::Lang_OBJC);
}

TEST(ASTSelectionFinder, SimpleCodeRangeASTSelection) {
  StringRef Source = R"(void f(int x, int y) {
  int z = x;
  f(2, 3);
  if (x == 0) {
    return;
  }
  x = 1;
  return;
}
void f2() {
  int m = 0;
}
)";
  // No selection range.
  findSelectedASTNodesWithRange(
      Source, {2, 2}, None,
      [](SourceRange SelectionRange, Optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        Optional<CodeRangeASTSelection> SelectedCode =
            CodeRangeASTSelection::create(SelectionRange, std::move(*Node));
        EXPECT_FALSE(SelectedCode);
      });
  findSelectedASTNodesWithRange(
      Source, {2, 2}, FileRange{{2, 2}, {2, 2}},
      [](SourceRange SelectionRange, Optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        Optional<CodeRangeASTSelection> SelectedCode =
            CodeRangeASTSelection::create(SelectionRange, std::move(*Node));
        EXPECT_FALSE(SelectedCode);
      });
  // Range that spans multiple functions is an invalid code range.
  findSelectedASTNodesWithRange(
      Source, {2, 2}, FileRange{{7, 2}, {12, 1}},
      [](SourceRange SelectionRange, Optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        Optional<CodeRangeASTSelection> SelectedCode =
            CodeRangeASTSelection::create(SelectionRange, std::move(*Node));
        EXPECT_FALSE(SelectedCode);
      });
  // Just 'z = x;':
  findSelectedASTNodesWithRange(
      Source, {2, 2}, FileRange{{2, 2}, {2, 13}},
      [](SourceRange SelectionRange, Optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        Optional<CodeRangeASTSelection> SelectedCode =
            CodeRangeASTSelection::create(SelectionRange, std::move(*Node));
        EXPECT_TRUE(SelectedCode);
        EXPECT_EQ(SelectedCode->size(), 1u);
        EXPECT_TRUE(isa<DeclStmt>((*SelectedCode)[0]));
        ArrayRef<SelectedASTNode::ReferenceType> Parents =
            SelectedCode->getParents();
        EXPECT_EQ(Parents.size(), 3u);
        EXPECT_TRUE(
            isa<TranslationUnitDecl>(Parents[0].get().Node.get<Decl>()));
        // Function 'f' definition.
        EXPECT_TRUE(isa<FunctionDecl>(Parents[1].get().Node.get<Decl>()));
        // Function body of function 'F'.
        EXPECT_TRUE(isa<CompoundStmt>(Parents[2].get().Node.get<Stmt>()));
      });
  // From 'f(2,3)' until just before 'x = 1;':
  findSelectedASTNodesWithRange(
      Source, {3, 2}, FileRange{{3, 2}, {7, 1}},
      [](SourceRange SelectionRange, Optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        Optional<CodeRangeASTSelection> SelectedCode =
            CodeRangeASTSelection::create(SelectionRange, std::move(*Node));
        EXPECT_TRUE(SelectedCode);
        EXPECT_EQ(SelectedCode->size(), 2u);
        EXPECT_TRUE(isa<CallExpr>((*SelectedCode)[0]));
        EXPECT_TRUE(isa<IfStmt>((*SelectedCode)[1]));
        ArrayRef<SelectedASTNode::ReferenceType> Parents =
            SelectedCode->getParents();
        EXPECT_EQ(Parents.size(), 3u);
        EXPECT_TRUE(
            isa<TranslationUnitDecl>(Parents[0].get().Node.get<Decl>()));
        // Function 'f' definition.
        EXPECT_TRUE(isa<FunctionDecl>(Parents[1].get().Node.get<Decl>()));
        // Function body of function 'F'.
        EXPECT_TRUE(isa<CompoundStmt>(Parents[2].get().Node.get<Stmt>()));
      });
  // From 'f(2,3)' until just before ';' in 'x = 1;':
  findSelectedASTNodesWithRange(
      Source, {3, 2}, FileRange{{3, 2}, {7, 8}},
      [](SourceRange SelectionRange, Optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        Optional<CodeRangeASTSelection> SelectedCode =
            CodeRangeASTSelection::create(SelectionRange, std::move(*Node));
        EXPECT_TRUE(SelectedCode);
        EXPECT_EQ(SelectedCode->size(), 3u);
        EXPECT_TRUE(isa<CallExpr>((*SelectedCode)[0]));
        EXPECT_TRUE(isa<IfStmt>((*SelectedCode)[1]));
        EXPECT_TRUE(isa<BinaryOperator>((*SelectedCode)[2]));
      });
  // From the middle of 'int z = 3' until the middle of 'x = 1;':
  findSelectedASTNodesWithRange(
      Source, {2, 10}, FileRange{{2, 10}, {7, 5}},
      [](SourceRange SelectionRange, Optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        EXPECT_TRUE(Node);
        Optional<CodeRangeASTSelection> SelectedCode =
            CodeRangeASTSelection::create(SelectionRange, std::move(*Node));
        EXPECT_TRUE(SelectedCode);
        EXPECT_EQ(SelectedCode->size(), 4u);
        EXPECT_TRUE(isa<DeclStmt>((*SelectedCode)[0]));
        EXPECT_TRUE(isa<CallExpr>((*SelectedCode)[1]));
        EXPECT_TRUE(isa<IfStmt>((*SelectedCode)[2]));
        EXPECT_TRUE(isa<BinaryOperator>((*SelectedCode)[3]));
      });
}

TEST(ASTSelectionFinder, OutOfBodyCodeRange) {
  StringRef Source = R"(
int codeRange = 2 + 3;
)";
  // '2+3' expression.
  findSelectedASTNodesWithRange(
      Source, {2, 17}, FileRange{{2, 17}, {2, 22}},
      [](SourceRange SelectionRange, Optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        Optional<CodeRangeASTSelection> SelectedCode =
            CodeRangeASTSelection::create(SelectionRange, std::move(*Node));
        EXPECT_TRUE(SelectedCode);
        EXPECT_EQ(SelectedCode->size(), 1u);
        EXPECT_TRUE(isa<BinaryOperator>((*SelectedCode)[0]));
        ArrayRef<SelectedASTNode::ReferenceType> Parents =
            SelectedCode->getParents();
        EXPECT_EQ(Parents.size(), 2u);
        EXPECT_TRUE(
            isa<TranslationUnitDecl>(Parents[0].get().Node.get<Decl>()));
        // Variable 'codeRange'.
        EXPECT_TRUE(isa<VarDecl>(Parents[1].get().Node.get<Decl>()));
      });
}

TEST(ASTSelectionFinder, SelectVarDeclStmt) {
  StringRef Source = R"(
void f() {
   {
       int a;
   }
}
)";
  // 'int a'
  findSelectedASTNodesWithRange(
      Source, {4, 8}, FileRange{{4, 8}, {4, 14}},
      [](SourceRange SelectionRange, Optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        Optional<CodeRangeASTSelection> SelectedCode =
            CodeRangeASTSelection::create(SelectionRange, std::move(*Node));
        EXPECT_TRUE(SelectedCode);
        EXPECT_EQ(SelectedCode->size(), 1u);
        EXPECT_TRUE(isa<DeclStmt>((*SelectedCode)[0]));
        ArrayRef<SelectedASTNode::ReferenceType> Parents =
            SelectedCode->getParents();
        EXPECT_EQ(Parents.size(), 4u);
        EXPECT_TRUE(
            isa<TranslationUnitDecl>(Parents[0].get().Node.get<Decl>()));
        // Function 'f' definition.
        EXPECT_TRUE(isa<FunctionDecl>(Parents[1].get().Node.get<Decl>()));
        // Function body of function 'F'.
        EXPECT_TRUE(isa<CompoundStmt>(Parents[2].get().Node.get<Stmt>()));
        // Compound statement in body of 'F'.
        EXPECT_TRUE(isa<CompoundStmt>(Parents[3].get().Node.get<Stmt>()));
      });
}

TEST(ASTSelectionFinder, SelectEntireDeclStmtRange) {
  StringRef Source = R"(
void f(int x, int y) {
   int a = x * y;
}
)";
  // 'int a = x * y'
  findSelectedASTNodesWithRange(
      Source, {3, 4}, FileRange{{3, 4}, {3, 17}},
      [](SourceRange SelectionRange, Optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        Optional<CodeRangeASTSelection> SelectedCode =
            CodeRangeASTSelection::create(SelectionRange, std::move(*Node));
        EXPECT_TRUE(SelectedCode);
        EXPECT_EQ(SelectedCode->size(), 1u);
        EXPECT_TRUE(isa<DeclStmt>((*SelectedCode)[0]));
        ArrayRef<SelectedASTNode::ReferenceType> Parents =
            SelectedCode->getParents();
        EXPECT_EQ(Parents.size(), 3u);
        EXPECT_TRUE(
            isa<TranslationUnitDecl>(Parents[0].get().Node.get<Decl>()));
        // Function 'f' definition.
        EXPECT_TRUE(isa<FunctionDecl>(Parents[1].get().Node.get<Decl>()));
        // Function body of function 'F'.
        EXPECT_TRUE(isa<CompoundStmt>(Parents[2].get().Node.get<Stmt>()));
      });
}

TEST(ASTSelectionFinder, SelectEntireDeclStmtRangeWithMultipleDecls) {
  StringRef Source = R"(
void f(int x, int y) {
   int a = x * y, b = x - y;
}
)";
  // 'b = x - y'
  findSelectedASTNodesWithRange(
      Source, {3, 19}, FileRange{{3, 19}, {3, 28}},
      [](SourceRange SelectionRange, Optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        Optional<CodeRangeASTSelection> SelectedCode =
            CodeRangeASTSelection::create(SelectionRange, std::move(*Node));
        EXPECT_TRUE(SelectedCode);
        EXPECT_EQ(SelectedCode->size(), 1u);
        EXPECT_TRUE(isa<DeclStmt>((*SelectedCode)[0]));
        ArrayRef<SelectedASTNode::ReferenceType> Parents =
            SelectedCode->getParents();
        EXPECT_EQ(Parents.size(), 3u);
        EXPECT_TRUE(
            isa<TranslationUnitDecl>(Parents[0].get().Node.get<Decl>()));
        // Function 'f' definition.
        EXPECT_TRUE(isa<FunctionDecl>(Parents[1].get().Node.get<Decl>()));
        // Function body of function 'F'.
        EXPECT_TRUE(isa<CompoundStmt>(Parents[2].get().Node.get<Stmt>()));
      });
}

TEST(ASTSelectionFinder, SimpleCodeRangeASTSelectionInObjCMethod) {
  StringRef Source = R"(@interface I @end
@implementation I
- (void) f:(int)x with:(int) y {
  int z = x;
  [self f: 2 with: 3];
  if (x == 0) {
    return;
  }
  x = 1;
  return;
}
- (void)f2 {
  int m = 0;
}
@end
)";
  // Range that spans multiple methods is an invalid code range.
  findSelectedASTNodesWithRange(
      Source, {9, 2}, FileRange{{9, 2}, {13, 1}},
      [](SourceRange SelectionRange, Optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        Optional<CodeRangeASTSelection> SelectedCode =
            CodeRangeASTSelection::create(SelectionRange, std::move(*Node));
        EXPECT_FALSE(SelectedCode);
      },
      SelectionFinderVisitor::Lang_OBJC);
  // Just 'z = x;':
  findSelectedASTNodesWithRange(
      Source, {4, 2}, FileRange{{4, 2}, {4, 13}},
      [](SourceRange SelectionRange, Optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        Optional<CodeRangeASTSelection> SelectedCode =
            CodeRangeASTSelection::create(SelectionRange, std::move(*Node));
        EXPECT_TRUE(SelectedCode);
        EXPECT_EQ(SelectedCode->size(), 1u);
        EXPECT_TRUE(isa<DeclStmt>((*SelectedCode)[0]));
        ArrayRef<SelectedASTNode::ReferenceType> Parents =
            SelectedCode->getParents();
        EXPECT_EQ(Parents.size(), 4u);
        EXPECT_TRUE(
            isa<TranslationUnitDecl>(Parents[0].get().Node.get<Decl>()));
        // 'I' @implementation.
        EXPECT_TRUE(isa<ObjCImplDecl>(Parents[1].get().Node.get<Decl>()));
        // Function 'f' definition.
        EXPECT_TRUE(isa<ObjCMethodDecl>(Parents[2].get().Node.get<Decl>()));
        // Function body of function 'F'.
        EXPECT_TRUE(isa<CompoundStmt>(Parents[3].get().Node.get<Stmt>()));
      },
      SelectionFinderVisitor::Lang_OBJC);
  // From '[self f: 2 with: 3]' until just before 'x = 1;':
  findSelectedASTNodesWithRange(
      Source, {5, 2}, FileRange{{5, 2}, {9, 1}},
      [](SourceRange SelectionRange, Optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        Optional<CodeRangeASTSelection> SelectedCode =
            CodeRangeASTSelection::create(SelectionRange, std::move(*Node));
        EXPECT_TRUE(SelectedCode);
        EXPECT_EQ(SelectedCode->size(), 2u);
        EXPECT_TRUE(isa<ObjCMessageExpr>((*SelectedCode)[0]));
        EXPECT_TRUE(isa<IfStmt>((*SelectedCode)[1]));
        ArrayRef<SelectedASTNode::ReferenceType> Parents =
            SelectedCode->getParents();
        EXPECT_EQ(Parents.size(), 4u);
        EXPECT_TRUE(
            isa<TranslationUnitDecl>(Parents[0].get().Node.get<Decl>()));
        // 'I' @implementation.
        EXPECT_TRUE(isa<ObjCImplDecl>(Parents[1].get().Node.get<Decl>()));
        // Function 'f' definition.
        EXPECT_TRUE(isa<ObjCMethodDecl>(Parents[2].get().Node.get<Decl>()));
        // Function body of function 'F'.
        EXPECT_TRUE(isa<CompoundStmt>(Parents[3].get().Node.get<Stmt>()));
      },
      SelectionFinderVisitor::Lang_OBJC);
}

} // end anonymous namespace
