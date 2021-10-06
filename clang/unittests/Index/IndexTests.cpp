//===--- IndexTests.cpp - Test indexing actions -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Index/IndexSymbol.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>

namespace clang {
namespace index {
namespace {
struct Position {
  size_t Line = 0;
  size_t Column = 0;

  Position(size_t Line = 0, size_t Column = 0) : Line(Line), Column(Column) {}

  static Position fromSourceLocation(SourceLocation Loc,
                                     const SourceManager &SM) {
    FileID FID;
    unsigned Offset;
    std::tie(FID, Offset) = SM.getDecomposedSpellingLoc(Loc);
    Position P;
    P.Line = SM.getLineNumber(FID, Offset);
    P.Column = SM.getColumnNumber(FID, Offset);
    return P;
  }
};

bool operator==(const Position &LHS, const Position &RHS) {
  return std::tie(LHS.Line, LHS.Column) == std::tie(RHS.Line, RHS.Column);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Position &Pos) {
  return OS << Pos.Line << ':' << Pos.Column;
}

struct TestSymbol {
  std::string QName;
  Position WrittenPos;
  Position DeclPos;
  SymbolInfo SymInfo;
  SymbolRoleSet Roles;
  // FIXME: add more information.
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const TestSymbol &S) {
  return OS << S.QName << '[' << S.WrittenPos << ']' << '@' << S.DeclPos << '('
            << static_cast<unsigned>(S.SymInfo.Kind) << ')';
}

class Indexer : public IndexDataConsumer {
public:
  void initialize(ASTContext &Ctx) override {
    AST = &Ctx;
    IndexDataConsumer::initialize(Ctx);
  }

  bool handleDeclOccurrence(const Decl *D, SymbolRoleSet Roles,
                            ArrayRef<SymbolRelation>, SourceLocation Loc,
                            ASTNodeInfo) override {
    const auto *ND = llvm::dyn_cast<NamedDecl>(D);
    if (!ND)
      return true;
    TestSymbol S;
    S.SymInfo = getSymbolInfo(D);
    S.QName = ND->getQualifiedNameAsString();
    S.WrittenPos = Position::fromSourceLocation(Loc, AST->getSourceManager());
    S.DeclPos =
        Position::fromSourceLocation(D->getLocation(), AST->getSourceManager());
    S.Roles = Roles;
    Symbols.push_back(std::move(S));
    return true;
  }

  bool handleMacroOccurrence(const IdentifierInfo *Name, const MacroInfo *MI,
                             SymbolRoleSet Roles, SourceLocation Loc) override {
    TestSymbol S;
    S.SymInfo = getSymbolInfoForMacro(*MI);
    S.QName = std::string(Name->getName());
    S.WrittenPos = Position::fromSourceLocation(Loc, AST->getSourceManager());
    S.DeclPos = Position::fromSourceLocation(MI->getDefinitionLoc(),
                                             AST->getSourceManager());
    S.Roles = Roles;
    Symbols.push_back(std::move(S));
    return true;
  }

  std::vector<TestSymbol> Symbols;
  const ASTContext *AST = nullptr;
};

class IndexAction : public ASTFrontendAction {
public:
  IndexAction(std::shared_ptr<Indexer> Index,
              IndexingOptions Opts = IndexingOptions())
      : Index(std::move(Index)), Opts(Opts) {}

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    class Consumer : public ASTConsumer {
      std::shared_ptr<Indexer> Index;
      std::shared_ptr<Preprocessor> PP;
      IndexingOptions Opts;

    public:
      Consumer(std::shared_ptr<Indexer> Index, std::shared_ptr<Preprocessor> PP,
               IndexingOptions Opts)
          : Index(std::move(Index)), PP(std::move(PP)), Opts(Opts) {}

      void HandleTranslationUnit(ASTContext &Ctx) override {
        std::vector<Decl *> DeclsToIndex(
            Ctx.getTranslationUnitDecl()->decls().begin(),
            Ctx.getTranslationUnitDecl()->decls().end());
        indexTopLevelDecls(Ctx, *PP, DeclsToIndex, *Index, Opts);
      }
    };
    return std::make_unique<Consumer>(Index, CI.getPreprocessorPtr(), Opts);
  }

private:
  std::shared_ptr<Indexer> Index;
  IndexingOptions Opts;
};

using testing::AllOf;
using testing::Contains;
using testing::Not;
using testing::UnorderedElementsAre;

MATCHER_P(QName, Name, "") { return arg.QName == Name; }
MATCHER_P(WrittenAt, Pos, "") { return arg.WrittenPos == Pos; }
MATCHER_P(DeclAt, Pos, "") { return arg.DeclPos == Pos; }
MATCHER_P(Kind, SymKind, "") { return arg.SymInfo.Kind == SymKind; }
MATCHER_P(HasRole, Role, "") { return arg.Roles & static_cast<unsigned>(Role); }

TEST(IndexTest, Simple) {
  auto Index = std::make_shared<Indexer>();
  tooling::runToolOnCode(std::make_unique<IndexAction>(Index),
                         "class X {}; void f() {}");
  EXPECT_THAT(Index->Symbols, UnorderedElementsAre(QName("X"), QName("f")));
}

TEST(IndexTest, IndexPreprocessorMacros) {
  std::string Code = R"cpp(
    #define INDEX_MAC 1
    #define INDEX_MAC_UNDEF 1
    #undef INDEX_MAC_UNDEF
    #define INDEX_MAC_REDEF 1
    #undef INDEX_MAC_REDEF
    #define INDEX_MAC_REDEF 2
  )cpp";
  auto Index = std::make_shared<Indexer>();
  IndexingOptions Opts;
  Opts.IndexMacrosInPreprocessor = true;
  tooling::runToolOnCode(std::make_unique<IndexAction>(Index, Opts), Code);
  EXPECT_THAT(Index->Symbols,
              Contains(AllOf(QName("INDEX_MAC"), WrittenAt(Position(2, 13)),
                             DeclAt(Position(2, 13)),
                             HasRole(SymbolRole::Definition))));
  EXPECT_THAT(
      Index->Symbols,
      AllOf(Contains(AllOf(QName("INDEX_MAC_UNDEF"), WrittenAt(Position(3, 13)),
                           DeclAt(Position(3, 13)),
                           HasRole(SymbolRole::Definition))),
            Contains(AllOf(QName("INDEX_MAC_UNDEF"), WrittenAt(Position(4, 12)),
                           DeclAt(Position(3, 13)),
                           HasRole(SymbolRole::Undefinition)))));
  EXPECT_THAT(
      Index->Symbols,
      AllOf(Contains(AllOf(QName("INDEX_MAC_REDEF"), WrittenAt(Position(5, 13)),
                           DeclAt(Position(5, 13)),
                           HasRole(SymbolRole::Definition))),
            Contains(AllOf(QName("INDEX_MAC_REDEF"), WrittenAt(Position(6, 12)),
                           DeclAt(Position(5, 13)),
                           HasRole(SymbolRole::Undefinition))),
            Contains(AllOf(QName("INDEX_MAC_REDEF"), WrittenAt(Position(7, 13)),
                           DeclAt(Position(7, 13)),
                           HasRole(SymbolRole::Definition)))));

  Opts.IndexMacrosInPreprocessor = false;
  Index->Symbols.clear();
  tooling::runToolOnCode(std::make_unique<IndexAction>(Index, Opts), Code);
  EXPECT_THAT(Index->Symbols, UnorderedElementsAre());
}

TEST(IndexTest, IndexParametersInDecls) {
  std::string Code = "void foo(int bar);";
  auto Index = std::make_shared<Indexer>();
  IndexingOptions Opts;
  Opts.IndexFunctionLocals = true;
  Opts.IndexParametersInDeclarations = true;
  tooling::runToolOnCode(std::make_unique<IndexAction>(Index, Opts), Code);
  EXPECT_THAT(Index->Symbols, Contains(QName("bar")));

  Opts.IndexParametersInDeclarations = false;
  Index->Symbols.clear();
  tooling::runToolOnCode(std::make_unique<IndexAction>(Index, Opts), Code);
  EXPECT_THAT(Index->Symbols, Not(Contains(QName("bar"))));
}

TEST(IndexTest, IndexExplicitTemplateInstantiation) {
  std::string Code = R"cpp(
    template <typename T>
    struct Foo { void bar() {} };
    template <>
    struct Foo<int> { void bar() {} };
    void foo() {
      Foo<char> abc;
      Foo<int> b;
    }
  )cpp";
  auto Index = std::make_shared<Indexer>();
  IndexingOptions Opts;
  tooling::runToolOnCode(std::make_unique<IndexAction>(Index, Opts), Code);
  EXPECT_THAT(Index->Symbols,
              AllOf(Contains(AllOf(QName("Foo"), WrittenAt(Position(8, 7)),
                                   DeclAt(Position(5, 12)))),
                    Contains(AllOf(QName("Foo"), WrittenAt(Position(7, 7)),
                                   DeclAt(Position(3, 12))))));
}

TEST(IndexTest, IndexTemplateInstantiationPartial) {
  std::string Code = R"cpp(
    template <typename T1, typename T2>
    struct Foo { void bar() {} };
    template <typename T>
    struct Foo<T, int> { void bar() {} };
    void foo() {
      Foo<char, char> abc;
      Foo<int, int> b;
    }
  )cpp";
  auto Index = std::make_shared<Indexer>();
  IndexingOptions Opts;
  tooling::runToolOnCode(std::make_unique<IndexAction>(Index, Opts), Code);
  EXPECT_THAT(Index->Symbols,
              Contains(AllOf(QName("Foo"), WrittenAt(Position(8, 7)),
                             DeclAt(Position(5, 12)))));
}

TEST(IndexTest, IndexTypeParmDecls) {
  std::string Code = R"cpp(
    template <typename T, int I, template<typename> class C, typename NoRef>
    struct Foo {
      T t = I;
      C<int> x;
    };
  )cpp";
  auto Index = std::make_shared<Indexer>();
  IndexingOptions Opts;
  tooling::runToolOnCode(std::make_unique<IndexAction>(Index, Opts), Code);
  EXPECT_THAT(Index->Symbols, AllOf(Not(Contains(QName("Foo::T"))),
                                    Not(Contains(QName("Foo::I"))),
                                    Not(Contains(QName("Foo::C"))),
                                    Not(Contains(QName("Foo::NoRef")))));

  Opts.IndexTemplateParameters = true;
  Index->Symbols.clear();
  tooling::runToolOnCode(std::make_unique<IndexAction>(Index, Opts), Code);
  EXPECT_THAT(Index->Symbols,
              AllOf(Contains(AllOf(QName("Foo::T"),
                                   Kind(SymbolKind::TemplateTypeParm))),
                    Contains(AllOf(QName("Foo::I"),
                                   Kind(SymbolKind::NonTypeTemplateParm))),
                    Contains(AllOf(QName("Foo::C"),
                                   Kind(SymbolKind::TemplateTemplateParm))),
                    Contains(QName("Foo::NoRef"))));
}

TEST(IndexTest, UsingDecls) {
  std::string Code = R"cpp(
    void foo(int bar);
    namespace std {
      using ::foo;
    }
  )cpp";
  auto Index = std::make_shared<Indexer>();
  IndexingOptions Opts;
  tooling::runToolOnCode(std::make_unique<IndexAction>(Index, Opts), Code);
  EXPECT_THAT(Index->Symbols,
              Contains(AllOf(QName("std::foo"), Kind(SymbolKind::Using))));
}

TEST(IndexTest, Constructors) {
  std::string Code = R"cpp(
    struct Foo {
      Foo(int);
      ~Foo();
    };
  )cpp";
  auto Index = std::make_shared<Indexer>();
  IndexingOptions Opts;
  tooling::runToolOnCode(std::make_unique<IndexAction>(Index, Opts), Code);
  EXPECT_THAT(
      Index->Symbols,
      UnorderedElementsAre(
          AllOf(QName("Foo"), Kind(SymbolKind::Struct),
                WrittenAt(Position(2, 12))),
          AllOf(QName("Foo::Foo"), Kind(SymbolKind::Constructor),
                WrittenAt(Position(3, 7))),
          AllOf(QName("Foo"), Kind(SymbolKind::Struct),
                HasRole(SymbolRole::NameReference), WrittenAt(Position(3, 7))),
          AllOf(QName("Foo::~Foo"), Kind(SymbolKind::Destructor),
                WrittenAt(Position(4, 7))),
          AllOf(QName("Foo"), Kind(SymbolKind::Struct),
                HasRole(SymbolRole::NameReference),
                WrittenAt(Position(4, 8)))));
}

TEST(IndexTest, InjecatedNameClass) {
  std::string Code = R"cpp(
    template <typename T>
    class Foo {
      void f(Foo x);
    };
  )cpp";
  auto Index = std::make_shared<Indexer>();
  IndexingOptions Opts;
  tooling::runToolOnCode(std::make_unique<IndexAction>(Index, Opts), Code);
  EXPECT_THAT(Index->Symbols,
              UnorderedElementsAre(AllOf(QName("Foo"), Kind(SymbolKind::Class),
                                         WrittenAt(Position(3, 11))),
                                   AllOf(QName("Foo::f"),
                                         Kind(SymbolKind::InstanceMethod),
                                         WrittenAt(Position(4, 12))),
                                   AllOf(QName("Foo"), Kind(SymbolKind::Class),
                                         HasRole(SymbolRole::Reference),
                                         WrittenAt(Position(4, 14)))));
}

TEST(IndexTest, VisitDefaultArgs) {
  std::string Code = R"cpp(
    int var = 0;
    void f(int s = var) {}
  )cpp";
  auto Index = std::make_shared<Indexer>();
  IndexingOptions Opts;
  Opts.IndexFunctionLocals = true;
  Opts.IndexParametersInDeclarations = true;
  tooling::runToolOnCode(std::make_unique<IndexAction>(Index, Opts), Code);
  EXPECT_THAT(Index->Symbols,
              Contains(AllOf(QName("var"), HasRole(SymbolRole::Reference),
                             WrittenAt(Position(3, 20)))));
}

TEST(IndexTest, RelationBaseOf) {
  std::string Code = R"cpp(
    class A {};
    template <typename> class B {};
    class C : B<A> {};
  )cpp";
  auto Index = std::make_shared<Indexer>();
  tooling::runToolOnCode(std::make_unique<IndexAction>(Index), Code);
  // A should not be the base of anything.
  EXPECT_THAT(Index->Symbols,
              Contains(AllOf(QName("A"), HasRole(SymbolRole::Reference),
                             Not(HasRole(SymbolRole::RelationBaseOf)))));
}

TEST(IndexTest, EnumBase) {
  std::string Code = R"cpp(
    typedef int MyTypedef;
    enum Foo : MyTypedef;
    enum Foo : MyTypedef {};
  )cpp";
  auto Index = std::make_shared<Indexer>();
  tooling::runToolOnCode(std::make_unique<IndexAction>(Index), Code);
  EXPECT_THAT(
      Index->Symbols,
      AllOf(Contains(AllOf(QName("MyTypedef"), HasRole(SymbolRole::Reference),
                           WrittenAt(Position(3, 16)))),
            Contains(AllOf(QName("MyTypedef"), HasRole(SymbolRole::Reference),
                           WrittenAt(Position(4, 16))))));
}
} // namespace
} // namespace index
} // namespace clang
