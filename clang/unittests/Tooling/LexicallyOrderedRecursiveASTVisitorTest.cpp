//===- unittest/Tooling/LexicallyOrderedRecursiveASTVisitorTest.cpp -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"
#include "clang/AST/LexicallyOrderedRecursiveASTVisitor.h"
#include <stack>

using namespace clang;

namespace {

class DummyMatchVisitor;

class LexicallyOrderedDeclVisitor
    : public LexicallyOrderedRecursiveASTVisitor<LexicallyOrderedDeclVisitor> {
public:
  LexicallyOrderedDeclVisitor(DummyMatchVisitor &Matcher,
                              const SourceManager &SM)
      : LexicallyOrderedRecursiveASTVisitor(SM), Matcher(Matcher) {}

  bool TraverseDecl(Decl *D) {
    TraversalStack.push_back(D);
    LexicallyOrderedRecursiveASTVisitor::TraverseDecl(D);
    TraversalStack.pop_back();
    return true;
  }

  bool VisitNamedDecl(const NamedDecl *D);

private:
  DummyMatchVisitor &Matcher;
  llvm::SmallVector<Decl *, 8> TraversalStack;
};

class DummyMatchVisitor : public ExpectedLocationVisitor<DummyMatchVisitor> {
public:
  bool VisitTranslationUnitDecl(TranslationUnitDecl *TU) {
    const ASTContext &Context = TU->getASTContext();
    const SourceManager &SM = Context.getSourceManager();
    LexicallyOrderedDeclVisitor SubVisitor(*this, SM);
    SubVisitor.TraverseDecl(TU);
    return false;
  }

  void match(StringRef Path, const Decl *D) { Match(Path, D->getLocStart()); }
};

bool LexicallyOrderedDeclVisitor::VisitNamedDecl(const NamedDecl *D) {
  std::string Path;
  llvm::raw_string_ostream OS(Path);
  assert(TraversalStack.back() == D);
  for (const Decl *D : TraversalStack) {
    if (isa<TranslationUnitDecl>(D)) {
      OS << "/";
      continue;
    }
    if (const auto *ND = dyn_cast<NamedDecl>(D))
      OS << ND->getNameAsString();
    else
      OS << "???";
    if (isa<DeclContext>(D))
      OS << "/";
  }
  Matcher.match(OS.str(), D);
  return true;
}

TEST(LexicallyOrderedRecursiveASTVisitor, VisitDeclsInImplementation) {
  StringRef Source = R"(
@interface I
@end
@implementation I

int nestedFunction() { }

- (void) method{ }

int anotherNestedFunction(int x) {
  return x;
}

int innerVariable = 0;

@end

int outerVariable = 0;

@implementation I(Cat)

void catF() { }

@end

void outerFunction() { }
)";
  DummyMatchVisitor Visitor;
  Visitor.DisallowMatch("/nestedFunction/", 6, 1);
  Visitor.ExpectMatch("/I/nestedFunction/", 6, 1);
  Visitor.ExpectMatch("/I/method/", 8, 1);
  Visitor.DisallowMatch("/anotherNestedFunction/", 10, 1);
  Visitor.ExpectMatch("/I/anotherNestedFunction/", 10, 1);
  Visitor.DisallowMatch("/innerVariable", 14, 1);
  Visitor.ExpectMatch("/I/innerVariable", 14, 1);
  Visitor.ExpectMatch("/outerVariable", 18, 1);
  Visitor.DisallowMatch("/catF/", 22, 1);
  Visitor.ExpectMatch("/Cat/catF/", 22, 1);
  Visitor.ExpectMatch("/outerFunction/", 26, 1);
  EXPECT_TRUE(Visitor.runOver(Source, DummyMatchVisitor::Lang_OBJC));
}

TEST(LexicallyOrderedRecursiveASTVisitor, VisitMacroDeclsInImplementation) {
  StringRef Source = R"(
@interface I
@end

void outerFunction() { }

#define MACRO_F(x) void nestedFunction##x() { }

@implementation I

MACRO_F(1)

@end

MACRO_F(2)
)";
  DummyMatchVisitor Visitor;
  Visitor.ExpectMatch("/outerFunction/", 5, 1);
  Visitor.ExpectMatch("/I/nestedFunction1/", 7, 20);
  Visitor.ExpectMatch("/nestedFunction2/", 7, 20);
  EXPECT_TRUE(Visitor.runOver(Source, DummyMatchVisitor::Lang_OBJC));
}

} // end anonymous namespace
