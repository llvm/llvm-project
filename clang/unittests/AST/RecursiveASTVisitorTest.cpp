//===- unittest/AST/RecursiveASTVisitorTest.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cassert>

using namespace clang;
using ::testing::ElementsAre;

namespace {
class ProcessASTAction : public clang::ASTFrontendAction {
public:
  ProcessASTAction(llvm::unique_function<void(clang::ASTContext &)> Process)
      : Process(std::move(Process)) {
    assert(this->Process);
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) {
    class Consumer : public ASTConsumer {
    public:
      Consumer(llvm::function_ref<void(ASTContext &CTx)> Process)
          : Process(Process) {}

      void HandleTranslationUnit(ASTContext &Ctx) override { Process(Ctx); }

    private:
      llvm::function_ref<void(ASTContext &CTx)> Process;
    };

    return std::make_unique<Consumer>(Process);
  }

private:
  llvm::unique_function<void(clang::ASTContext &)> Process;
};

enum class VisitEvent {
  StartTraverseFunction,
  EndTraverseFunction,
  StartTraverseAttr,
  EndTraverseAttr,
  StartTraverseEnum,
  EndTraverseEnum,
  StartTraverseTypedefType,
  EndTraverseTypedefType,
  StartTraverseObjCInterface,
  EndTraverseObjCInterface,
  StartTraverseObjCProtocol,
  EndTraverseObjCProtocol,
  StartTraverseObjCProtocolLoc,
  EndTraverseObjCProtocolLoc,
};

class CollectInterestingEvents
    : public RecursiveASTVisitor<CollectInterestingEvents> {
public:
  bool TraverseFunctionDecl(FunctionDecl *D) {
    Events.push_back(VisitEvent::StartTraverseFunction);
    bool Ret = RecursiveASTVisitor::TraverseFunctionDecl(D);
    Events.push_back(VisitEvent::EndTraverseFunction);

    return Ret;
  }

  bool TraverseAttr(Attr *A) {
    Events.push_back(VisitEvent::StartTraverseAttr);
    bool Ret = RecursiveASTVisitor::TraverseAttr(A);
    Events.push_back(VisitEvent::EndTraverseAttr);

    return Ret;
  }

  bool TraverseEnumDecl(EnumDecl *D) {
    Events.push_back(VisitEvent::StartTraverseEnum);
    bool Ret = RecursiveASTVisitor::TraverseEnumDecl(D);
    Events.push_back(VisitEvent::EndTraverseEnum);

    return Ret;
  }

  bool TraverseTypedefTypeLoc(TypedefTypeLoc TL, bool TraverseQualifier) {
    Events.push_back(VisitEvent::StartTraverseTypedefType);
    bool Ret =
        RecursiveASTVisitor::TraverseTypedefTypeLoc(TL, TraverseQualifier);
    Events.push_back(VisitEvent::EndTraverseTypedefType);

    return Ret;
  }

  bool TraverseObjCInterfaceDecl(ObjCInterfaceDecl *ID) {
    Events.push_back(VisitEvent::StartTraverseObjCInterface);
    bool Ret = RecursiveASTVisitor::TraverseObjCInterfaceDecl(ID);
    Events.push_back(VisitEvent::EndTraverseObjCInterface);

    return Ret;
  }

  bool TraverseObjCProtocolDecl(ObjCProtocolDecl *PD) {
    Events.push_back(VisitEvent::StartTraverseObjCProtocol);
    bool Ret = RecursiveASTVisitor::TraverseObjCProtocolDecl(PD);
    Events.push_back(VisitEvent::EndTraverseObjCProtocol);

    return Ret;
  }

  bool TraverseObjCProtocolLoc(ObjCProtocolLoc ProtocolLoc) {
    Events.push_back(VisitEvent::StartTraverseObjCProtocolLoc);
    bool Ret = RecursiveASTVisitor::TraverseObjCProtocolLoc(ProtocolLoc);
    Events.push_back(VisitEvent::EndTraverseObjCProtocolLoc);

    return Ret;
  }

  std::vector<VisitEvent> takeEvents() && { return std::move(Events); }

private:
  std::vector<VisitEvent> Events;
};

std::vector<VisitEvent> collectEvents(llvm::StringRef Code,
                                      const Twine &FileName = "input.cc") {
  CollectInterestingEvents Visitor;
  clang::tooling::runToolOnCode(
      std::make_unique<ProcessASTAction>(
          [&](clang::ASTContext &Ctx) { Visitor.TraverseAST(Ctx); }),
      Code, FileName);
  return std::move(Visitor).takeEvents();
}
class ConstCollectInterestingEvents
    : public ConstRecursiveASTVisitor<ConstCollectInterestingEvents> {
public:
  bool TraverseFunctionDecl(const FunctionDecl *D) {
    Events.push_back(VisitEvent::StartTraverseFunction);
    bool Ret = ConstRecursiveASTVisitor::TraverseFunctionDecl(D);
    Events.push_back(VisitEvent::EndTraverseFunction);

    return Ret;
  }

  bool TraverseAttr(const Attr *A) {
    Events.push_back(VisitEvent::StartTraverseAttr);
    bool Ret = ConstRecursiveASTVisitor::TraverseAttr(A);
    Events.push_back(VisitEvent::EndTraverseAttr);

    return Ret;
  }

  bool TraverseEnumDecl(const EnumDecl *D) {
    Events.push_back(VisitEvent::StartTraverseEnum);
    bool Ret = ConstRecursiveASTVisitor::TraverseEnumDecl(D);
    Events.push_back(VisitEvent::EndTraverseEnum);

    return Ret;
  }

  bool TraverseTypedefTypeLoc(TypedefTypeLoc TL, bool TraverseQualifier) {
    Events.push_back(VisitEvent::StartTraverseTypedefType);
    bool Ret =
        ConstRecursiveASTVisitor::TraverseTypedefTypeLoc(TL, TraverseQualifier);
    Events.push_back(VisitEvent::EndTraverseTypedefType);

    return Ret;
  }

  bool TraverseObjCInterfaceDecl(const ObjCInterfaceDecl *ID) {
    Events.push_back(VisitEvent::StartTraverseObjCInterface);
    bool Ret = ConstRecursiveASTVisitor::TraverseObjCInterfaceDecl(ID);
    Events.push_back(VisitEvent::EndTraverseObjCInterface);

    return Ret;
  }

  bool TraverseObjCProtocolDecl(const ObjCProtocolDecl *PD) {
    Events.push_back(VisitEvent::StartTraverseObjCProtocol);
    bool Ret = ConstRecursiveASTVisitor::TraverseObjCProtocolDecl(PD);
    Events.push_back(VisitEvent::EndTraverseObjCProtocol);

    return Ret;
  }

  bool TraverseObjCProtocolLoc(ObjCProtocolLoc ProtocolLoc) {
    Events.push_back(VisitEvent::StartTraverseObjCProtocolLoc);
    bool Ret = ConstRecursiveASTVisitor::TraverseObjCProtocolLoc(ProtocolLoc);
    Events.push_back(VisitEvent::EndTraverseObjCProtocolLoc);

    return Ret;
  }

  std::vector<VisitEvent> takeEvents() && { return std::move(Events); }

private:
  std::vector<VisitEvent> Events;
};

std::vector<VisitEvent> collectConstEvents(llvm::StringRef Code,
                                           const Twine &FileName = "input.cc") {
  ConstCollectInterestingEvents Visitor;
  clang::tooling::runToolOnCode(
      std::make_unique<ProcessASTAction>(
          [&](const clang::ASTContext &Ctx) { Visitor.TraverseAST(Ctx); }),
      Code, FileName);
  return std::move(Visitor).takeEvents();
}

} // namespace

TEST(RecursiveASTVisitorTest, AttributesInsideDecls) {
  /// Check attributes are traversed inside TraverseFunctionDecl.
  llvm::StringRef Code = R"cpp(
__attribute__((annotate("something"))) int foo() { return 10; }
  )cpp";

  EXPECT_EQ(collectEvents(Code), collectConstEvents(Code));
  EXPECT_THAT(collectEvents(Code),
              ElementsAre(VisitEvent::StartTraverseFunction,
                          VisitEvent::StartTraverseAttr,
                          VisitEvent::EndTraverseAttr,
                          VisitEvent::EndTraverseFunction));
}

TEST(RecursiveASTVisitorTest, EnumDeclWithBase) {
  // Check enum and its integer base is visited.
  llvm::StringRef Code = R"cpp(
  typedef int Foo;
  enum Bar : Foo;
  )cpp";

  EXPECT_EQ(collectEvents(Code), collectConstEvents(Code));
  EXPECT_THAT(collectEvents(Code),
              ElementsAre(VisitEvent::StartTraverseEnum,
                          VisitEvent::StartTraverseTypedefType,
                          VisitEvent::EndTraverseTypedefType,
                          VisitEvent::EndTraverseEnum));
}

TEST(RecursiveASTVisitorTest, InterfaceDeclWithProtocols) {
  // Check interface and its protocols are visited.
  llvm::StringRef Code = R"cpp(
  @protocol Foo
  @end
  @protocol Bar
  @end

  @interface SomeObject <Foo, Bar>
  @end
  )cpp";

  EXPECT_EQ(collectEvents(Code), collectConstEvents(Code));
  EXPECT_THAT(collectEvents(Code, "input.m"),
              ElementsAre(VisitEvent::StartTraverseObjCProtocol,
                          VisitEvent::EndTraverseObjCProtocol,
                          VisitEvent::StartTraverseObjCProtocol,
                          VisitEvent::EndTraverseObjCProtocol,
                          VisitEvent::StartTraverseObjCInterface,
                          VisitEvent::StartTraverseObjCProtocolLoc,
                          VisitEvent::EndTraverseObjCProtocolLoc,
                          VisitEvent::StartTraverseObjCProtocolLoc,
                          VisitEvent::EndTraverseObjCProtocolLoc,
                          VisitEvent::EndTraverseObjCInterface));
}

TEST(ConstRecursiveASTVisitorTest, ConstCorrectness) {
  // This test verifies that ConstRecursiveASTVisitor properly enforces
  // const-correctness.
  // The derived class defines const versions of the Visit* methods,
  // and they should correctly override the default implementations,
  // which is demonstrated by non-0 counters.

  class ConstCorrectnessValidator
      : public ConstRecursiveASTVisitor<ConstCorrectnessValidator> {
  public:
    bool VisitFunctionDecl(const FunctionDecl *D) {
      FunctionDeclCount++;
      return true;
    }

    bool VisitStmt(const Stmt *S) {
      StmtCount++;
      return true;
    }

    int getFunctionDeclCount() const { return FunctionDeclCount; }
    int getStmtCount() const { return StmtCount; }

  private:
    int FunctionDeclCount = 0;
    int StmtCount = 0;
  };

  llvm::StringRef Code = R"cpp(
    int foo() {
      return 42;
    }
    void bar() {
      int x = 0;
      x += 2;
    }
  )cpp";

  ConstCorrectnessValidator Visitor;
  clang::tooling::runToolOnCode(
      std::make_unique<ProcessASTAction>(
          [&](clang::ASTContext &Ctx) { Visitor.TraverseAST(Ctx); }),
      Code);

  // Verify that the visitor found the expected number of nodes
  EXPECT_EQ(Visitor.getFunctionDeclCount(), 2); // foo and bar
  // There are at least 3 statements: return 42; int x = 0; x += 2;
  EXPECT_GE(Visitor.getStmtCount(), 3);
}
