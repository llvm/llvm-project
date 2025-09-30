//===- unittest/AST/RecursiveASTVisitorTest.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
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

class CollectInterestingEvents : public DynamicRecursiveASTVisitor {
public:
  bool TraverseFunctionDecl(FunctionDecl *D) override {
    Events.push_back(VisitEvent::StartTraverseFunction);
    bool Ret = DynamicRecursiveASTVisitor::TraverseFunctionDecl(D);
    Events.push_back(VisitEvent::EndTraverseFunction);

    return Ret;
  }

  bool TraverseAttr(Attr *A) override {
    Events.push_back(VisitEvent::StartTraverseAttr);
    bool Ret = DynamicRecursiveASTVisitor::TraverseAttr(A);
    Events.push_back(VisitEvent::EndTraverseAttr);

    return Ret;
  }

  bool TraverseEnumDecl(EnumDecl *D) override {
    Events.push_back(VisitEvent::StartTraverseEnum);
    bool Ret = DynamicRecursiveASTVisitor::TraverseEnumDecl(D);
    Events.push_back(VisitEvent::EndTraverseEnum);

    return Ret;
  }

  bool TraverseTypedefTypeLoc(TypedefTypeLoc TL,
                              bool TraverseQualifier) override {
    Events.push_back(VisitEvent::StartTraverseTypedefType);
    bool Ret = DynamicRecursiveASTVisitor::TraverseTypedefTypeLoc(
        TL, TraverseQualifier);
    Events.push_back(VisitEvent::EndTraverseTypedefType);

    return Ret;
  }

  bool TraverseObjCInterfaceDecl(ObjCInterfaceDecl *ID) override {
    Events.push_back(VisitEvent::StartTraverseObjCInterface);
    bool Ret = DynamicRecursiveASTVisitor::TraverseObjCInterfaceDecl(ID);
    Events.push_back(VisitEvent::EndTraverseObjCInterface);

    return Ret;
  }

  bool TraverseObjCProtocolDecl(ObjCProtocolDecl *PD) override {
    Events.push_back(VisitEvent::StartTraverseObjCProtocol);
    bool Ret = DynamicRecursiveASTVisitor::TraverseObjCProtocolDecl(PD);
    Events.push_back(VisitEvent::EndTraverseObjCProtocol);

    return Ret;
  }

  bool TraverseObjCProtocolLoc(ObjCProtocolLoc ProtocolLoc) override {
    Events.push_back(VisitEvent::StartTraverseObjCProtocolLoc);
    bool Ret = DynamicRecursiveASTVisitor::TraverseObjCProtocolLoc(ProtocolLoc);
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
} // namespace

TEST(RecursiveASTVisitorTest, AttributesInsideDecls) {
  /// Check attributes are traversed inside TraverseFunctionDecl.
  llvm::StringRef Code = R"cpp(
__attribute__((annotate("something"))) int foo() { return 10; }
  )cpp";

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
