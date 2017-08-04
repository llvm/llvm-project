//===- unittest/Tooling/LookupTest.cpp ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"
#include "clang/Tooling/Core/Lookup.h"
using namespace clang;

namespace {
struct GetDeclsVisitor : TestVisitor<GetDeclsVisitor> {
  std::function<void(CallExpr *)> OnCall;
  std::function<void(RecordTypeLoc)> OnRecordTypeLoc;
  SmallVector<Decl *, 4> DeclStack;

  bool VisitCallExpr(CallExpr *Expr) {
    if (OnCall)
      OnCall(Expr);
    return true;
  }

  bool VisitRecordTypeLoc(RecordTypeLoc Loc) {
    if (OnRecordTypeLoc)
      OnRecordTypeLoc(Loc);
    return true;
  }

  bool TraverseDecl(Decl *D) {
    DeclStack.push_back(D);
    bool Ret = TestVisitor::TraverseDecl(D);
    DeclStack.pop_back();
    return Ret;
  }
};

TEST(LookupTest, replaceNestedFunctionName) {
  GetDeclsVisitor Visitor;

  auto replaceCallExpr = [&](const CallExpr *Expr,
                             StringRef ReplacementString) {
    const auto *Callee = cast<DeclRefExpr>(Expr->getCallee()->IgnoreImplicit());
    const ValueDecl *FD = Callee->getDecl();
    return tooling::replaceNestedName(
        Callee->getQualifier(), Visitor.DeclStack.back()->getDeclContext(), FD,
        ReplacementString);
  };

  Visitor.OnCall = [&](CallExpr *Expr) {
    EXPECT_EQ("bar", replaceCallExpr(Expr, "::bar"));
  };
  Visitor.runOver("namespace a { void foo(); }\n"
                  "namespace a { void f() { foo(); } }\n");

  Visitor.OnCall = [&](CallExpr *Expr) {
    EXPECT_EQ("bar", replaceCallExpr(Expr, "::a::bar"));
  };
  Visitor.runOver("namespace a { void foo(); }\n"
                  "namespace a { void f() { foo(); } }\n");

  Visitor.OnCall = [&](CallExpr *Expr) {
    EXPECT_EQ("a::bar", replaceCallExpr(Expr, "::a::bar"));
  };
  Visitor.runOver("namespace a { void foo(); }\n"
                  "namespace b { void f() { a::foo(); } }\n");

  Visitor.OnCall = [&](CallExpr *Expr) {
    EXPECT_EQ("a::bar", replaceCallExpr(Expr, "::a::bar"));
  };
  Visitor.runOver("namespace a { void foo(); }\n"
                  "namespace b { namespace a { void foo(); }\n"
                  "void f() { a::foo(); } }\n");

  Visitor.OnCall = [&](CallExpr *Expr) {
    EXPECT_EQ("c::bar", replaceCallExpr(Expr, "::a::c::bar"));
  };
  Visitor.runOver("namespace a { namespace b { void foo(); }\n"
                  "void f() { b::foo(); } }\n");

  Visitor.OnCall = [&](CallExpr *Expr) {
    EXPECT_EQ("bar", replaceCallExpr(Expr, "::a::bar"));
  };
  Visitor.runOver("namespace a { namespace b { void foo(); }\n"
                  "void f() { b::foo(); } }\n");

  Visitor.OnCall = [&](CallExpr *Expr) {
    EXPECT_EQ("bar", replaceCallExpr(Expr, "::bar"));
  };
  Visitor.runOver("void foo(); void f() { foo(); }\n");

  Visitor.OnCall = [&](CallExpr *Expr) {
    EXPECT_EQ("::bar", replaceCallExpr(Expr, "::bar"));
  };
  Visitor.runOver("void foo(); void f() { ::foo(); }\n");

  Visitor.OnCall = [&](CallExpr *Expr) {
    EXPECT_EQ("a::bar", replaceCallExpr(Expr, "::a::bar"));
  };
  Visitor.runOver("namespace a { void foo(); }\nvoid f() { a::foo(); }\n");

  Visitor.OnCall = [&](CallExpr *Expr) {
    EXPECT_EQ("a::bar", replaceCallExpr(Expr, "::a::bar"));
  };
  Visitor.runOver("namespace a { int foo(); }\nauto f = a::foo();\n");

  Visitor.OnCall = [&](CallExpr *Expr) {
    EXPECT_EQ("bar", replaceCallExpr(Expr, "::a::bar"));
  };
  Visitor.runOver(
      "namespace a { int foo(); }\nusing a::foo;\nauto f = foo();\n");

  Visitor.OnCall = [&](CallExpr *Expr) {
    EXPECT_EQ("c::bar", replaceCallExpr(Expr, "::a::c::bar"));
  };
  Visitor.runOver("namespace a { namespace b { void foo(); } }\n"
                  "namespace a { namespace b { namespace {"
                  "void f() { foo(); }"
                  "} } }\n");

  Visitor.OnCall = [&](CallExpr *Expr) {
    EXPECT_EQ("x::bar", replaceCallExpr(Expr, "::a::x::bar"));
  };
  Visitor.runOver("namespace a { namespace b { void foo(); } }\n"
                  "namespace a { namespace b { namespace c {"
                  "void f() { foo(); }"
                  "} } }\n");
}

TEST(LookupTest, replaceNestedClassName) {
  GetDeclsVisitor Visitor;

  auto replaceRecordTypeLoc = [&](RecordTypeLoc Loc,
                                  StringRef ReplacementString) {
    const auto *FD = cast<CXXRecordDecl>(Loc.getDecl());
    return tooling::replaceNestedName(
        nullptr, Visitor.DeclStack.back()->getDeclContext(), FD,
        ReplacementString);
  };

  Visitor.OnRecordTypeLoc = [&](RecordTypeLoc Type) {
    // Filter Types by name since there are other `RecordTypeLoc` in the test
    // file.
    if (Type.getDecl()->getQualifiedNameAsString() == "a::b::Foo") {
      EXPECT_EQ("x::Bar", replaceRecordTypeLoc(Type, "::a::x::Bar"));
    }
  };
  Visitor.runOver("namespace a { namespace b {\n"
                  "class Foo;\n"
                  "namespace c { Foo f();; }\n"
                  "} }\n");

  Visitor.OnRecordTypeLoc = [&](RecordTypeLoc Type) {
    // Filter Types by name since there are other `RecordTypeLoc` in the test
    // file.
    // `a::b::Foo` in using shadow decl is not `TypeLoc`.
    if (Type.getDecl()->getQualifiedNameAsString() == "a::b::Foo") {
      EXPECT_EQ("Bar", replaceRecordTypeLoc(Type, "::a::x::Bar"));
    }
  };
  Visitor.runOver("namespace a { namespace b { class Foo {}; } }\n"
                  "namespace c { using a::b::Foo; Foo f();; }\n");
}

} // end anonymous namespace
