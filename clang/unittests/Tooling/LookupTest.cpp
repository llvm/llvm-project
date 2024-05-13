//===- unittest/Tooling/LookupTest.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactoring/Lookup.h"
#include "TestVisitor.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/SourceLocation.h"
using namespace clang;

namespace {
struct GetDeclsVisitor : TestVisitor<GetDeclsVisitor> {
  std::function<void(CallExpr *)> OnCall;
  std::function<void(RecordTypeLoc)> OnRecordTypeLoc;
  std::function<void(UsingTypeLoc)> OnUsingTypeLoc;
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

  bool VisitUsingTypeLoc(UsingTypeLoc Loc) {
    if (OnUsingTypeLoc)
      OnUsingTypeLoc(Loc);
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
        Callee->getQualifier(), Callee->getLocation(),
        Visitor.DeclStack.back()->getDeclContext(), FD, ReplacementString);
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
    EXPECT_EQ("::a::bar", replaceCallExpr(Expr, "::a::bar"));
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

  // If the shortest name is ambiguous, we need to add more qualifiers.
  Visitor.OnCall = [&](CallExpr *Expr) {
    EXPECT_EQ("a::y::bar", replaceCallExpr(Expr, "::a::y::bar"));
  };
  Visitor.runOver(R"(
    namespace a {
     namespace b {
      namespace x { void foo() {} }
      namespace y { void foo() {} }
     }
    }

    namespace a {
     namespace b {
      void f() { x::foo(); }
     }
    })");

  Visitor.OnCall = [&](CallExpr *Expr) {
    // y::bar would be ambiguous due to "a::b::y".
    EXPECT_EQ("::y::bar", replaceCallExpr(Expr, "::y::bar"));
  };
  Visitor.runOver(R"(
    namespace a {
     namespace b {
      void foo() {}
      namespace y { }
     }
    }

    namespace a {
     namespace b {
      void f() { foo(); }
     }
    })");

  Visitor.OnCall = [&](CallExpr *Expr) {
    EXPECT_EQ("y::bar", replaceCallExpr(Expr, "::y::bar"));
  };
  Visitor.runOver(R"(
    namespace a {
    namespace b {
    namespace x { void foo() {} }
    namespace y { void foo() {} }
    }
    }

    void f() { a::b::x::foo(); }
    )");
}

TEST(LookupTest, replaceNestedClassName) {
  GetDeclsVisitor Visitor;

  auto replaceTypeLoc = [&](const NamedDecl *ND, SourceLocation Loc,
                            StringRef ReplacementString) {
    return tooling::replaceNestedName(
        nullptr, Loc, Visitor.DeclStack.back()->getDeclContext(), ND,
        ReplacementString);
  };

  Visitor.OnRecordTypeLoc = [&](RecordTypeLoc Type) {
    // Filter Types by name since there are other `RecordTypeLoc` in the test
    // file.
    if (Type.getDecl()->getQualifiedNameAsString() == "a::b::Foo") {
      EXPECT_EQ("x::Bar", replaceTypeLoc(Type.getDecl(), Type.getBeginLoc(),
                                         "::a::x::Bar"));
    }
  };
  Visitor.runOver("namespace a { namespace b {\n"
                  "class Foo;\n"
                  "namespace c { Foo f();; }\n"
                  "} }\n");

  Visitor.OnUsingTypeLoc = [&](UsingTypeLoc Type) {
    // Filter Types by name since there are other `RecordTypeLoc` in the test
    // file.
    // `a::b::Foo` in using shadow decl is not `TypeLoc`.
    auto *TD = Type.getFoundDecl()->getTargetDecl();
    if (TD->getQualifiedNameAsString() == "a::b::Foo") {
      EXPECT_EQ("Bar", replaceTypeLoc(TD, Type.getBeginLoc(), "::a::x::Bar"));
    }
  };
  Visitor.runOver("namespace a { namespace b { class Foo {}; } }\n"
                  "namespace c { using a::b::Foo; Foo f();; }\n");

  // Rename TypeLoc `x::y::Old` to new name `x::Foo` at [0] and check that the
  // type is replaced with "Foo" instead of "x::Foo". Although there is a symbol
  // `x::y::Foo` in c.cc [1], it should not make "Foo" at [0] ambiguous because
  // it's not visible at [0].
  Visitor.OnRecordTypeLoc = [&](RecordTypeLoc Type) {
    if (Type.getDecl()->getQualifiedNameAsString() == "x::y::Old") {
      EXPECT_EQ("Foo",
                replaceTypeLoc(Type.getDecl(), Type.getBeginLoc(), "::x::Foo"));
    }
  };
  Visitor.runOver(R"(
    // a.h
    namespace x {
     namespace y {
      class Old {};
      class Other {};
     }
    }

    // b.h
    namespace x {
     namespace y {
      // This is to be renamed to x::Foo
      // The expected replacement is "Foo".
      Old f;  // [0].
     }
    }

    // c.cc
    namespace x {
    namespace y {
     using Foo = ::x::y::Other; // [1]
    }
    }
    )");
}

} // end anonymous namespace
