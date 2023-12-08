//===-- InsertionPointTess.cpp  -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "Protocol.h"
#include "SourceCode.h"
#include "TestTU.h"
#include "XRefs.h"
#include "refactor/InsertionPoint.h"
#include "clang/AST/DeclBase.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {
using llvm::HasValue;

TEST(InsertionPointTests, Generic) {
  Annotations Code(R"cpp(
  namespace ns {
    $a^int a1;
    $b^// leading comment
    int b;
    $c^int c1; // trailing comment
    int c2;
    $a2^int a2;
  $end^};
  )cpp");

  auto StartsWith =
      [&](llvm::StringLiteral S) -> std::function<bool(const Decl *)> {
    return [S](const Decl *D) {
      if (const auto *ND = llvm::dyn_cast<NamedDecl>(D))
        return llvm::StringRef(ND->getNameAsString()).startswith(S);
      return false;
    };
  };

  auto AST = TestTU::withCode(Code.code()).build();
  auto &NS = cast<NamespaceDecl>(findDecl(AST, "ns"));

  // Test single anchors.
  auto Point = [&](llvm::StringLiteral Prefix, Anchor::Dir Direction) {
    auto Loc = insertionPoint(NS, {Anchor{StartsWith(Prefix), Direction}});
    return sourceLocToPosition(AST.getSourceManager(), Loc);
  };
  EXPECT_EQ(Point("a", Anchor::Above), Code.point("a"));
  EXPECT_EQ(Point("a", Anchor::Below), Code.point("b"));
  EXPECT_EQ(Point("b", Anchor::Above), Code.point("b"));
  EXPECT_EQ(Point("b", Anchor::Below), Code.point("c"));
  EXPECT_EQ(Point("c", Anchor::Above), Code.point("c"));
  EXPECT_EQ(Point("c", Anchor::Below), Code.point("a2"));
  EXPECT_EQ(Point("", Anchor::Above), Code.point("a"));
  EXPECT_EQ(Point("", Anchor::Below), Code.point("end"));
  EXPECT_EQ(Point("no_match", Anchor::Below), Position{});

  // Test anchor chaining.
  auto Chain = [&](llvm::StringLiteral P1, llvm::StringLiteral P2) {
    auto Loc = insertionPoint(NS, {Anchor{StartsWith(P1), Anchor::Above},
                                   Anchor{StartsWith(P2), Anchor::Above}});
    return sourceLocToPosition(AST.getSourceManager(), Loc);
  };
  EXPECT_EQ(Chain("a", "b"), Code.point("a"));
  EXPECT_EQ(Chain("b", "a"), Code.point("b"));
  EXPECT_EQ(Chain("no_match", "a"), Code.point("a"));

  // Test edit generation.
  auto Edit = insertDecl("foo;", NS, {Anchor{StartsWith("a"), Anchor::Below}});
  ASSERT_THAT_EXPECTED(Edit, llvm::Succeeded());
  EXPECT_EQ(offsetToPosition(Code.code(), Edit->getOffset()), Code.point("b"));
  EXPECT_EQ(Edit->getReplacementText(), "foo;");
  // If no match, the edit is inserted at the end.
  Edit = insertDecl("x;", NS, {Anchor{StartsWith("no_match"), Anchor::Below}});
  ASSERT_THAT_EXPECTED(Edit, llvm::Succeeded());
  EXPECT_EQ(offsetToPosition(Code.code(), Edit->getOffset()),
            Code.point("end"));
}

// For CXX, we should check:
// - special handling for access specifiers
// - unwrapping of template decls
TEST(InsertionPointTests, CXX) {
  Annotations Code(R"cpp(
    class C {
    public:
      $Method^void pubMethod();
      $Field^int PubField;

    $private^private:
      $field^int PrivField;
      $method^void privMethod();
      template <typename T> void privTemplateMethod();
    $end^};
  )cpp");

  auto AST = TestTU::withCode(Code.code()).build();
  const CXXRecordDecl &C = cast<CXXRecordDecl>(findDecl(AST, "C"));

  auto IsMethod = [](const Decl *D) { return llvm::isa<CXXMethodDecl>(D); };
  auto Any = [](const Decl *D) { return true; };

  // Test single anchors.
  auto Point = [&](Anchor A, AccessSpecifier Protection) {
    auto Loc = insertionPoint(C, {A}, Protection);
    return sourceLocToPosition(AST.getSourceManager(), Loc);
  };
  EXPECT_EQ(Point({IsMethod, Anchor::Above}, AS_public), Code.point("Method"));
  EXPECT_EQ(Point({IsMethod, Anchor::Below}, AS_public), Code.point("Field"));
  EXPECT_EQ(Point({Any, Anchor::Above}, AS_public), Code.point("Method"));
  EXPECT_EQ(Point({Any, Anchor::Below}, AS_public), Code.point("private"));
  EXPECT_EQ(Point({IsMethod, Anchor::Above}, AS_private), Code.point("method"));
  EXPECT_EQ(Point({IsMethod, Anchor::Below}, AS_private), Code.point("end"));
  EXPECT_EQ(Point({Any, Anchor::Above}, AS_private), Code.point("field"));
  EXPECT_EQ(Point({Any, Anchor::Below}, AS_private), Code.point("end"));
  EXPECT_EQ(Point({IsMethod, Anchor::Above}, AS_protected), Position{});
  EXPECT_EQ(Point({IsMethod, Anchor::Below}, AS_protected), Position{});
  EXPECT_EQ(Point({Any, Anchor::Above}, AS_protected), Position{});
  EXPECT_EQ(Point({Any, Anchor::Below}, AS_protected), Position{});

  // Edits when there's no match --> end of matching access control section.
  auto Edit = insertDecl("x", C, {}, AS_public);
  ASSERT_THAT_EXPECTED(Edit, llvm::Succeeded());
  EXPECT_EQ(offsetToPosition(Code.code(), Edit->getOffset()),
            Code.point("private"));

  Edit = insertDecl("x", C, {}, AS_private);
  ASSERT_THAT_EXPECTED(Edit, llvm::Succeeded());
  EXPECT_EQ(offsetToPosition(Code.code(), Edit->getOffset()),
            Code.point("end"));

  Edit = insertDecl("x", C, {}, AS_protected);
  ASSERT_THAT_EXPECTED(Edit, llvm::Succeeded());
  EXPECT_EQ(offsetToPosition(Code.code(), Edit->getOffset()),
            Code.point("end"));
  EXPECT_EQ(Edit->getReplacementText(), "protected:\nx");
}

MATCHER_P(replacementText, Text, "") {
  if (arg.getReplacementText() != Text) {
    *result_listener << "replacement is " << arg.getReplacementText().str();
    return false;
  }
  return true;
}

TEST(InsertionPointTests, CXXAccessProtection) {
  // Empty class uses default access.
  auto AST = TestTU::withCode("struct S{};").build();
  const CXXRecordDecl &S = cast<CXXRecordDecl>(findDecl(AST, "S"));
  ASSERT_THAT_EXPECTED(insertDecl("x", S, {}, AS_public),
                       HasValue(replacementText("x")));
  ASSERT_THAT_EXPECTED(insertDecl("x", S, {}, AS_private),
                       HasValue(replacementText("private:\nx")));

  // We won't insert above the first access specifier if there's nothing there.
  AST = TestTU::withCode("struct T{private:};").build();
  const CXXRecordDecl &T = cast<CXXRecordDecl>(findDecl(AST, "T"));
  ASSERT_THAT_EXPECTED(insertDecl("x", T, {}, AS_public),
                       HasValue(replacementText("public:\nx")));
  ASSERT_THAT_EXPECTED(insertDecl("x", T, {}, AS_private),
                       HasValue(replacementText("x")));

  // But we will if there are declarations.
  AST = TestTU::withCode("struct U{int i;private:};").build();
  const CXXRecordDecl &U = cast<CXXRecordDecl>(findDecl(AST, "U"));
  ASSERT_THAT_EXPECTED(insertDecl("x", U, {}, AS_public),
                       HasValue(replacementText("x")));
  ASSERT_THAT_EXPECTED(insertDecl("x", U, {}, AS_private),
                       HasValue(replacementText("x")));
}

// In ObjC we need to take care to get the @end fallback right.
TEST(InsertionPointTests, ObjC) {
  Annotations Code(R"objc(
    @interface Foo
     -(void) v;
    $endIface^@end
    @implementation Foo
     -(void) v {}
    $endImpl^@end
  )objc");
  auto TU = TestTU::withCode(Code.code());
  TU.Filename = "TestTU.m";
  auto AST = TU.build();

  auto &Impl =
      cast<ObjCImplementationDecl>(findDecl(AST, [&](const NamedDecl &D) {
        return llvm::isa<ObjCImplementationDecl>(D);
      }));
  auto &Iface = *Impl.getClassInterface();
  Anchor End{[](const Decl *) { return true; }, Anchor::Below};

  const auto &SM = AST.getSourceManager();
  EXPECT_EQ(sourceLocToPosition(SM, insertionPoint(Iface, {End})),
            Code.point("endIface"));
  EXPECT_EQ(sourceLocToPosition(SM, insertionPoint(Impl, {End})),
            Code.point("endImpl"));
}

} // namespace
} // namespace clangd
} // namespace clang
