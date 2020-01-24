//===-- ASTTests.cpp --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AST.h"

#include "Annotations.h"
#include "ParsedAST.h"
#include "TestTU.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstddef>
#include <string>
#include <vector>

namespace clang {
namespace clangd {
namespace {

TEST(GetDeducedType, KwAutoExpansion) {
  struct Test {
    StringRef AnnotatedCode;
    const char *DeducedType;
  } Tests[] = {
      {"^auto i = 0;", "int"},
      {"^auto f(){ return 1;};", "int"},
  };
  for (Test T : Tests) {
    Annotations File(T.AnnotatedCode);
    auto AST = TestTU::withCode(File.code()).build();
    SourceManagerForFile SM("foo.cpp", File.code());

    for (Position Pos : File.points()) {
      auto Location = sourceLocationInMainFile(SM.get(), Pos);
      ASSERT_TRUE(!!Location) << llvm::toString(Location.takeError());
      auto DeducedType = getDeducedType(AST.getASTContext(), *Location);
      EXPECT_EQ(DeducedType->getAsString(), T.DeducedType);
    }
  }
}

TEST(ClangdAST, GetQualification) {
  // Tries to insert the decl `Foo` into position of each decl named `insert`.
  // This is done to get an appropriate DeclContext for the insertion location.
  // Qualifications are the required nested name specifier to spell `Foo` at the
  // `insert`ion location.
  // VisibleNamespaces are assumed to be visible at every insertion location.
  const struct {
    llvm::StringRef Test;
    std::vector<llvm::StringRef> Qualifications;
    std::vector<std::string> VisibleNamespaces;
  } Cases[] = {
      {
          R"cpp(
            namespace ns1 { namespace ns2 { class Foo {}; } }
            void insert(); // ns1::ns2::Foo
            namespace ns1 {
              void insert(); // ns2::Foo
              namespace ns2 {
                void insert(); // Foo
              }
              using namespace ns2;
              void insert(); // Foo
            }
            using namespace ns1;
            void insert(); // ns2::Foo
            using namespace ns2;
            void insert(); // Foo
          )cpp",
          {"ns1::ns2::", "ns2::", "", "", "ns2::", ""},
          {},
      },
      {
          R"cpp(
            namespace ns1 { namespace ns2 { class Bar { void Foo(); }; } }
            void insert(); // ns1::ns2::Bar::Foo
            namespace ns1 {
              void insert(); // ns2::Bar::Foo
              namespace ns2 {
                void insert(); // Bar::Foo
              }
              using namespace ns2;
              void insert(); // Bar::Foo
            }
            using namespace ns1;
            void insert(); // ns2::Bar::Foo
            using namespace ns2;
            void insert(); // Bar::Foo
          )cpp",
          {"ns1::ns2::Bar::", "ns2::Bar::", "Bar::", "Bar::", "ns2::Bar::",
           "Bar::"},
          {},
      },
      {
          R"cpp(
            namespace ns1 { namespace ns2 { void Foo(); } }
            void insert(); // ns2::Foo
            namespace ns1 {
              void insert(); // ns2::Foo
              namespace ns2 {
                void insert(); // Foo
              }
            }
          )cpp",
          {"ns2::", "ns2::", ""},
          {"ns1::"},
      },
  };
  for (const auto &Case : Cases) {
    Annotations Test(Case.Test);
    TestTU TU = TestTU::withCode(Test.code());
    ParsedAST AST = TU.build();
    std::vector<const Decl *> InsertionPoints;
    const NamedDecl *TargetDecl;
    findDecl(AST, [&](const NamedDecl &ND) {
      if (ND.getNameAsString() == "Foo") {
        TargetDecl = &ND;
        return true;
      }

      if (ND.getNameAsString() == "insert")
        InsertionPoints.push_back(&ND);
      return false;
    });

    ASSERT_EQ(InsertionPoints.size(), Case.Qualifications.size());
    for (size_t I = 0, E = InsertionPoints.size(); I != E; ++I) {
      const Decl *D = InsertionPoints[I];
      if (Case.VisibleNamespaces.empty()) {
        EXPECT_EQ(getQualification(AST.getASTContext(),
                                   D->getLexicalDeclContext(), D->getBeginLoc(),
                                   TargetDecl),
                  Case.Qualifications[I]);
      } else {
        EXPECT_EQ(getQualification(AST.getASTContext(),
                                   D->getLexicalDeclContext(), TargetDecl,
                                   Case.VisibleNamespaces),
                  Case.Qualifications[I]);
      }
    }
  }
}

TEST(ClangdAST, PrintType) {
  const struct {
    llvm::StringRef Test;
    std::vector<llvm::StringRef> Types;
  } Cases[] = {
      {
          R"cpp(
            namespace ns1 { namespace ns2 { class Foo {}; } }
            void insert(); // ns1::ns2::Foo
            namespace ns1 {
              void insert(); // ns2::Foo
              namespace ns2 {
                void insert(); // Foo
              }
            }
          )cpp",
          {"ns1::ns2::Foo", "ns2::Foo", "Foo"},
      },
      {
          R"cpp(
            namespace ns1 {
              typedef int Foo;
            }
            void insert(); // ns1::Foo
            namespace ns1 {
              void insert(); // Foo
            }
          )cpp",
          {"ns1::Foo", "Foo"},
      },
  };
  for (const auto &Case : Cases) {
    Annotations Test(Case.Test);
    TestTU TU = TestTU::withCode(Test.code());
    ParsedAST AST = TU.build();
    std::vector<const DeclContext *> InsertionPoints;
    const TypeDecl *TargetDecl = nullptr;
    findDecl(AST, [&](const NamedDecl &ND) {
      if (ND.getNameAsString() == "Foo") {
        if (const auto *TD = llvm::dyn_cast<TypeDecl>(&ND)) {
          TargetDecl = TD;
          return true;
        }
      } else if (ND.getNameAsString() == "insert")
        InsertionPoints.push_back(ND.getDeclContext());
      return false;
    });

    ASSERT_EQ(InsertionPoints.size(), Case.Types.size());
    for (size_t I = 0, E = InsertionPoints.size(); I != E; ++I) {
      const auto *DC = InsertionPoints[I];
      EXPECT_EQ(printType(AST.getASTContext().getTypeDeclType(TargetDecl), *DC),
                Case.Types[I]);
    }
  }
}
} // namespace
} // namespace clangd
} // namespace clang
