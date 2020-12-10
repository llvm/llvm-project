//===-- CallHierarchyTests.cpp  ---------------------------*- C++ -*-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Annotations.h"
#include "Compiler.h"
#include "Matchers.h"
#include "ParsedAST.h"
#include "SyncAPI.h"
#include "TestFS.h"
#include "TestTU.h"
#include "TestWorkspace.h"
#include "XRefs.h"
#include "index/FileIndex.h"
#include "index/SymbolCollector.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Index/IndexingAction.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ScopedPrinter.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::Matcher;
using ::testing::UnorderedElementsAre;

// Helpers for matching call hierarchy data structures.
MATCHER_P(WithName, N, "") { return arg.name == N; }
MATCHER_P(WithSelectionRange, R, "") { return arg.selectionRange == R; }

template <class ItemMatcher>
::testing::Matcher<CallHierarchyIncomingCall> From(ItemMatcher M) {
  return Field(&CallHierarchyIncomingCall::from, M);
}
template <class... RangeMatchers>
::testing::Matcher<CallHierarchyIncomingCall> FromRanges(RangeMatchers... M) {
  return Field(&CallHierarchyIncomingCall::fromRanges,
               UnorderedElementsAre(M...));
}

TEST(CallHierarchy, IncomingOneFile) {
  Annotations Source(R"cpp(
    void call^ee(int);
    void caller1() {
      $Callee[[callee]](42);
    }
    void caller2() {
      $Caller1A[[caller1]]();
      $Caller1B[[caller1]]();
    }
    void caller3() {
      $Caller1C[[caller1]]();
      $Caller2[[caller2]]();
    }
  )cpp");
  TestTU TU = TestTU::withCode(Source.code());
  auto AST = TU.build();
  auto Index = TU.index();

  std::vector<CallHierarchyItem> Items =
      prepareCallHierarchy(AST, Source.point(), testPath(TU.Filename));
  ASSERT_THAT(Items, ElementsAre(WithName("callee")));
  auto IncomingLevel1 = incomingCalls(Items[0], Index.get());
  ASSERT_THAT(IncomingLevel1,
              ElementsAre(AllOf(From(WithName("caller1")),
                                FromRanges(Source.range("Callee")))));

  auto IncomingLevel2 = incomingCalls(IncomingLevel1[0].from, Index.get());
  ASSERT_THAT(IncomingLevel2,
              ElementsAre(AllOf(From(WithName("caller2")),
                                FromRanges(Source.range("Caller1A"),
                                           Source.range("Caller1B"))),
                          AllOf(From(WithName("caller3")),
                                FromRanges(Source.range("Caller1C")))));

  auto IncomingLevel3 = incomingCalls(IncomingLevel2[0].from, Index.get());
  ASSERT_THAT(IncomingLevel3,
              ElementsAre(AllOf(From(WithName("caller3")),
                                FromRanges(Source.range("Caller2")))));

  auto IncomingLevel4 = incomingCalls(IncomingLevel3[0].from, Index.get());
  EXPECT_THAT(IncomingLevel4, IsEmpty());
}

TEST(CallHierarchy, MainFileOnlyRef) {
  // In addition to testing that we store refs to main-file only symbols,
  // this tests that anonymous namespaces do not interfere with the
  // symbol re-identification process in callHierarchyItemToSymbo().
  Annotations Source(R"cpp(
    void call^ee(int);
    namespace {
      void caller1() {
        $Callee[[callee]](42);
      }
    }
    void caller2() {
      $Caller1[[caller1]]();
    }
  )cpp");
  TestTU TU = TestTU::withCode(Source.code());
  auto AST = TU.build();
  auto Index = TU.index();

  std::vector<CallHierarchyItem> Items =
      prepareCallHierarchy(AST, Source.point(), testPath(TU.Filename));
  ASSERT_THAT(Items, ElementsAre(WithName("callee")));
  auto IncomingLevel1 = incomingCalls(Items[0], Index.get());
  ASSERT_THAT(IncomingLevel1,
              ElementsAre(AllOf(From(WithName("caller1")),
                                FromRanges(Source.range("Callee")))));

  auto IncomingLevel2 = incomingCalls(IncomingLevel1[0].from, Index.get());
  EXPECT_THAT(IncomingLevel2,
              ElementsAre(AllOf(From(WithName("caller2")),
                                FromRanges(Source.range("Caller1")))));
}

TEST(CallHierarchy, IncomingQualified) {
  Annotations Source(R"cpp(
    namespace ns {
    struct Waldo {
      void find();
    };
    void Waldo::find() {}
    void caller1(Waldo &W) {
      W.$Caller1[[f^ind]]();
    }
    void caller2(Waldo &W) {
      W.$Caller2[[find]]();
    }
    }
  )cpp");
  TestTU TU = TestTU::withCode(Source.code());
  auto AST = TU.build();
  auto Index = TU.index();

  std::vector<CallHierarchyItem> Items =
      prepareCallHierarchy(AST, Source.point(), testPath(TU.Filename));
  ASSERT_THAT(Items, ElementsAre(WithName("Waldo::find")));
  auto Incoming = incomingCalls(Items[0], Index.get());
  EXPECT_THAT(Incoming,
              ElementsAre(AllOf(From(WithName("caller1")),
                                FromRanges(Source.range("Caller1"))),
                          AllOf(From(WithName("caller2")),
                                FromRanges(Source.range("Caller2")))));
}

TEST(CallHierarchy, IncomingMultiFile) {
  // The test uses a .hh suffix for header files to get clang
  // to parse them in C++ mode. .h files are parsed in C mode
  // by default, which causes problems because e.g. symbol
  // USRs are different in C mode (do not include function signatures).

  Annotations CalleeH(R"cpp(
    void calle^e(int);
  )cpp");
  Annotations CalleeC(R"cpp(
    #include "callee.hh"
    void calle^e(int) {}
  )cpp");
  Annotations Caller1H(R"cpp(
    void caller1();
  )cpp");
  Annotations Caller1C(R"cpp(
    #include "callee.hh"
    #include "caller1.hh"
    void caller1() {
      [[calle^e]](42);
    }
  )cpp");
  Annotations Caller2H(R"cpp(
    void caller2();
  )cpp");
  Annotations Caller2C(R"cpp(
    #include "caller1.hh"
    #include "caller2.hh"
    void caller2() {
      $A[[caller1]]();
      $B[[caller1]]();
    }
  )cpp");
  Annotations Caller3C(R"cpp(
    #include "caller1.hh"
    #include "caller2.hh"
    void caller3() {
      $Caller1[[caller1]]();
      $Caller2[[caller2]]();
    }
  )cpp");

  TestWorkspace Workspace;
  Workspace.addSource("callee.hh", CalleeH.code());
  Workspace.addSource("caller1.hh", Caller1H.code());
  Workspace.addSource("caller2.hh", Caller2H.code());
  Workspace.addMainFile("callee.cc", CalleeC.code());
  Workspace.addMainFile("caller1.cc", Caller1C.code());
  Workspace.addMainFile("caller2.cc", Caller2C.code());
  Workspace.addMainFile("caller3.cc", Caller3C.code());

  auto Index = Workspace.index();

  auto CheckCallHierarchy = [&](ParsedAST &AST, Position Pos, PathRef TUPath) {
    std::vector<CallHierarchyItem> Items =
        prepareCallHierarchy(AST, Pos, TUPath);
    ASSERT_THAT(Items, ElementsAre(WithName("callee")));
    auto IncomingLevel1 = incomingCalls(Items[0], Index.get());
    ASSERT_THAT(IncomingLevel1,
                ElementsAre(AllOf(From(WithName("caller1")),
                                  FromRanges(Caller1C.range()))));

    auto IncomingLevel2 = incomingCalls(IncomingLevel1[0].from, Index.get());
    ASSERT_THAT(
        IncomingLevel2,
        ElementsAre(AllOf(From(WithName("caller2")),
                          FromRanges(Caller2C.range("A"), Caller2C.range("B"))),
                    AllOf(From(WithName("caller3")),
                          FromRanges(Caller3C.range("Caller1")))));

    auto IncomingLevel3 = incomingCalls(IncomingLevel2[0].from, Index.get());
    ASSERT_THAT(IncomingLevel3,
                ElementsAre(AllOf(From(WithName("caller3")),
                                  FromRanges(Caller3C.range("Caller2")))));

    auto IncomingLevel4 = incomingCalls(IncomingLevel3[0].from, Index.get());
    EXPECT_THAT(IncomingLevel4, IsEmpty());
  };

  // Check that invoking from a call site works.
  auto AST = Workspace.openFile("caller1.cc");
  ASSERT_TRUE(bool(AST));
  CheckCallHierarchy(*AST, Caller1C.point(), testPath("caller1.cc"));

  // Check that invoking from the declaration site works.
  AST = Workspace.openFile("callee.hh");
  ASSERT_TRUE(bool(AST));
  CheckCallHierarchy(*AST, CalleeH.point(), testPath("callee.hh"));

  // Check that invoking from the definition site works.
  AST = Workspace.openFile("callee.cc");
  ASSERT_TRUE(bool(AST));
  CheckCallHierarchy(*AST, CalleeC.point(), testPath("callee.cc"));
}

} // namespace
} // namespace clangd
} // namespace clang
