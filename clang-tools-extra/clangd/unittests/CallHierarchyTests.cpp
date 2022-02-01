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

llvm::raw_ostream &operator<<(llvm::raw_ostream &Stream,
                              const CallHierarchyItem &Item) {
  return Stream << Item.name << "@" << Item.selectionRange;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &Stream,
                              const CallHierarchyIncomingCall &Call) {
  Stream << "{ from: " << Call.from << ", ranges: [";
  for (const auto &R : Call.fromRanges) {
    Stream << R;
    Stream << ", ";
  }
  return Stream << "] }";
}

namespace {

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::Matcher;
using ::testing::UnorderedElementsAre;

// Helpers for matching call hierarchy data structures.
MATCHER_P(withName, N, "") { return arg.name == N; }
MATCHER_P(withSelectionRange, R, "") { return arg.selectionRange == R; }

template <class ItemMatcher>
::testing::Matcher<CallHierarchyIncomingCall> from(ItemMatcher M) {
  return Field(&CallHierarchyIncomingCall::from, M);
}
template <class... RangeMatchers>
::testing::Matcher<CallHierarchyIncomingCall> fromRanges(RangeMatchers... M) {
  return Field(&CallHierarchyIncomingCall::fromRanges,
               UnorderedElementsAre(M...));
}

TEST(CallHierarchy, IncomingOneFileCpp) {
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
  ASSERT_THAT(Items, ElementsAre(withName("callee")));
  auto IncomingLevel1 = incomingCalls(Items[0], Index.get());
  ASSERT_THAT(IncomingLevel1,
              ElementsAre(AllOf(from(withName("caller1")),
                                fromRanges(Source.range("Callee")))));
  auto IncomingLevel2 = incomingCalls(IncomingLevel1[0].from, Index.get());
  ASSERT_THAT(IncomingLevel2,
              ElementsAre(AllOf(from(withName("caller2")),
                                fromRanges(Source.range("Caller1A"),
                                           Source.range("Caller1B"))),
                          AllOf(from(withName("caller3")),
                                fromRanges(Source.range("Caller1C")))));

  auto IncomingLevel3 = incomingCalls(IncomingLevel2[0].from, Index.get());
  ASSERT_THAT(IncomingLevel3,
              ElementsAre(AllOf(from(withName("caller3")),
                                fromRanges(Source.range("Caller2")))));

  auto IncomingLevel4 = incomingCalls(IncomingLevel3[0].from, Index.get());
  EXPECT_THAT(IncomingLevel4, IsEmpty());
}

TEST(CallHierarchy, IncomingOneFileObjC) {
  Annotations Source(R"objc(
    @implementation MyClass {}
      +(void)call^ee {}
      +(void) caller1 {
        [MyClass $Callee[[callee]]];
      }
      +(void) caller2 {
        [MyClass $Caller1A[[caller1]]];
        [MyClass $Caller1B[[caller1]]];
      }
      +(void) caller3 {
        [MyClass $Caller1C[[caller1]]];
        [MyClass $Caller2[[caller2]]];
      }
    @end
  )objc");
  TestTU TU = TestTU::withCode(Source.code());
  TU.Filename = "TestTU.m";
  auto AST = TU.build();
  auto Index = TU.index();
  std::vector<CallHierarchyItem> Items =
      prepareCallHierarchy(AST, Source.point(), testPath(TU.Filename));
  ASSERT_THAT(Items, ElementsAre(withName("callee")));
  auto IncomingLevel1 = incomingCalls(Items[0], Index.get());
  ASSERT_THAT(IncomingLevel1,
              ElementsAre(AllOf(from(withName("caller1")),
                                fromRanges(Source.range("Callee")))));
  auto IncomingLevel2 = incomingCalls(IncomingLevel1[0].from, Index.get());
  ASSERT_THAT(IncomingLevel2,
              ElementsAre(AllOf(from(withName("caller2")),
                                fromRanges(Source.range("Caller1A"),
                                           Source.range("Caller1B"))),
                          AllOf(from(withName("caller3")),
                                fromRanges(Source.range("Caller1C")))));

  auto IncomingLevel3 = incomingCalls(IncomingLevel2[0].from, Index.get());
  ASSERT_THAT(IncomingLevel3,
              ElementsAre(AllOf(from(withName("caller3")),
                                fromRanges(Source.range("Caller2")))));

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
  ASSERT_THAT(Items, ElementsAre(withName("callee")));
  auto IncomingLevel1 = incomingCalls(Items[0], Index.get());
  ASSERT_THAT(IncomingLevel1,
              ElementsAre(AllOf(from(withName("caller1")),
                                fromRanges(Source.range("Callee")))));

  auto IncomingLevel2 = incomingCalls(IncomingLevel1[0].from, Index.get());
  EXPECT_THAT(IncomingLevel2,
              ElementsAre(AllOf(from(withName("caller2")),
                                fromRanges(Source.range("Caller1")))));
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
  ASSERT_THAT(Items, ElementsAre(withName("Waldo::find")));
  auto Incoming = incomingCalls(Items[0], Index.get());
  EXPECT_THAT(Incoming,
              ElementsAre(AllOf(from(withName("caller1")),
                                fromRanges(Source.range("Caller1"))),
                          AllOf(from(withName("caller2")),
                                fromRanges(Source.range("Caller2")))));
}

TEST(CallHierarchy, IncomingMultiFileCpp) {
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
    ASSERT_THAT(Items, ElementsAre(withName("callee")));
    auto IncomingLevel1 = incomingCalls(Items[0], Index.get());
    ASSERT_THAT(IncomingLevel1,
                ElementsAre(AllOf(from(withName("caller1")),
                                  fromRanges(Caller1C.range()))));

    auto IncomingLevel2 = incomingCalls(IncomingLevel1[0].from, Index.get());
    ASSERT_THAT(
        IncomingLevel2,
        ElementsAre(AllOf(from(withName("caller2")),
                          fromRanges(Caller2C.range("A"), Caller2C.range("B"))),
                    AllOf(from(withName("caller3")),
                          fromRanges(Caller3C.range("Caller1")))));

    auto IncomingLevel3 = incomingCalls(IncomingLevel2[0].from, Index.get());
    ASSERT_THAT(IncomingLevel3,
                ElementsAre(AllOf(from(withName("caller3")),
                                  fromRanges(Caller3C.range("Caller2")))));

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

TEST(CallHierarchy, IncomingMultiFileObjC) {
  // The test uses a .mi suffix for header files to get clang
  // to parse them in ObjC mode. .h files are parsed in C mode
  // by default, which causes problems because e.g. symbol
  // USRs are different in C mode (do not include function signatures).

  Annotations CalleeH(R"objc(
    @interface CalleeClass
      +(void)call^ee;
    @end
  )objc");
  Annotations CalleeC(R"objc(
    #import "callee.mi"
    @implementation CalleeClass {}
      +(void)call^ee {}
    @end
  )objc");
  Annotations Caller1H(R"objc(
    @interface Caller1Class
      +(void)caller1;
    @end
  )objc");
  Annotations Caller1C(R"objc(
    #import "callee.mi"
    #import "caller1.mi"
    @implementation Caller1Class {}
      +(void)caller1 {
        [CalleeClass [[calle^e]]];
      }
    @end
  )objc");
  Annotations Caller2H(R"objc(
    @interface Caller2Class
      +(void)caller2;
    @end
  )objc");
  Annotations Caller2C(R"objc(
    #import "caller1.mi"
    #import "caller2.mi"
    @implementation Caller2Class {}
      +(void)caller2 {
        [Caller1Class $A[[caller1]]];
        [Caller1Class $B[[caller1]]];
      }
    @end
  )objc");
  Annotations Caller3C(R"objc(
    #import "caller1.mi"
    #import "caller2.mi"
    @implementation Caller3Class {}
      +(void)caller3 {
        [Caller1Class $Caller1[[caller1]]];
        [Caller2Class $Caller2[[caller2]]];
      }
    @end
  )objc");

  TestWorkspace Workspace;
  Workspace.addSource("callee.mi", CalleeH.code());
  Workspace.addSource("caller1.mi", Caller1H.code());
  Workspace.addSource("caller2.mi", Caller2H.code());
  Workspace.addMainFile("callee.m", CalleeC.code());
  Workspace.addMainFile("caller1.m", Caller1C.code());
  Workspace.addMainFile("caller2.m", Caller2C.code());
  Workspace.addMainFile("caller3.m", Caller3C.code());
  auto Index = Workspace.index();

  auto CheckCallHierarchy = [&](ParsedAST &AST, Position Pos, PathRef TUPath) {
    std::vector<CallHierarchyItem> Items =
        prepareCallHierarchy(AST, Pos, TUPath);
    ASSERT_THAT(Items, ElementsAre(withName("callee")));
    auto IncomingLevel1 = incomingCalls(Items[0], Index.get());
    ASSERT_THAT(IncomingLevel1,
                ElementsAre(AllOf(from(withName("caller1")),
                                  fromRanges(Caller1C.range()))));

    auto IncomingLevel2 = incomingCalls(IncomingLevel1[0].from, Index.get());
    ASSERT_THAT(
        IncomingLevel2,
        ElementsAre(AllOf(from(withName("caller2")),
                          fromRanges(Caller2C.range("A"), Caller2C.range("B"))),
                    AllOf(from(withName("caller3")),
                          fromRanges(Caller3C.range("Caller1")))));

    auto IncomingLevel3 = incomingCalls(IncomingLevel2[0].from, Index.get());
    ASSERT_THAT(IncomingLevel3,
                ElementsAre(AllOf(from(withName("caller3")),
                                  fromRanges(Caller3C.range("Caller2")))));

    auto IncomingLevel4 = incomingCalls(IncomingLevel3[0].from, Index.get());
    EXPECT_THAT(IncomingLevel4, IsEmpty());
  };

  // Check that invoking from a call site works.
  auto AST = Workspace.openFile("caller1.m");
  ASSERT_TRUE(bool(AST));
  CheckCallHierarchy(*AST, Caller1C.point(), testPath("caller1.m"));

  // Check that invoking from the declaration site works.
  AST = Workspace.openFile("callee.mi");
  ASSERT_TRUE(bool(AST));
  CheckCallHierarchy(*AST, CalleeH.point(), testPath("callee.mi"));

  // Check that invoking from the definition site works.
  AST = Workspace.openFile("callee.m");
  ASSERT_TRUE(bool(AST));
  CheckCallHierarchy(*AST, CalleeC.point(), testPath("callee.m"));
}

TEST(CallHierarchy, CallInLocalVarDecl) {
  // Tests that local variable declarations are not treated as callers
  // (they're not indexed, so they can't be represented as call hierarchy
  // items); instead, the caller should be the containing function.
  // However, namespace-scope variable declarations should be treated as
  // callers because those are indexed and there is no enclosing entity
  // that would be a useful caller.
  Annotations Source(R"cpp(
    int call^ee();
    void caller1() {
      $call1[[callee]]();
    }
    void caller2() {
      int localVar = $call2[[callee]]();
    }
    int caller3 = $call3[[callee]]();
  )cpp");
  TestTU TU = TestTU::withCode(Source.code());
  auto AST = TU.build();
  auto Index = TU.index();

  std::vector<CallHierarchyItem> Items =
      prepareCallHierarchy(AST, Source.point(), testPath(TU.Filename));
  ASSERT_THAT(Items, ElementsAre(withName("callee")));

  auto Incoming = incomingCalls(Items[0], Index.get());
  ASSERT_THAT(
      Incoming,
      ElementsAre(
          AllOf(from(withName("caller1")), fromRanges(Source.range("call1"))),
          AllOf(from(withName("caller2")), fromRanges(Source.range("call2"))),
          AllOf(from(withName("caller3")), fromRanges(Source.range("call3")))));
}

} // namespace
} // namespace clangd
} // namespace clang
