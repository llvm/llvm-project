//===-- CallHierarchyTests.cpp  ---------------------------*- C++ -*-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Annotations.h"
#include "ParsedAST.h"
#include "TestFS.h"
#include "TestTU.h"
#include "TestWorkspace.h"
#include "XRefs.h"
#include "llvm/Support/Path.h"
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
MATCHER_P(withDetail, N, "") { return arg.detail == N; }
MATCHER_P(withSelectionRange, R, "") { return arg.selectionRange == R; }

template <class ItemMatcher>
::testing::Matcher<CallHierarchyIncomingCall> from(ItemMatcher M) {
  return Field(&CallHierarchyIncomingCall::from, M);
}
template <class ItemMatcher>
::testing::Matcher<CallHierarchyOutgoingCall> to(ItemMatcher M) {
  return Field(&CallHierarchyOutgoingCall::to, M);
}
template <class... RangeMatchers>
::testing::Matcher<CallHierarchyIncomingCall> iFromRanges(RangeMatchers... M) {
  return Field(&CallHierarchyIncomingCall::fromRanges,
               UnorderedElementsAre(M...));
}
template <class... RangeMatchers>
::testing::Matcher<CallHierarchyOutgoingCall> oFromRanges(RangeMatchers... M) {
  return Field(&CallHierarchyOutgoingCall::fromRanges,
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
  ASSERT_THAT(
      IncomingLevel1,
      ElementsAre(AllOf(from(AllOf(withName("caller1"), withDetail("caller1"))),
                        iFromRanges(Source.range("Callee")))));
  auto IncomingLevel2 = incomingCalls(IncomingLevel1[0].from, Index.get());
  ASSERT_THAT(
      IncomingLevel2,
      ElementsAre(AllOf(from(AllOf(withName("caller2"), withDetail("caller2"))),
                        iFromRanges(Source.range("Caller1A"),
                                    Source.range("Caller1B"))),
                  AllOf(from(AllOf(withName("caller3"), withDetail("caller3"))),
                        iFromRanges(Source.range("Caller1C")))));

  auto IncomingLevel3 = incomingCalls(IncomingLevel2[0].from, Index.get());
  ASSERT_THAT(
      IncomingLevel3,
      ElementsAre(AllOf(from(AllOf(withName("caller3"), withDetail("caller3"))),
                        iFromRanges(Source.range("Caller2")))));

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
              ElementsAre(AllOf(from(AllOf(withName("caller1"),
                                           withDetail("MyClass::caller1"))),
                                iFromRanges(Source.range("Callee")))));
  auto IncomingLevel2 = incomingCalls(IncomingLevel1[0].from, Index.get());
  ASSERT_THAT(IncomingLevel2,
              ElementsAre(AllOf(from(AllOf(withName("caller2"),
                                           withDetail("MyClass::caller2"))),
                                iFromRanges(Source.range("Caller1A"),
                                            Source.range("Caller1B"))),
                          AllOf(from(AllOf(withName("caller3"),
                                           withDetail("MyClass::caller3"))),
                                iFromRanges(Source.range("Caller1C")))));

  auto IncomingLevel3 = incomingCalls(IncomingLevel2[0].from, Index.get());
  ASSERT_THAT(IncomingLevel3,
              ElementsAre(AllOf(from(AllOf(withName("caller3"),
                                           withDetail("MyClass::caller3"))),
                                iFromRanges(Source.range("Caller2")))));

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
  ASSERT_THAT(
      IncomingLevel1,
      ElementsAre(AllOf(from(AllOf(withName("caller1"), withDetail("caller1"))),
                        iFromRanges(Source.range("Callee")))));

  auto IncomingLevel2 = incomingCalls(IncomingLevel1[0].from, Index.get());
  EXPECT_THAT(
      IncomingLevel2,
      ElementsAre(AllOf(from(AllOf(withName("caller2"), withDetail("caller2"))),
                        iFromRanges(Source.range("Caller1")))));
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
  EXPECT_THAT(
      Incoming,
      ElementsAre(
          AllOf(from(AllOf(withName("caller1"), withDetail("ns::caller1"))),
                iFromRanges(Source.range("Caller1"))),
          AllOf(from(AllOf(withName("caller2"), withDetail("ns::caller2"))),
                iFromRanges(Source.range("Caller2")))));
}

TEST(CallHierarchy, OutgoingOneFile) {
  // Test outgoing call on the main file, with namespaces and methods
  Annotations Source(R"cpp(
    void callee(int);
    namespace ns {
      struct Foo {
        void caller1();
      };
      void Foo::caller1() {
        $Callee[[callee]](42);
      }
    }
    namespace {
      void caller2(ns::Foo& F) {
        F.$Caller1A[[caller1]]();
        F.$Caller1B[[caller1]]();
      }
    }
    void call^er3(ns::Foo& F) {
      F.$Caller1C[[caller1]]();
      $Caller2[[caller2]](F);
    }
  )cpp");
  TestTU TU = TestTU::withCode(Source.code());
  auto AST = TU.build();
  auto Index = TU.index();

  std::vector<CallHierarchyItem> Items =
      prepareCallHierarchy(AST, Source.point(), testPath(TU.Filename));
  ASSERT_THAT(Items, ElementsAre(withName("caller3")));
  auto OugoingLevel1 = outgoingCalls(Items[0], Index.get());
  ASSERT_THAT(
      OugoingLevel1,
      ElementsAre(
          AllOf(to(AllOf(withName("caller1"), withDetail("ns::Foo::caller1"))),
                oFromRanges(Source.range("Caller1C"))),
          AllOf(to(AllOf(withName("caller2"), withDetail("caller2"))),
                oFromRanges(Source.range("Caller2")))));

  auto OutgoingLevel2 = outgoingCalls(OugoingLevel1[1].to, Index.get());
  ASSERT_THAT(
      OutgoingLevel2,
      ElementsAre(AllOf(
          to(AllOf(withName("caller1"), withDetail("ns::Foo::caller1"))),
          oFromRanges(Source.range("Caller1A"), Source.range("Caller1B")))));

  auto OutgoingLevel3 = outgoingCalls(OutgoingLevel2[0].to, Index.get());
  ASSERT_THAT(
      OutgoingLevel3,
      ElementsAre(AllOf(to(AllOf(withName("callee"), withDetail("callee"))),
                        oFromRanges(Source.range("Callee")))));

  auto OutgoingLevel4 = outgoingCalls(OutgoingLevel3[0].to, Index.get());
  EXPECT_THAT(OutgoingLevel4, IsEmpty());
}

TEST(CallHierarchy, MultiFileCpp) {
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
    namespace nsa {
      void caller1();
    }
  )cpp");
  Annotations Caller1C(R"cpp(
    #include "callee.hh"
    #include "caller1.hh"
    namespace nsa {
      void caller1() {
        [[calle^e]](42);
      }
    }
  )cpp");
  Annotations Caller2H(R"cpp(
    namespace nsb {
      void caller2();
    }
  )cpp");
  Annotations Caller2C(R"cpp(
    #include "caller1.hh"
    #include "caller2.hh"
    namespace nsb {
      void caller2() {
        nsa::$A[[caller1]]();
        nsa::$B[[caller1]]();
      }
    }
  )cpp");
  Annotations Caller3H(R"cpp(
    namespace nsa {
      void call^er3();
    }
  )cpp");
  Annotations Caller3C(R"cpp(
    #include "caller1.hh"
    #include "caller2.hh"
    namespace nsa {
      void call^er3() {
        $Caller1[[caller1]]();
        nsb::$Caller2[[caller2]]();
      }
    }
  )cpp");

  TestWorkspace Workspace;
  Workspace.addSource("callee.hh", CalleeH.code());
  Workspace.addSource("caller1.hh", Caller1H.code());
  Workspace.addSource("caller2.hh", Caller2H.code());
  Workspace.addSource("caller3.hh", Caller3H.code());
  Workspace.addMainFile("callee.cc", CalleeC.code());
  Workspace.addMainFile("caller1.cc", Caller1C.code());
  Workspace.addMainFile("caller2.cc", Caller2C.code());
  Workspace.addMainFile("caller3.cc", Caller3C.code());

  auto Index = Workspace.index();

  auto CheckIncomingCalls = [&](ParsedAST &AST, Position Pos, PathRef TUPath) {
    std::vector<CallHierarchyItem> Items =
        prepareCallHierarchy(AST, Pos, TUPath);
    ASSERT_THAT(Items, ElementsAre(withName("callee")));
    auto IncomingLevel1 = incomingCalls(Items[0], Index.get());
    ASSERT_THAT(IncomingLevel1,
                ElementsAre(AllOf(from(AllOf(withName("caller1"),
                                             withDetail("nsa::caller1"))),
                                  iFromRanges(Caller1C.range()))));

    auto IncomingLevel2 = incomingCalls(IncomingLevel1[0].from, Index.get());
    ASSERT_THAT(
        IncomingLevel2,
        ElementsAre(
            AllOf(from(AllOf(withName("caller2"), withDetail("nsb::caller2"))),
                  iFromRanges(Caller2C.range("A"), Caller2C.range("B"))),
            AllOf(from(AllOf(withName("caller3"), withDetail("nsa::caller3"))),
                  iFromRanges(Caller3C.range("Caller1")))));

    auto IncomingLevel3 = incomingCalls(IncomingLevel2[0].from, Index.get());
    ASSERT_THAT(IncomingLevel3,
                ElementsAre(AllOf(from(AllOf(withName("caller3"),
                                             withDetail("nsa::caller3"))),
                                  iFromRanges(Caller3C.range("Caller2")))));

    auto IncomingLevel4 = incomingCalls(IncomingLevel3[0].from, Index.get());
    EXPECT_THAT(IncomingLevel4, IsEmpty());
  };

  auto CheckOutgoingCalls = [&](ParsedAST &AST, Position Pos, PathRef TUPath) {
    std::vector<CallHierarchyItem> Items =
        prepareCallHierarchy(AST, Pos, TUPath);
    ASSERT_THAT(Items, ElementsAre(withName("caller3")));
    auto OutgoingLevel1 = outgoingCalls(Items[0], Index.get());
    ASSERT_THAT(
        OutgoingLevel1,
        ElementsAre(
            AllOf(to(AllOf(withName("caller1"), withDetail("nsa::caller1"))),
                  oFromRanges(Caller3C.range("Caller1"))),
            AllOf(to(AllOf(withName("caller2"), withDetail("nsb::caller2"))),
                  oFromRanges(Caller3C.range("Caller2")))));

    auto OutgoingLevel2 = outgoingCalls(OutgoingLevel1[1].to, Index.get());
    ASSERT_THAT(OutgoingLevel2,
                ElementsAre(AllOf(
                    to(AllOf(withName("caller1"), withDetail("nsa::caller1"))),
                    oFromRanges(Caller2C.range("A"), Caller2C.range("B")))));

    auto OutgoingLevel3 = outgoingCalls(OutgoingLevel2[0].to, Index.get());
    ASSERT_THAT(
        OutgoingLevel3,
        ElementsAre(AllOf(to(AllOf(withName("callee"), withDetail("callee"))),
                          oFromRanges(Caller1C.range()))));

    auto OutgoingLevel4 = outgoingCalls(OutgoingLevel3[0].to, Index.get());
    EXPECT_THAT(OutgoingLevel4, IsEmpty());
  };

  // Check that invoking from a call site works.
  auto AST = Workspace.openFile("caller1.cc");
  ASSERT_TRUE(bool(AST));
  CheckIncomingCalls(*AST, Caller1C.point(), testPath("caller1.cc"));

  // Check that invoking from the declaration site works.
  AST = Workspace.openFile("callee.hh");
  ASSERT_TRUE(bool(AST));
  CheckIncomingCalls(*AST, CalleeH.point(), testPath("callee.hh"));
  AST = Workspace.openFile("caller3.hh");
  ASSERT_TRUE(bool(AST));
  CheckOutgoingCalls(*AST, Caller3H.point(), testPath("caller3.hh"));

  // Check that invoking from the definition site works.
  AST = Workspace.openFile("callee.cc");
  ASSERT_TRUE(bool(AST));
  CheckIncomingCalls(*AST, CalleeC.point(), testPath("callee.cc"));
  AST = Workspace.openFile("caller3.cc");
  ASSERT_TRUE(bool(AST));
  CheckOutgoingCalls(*AST, Caller3C.point(), testPath("caller3.cc"));
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
                                  iFromRanges(Caller1C.range()))));

    auto IncomingLevel2 = incomingCalls(IncomingLevel1[0].from, Index.get());
    ASSERT_THAT(IncomingLevel2,
                ElementsAre(AllOf(from(withName("caller2")),
                                  iFromRanges(Caller2C.range("A"),
                                              Caller2C.range("B"))),
                            AllOf(from(withName("caller3")),
                                  iFromRanges(Caller3C.range("Caller1")))));

    auto IncomingLevel3 = incomingCalls(IncomingLevel2[0].from, Index.get());
    ASSERT_THAT(IncomingLevel3,
                ElementsAre(AllOf(from(withName("caller3")),
                                  iFromRanges(Caller3C.range("Caller2")))));

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
  ASSERT_THAT(Incoming, ElementsAre(AllOf(from(withName("caller1")),
                                          iFromRanges(Source.range("call1"))),
                                    AllOf(from(withName("caller2")),
                                          iFromRanges(Source.range("call2"))),
                                    AllOf(from(withName("caller3")),
                                          iFromRanges(Source.range("call3")))));
}

TEST(CallHierarchy, HierarchyOnField) {
  // Tests that the call hierarchy works on fields.
  Annotations Source(R"cpp(
    struct Vars {
      int v^ar1 = 1;
    };
    void caller() {
      Vars values;
      values.$Callee[[var1]];
    }
  )cpp");
  TestTU TU = TestTU::withCode(Source.code());
  auto AST = TU.build();
  auto Index = TU.index();

  std::vector<CallHierarchyItem> Items =
      prepareCallHierarchy(AST, Source.point(), testPath(TU.Filename));
  ASSERT_THAT(Items, ElementsAre(withName("var1")));
  auto IncomingLevel1 = incomingCalls(Items[0], Index.get());
  ASSERT_THAT(IncomingLevel1,
              ElementsAre(AllOf(from(withName("caller")),
                                iFromRanges(Source.range("Callee")))));
}

TEST(CallHierarchy, HierarchyOnVar) {
  // Tests that the call hierarchy works on non-local variables.
  Annotations Source(R"cpp(
    int v^ar = 1;
    void caller() {
      $Callee[[var]];
    }
  )cpp");
  TestTU TU = TestTU::withCode(Source.code());
  auto AST = TU.build();
  auto Index = TU.index();

  std::vector<CallHierarchyItem> Items =
      prepareCallHierarchy(AST, Source.point(), testPath(TU.Filename));
  ASSERT_THAT(Items, ElementsAre(withName("var")));
  auto IncomingLevel1 = incomingCalls(Items[0], Index.get());
  ASSERT_THAT(IncomingLevel1,
              ElementsAre(AllOf(from(withName("caller")),
                                iFromRanges(Source.range("Callee")))));
}

TEST(CallHierarchy, CallInDifferentFileThanCaller) {
  Annotations Header(R"cpp(
    #define WALDO void caller() {
  )cpp");
  Annotations Source(R"cpp(
    void call^ee();
    WALDO
      callee();
    }
  )cpp");
  auto TU = TestTU::withCode(Source.code());
  TU.HeaderCode = Header.code();
  auto AST = TU.build();
  auto Index = TU.index();

  std::vector<CallHierarchyItem> Items =
      prepareCallHierarchy(AST, Source.point(), testPath(TU.Filename));
  ASSERT_THAT(Items, ElementsAre(withName("callee")));

  auto Incoming = incomingCalls(Items[0], Index.get());

  // The only call site is in the source file, which is a different file from
  // the declaration of the function containing the call, which is in the
  // header. The protocol does not allow us to represent such calls, so we drop
  // them. (The call hierarchy item itself is kept.)
  EXPECT_THAT(Incoming,
              ElementsAre(AllOf(from(withName("caller")), iFromRanges())));
}

} // namespace
} // namespace clangd
} // namespace clang
