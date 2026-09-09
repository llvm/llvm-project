//===-- CallHierarchyMultipleCallersTest.cpp ---------------*- C++ -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Regression tests for call hierarchy when multiple callers share the same
// function signature (and thus the same SymbolID) but are defined in different
// files. This is a common scenario when multiple binaries each define their
// own main() or other identically-named helper functions calling a shared
// library function.
//
// See https://github.com/clangd/clangd/issues/2361
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "TestFS.h"
#include "TestWorkspace.h"
#include "XRefs.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

using ::testing::Field;
using ::testing::UnorderedElementsAre;

MATCHER_P(withName, N, "") { return arg.name == N; }

template <class ItemMatcher>
::testing::Matcher<CallHierarchyIncomingCall> from(ItemMatcher M) {
  return Field(&CallHierarchyIncomingCall::from, M);
}

// Reproduces a bug where multiple callers with the same function signature
// (e.g. main() in different binaries) only show up as a single caller in the
// call hierarchy.
//
// Scenario:
//   - lib.cpp defines a shared function: util::doWork()
//   - binary1_main.cpp has main() calling util::doWork()
//   - binary2_main.cpp has main() calling util::doWork()
//   - Expected: incomingCalls(util::doWork) shows main() from BOTH files
//   - Bug: only one main() appears because they share the same SymbolID
TEST(CallHierarchyMultipleCallers, IncomingSameSignatureDifferentFiles) {
  TestWorkspace Workspace;

  Workspace.addSource("util.h", R"cpp(
    namespace util {
      int doWork(int x);
    }
  )cpp");

  Workspace.addMainFile("lib.cpp", R"cpp(
    #include "util.h"
    namespace util {
      int doWork(int x) { return x * 2; }
    }
  )cpp");

  // Binary 1: defines its own main() calling util::doWork()
  Workspace.addMainFile("binary1_main.cpp", R"cpp(
    #include "util.h"
    int main() {
      return util::doWork(42);
    }
  )cpp");

  // Binary 2: defines its own main() calling util::doWork()
  // Both main() functions have the same signature -> same SymbolID
  Workspace.addMainFile("binary2_main.cpp", R"cpp(
    #include "util.h"
    int main() {
      return util::doWork(99);
    }
  )cpp");

  auto Index = Workspace.index();
  auto AST = Workspace.openFile("lib.cpp");
  ASSERT_TRUE(bool(AST));

  Annotations Source(R"cpp(
    #include "util.h"
    namespace util {
      int doW^ork(int x) { return x * 2; }
    }
  )cpp");

  auto Items = prepareCallHierarchy(*AST, Source.point(), testPath("lib.cpp"));
  ASSERT_EQ(Items.size(), 1u);
  EXPECT_EQ(Items[0].name, "doWork");

  auto Incoming = incomingCalls(Items[0], Index.get());

  // main() from both binary1_main.cpp and binary2_main.cpp should appear.
  EXPECT_EQ(Incoming.size(), 2u)
      << "Expected 2 callers (main from binary1_main and binary2_main), got "
      << Incoming.size();
  if (Incoming.size() >= 2) {
    EXPECT_THAT(Incoming,
                UnorderedElementsAre(from(withName("main")),
                                     from(withName("main"))));
  }
}

// Variant where the callers are not main() but ordinary functions with the
// same name in different files, and each caller's URI points to the correct
// file.
TEST(CallHierarchyMultipleCallers, IncomingSameHelperInDifferentBinaries) {
  TestWorkspace Workspace;

  Workspace.addSource("util.h", R"cpp(
    namespace util {
      int add(int a, int b);
    }
  )cpp");

  Workspace.addMainFile("lib.cpp", R"cpp(
    #include "util.h"
    namespace util {
      int add(int a, int b) { return a + b; }
    }
  )cpp");

  // Binary 1: defines process() calling util::add()
  Workspace.addMainFile("binary1_process.cpp", R"cpp(
    #include "util.h"
    int process() {
      return util::add(1, 2);
    }
  )cpp");

  // Binary 2: defines process() calling util::add()
  // Same function name & signature -> same SymbolID
  Workspace.addMainFile("binary2_process.cpp", R"cpp(
    #include "util.h"
    int process() {
      return util::add(3, 4);
    }
  )cpp");

  auto Index = Workspace.index();
  auto AST = Workspace.openFile("lib.cpp");
  ASSERT_TRUE(bool(AST));

  Annotations Source(R"cpp(
    #include "util.h"
    namespace util {
      int ad^d(int a, int b) { return a + b; }
    }
  )cpp");

  auto Items = prepareCallHierarchy(*AST, Source.point(), testPath("lib.cpp"));
  ASSERT_EQ(Items.size(), 1u);
  EXPECT_EQ(Items[0].name, "add");

  auto Incoming = incomingCalls(Items[0], Index.get());

  // process() from both binary1_process.cpp and binary2_process.cpp.
  EXPECT_EQ(Incoming.size(), 2u)
      << "Expected 2 callers (process from binary1 and binary2), got "
      << Incoming.size();
  if (Incoming.size() >= 2) {
    EXPECT_THAT(Incoming,
                UnorderedElementsAre(from(withName("process")),
                                     from(withName("process"))));
  }
}

// Verifies that each caller's URI points to the correct file (not both to the
// same one).
TEST(CallHierarchyMultipleCallers, IncomingCallerURIPointsToCorrectFile) {
  TestWorkspace Workspace;

  Workspace.addSource("util.h", R"cpp(
    namespace util {
      int add(int a, int b);
    }
  )cpp");

  Workspace.addMainFile("lib.cpp", R"cpp(
    #include "util.h"
    namespace util {
      int add(int a, int b) { return a + b; }
    }
  )cpp");

  Workspace.addMainFile("binary1_main.cpp", R"cpp(
    #include "util.h"
    int main() {
      return util::add(1, 2);
    }
  )cpp");

  Workspace.addMainFile("binary2_main.cpp", R"cpp(
    #include "util.h"
    int main() {
      return util::add(3, 4);
    }
  )cpp");

  auto Index = Workspace.index();
  auto AST = Workspace.openFile("lib.cpp");
  ASSERT_TRUE(bool(AST));

  Annotations Source(R"cpp(
    #include "util.h"
    namespace util {
      int ad^d(int a, int b) { return a + b; }
    }
  )cpp");

  auto Items = prepareCallHierarchy(*AST, Source.point(), testPath("lib.cpp"));
  ASSERT_EQ(Items.size(), 1u);

  auto Incoming = incomingCalls(Items[0], Index.get());

  ASSERT_EQ(Incoming.size(), 2u)
      << "Expected 2 callers, got " << Incoming.size();

  std::vector<std::string> Files;
  for (const auto &Call : Incoming)
    Files.push_back(Call.from.uri.file().str());

  // Each caller should point to a different file.
  ASSERT_NE(Files[0], Files[1])
      << "Both callers incorrectly point to the same file: " << Files[0];

  auto Binary1Path = testPath("binary1_main.cpp");
  auto Binary2Path = testPath("binary2_main.cpp");
  EXPECT_EQ(Files[0], Binary1Path);
  EXPECT_EQ(Files[1], Binary2Path);
}

// Distinct callers with different names in the same file still work correctly
// (non-regression for the basic single-file case).
TEST(CallHierarchyMultipleCallers, IncomingDistinctCallersInSameFile) {
  TestWorkspace Workspace;

  Workspace.addSource("util.h", R"cpp(
    namespace util {
      int add(int a, int b);
    }
  )cpp");

  Workspace.addMainFile("lib.cpp", R"cpp(
    #include "util.h"
    namespace util {
      int add(int a, int b) { return a + b; }
    }
  )cpp");

  Workspace.addMainFile("main.cpp", R"cpp(
    #include "util.h"
    int caller1() {
      return util::add(1, 2);
    }
    int caller2() {
      return util::add(3, 4);
    }
  )cpp");

  auto Index = Workspace.index();
  auto AST = Workspace.openFile("lib.cpp");
  ASSERT_TRUE(bool(AST));

  Annotations Source(R"cpp(
    #include "util.h"
    namespace util {
      int ad^d(int a, int b) { return a + b; }
    }
  )cpp");

  auto Items = prepareCallHierarchy(*AST, Source.point(), testPath("lib.cpp"));
  ASSERT_EQ(Items.size(), 1u);

  auto Incoming = incomingCalls(Items[0], Index.get());

  EXPECT_EQ(Incoming.size(), 2u)
      << "Expected 2 callers (caller1 and caller2), got " << Incoming.size();
  if (Incoming.size() >= 2) {
    EXPECT_THAT(Incoming,
                UnorderedElementsAre(from(withName("caller1")),
                                     from(withName("caller2"))));
  }
}

// Stress test: many binaries with same-named functions calling the same
// library function. Each should appear as a separate caller.
TEST(CallHierarchyMultipleCallers, IncomingManySameSignatureCallers) {
  TestWorkspace Workspace;

  Workspace.addSource("util.h", R"cpp(
    namespace util {
      int add(int a, int b);
    }
  )cpp");

  Workspace.addMainFile("lib.cpp", R"cpp(
    #include "util.h"
    namespace util {
      int add(int a, int b) { return a + b; }
    }
  )cpp");

  // Create 5 binaries, each with their own process() calling util::add().
  const int NumBinaries = 5;
  for (int I = 0; I < NumBinaries; ++I) {
    std::string Filename = "binary" + std::to_string(I) + "_process.cpp";
    std::string Code =
        "#include \"util.h\"\n"
        "int process() {\n"
        "  return util::add(" + std::to_string(I) + ", " +
        std::to_string(I + 1) + ");\n"
        "}\n";
    Workspace.addMainFile(Filename, Code);
  }

  auto Index = Workspace.index();
  auto AST = Workspace.openFile("lib.cpp");
  ASSERT_TRUE(bool(AST));

  Annotations Source(R"cpp(
    #include "util.h"
    namespace util {
      int ad^d(int a, int b) { return a + b; }
    }
  )cpp");

  auto Items = prepareCallHierarchy(*AST, Source.point(), testPath("lib.cpp"));
  ASSERT_EQ(Items.size(), 1u);

  auto Incoming = incomingCalls(Items[0], Index.get());

  EXPECT_EQ(Incoming.size(), static_cast<size_t>(NumBinaries))
      << "Expected " << NumBinaries
      << " callers (process from each binary), got " << Incoming.size();
}

} // namespace
} // namespace clangd
} // namespace clang