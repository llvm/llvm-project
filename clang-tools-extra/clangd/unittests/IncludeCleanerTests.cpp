//===--- IncludeCleanerTests.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "Config.h"
#include "Diagnostics.h"
#include "IncludeCleaner.h"
#include "ParsedAST.h"
#include "SourceCode.h"
#include "TestFS.h"
#include "TestTU.h"
#include "clang-include-cleaner/Analysis.h"
#include "clang-include-cleaner/Types.h"
#include "support/Context.h"
#include "clang/AST/DeclBase.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace clang {
namespace clangd {
namespace {

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Matcher;
using ::testing::Pointee;
using ::testing::UnorderedElementsAre;

Matcher<const Diag &> withFix(::testing::Matcher<Fix> FixMatcher) {
  return Field(&Diag::Fixes, ElementsAre(FixMatcher));
}

MATCHER_P2(Diag, Range, Message,
           "Diag at " + llvm::to_string(Range) + " = [" + Message + "]") {
  return arg.Range == Range && arg.Message == Message;
}

MATCHER_P3(Fix, Range, Replacement, Message,
           "Fix " + llvm::to_string(Range) + " => " +
               ::testing::PrintToString(Replacement) + " = [" + Message + "]") {
  return arg.Message == Message && arg.Edits.size() == 1 &&
         arg.Edits[0].range == Range && arg.Edits[0].newText == Replacement;
}

std::string guard(llvm::StringRef Code) {
  return "#pragma once\n" + Code.str();
}

MATCHER_P(writtenInclusion, Written, "") {
  if (arg.Written != Written)
    *result_listener << arg.Written;
  return arg.Written == Written;
}

TEST(IncludeCleaner, StdlibUnused) {
  setIncludeCleanerAnalyzesStdlib(true);
  auto Cleanup =
      llvm::make_scope_exit([] { setIncludeCleanerAnalyzesStdlib(false); });

  auto TU = TestTU::withCode(R"cpp(
    #include <list>
    #include <queue>
    std::list<int> x;
  )cpp");
  // Layout of std library impl is not relevant.
  TU.AdditionalFiles["bits"] = R"cpp(
    #pragma once
    namespace std {
      template <typename> class list {};
      template <typename> class queue {};
    }
  )cpp";
  TU.AdditionalFiles["list"] = "#include <bits>";
  TU.AdditionalFiles["queue"] = "#include <bits>";
  TU.ExtraArgs = {"-isystem", testRoot()};
  auto AST = TU.build();
  IncludeCleanerFindings Findings = computeIncludeCleanerFindings(AST);
  EXPECT_THAT(Findings.UnusedIncludes,
              ElementsAre(Pointee(writtenInclusion("<queue>"))));
}

TEST(IncludeCleaner, GetUnusedHeaders) {
  llvm::StringLiteral MainFile = R"cpp(
    #include "a.h"
    #include "b.h"
    #include "dir/c.h"
    #include "dir/unused.h"
    #include "unguarded.h"
    #include "unused.h"
    #include <system_header.h>
    void foo() {
      a();
      b();
      c();
    })cpp";
  // Build expected ast with symbols coming from headers.
  TestTU TU;
  TU.Filename = "foo.cpp";
  TU.AdditionalFiles["foo.h"] = guard("void foo();");
  TU.AdditionalFiles["a.h"] = guard("void a();");
  TU.AdditionalFiles["b.h"] = guard("void b();");
  TU.AdditionalFiles["dir/c.h"] = guard("void c();");
  TU.AdditionalFiles["unused.h"] = guard("void unused();");
  TU.AdditionalFiles["dir/unused.h"] = guard("void dirUnused();");
  TU.AdditionalFiles["system/system_header.h"] = guard("");
  TU.AdditionalFiles["unguarded.h"] = "";
  TU.ExtraArgs.push_back("-I" + testPath("dir"));
  TU.ExtraArgs.push_back("-isystem" + testPath("system"));
  TU.Code = MainFile.str();
  ParsedAST AST = TU.build();
  IncludeCleanerFindings Findings = computeIncludeCleanerFindings(AST);
  EXPECT_THAT(
      Findings.UnusedIncludes,
      UnorderedElementsAre(Pointee(writtenInclusion("\"unused.h\"")),
                           Pointee(writtenInclusion("\"dir/unused.h\""))));
}

TEST(IncludeCleaner, ComputeMissingHeaders) {
  Annotations MainFile(R"cpp(
    #include "a.h"

    void foo() {
      $b[[b]]();
    })cpp");
  TestTU TU;
  TU.Filename = "foo.cpp";
  TU.AdditionalFiles["a.h"] = guard("#include \"b.h\"");
  TU.AdditionalFiles["b.h"] = guard("void b();");

  TU.Code = MainFile.code();
  ParsedAST AST = TU.build();

  IncludeCleanerFindings Findings = computeIncludeCleanerFindings(AST);
  const SourceManager &SM = AST.getSourceManager();
  const NamedDecl *BDecl = nullptr;
  for (Decl *D : AST.getASTContext().getTranslationUnitDecl()->decls()) {
    const NamedDecl *CandidateDecl = llvm::dyn_cast<NamedDecl>(D);
    std::string Name = CandidateDecl->getQualifiedNameAsString();
    if (Name != "b")
      continue;
    BDecl = CandidateDecl;
  }
  ASSERT_TRUE(BDecl);
  include_cleaner::Symbol B{*BDecl};
  auto Range = MainFile.range("b");
  size_t Start = llvm::cantFail(positionToOffset(MainFile.code(), Range.start));
  size_t End = llvm::cantFail(positionToOffset(MainFile.code(), Range.end));
  syntax::FileRange BRange{SM.getMainFileID(), static_cast<unsigned int>(Start),
                           static_cast<unsigned int>(End)};
  include_cleaner::Header Header{*SM.getFileManager().getFile("b.h")};
  MissingIncludeDiagInfo BInfo{B, BRange, {Header}};
  EXPECT_THAT(Findings.MissingIncludes, ElementsAre(BInfo));
}

TEST(IncludeCleaner, GenerateMissingHeaderDiags) {
  Config Cfg;
  Cfg.Diagnostics.MissingIncludes = Config::IncludesPolicy::Strict;
  Cfg.Diagnostics.Includes.IgnoreHeader = {
      [](llvm::StringRef Header) { return Header.ends_with("buzz.h"); }};
  WithContextValue Ctx(Config::Key, std::move(Cfg));
  Annotations MainFile(R"cpp(
#include "a.h"
#include "all.h"
$insert_b[[]]#include "baz.h"
#include "dir/c.h"
$insert_d[[]]$insert_foo[[]]#include "fuzz.h"
#include "header.h"
$insert_foobar[[]]#include <e.h>
$insert_f[[]]$insert_vector[[]]

#define DEF(X) const Foo *X;
#define BAZ(X) const X x

  void foo() {
    $b[[b]]();

    ns::$bar[[Bar]] bar;
    bar.d();
    $f[[f]](); 

    // this should not be diagnosed, because it's ignored in the config
    buzz(); 

    $foobar[[foobar]]();

    std::$vector[[vector]] v;

    int var = $FOO[[FOO]];

    $DEF[[DEF]](a);

    $BAR[[BAR]](b);

    BAZ($Foo[[Foo]]);
})cpp");

  TestTU TU;
  TU.Filename = "foo.cpp";
  TU.AdditionalFiles["a.h"] = guard("#include \"b.h\"");
  TU.AdditionalFiles["b.h"] = guard("void b();");

  TU.AdditionalFiles["dir/c.h"] = guard("#include \"d.h\"");
  TU.AdditionalFiles["dir/d.h"] =
      guard("namespace ns { struct Bar { void d(); }; }");

  TU.AdditionalFiles["system/e.h"] = guard("#include <f.h>");
  TU.AdditionalFiles["system/f.h"] = guard("void f();");
  TU.ExtraArgs.push_back("-isystem" + testPath("system"));

  TU.AdditionalFiles["fuzz.h"] = guard("#include \"buzz.h\"");
  TU.AdditionalFiles["buzz.h"] = guard("void buzz();");

  TU.AdditionalFiles["baz.h"] = guard("#include \"private.h\"");
  TU.AdditionalFiles["private.h"] = guard(R"cpp(
    // IWYU pragma: private, include "public.h"
    void foobar();
  )cpp");
  TU.AdditionalFiles["header.h"] = guard(R"cpp(
  namespace std { class vector {}; }
  )cpp");

  TU.AdditionalFiles["all.h"] = guard("#include \"foo.h\"");
  TU.AdditionalFiles["foo.h"] = guard(R"cpp(
    #define BAR(x) Foo *x
    #define FOO 1
    struct Foo{}; 
  )cpp");

  TU.Code = MainFile.code();
  ParsedAST AST = TU.build();

  std::vector<clangd::Diag> Diags =
      issueIncludeCleanerDiagnostics(AST, TU.Code);
  EXPECT_THAT(
      Diags,
      UnorderedElementsAre(
          AllOf(Diag(MainFile.range("b"),
                     "No header providing \"b\" is directly included"),
                withFix(Fix(MainFile.range("insert_b"), "#include \"b.h\"\n",
                            "#include \"b.h\""))),
          AllOf(Diag(MainFile.range("bar"),
                     "No header providing \"ns::Bar\" is directly included"),
                withFix(Fix(MainFile.range("insert_d"),
                            "#include \"dir/d.h\"\n", "#include \"dir/d.h\""))),
          AllOf(Diag(MainFile.range("f"),
                     "No header providing \"f\" is directly included"),
                withFix(Fix(MainFile.range("insert_f"), "#include <f.h>\n",
                            "#include <f.h>"))),
          AllOf(
              Diag(MainFile.range("foobar"),
                   "No header providing \"foobar\" is directly included"),
              withFix(Fix(MainFile.range("insert_foobar"),
                          "#include \"public.h\"\n", "#include \"public.h\""))),
          AllOf(
              Diag(MainFile.range("vector"),
                   "No header providing \"std::vector\" is directly included"),
              withFix(Fix(MainFile.range("insert_vector"),
                          "#include <vector>\n", "#include <vector>"))),
          AllOf(Diag(MainFile.range("FOO"),
                     "No header providing \"FOO\" is directly included"),
                withFix(Fix(MainFile.range("insert_foo"),
                            "#include \"foo.h\"\n", "#include \"foo.h\""))),
          AllOf(Diag(MainFile.range("DEF"),
                     "No header providing \"Foo\" is directly included"),
                withFix(Fix(MainFile.range("insert_foo"),
                            "#include \"foo.h\"\n", "#include \"foo.h\""))),
          AllOf(Diag(MainFile.range("BAR"),
                     "No header providing \"BAR\" is directly included"),
                withFix(Fix(MainFile.range("insert_foo"),
                            "#include \"foo.h\"\n", "#include \"foo.h\""))),
          AllOf(Diag(MainFile.range("Foo"),
                     "No header providing \"Foo\" is directly included"),
                withFix(Fix(MainFile.range("insert_foo"),
                            "#include \"foo.h\"\n", "#include \"foo.h\"")))));
}

TEST(IncludeCleaner, IWYUPragmas) {
  TestTU TU;
  TU.Code = R"cpp(
    #include "behind_keep.h" // IWYU pragma: keep
    #include "exported.h" // IWYU pragma: export
    #include "public.h"

    void bar() { foo(); }
    )cpp";
  TU.AdditionalFiles["behind_keep.h"] = guard("");
  TU.AdditionalFiles["exported.h"] = guard("");
  TU.AdditionalFiles["public.h"] = guard("#include \"private.h\"");
  TU.AdditionalFiles["private.h"] = guard(R"cpp(
    // IWYU pragma: private, include "public.h"
    void foo() {}
  )cpp");
  Config Cfg;
  Cfg.Diagnostics.UnusedIncludes = Config::IncludesPolicy::Strict;
  WithContextValue Ctx(Config::Key, std::move(Cfg));
  ParsedAST AST = TU.build();
  EXPECT_THAT(AST.getDiagnostics(), llvm::ValueIs(IsEmpty()));
  IncludeCleanerFindings Findings = computeIncludeCleanerFindings(AST);
  EXPECT_THAT(Findings.UnusedIncludes, IsEmpty());
}

TEST(IncludeCleaner, IWYUPragmaExport) {
  TestTU TU;
  TU.Code = R"cpp(
    #include "foo.h"
    )cpp";
  TU.AdditionalFiles["foo.h"] = R"cpp(
    #ifndef FOO_H
    #define FOO_H

    #include "bar.h" // IWYU pragma: export

    #endif
  )cpp";
  TU.AdditionalFiles["bar.h"] = guard(R"cpp(
    void bar() {}
  )cpp");
  ParsedAST AST = TU.build();

  EXPECT_THAT(AST.getDiagnostics(), llvm::ValueIs(IsEmpty()));
  IncludeCleanerFindings Findings = computeIncludeCleanerFindings(AST);
  EXPECT_THAT(Findings.UnusedIncludes,
              ElementsAre(Pointee(writtenInclusion("\"foo.h\""))));
}

TEST(IncludeCleaner, NoDiagsForObjC) {
  TestTU TU;
  TU.Code = R"cpp(
    #include "foo.h"

    void bar() {}
    )cpp";
  TU.AdditionalFiles["foo.h"] = R"cpp(
    #ifndef FOO_H
    #define FOO_H

    #endif
  )cpp";
  TU.ExtraArgs.emplace_back("-xobjective-c");

  Config Cfg;

  Cfg.Diagnostics.UnusedIncludes = Config::IncludesPolicy::Strict;
  Cfg.Diagnostics.MissingIncludes = Config::IncludesPolicy::Strict;
  WithContextValue Ctx(Config::Key, std::move(Cfg));
  ParsedAST AST = TU.build();
  EXPECT_THAT(AST.getDiagnostics(), llvm::ValueIs(IsEmpty()));
}

TEST(IncludeCleaner, UmbrellaUsesPrivate) {
  TestTU TU;
  TU.Code = R"cpp(
    #include "private.h"
    )cpp";
  TU.AdditionalFiles["private.h"] = guard(R"cpp(
    // IWYU pragma: private, include "public.h"
    void foo() {}
  )cpp");
  TU.Filename = "public.h";
  Config Cfg;
  Cfg.Diagnostics.UnusedIncludes = Config::IncludesPolicy::Strict;
  WithContextValue Ctx(Config::Key, std::move(Cfg));
  ParsedAST AST = TU.build();
  EXPECT_THAT(AST.getDiagnostics(), llvm::ValueIs(IsEmpty()));
  IncludeCleanerFindings Findings = computeIncludeCleanerFindings(AST);
  EXPECT_THAT(Findings.UnusedIncludes, IsEmpty());
}

} // namespace
} // namespace clangd
} // namespace clang
