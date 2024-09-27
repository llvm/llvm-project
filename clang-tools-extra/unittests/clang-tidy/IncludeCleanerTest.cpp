//===--- IncludeCleanerTest.cpp - clang-tidy -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangTidyDiagnosticConsumer.h"
#include "ClangTidyOptions.h"
#include "ClangTidyTest.h"
#include "misc/IncludeCleanerCheck.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include "llvm/Testing/Annotations/Annotations.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <initializer_list>

#include <optional>
#include <vector>

using namespace clang::tidy::misc;

namespace clang {
namespace tidy {
namespace test {
namespace {

std::string
appendPathFileSystemIndependent(std::initializer_list<std::string> Segments) {
  llvm::SmallString<32> Result;
  for (const auto &Segment : Segments)
    llvm::sys::path::append(Result, llvm::sys::path::Style::native, Segment);
  return std::string(Result.str());
}

TEST(IncludeCleanerCheckTest, BasicUnusedIncludes) {
  const char *PreCode = R"(
#include "bar.h"
#include <vector>
#include "bar.h"
)";
  const char *PostCode = "\n";

  std::vector<ClangTidyError> Errors;
  EXPECT_EQ(PostCode,
            runCheckOnCode<IncludeCleanerCheck>(
                PreCode, &Errors, "file.cpp", {}, ClangTidyOptions(),
                {{"bar.h", "#pragma once"}, {"vector", "#pragma once"}}));
}

TEST(IncludeCleanerCheckTest, SuppressUnusedIncludes) {
  const char *PreCode = R"(
#include "bar.h"
#include "foo/qux.h"
#include "baz/qux/qux.h"
#include <vector>
#include <list>
)";

  const char *PostCode = R"(
#include "bar.h"
#include "foo/qux.h"
#include <vector>
#include <list>
)";

  std::vector<ClangTidyError> Errors;
  ClangTidyOptions Opts;
  Opts.CheckOptions["IgnoreHeaders"] = llvm::StringRef{llvm::formatv(
      "bar.h;{0};{1};vector;<list>;",
      llvm::Regex::escape(appendPathFileSystemIndependent({"foo", "qux.h"})),
      llvm::Regex::escape(appendPathFileSystemIndependent({"baz", "qux"})))};
  EXPECT_EQ(
      PostCode,
      runCheckOnCode<IncludeCleanerCheck>(
          PreCode, &Errors, "file.cpp", {}, Opts,
          {{"bar.h", "#pragma once"},
           {"vector", "#pragma once"},
           {"list", "#pragma once"},
           {appendPathFileSystemIndependent({"foo", "qux.h"}), "#pragma once"},
           {appendPathFileSystemIndependent({"baz", "qux", "qux.h"}),
            "#pragma once"}}));
}

TEST(IncludeCleanerCheckTest, BasicMissingIncludes) {
  const char *PreCode = R"(
#include "bar.h"

int BarResult = bar();
int BazResult = baz();
)";
  const char *PostCode = R"(
#include "bar.h"
#include "baz.h"

int BarResult = bar();
int BazResult = baz();
)";

  std::vector<ClangTidyError> Errors;
  EXPECT_EQ(PostCode, runCheckOnCode<IncludeCleanerCheck>(
                          PreCode, &Errors, "file.cpp", {}, ClangTidyOptions(),
                          {{"bar.h", R"(#pragma once
                              #include "baz.h"
                              int bar();
                           )"},
                           {"baz.h", R"(#pragma once
                              int baz();
                           )"}}));
}

TEST(IncludeCleanerCheckTest, DedupsMissingIncludes) {
  llvm::Annotations Code(R"(
#include "baz.h" // IWYU pragma: keep

int BarResult1 = $diag1^bar();
int BarResult2 = $diag2^bar();)");

  {
    std::vector<ClangTidyError> Errors;
    runCheckOnCode<IncludeCleanerCheck>(Code.code(), &Errors, "file.cpp", {},
                                        ClangTidyOptions(),
                                        {{"baz.h", R"(#pragma once
                              #include "bar.h"
                           )"},
                                         {"bar.h", R"(#pragma once
                              int bar();
                           )"}});
    ASSERT_THAT(Errors.size(), testing::Eq(1U));
    EXPECT_EQ(Errors.front().Message.Message,
              "no header providing \"bar\" is directly included");
    EXPECT_EQ(Errors.front().Message.FileOffset, Code.point("diag1"));
  }
  {
    std::vector<ClangTidyError> Errors;
    ClangTidyOptions Opts;
    Opts.CheckOptions.insert({"DeduplicateFindings", "false"});
    runCheckOnCode<IncludeCleanerCheck>(Code.code(), &Errors, "file.cpp", {},
                                        Opts,
                                        {{"baz.h", R"(#pragma once
                              #include "bar.h"
                           )"},
                                         {"bar.h", R"(#pragma once
                              int bar();
                           )"}});
    ASSERT_THAT(Errors.size(), testing::Eq(2U));
    EXPECT_EQ(Errors.front().Message.Message,
              "no header providing \"bar\" is directly included");
    EXPECT_EQ(Errors.front().Message.FileOffset, Code.point("diag1"));
    EXPECT_EQ(Errors.back().Message.Message,
              "no header providing \"bar\" is directly included");
    EXPECT_EQ(Errors.back().Message.FileOffset, Code.point("diag2"));
  }
}

TEST(IncludeCleanerCheckTest, SuppressMissingIncludes) {
  const char *PreCode = R"(
#include "bar.h"

int BarResult = bar();
int BazResult = baz();
int QuxResult = qux();
int PrivResult = test();
std::vector x;
)";

  ClangTidyOptions Opts;
  Opts.CheckOptions["IgnoreHeaders"] = llvm::StringRef{
      "public.h;<vector>;baz.h;" +
      llvm::Regex::escape(appendPathFileSystemIndependent({"foo", "qux.h"}))};
  std::vector<ClangTidyError> Errors;
  EXPECT_EQ(PreCode, runCheckOnCode<IncludeCleanerCheck>(
                         PreCode, &Errors, "file.cpp", {}, Opts,
                         {{"bar.h", R"(#pragma once
                              #include "baz.h"
                              #include "foo/qux.h"
                              #include "private.h"
                              int bar();
                              namespace std { struct vector {}; }
                           )"},
                          {"baz.h", R"(#pragma once
                              int baz();
                           )"},
                          {"private.h", R"(#pragma once
                              // IWYU pragma: private, include "public.h"
                              int test();
                           )"},
                          {appendPathFileSystemIndependent({"foo", "qux.h"}),
                           R"(#pragma once
                              int qux();
                           )"}}));
}

TEST(IncludeCleanerCheckTest, MultipleTimeMissingInclude) {
  const char *PreCode = R"(
#include "bar.h"

int BarResult = bar();
int BazResult_0 = baz_0();
int BazResult_1 = baz_1();
)";
  const char *PostCode = R"(
#include "bar.h"
#include "baz.h"

int BarResult = bar();
int BazResult_0 = baz_0();
int BazResult_1 = baz_1();
)";

  std::vector<ClangTidyError> Errors;
  EXPECT_EQ(PostCode, runCheckOnCode<IncludeCleanerCheck>(
                          PreCode, &Errors, "file.cpp", {}, ClangTidyOptions(),
                          {{"bar.h", R"(#pragma once
                              #include "baz.h"
                              int bar();
                           )"},
                           {"baz.h", R"(#pragma once
                              int baz_0();
                              int baz_1();
                           )"}}));
}

TEST(IncludeCleanerCheckTest, SystemMissingIncludes) {
  const char *PreCode = R"(
#include <vector>

std::string HelloString;
std::vector Vec;
)";
  const char *PostCode = R"(
#include <string>
#include <vector>

std::string HelloString;
std::vector Vec;
)";

  std::vector<ClangTidyError> Errors;
  EXPECT_EQ(PostCode, runCheckOnCode<IncludeCleanerCheck>(
                          PreCode, &Errors, "file.cpp", {}, ClangTidyOptions(),
                          {{"string", R"(#pragma once
                              namespace std { class string {}; }
                            )"},
                           {"vector", R"(#pragma once
                              #include <string>
                              namespace std { class vector {}; }
                            )"}}));
}

TEST(IncludeCleanerCheckTest, PragmaMissingIncludes) {
  const char *PreCode = R"(
#include "bar.h"

int BarResult = bar();
int FooBarResult = foobar();
)";
  const char *PostCode = R"(
#include "bar.h"
#include "public.h"

int BarResult = bar();
int FooBarResult = foobar();
)";

  std::vector<ClangTidyError> Errors;
  EXPECT_EQ(PostCode, runCheckOnCode<IncludeCleanerCheck>(
                          PreCode, &Errors, "file.cpp", {}, ClangTidyOptions(),
                          {{"bar.h", R"(#pragma once
                              #include "private.h"
                              int bar();
                           )"},
                           {"private.h", R"(#pragma once
                                // IWYU pragma: private, include "public.h"
                                int foobar();
                               )"}}));
}

TEST(IncludeCleanerCheckTest, DeclFromMacroExpansion) {
  const char *PreCode = R"(
#include "foo.h"

DECLARE(myfunc) {
   int a;
}
)";

  std::vector<ClangTidyError> Errors;
  EXPECT_EQ(PreCode, runCheckOnCode<IncludeCleanerCheck>(
                         PreCode, &Errors, "file.cpp", {}, ClangTidyOptions(),
                         {{"foo.h",
                           R"(#pragma once
                     #define DECLARE(X) void X()
                  )"}}));

  PreCode = R"(
#include "foo.h"

DECLARE {
   int a;
}
)";

  EXPECT_EQ(PreCode, runCheckOnCode<IncludeCleanerCheck>(
                         PreCode, &Errors, "file.cpp", {}, ClangTidyOptions(),
                         {{"foo.h",
                           R"(#pragma once
                     #define DECLARE void myfunc()
                  )"}}));
}

} // namespace
} // namespace test
} // namespace tidy
} // namespace clang
