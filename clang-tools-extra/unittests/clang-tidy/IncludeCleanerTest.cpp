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
  EXPECT_EQ(PostCode, runCheckOnCode<IncludeCleanerCheck>(
                          PreCode, &Errors, "file.cpp", std::nullopt,
                          ClangTidyOptions(), {{"bar.h", ""}, {"vector", ""}}));
}

TEST(IncludeCleanerCheckTest, SuppressUnusedIncludes) {
  const char *PreCode = R"(
#include "bar.h"
#include "foo/qux.h"
#include "baz/qux/qux.h"
#include <vector>
)";

  const char *PostCode = R"(
#include "bar.h"
#include "foo/qux.h"
#include <vector>
)";

  std::vector<ClangTidyError> Errors;
  ClangTidyOptions Opts;
  Opts.CheckOptions["IgnoreHeaders"] = llvm::StringRef{llvm::formatv(
      "bar.h;{0};{1};vector",
      llvm::Regex::escape(appendPathFileSystemIndependent({"foo", "qux.h"})),
      llvm::Regex::escape(appendPathFileSystemIndependent({"baz", "qux"})))};
  EXPECT_EQ(
      PostCode,
      runCheckOnCode<IncludeCleanerCheck>(
          PreCode, &Errors, "file.cpp", std::nullopt, Opts,
          {{"bar.h", ""},
           {"vector", ""},
           {appendPathFileSystemIndependent({"foo", "qux.h"}), ""},
           {appendPathFileSystemIndependent({"baz", "qux", "qux.h"}), ""}}));
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
  EXPECT_EQ(PostCode,
            runCheckOnCode<IncludeCleanerCheck>(
                PreCode, &Errors, "file.cpp", std::nullopt, ClangTidyOptions(),
                {{"bar.h", R"(#pragma once
                              #include "baz.h"
                              int bar();
                           )"},
                 {"baz.h", R"(#pragma once
                              int baz();
                           )"}}));
}

TEST(IncludeCleanerCheckTest, SuppressMissingIncludes) {
  const char *PreCode = R"(
#include "bar.h"

int BarResult = bar();
int BazResult = baz();
int QuxResult = qux();
)";

  ClangTidyOptions Opts;
  Opts.CheckOptions["IgnoreHeaders"] = llvm::StringRef{
      "baz.h;" +
      llvm::Regex::escape(appendPathFileSystemIndependent({"foo", "qux.h"}))};
  std::vector<ClangTidyError> Errors;
  EXPECT_EQ(PreCode, runCheckOnCode<IncludeCleanerCheck>(
                         PreCode, &Errors, "file.cpp", std::nullopt, Opts,
                         {{"bar.h", R"(#pragma once
                              #include "baz.h"
                              #include "foo/qux.h"
                              int bar();
                           )"},
                          {"baz.h", R"(#pragma once
                              int baz();
                           )"},
                          {appendPathFileSystemIndependent({"foo", "qux.h"}),
                           R"(#pragma once
                              int qux();
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
  EXPECT_EQ(PostCode,
            runCheckOnCode<IncludeCleanerCheck>(
                PreCode, &Errors, "file.cpp", std::nullopt, ClangTidyOptions(),
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
  EXPECT_EQ(PostCode,
            runCheckOnCode<IncludeCleanerCheck>(
                PreCode, &Errors, "file.cpp", std::nullopt, ClangTidyOptions(),
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
  EXPECT_EQ(PreCode,
            runCheckOnCode<IncludeCleanerCheck>(
                PreCode, &Errors, "file.cpp", std::nullopt, ClangTidyOptions(),
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

  EXPECT_EQ(PreCode,
            runCheckOnCode<IncludeCleanerCheck>(
                PreCode, &Errors, "file.cpp", std::nullopt, ClangTidyOptions(),
                {{"foo.h",
                  R"(#pragma once
                     #define DECLARE void myfunc()
                  )"}}));
}

} // namespace
} // namespace test
} // namespace tidy
} // namespace clang
