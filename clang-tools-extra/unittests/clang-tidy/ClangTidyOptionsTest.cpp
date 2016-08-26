#include "ClangTidyOptions.h"
#include "gtest/gtest.h"
#include "llvm/ADT/StringExtras.h"

namespace clang {
namespace tidy {
namespace test {

TEST(ParseLineFilter, EmptyFilter) {
  ClangTidyGlobalOptions Options;
  EXPECT_FALSE(parseLineFilter("", Options));
  EXPECT_TRUE(Options.LineFilter.empty());
  EXPECT_FALSE(parseLineFilter("[]", Options));
  EXPECT_TRUE(Options.LineFilter.empty());
}

TEST(ParseLineFilter, InvalidFilter) {
  ClangTidyGlobalOptions Options;
  EXPECT_TRUE(!!parseLineFilter("asdf", Options));
  EXPECT_TRUE(Options.LineFilter.empty());

  EXPECT_TRUE(!!parseLineFilter("[{}]", Options));
  EXPECT_TRUE(!!parseLineFilter("[{\"name\":\"\"}]", Options));
  EXPECT_TRUE(
      !!parseLineFilter("[{\"name\":\"test\",\"lines\":[[1]]}]", Options));
  EXPECT_TRUE(
      !!parseLineFilter("[{\"name\":\"test\",\"lines\":[[1,2,3]]}]", Options));
  EXPECT_TRUE(
      !!parseLineFilter("[{\"name\":\"test\",\"lines\":[[1,-1]]}]", Options));
}

TEST(ParseLineFilter, ValidFilter) {
  ClangTidyGlobalOptions Options;
  std::error_code Error = parseLineFilter(
      "[{\"name\":\"file1.cpp\",\"lines\":[[3,15],[20,30],[42,42]]},"
      "{\"name\":\"file2.h\"},"
      "{\"name\":\"file3.cc\",\"lines\":[[100,1000]]}]",
      Options);
  EXPECT_FALSE(Error);
  EXPECT_EQ(3u, Options.LineFilter.size());
  EXPECT_EQ("file1.cpp", Options.LineFilter[0].Name);
  EXPECT_EQ(3u, Options.LineFilter[0].LineRanges.size());
  EXPECT_EQ(3u, Options.LineFilter[0].LineRanges[0].first);
  EXPECT_EQ(15u, Options.LineFilter[0].LineRanges[0].second);
  EXPECT_EQ(20u, Options.LineFilter[0].LineRanges[1].first);
  EXPECT_EQ(30u, Options.LineFilter[0].LineRanges[1].second);
  EXPECT_EQ(42u, Options.LineFilter[0].LineRanges[2].first);
  EXPECT_EQ(42u, Options.LineFilter[0].LineRanges[2].second);
  EXPECT_EQ("file2.h", Options.LineFilter[1].Name);
  EXPECT_EQ(0u, Options.LineFilter[1].LineRanges.size());
  EXPECT_EQ("file3.cc", Options.LineFilter[2].Name);
  EXPECT_EQ(1u, Options.LineFilter[2].LineRanges.size());
  EXPECT_EQ(100u, Options.LineFilter[2].LineRanges[0].first);
  EXPECT_EQ(1000u, Options.LineFilter[2].LineRanges[0].second);
}

TEST(ParseConfiguration, ValidConfiguration) {
  llvm::ErrorOr<ClangTidyOptions> Options =
      parseConfiguration("Checks: \"-*,misc-*\"\n"
                         "HeaderFilterRegex: \".*\"\n"
                         "AnalyzeTemporaryDtors: true\n"
                         "User: some.user");
  EXPECT_TRUE(!!Options);
  EXPECT_EQ("-*,misc-*", *Options->Checks);
  EXPECT_EQ(".*", *Options->HeaderFilterRegex);
  EXPECT_TRUE(*Options->AnalyzeTemporaryDtors);
  EXPECT_EQ("some.user", *Options->User);
}

TEST(ParseConfiguration, MergeConfigurations) {
  llvm::ErrorOr<ClangTidyOptions> Options1 = parseConfiguration(R"(
      Checks: "check1,check2"
      HeaderFilterRegex: "filter1"
      AnalyzeTemporaryDtors: true
      User: user1
      ExtraArgs: ['arg1', 'arg2']
      ExtraArgsBefore: ['arg-before1', 'arg-before2']
  )");
  ASSERT_TRUE(!!Options1);
  llvm::ErrorOr<ClangTidyOptions> Options2 = parseConfiguration(R"(
      Checks: "check3,check4"
      HeaderFilterRegex: "filter2"
      AnalyzeTemporaryDtors: false
      User: user2
      ExtraArgs: ['arg3', 'arg4']
      ExtraArgsBefore: ['arg-before3', 'arg-before4']
  )");
  ASSERT_TRUE(!!Options2);
  ClangTidyOptions Options = Options1->mergeWith(*Options2);
  EXPECT_EQ("check1,check2,check3,check4", *Options.Checks);
  EXPECT_EQ("filter2", *Options.HeaderFilterRegex);
  EXPECT_FALSE(*Options.AnalyzeTemporaryDtors);
  EXPECT_EQ("user2", *Options.User);
  ASSERT_TRUE(Options.ExtraArgs.hasValue());
  EXPECT_EQ("arg1,arg2,arg3,arg4", llvm::join(Options.ExtraArgs->begin(),
                                              Options.ExtraArgs->end(), ","));
  ASSERT_TRUE(Options.ExtraArgsBefore.hasValue());
  EXPECT_EQ("arg-before1,arg-before2,arg-before3,arg-before4",
            llvm::join(Options.ExtraArgsBefore->begin(),
                       Options.ExtraArgsBefore->end(), ","));
}
} // namespace test
} // namespace tidy
} // namespace clang
