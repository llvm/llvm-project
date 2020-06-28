#include "ClangTidyOptions.h"
#include "ClangTidyCheck.h"
#include "ClangTidyDiagnosticConsumer.h"
#include "llvm/ADT/StringExtras.h"
#include "gtest/gtest.h"

namespace clang {
namespace tidy {

enum class Colours { Red, Orange, Yellow, Green, Blue, Indigo, Violet };

template <> struct OptionEnumMapping<Colours> {
  static llvm::ArrayRef<std::pair<Colours, StringRef>> getEnumMapping() {
    static constexpr std::pair<Colours, StringRef> Mapping[] = {
        {Colours::Red, "Red"},       {Colours::Orange, "Orange"},
        {Colours::Yellow, "Yellow"}, {Colours::Green, "Green"},
        {Colours::Blue, "Blue"},     {Colours::Indigo, "Indigo"},
        {Colours::Violet, "Violet"}};
    return makeArrayRef(Mapping);
  }
};

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
      UseColor: false
  )");
  ASSERT_TRUE(!!Options1);
  llvm::ErrorOr<ClangTidyOptions> Options2 = parseConfiguration(R"(
      Checks: "check3,check4"
      HeaderFilterRegex: "filter2"
      AnalyzeTemporaryDtors: false
      User: user2
      ExtraArgs: ['arg3', 'arg4']
      ExtraArgsBefore: ['arg-before3', 'arg-before4']
      UseColor: true
  )");
  ASSERT_TRUE(!!Options2);
  ClangTidyOptions Options = Options1->mergeWith(*Options2, 0);
  EXPECT_EQ("check1,check2,check3,check4", *Options.Checks);
  EXPECT_EQ("filter2", *Options.HeaderFilterRegex);
  EXPECT_EQ("user2", *Options.User);
  ASSERT_TRUE(Options.ExtraArgs.hasValue());
  EXPECT_EQ("arg1,arg2,arg3,arg4", llvm::join(Options.ExtraArgs->begin(),
                                              Options.ExtraArgs->end(), ","));
  ASSERT_TRUE(Options.ExtraArgsBefore.hasValue());
  EXPECT_EQ("arg-before1,arg-before2,arg-before3,arg-before4",
            llvm::join(Options.ExtraArgsBefore->begin(),
                       Options.ExtraArgsBefore->end(), ","));
  ASSERT_TRUE(Options.UseColor.hasValue());
  EXPECT_TRUE(*Options.UseColor);
}

class TestCheck : public ClangTidyCheck {
public:
  TestCheck(ClangTidyContext *Context) : ClangTidyCheck("test", Context) {}

  template <typename... Args> auto getLocal(Args &&... Arguments) {
    return Options.get(std::forward<Args>(Arguments)...);
  }

  template <typename... Args> auto getGlobal(Args &&... Arguments) {
    return Options.getLocalOrGlobal(std::forward<Args>(Arguments)...);
  }

  template <typename IntType = int, typename... Args>
  auto getIntLocal(Args &&... Arguments) {
    return Options.get<IntType>(std::forward<Args>(Arguments)...);
  }

  template <typename IntType = int, typename... Args>
  auto getIntGlobal(Args &&... Arguments) {
    return Options.getLocalOrGlobal<IntType>(std::forward<Args>(Arguments)...);
  }
};

#define CHECK_VAL(Value, Expected)                                             \
  do {                                                                         \
    auto Item = Value;                                                         \
    ASSERT_TRUE(!!Item);                                                       \
    EXPECT_EQ(*Item, Expected);                                                \
  } while (false)

#define CHECK_ERROR(Value, ErrorType, ExpectedMessage)                         \
  do {                                                                         \
    auto Item = Value;                                                         \
    ASSERT_FALSE(Item);                                                        \
    ASSERT_TRUE(Item.errorIsA<ErrorType>());                                   \
    ASSERT_FALSE(llvm::handleErrors(                                           \
        Item.takeError(), [&](const ErrorType &Err) -> llvm::Error {           \
          EXPECT_EQ(Err.message(), ExpectedMessage);                           \
          return llvm::Error::success();                                       \
        }));                                                                   \
  } while (false)

TEST(CheckOptionsValidation, MissingOptions) {
  ClangTidyOptions Options;
  ClangTidyContext Context(std::make_unique<DefaultOptionsProvider>(
      ClangTidyGlobalOptions(), Options));
  TestCheck TestCheck(&Context);
  CHECK_ERROR(TestCheck.getLocal("Opt"), MissingOptionError,
              "option not found 'test.Opt'");
  EXPECT_EQ(TestCheck.getLocal("Opt", "Unknown"), "Unknown");
}

TEST(CheckOptionsValidation, ValidIntOptions) {
  ClangTidyOptions Options;
  auto &CheckOptions = Options.CheckOptions;
  CheckOptions["test.IntExpected1"] = "1";
  CheckOptions["test.IntExpected2"] = "1WithMore";
  CheckOptions["test.IntExpected3"] = "NoInt";
  CheckOptions["GlobalIntExpected1"] = "1";
  CheckOptions["GlobalIntExpected2"] = "NoInt";
  CheckOptions["test.DefaultedIntInvalid"] = "NoInt";
  CheckOptions["GlobalIntInvalid"] = "NoInt";
  CheckOptions["test.BoolITrueValue"] = "1";
  CheckOptions["test.BoolIFalseValue"] = "0";
  CheckOptions["test.BoolTrueValue"] = "true";
  CheckOptions["test.BoolFalseValue"] = "false";
  CheckOptions["test.BoolUnparseable"] = "Nothing";
  CheckOptions["test.BoolCaseMismatch"] = "True";

  ClangTidyContext Context(std::make_unique<DefaultOptionsProvider>(
      ClangTidyGlobalOptions(), Options));
  TestCheck TestCheck(&Context);

#define CHECK_ERROR_INT(Name, Expected)                                        \
  CHECK_ERROR(Name, UnparseableIntegerOptionError, Expected)

  CHECK_VAL(TestCheck.getIntLocal("IntExpected1"), 1);
  CHECK_VAL(TestCheck.getIntGlobal("GlobalIntExpected1"), 1);
  CHECK_ERROR_INT(TestCheck.getIntLocal("IntExpected2"),
                  "invalid configuration value '1WithMore' for option "
                  "'test.IntExpected2'; expected an integer value");
  CHECK_ERROR_INT(TestCheck.getIntLocal("IntExpected3"),
                  "invalid configuration value 'NoInt' for option "
                  "'test.IntExpected3'; expected an integer value");
  CHECK_ERROR_INT(TestCheck.getIntGlobal("GlobalIntExpected2"),
                  "invalid configuration value 'NoInt' for option "
                  "'GlobalIntExpected2'; expected an integer value");
  ASSERT_EQ(TestCheck.getIntLocal("DefaultedIntInvalid", 1), 1);
  ASSERT_EQ(TestCheck.getIntGlobal("GlobalIntInvalid", 1), 1);

  CHECK_VAL(TestCheck.getIntLocal<bool>("BoolITrueValue"), true);
  CHECK_VAL(TestCheck.getIntLocal<bool>("BoolIFalseValue"), false);
  CHECK_VAL(TestCheck.getIntLocal<bool>("BoolTrueValue"), true);
  CHECK_VAL(TestCheck.getIntLocal<bool>("BoolFalseValue"), false);
  CHECK_ERROR_INT(TestCheck.getIntLocal<bool>("BoolUnparseable"),
                  "invalid configuration value 'Nothing' for option "
                  "'test.BoolUnparseable'; expected a bool");
  CHECK_ERROR_INT(TestCheck.getIntLocal<bool>("BoolCaseMismatch"),
                  "invalid configuration value 'True' for option "
                  "'test.BoolCaseMismatch'; expected a bool");

#undef CHECK_ERROR_INT
}

TEST(ValidConfiguration, ValidEnumOptions) {
  ClangTidyOptions Options;
  auto &CheckOptions = Options.CheckOptions;

  CheckOptions["test.Valid"] = "Red";
  CheckOptions["test.Invalid"] = "Scarlet";
  CheckOptions["test.ValidWrongCase"] = "rED";
  CheckOptions["test.NearMiss"] = "Oragne";
  CheckOptions["GlobalValid"] = "Violet";
  CheckOptions["GlobalInvalid"] = "Purple";
  CheckOptions["GlobalValidWrongCase"] = "vIOLET";
  CheckOptions["GlobalNearMiss"] = "Yelow";

  ClangTidyContext Context(std::make_unique<DefaultOptionsProvider>(
      ClangTidyGlobalOptions(), Options));
  TestCheck TestCheck(&Context);

#define CHECK_ERROR_ENUM(Name, Expected)                                       \
  CHECK_ERROR(Name, UnparseableEnumOptionError, Expected)

  CHECK_VAL(TestCheck.getIntLocal<Colours>("Valid"), Colours::Red);
  CHECK_VAL(TestCheck.getIntGlobal<Colours>("GlobalValid"), Colours::Violet);
  CHECK_VAL(
      TestCheck.getIntLocal<Colours>("ValidWrongCase", /*IgnoreCase*/ true),
      Colours::Red);
  CHECK_VAL(TestCheck.getIntGlobal<Colours>("GlobalValidWrongCase",
                                            /*IgnoreCase*/ true),
            Colours::Violet);
  CHECK_ERROR_ENUM(TestCheck.getIntLocal<Colours>("Invalid"),
                   "invalid configuration value "
                   "'Scarlet' for option 'test.Invalid'");
  CHECK_ERROR_ENUM(TestCheck.getIntLocal<Colours>("ValidWrongCase"),
                   "invalid configuration value 'rED' for option "
                   "'test.ValidWrongCase'; did you mean 'Red'?");
  CHECK_ERROR_ENUM(TestCheck.getIntLocal<Colours>("NearMiss"),
                   "invalid configuration value 'Oragne' for option "
                   "'test.NearMiss'; did you mean 'Orange'?");
  CHECK_ERROR_ENUM(TestCheck.getIntGlobal<Colours>("GlobalInvalid"),
                   "invalid configuration value "
                   "'Purple' for option 'GlobalInvalid'");
  CHECK_ERROR_ENUM(TestCheck.getIntGlobal<Colours>("GlobalValidWrongCase"),
                   "invalid configuration value 'vIOLET' for option "
                   "'GlobalValidWrongCase'; did you mean 'Violet'?");
  CHECK_ERROR_ENUM(TestCheck.getIntGlobal<Colours>("GlobalNearMiss"),
                   "invalid configuration value 'Yelow' for option "
                   "'GlobalNearMiss'; did you mean 'Yellow'?");

#undef CHECK_ERROR_ENUM
}

#undef CHECK_VAL
#undef CHECK_ERROR

} // namespace test
} // namespace tidy
} // namespace clang
