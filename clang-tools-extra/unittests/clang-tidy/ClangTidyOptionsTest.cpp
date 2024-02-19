#include "ClangTidyOptions.h"
#include "ClangTidyCheck.h"
#include "ClangTidyDiagnosticConsumer.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Testing/Annotations/Annotations.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <optional>

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
    return ArrayRef(Mapping);
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
      parseConfiguration(llvm::MemoryBufferRef(
          "Checks: \"-*,misc-*\"\n"
          "HeaderFileExtensions: [\"\",\"h\",\"hh\",\"hpp\",\"hxx\"]\n"
          "ImplementationFileExtensions: [\"c\",\"cc\",\"cpp\",\"cxx\"]\n"
          "HeaderFilterRegex: \".*\"\n"
          "User: some.user",
          "Options"));
  EXPECT_TRUE(!!Options);
  EXPECT_EQ("-*,misc-*", *Options->Checks);
  EXPECT_EQ(std::vector<std::string>({"", "h", "hh", "hpp", "hxx"}),
            *Options->HeaderFileExtensions);
  EXPECT_EQ(std::vector<std::string>({"c", "cc", "cpp", "cxx"}),
            *Options->ImplementationFileExtensions);
  EXPECT_EQ(".*", *Options->HeaderFilterRegex);
  EXPECT_EQ("some.user", *Options->User);
}

TEST(ParseConfiguration, ChecksSeparatedByNewlines) {
  auto MemoryBuffer = llvm::MemoryBufferRef("Checks: |\n"
                                            "  -*,misc-*\n"
                                            "  llvm-*\n"
                                            "  -clang-*,\n"
                                            "  google-*",
                                            "Options");

  auto Options = parseConfiguration(MemoryBuffer);

  EXPECT_TRUE(!!Options);
  EXPECT_EQ("-*,misc-*\nllvm-*\n-clang-*,\ngoogle-*\n", *Options->Checks);
}

TEST(ParseConfiguration, MergeConfigurations) {
  llvm::ErrorOr<ClangTidyOptions> Options1 =
      parseConfiguration(llvm::MemoryBufferRef(R"(
      Checks: "check1,check2"
      HeaderFileExtensions: ["h","hh"]
      ImplementationFileExtensions: ["c","cc"]
      HeaderFilterRegex: "filter1"
      User: user1
      ExtraArgs: ['arg1', 'arg2']
      ExtraArgsBefore: ['arg-before1', 'arg-before2']
      UseColor: false
      SystemHeaders: false
  )",
                                               "Options1"));
  ASSERT_TRUE(!!Options1);
  llvm::ErrorOr<ClangTidyOptions> Options2 =
      parseConfiguration(llvm::MemoryBufferRef(R"(
      Checks: "check3,check4"
      HeaderFileExtensions: ["hpp","hxx"]
      ImplementationFileExtensions: ["cpp","cxx"]
      HeaderFilterRegex: "filter2"
      User: user2
      ExtraArgs: ['arg3', 'arg4']
      ExtraArgsBefore: ['arg-before3', 'arg-before4']
      UseColor: true
      SystemHeaders: true
  )",
                                               "Options2"));
  ASSERT_TRUE(!!Options2);
  ClangTidyOptions Options = Options1->merge(*Options2, 0);
  EXPECT_EQ("check1,check2,check3,check4", *Options.Checks);
  EXPECT_EQ(std::vector<std::string>({"hpp", "hxx"}),
            *Options.HeaderFileExtensions);
  EXPECT_EQ(std::vector<std::string>({"cpp", "cxx"}),
            *Options.ImplementationFileExtensions);
  EXPECT_EQ("filter2", *Options.HeaderFilterRegex);
  EXPECT_EQ("user2", *Options.User);
  ASSERT_TRUE(Options.ExtraArgs.has_value());
  EXPECT_EQ("arg1,arg2,arg3,arg4", llvm::join(Options.ExtraArgs->begin(),
                                              Options.ExtraArgs->end(), ","));
  ASSERT_TRUE(Options.ExtraArgsBefore.has_value());
  EXPECT_EQ("arg-before1,arg-before2,arg-before3,arg-before4",
            llvm::join(Options.ExtraArgsBefore->begin(),
                       Options.ExtraArgsBefore->end(), ","));
  ASSERT_TRUE(Options.UseColor.has_value());
  EXPECT_TRUE(*Options.UseColor);

  ASSERT_TRUE(Options.SystemHeaders.has_value());
  EXPECT_TRUE(*Options.SystemHeaders);
}

namespace {
class DiagCollecter {
public:
  struct Diag {
  private:
    static size_t posToOffset(const llvm::SMLoc Loc,
                              const llvm::SourceMgr *Src) {
      return Loc.getPointer() -
             Src->getMemoryBuffer(Src->FindBufferContainingLoc(Loc))
                 ->getBufferStart();
    }

  public:
    Diag(const llvm::SMDiagnostic &D)
        : Message(D.getMessage()), Kind(D.getKind()),
          Pos(posToOffset(D.getLoc(), D.getSourceMgr())) {
      if (!D.getRanges().empty()) {
        // Ranges are stored as column numbers on the line that has the error.
        unsigned Offset = Pos - D.getColumnNo();
        Range.emplace();
        Range->Begin = Offset + D.getRanges().front().first,
        Range->End = Offset + D.getRanges().front().second;
      }
    }
    std::string Message;
    llvm::SourceMgr::DiagKind Kind;
    size_t Pos;
    std::optional<llvm::Annotations::Range> Range;

    friend void PrintTo(const Diag &D, std::ostream *OS) {
      *OS << (D.Kind == llvm::SourceMgr::DK_Error ? "error: " : "warning: ")
          << D.Message << "@" << llvm::to_string(D.Pos);
      if (D.Range)
        *OS << ":[" << D.Range->Begin << ", " << D.Range->End << ")";
    }
  };

  DiagCollecter() = default;
  DiagCollecter(const DiagCollecter &) = delete;

  std::function<void(const llvm::SMDiagnostic &)>
  getCallback(bool Clear = true) & {
    if (Clear)
      Diags.clear();
    return [&](const llvm::SMDiagnostic &Diag) { Diags.emplace_back(Diag); };
  }

  std::function<void(const llvm::SMDiagnostic &)>
  getCallback(bool Clear = true) && = delete;

  llvm::ArrayRef<Diag> getDiags() const { return Diags; }

private:
  std::vector<Diag> Diags;
};

MATCHER_P(DiagMessage, M, "") { return arg.Message == M; }
MATCHER_P(DiagKind, K, "") { return arg.Kind == K; }
MATCHER_P(DiagPos, P, "") { return arg.Pos == P; }
MATCHER_P(DiagRange, P, "") { return arg.Range && *arg.Range == P; }
} // namespace

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;

TEST(ParseConfiguration, CollectDiags) {
  DiagCollecter Collector;
  auto ParseWithDiags = [&](llvm::StringRef Buffer) {
    return parseConfigurationWithDiags(llvm::MemoryBufferRef(Buffer, "Options"),
                                       Collector.getCallback());
  };
  llvm::Annotations Options(R"(
    [[Check]]: llvm-include-order
  )");
  llvm::ErrorOr<ClangTidyOptions> ParsedOpt = ParseWithDiags(Options.code());
  EXPECT_TRUE(!ParsedOpt);
  EXPECT_THAT(Collector.getDiags(),
              testing::ElementsAre(AllOf(DiagMessage("unknown key 'Check'"),
                                         DiagKind(llvm::SourceMgr::DK_Error),
                                         DiagPos(Options.range().Begin),
                                         DiagRange(Options.range()))));

  Options = llvm::Annotations(R"(
    UseColor: [[NotABool]]
  )");
  ParsedOpt = ParseWithDiags(Options.code());
  EXPECT_TRUE(!ParsedOpt);
  EXPECT_THAT(Collector.getDiags(),
              testing::ElementsAre(AllOf(DiagMessage("invalid boolean"),
                                         DiagKind(llvm::SourceMgr::DK_Error),
                                         DiagPos(Options.range().Begin),
                                         DiagRange(Options.range()))));

  Options = llvm::Annotations(R"(
    SystemHeaders: [[NotABool]]
  )");
  ParsedOpt = ParseWithDiags(Options.code());
  EXPECT_TRUE(!ParsedOpt);
  EXPECT_THAT(Collector.getDiags(),
              testing::ElementsAre(AllOf(DiagMessage("invalid boolean"),
                                         DiagKind(llvm::SourceMgr::DK_Error),
                                         DiagPos(Options.range().Begin),
                                         DiagRange(Options.range()))));
}

namespace {
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

MATCHER_P(ToolDiagMessage, M, "") { return arg.Message.Message == M; }
MATCHER_P(ToolDiagLevel, L, "") { return arg.DiagLevel == L; }

} // namespace

} // namespace test

static constexpr auto Warning = tooling::Diagnostic::Warning;
static constexpr auto Error = tooling::Diagnostic::Error;

static void PrintTo(const ClangTidyError &Err, ::std::ostream *OS) {
  *OS << (Err.DiagLevel == Error ? "error: " : "warning: ")
      << Err.Message.Message;
}

namespace test {

TEST(CheckOptionsValidation, MissingOptions) {
  ClangTidyOptions Options;
  ClangTidyContext Context(std::make_unique<DefaultOptionsProvider>(
      ClangTidyGlobalOptions(), Options));
  ClangTidyDiagnosticConsumer DiagConsumer(Context);
  DiagnosticsEngine DE(new DiagnosticIDs(), new DiagnosticOptions,
                       &DiagConsumer, false);
  Context.setDiagnosticsEngine(&DE);
  TestCheck TestCheck(&Context);
  EXPECT_FALSE(TestCheck.getLocal("Opt"));
  EXPECT_EQ(TestCheck.getLocal("Opt", "Unknown"), "Unknown");
  // Missing options aren't errors.
  EXPECT_TRUE(DiagConsumer.take().empty());
}

TEST(CheckOptionsValidation, ValidIntOptions) {
  ClangTidyOptions Options;
  auto &CheckOptions = Options.CheckOptions;
  CheckOptions["test.IntExpected"] = "1";
  CheckOptions["test.IntInvalid1"] = "1WithMore";
  CheckOptions["test.IntInvalid2"] = "NoInt";
  CheckOptions["GlobalIntExpected"] = "1";
  CheckOptions["GlobalIntInvalid"] = "NoInt";
  CheckOptions["test.DefaultedIntInvalid"] = "NoInt";
  CheckOptions["test.BoolITrueValue"] = "1";
  CheckOptions["test.BoolIFalseValue"] = "0";
  CheckOptions["test.BoolTrueValue"] = "true";
  CheckOptions["test.BoolFalseValue"] = "false";
  CheckOptions["test.BoolTrueShort"] = "Y";
  CheckOptions["test.BoolFalseShort"] = "N";
  CheckOptions["test.BoolUnparseable"] = "Nothing";

  ClangTidyContext Context(std::make_unique<DefaultOptionsProvider>(
      ClangTidyGlobalOptions(), Options));
  ClangTidyDiagnosticConsumer DiagConsumer(Context);
  DiagnosticsEngine DE(new DiagnosticIDs(), new DiagnosticOptions,
                       &DiagConsumer, false);
  Context.setDiagnosticsEngine(&DE);
  TestCheck TestCheck(&Context);

  CHECK_VAL(TestCheck.getIntLocal("IntExpected"), 1);
  CHECK_VAL(TestCheck.getIntGlobal("GlobalIntExpected"), 1);
  EXPECT_FALSE(TestCheck.getIntLocal("IntInvalid1").has_value());
  EXPECT_FALSE(TestCheck.getIntLocal("IntInvalid2").has_value());
  EXPECT_FALSE(TestCheck.getIntGlobal("GlobalIntInvalid").has_value());
  ASSERT_EQ(TestCheck.getIntLocal("DefaultedIntInvalid", 1), 1);

  CHECK_VAL(TestCheck.getIntLocal<bool>("BoolITrueValue"), true);
  CHECK_VAL(TestCheck.getIntLocal<bool>("BoolIFalseValue"), false);
  CHECK_VAL(TestCheck.getIntLocal<bool>("BoolTrueValue"), true);
  CHECK_VAL(TestCheck.getIntLocal<bool>("BoolFalseValue"), false);
  CHECK_VAL(TestCheck.getIntLocal<bool>("BoolTrueShort"), true);
  CHECK_VAL(TestCheck.getIntLocal<bool>("BoolFalseShort"), false);
  EXPECT_FALSE(TestCheck.getIntLocal<bool>("BoolUnparseable"));

  EXPECT_THAT(
      DiagConsumer.take(),
      UnorderedElementsAre(
          AllOf(ToolDiagMessage(
                    "invalid configuration value '1WithMore' for option "
                    "'test.IntInvalid1'; expected an integer"),
                ToolDiagLevel(Warning)),
          AllOf(
              ToolDiagMessage("invalid configuration value 'NoInt' for option "
                              "'test.IntInvalid2'; expected an integer"),
              ToolDiagLevel(Warning)),
          AllOf(
              ToolDiagMessage("invalid configuration value 'NoInt' for option "
                              "'GlobalIntInvalid'; expected an integer"),
              ToolDiagLevel(Warning)),
          AllOf(ToolDiagMessage(
                    "invalid configuration value 'NoInt' for option "
                    "'test.DefaultedIntInvalid'; expected an integer"),
                ToolDiagLevel(Warning)),
          AllOf(ToolDiagMessage(
                    "invalid configuration value 'Nothing' for option "
                    "'test.BoolUnparseable'; expected a bool"),
                ToolDiagLevel(Warning))));
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
  ClangTidyDiagnosticConsumer DiagConsumer(Context);
  DiagnosticsEngine DE(new DiagnosticIDs(), new DiagnosticOptions,
                       &DiagConsumer, false);
  Context.setDiagnosticsEngine(&DE);
  TestCheck TestCheck(&Context);

  CHECK_VAL(TestCheck.getIntLocal<Colours>("Valid"), Colours::Red);
  CHECK_VAL(TestCheck.getIntGlobal<Colours>("GlobalValid"), Colours::Violet);

  CHECK_VAL(
      TestCheck.getIntLocal<Colours>("ValidWrongCase", /*IgnoreCase*/ true),
      Colours::Red);
  CHECK_VAL(TestCheck.getIntGlobal<Colours>("GlobalValidWrongCase",
                                            /*IgnoreCase*/ true),
            Colours::Violet);

  EXPECT_FALSE(TestCheck.getIntLocal<Colours>("ValidWrongCase").has_value());
  EXPECT_FALSE(TestCheck.getIntLocal<Colours>("NearMiss").has_value());
  EXPECT_FALSE(TestCheck.getIntGlobal<Colours>("GlobalInvalid").has_value());
  EXPECT_FALSE(
      TestCheck.getIntGlobal<Colours>("GlobalValidWrongCase").has_value());
  EXPECT_FALSE(TestCheck.getIntGlobal<Colours>("GlobalNearMiss").has_value());

  EXPECT_FALSE(TestCheck.getIntLocal<Colours>("Invalid").has_value());
  EXPECT_THAT(
      DiagConsumer.take(),
      UnorderedElementsAre(
          AllOf(ToolDiagMessage("invalid configuration value "
                                "'Scarlet' for option 'test.Invalid'"),
                ToolDiagLevel(Warning)),
          AllOf(ToolDiagMessage("invalid configuration value 'rED' for option "
                                "'test.ValidWrongCase'; did you mean 'Red'?"),
                ToolDiagLevel(Warning)),
          AllOf(
              ToolDiagMessage("invalid configuration value 'Oragne' for option "
                              "'test.NearMiss'; did you mean 'Orange'?"),
              ToolDiagLevel(Warning)),
          AllOf(ToolDiagMessage("invalid configuration value "
                                "'Purple' for option 'GlobalInvalid'"),
                ToolDiagLevel(Warning)),
          AllOf(
              ToolDiagMessage("invalid configuration value 'vIOLET' for option "
                              "'GlobalValidWrongCase'; did you mean 'Violet'?"),
              ToolDiagLevel(Warning)),
          AllOf(
              ToolDiagMessage("invalid configuration value 'Yelow' for option "
                              "'GlobalNearMiss'; did you mean 'Yellow'?"),
              ToolDiagLevel(Warning))));
}

#undef CHECK_VAL

} // namespace test
} // namespace tidy
} // namespace clang
