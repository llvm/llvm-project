#include "ClangTidyOptions.h"
#include "ClangTidyTest.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/HeaderGuardCheck.h"
#include "llvm/IncludeOrderCheck.h"
#include "gtest/gtest.h"
#include <optional>

using namespace clang::tidy::llvm_check;

namespace clang {
namespace tidy {
namespace test {

template <typename T>
static std::string runCheck(StringRef Code, const Twine &Filename,
                            std::optional<StringRef> ExpectedWarning,
                            std::map<StringRef, StringRef> PathsToContent =
                                std::map<StringRef, StringRef>()) {
  std::vector<ClangTidyError> Errors;
  std::string Result = test::runCheckOnCode<T>(
      Code, &Errors, Filename, std::string("-xc++-header"), ClangTidyOptions{},
      std::move(PathsToContent));
  if (Errors.size() != (size_t)ExpectedWarning.has_value())
    return "invalid error count";
  if (ExpectedWarning && *ExpectedWarning != Errors.back().Message.Message)
    return "expected: '" + ExpectedWarning->str() + "', saw: '" +
           Errors.back().Message.Message + "'";
  return Result;
}

static std::string
runHeaderGuardCheck(StringRef Code, const Twine &Filename,
                    std::optional<StringRef> ExpectedWarning) {
  return runCheck<LLVMHeaderGuardCheck>(Code, Filename,
                                        std::move(ExpectedWarning));
}

static std::string
runIncludeOrderCheck(StringRef Code, const Twine &Filename,
                     std::optional<StringRef> ExpectedWarning,
                     llvm::ArrayRef<llvm::StringLiteral> Includes) {
  std::map<StringRef, StringRef> PathsToContent;
  for (auto Include : Includes)
    PathsToContent.emplace(Include, "");
  return runCheck<IncludeOrderCheck>(Code, Filename, std::move(ExpectedWarning),
                                     PathsToContent);
}

namespace {
struct WithEndifComment : public LLVMHeaderGuardCheck {
  WithEndifComment(StringRef Name, ClangTidyContext *Context)
      : LLVMHeaderGuardCheck(Name, Context) {}
  bool shouldSuggestEndifComment(StringRef Filename) override { return true; }
};

static std::string
runHeaderGuardCheckWithEndif(StringRef Code, const Twine &Filename,
                             std::optional<StringRef> ExpectedWarning) {
  return runCheck<WithEndifComment>(Code, Filename, std::move(ExpectedWarning));
}
} // namespace

TEST(LLVMHeaderGuardCheckTest, FixHeaderGuards) {
  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H\n"
            "#define LLVM_ADT_FOO_H\n"
            "#endif\n",
            runHeaderGuardCheck(
                "#ifndef FOO\n"
                "#define FOO\n"
                "#endif\n",
                "include/llvm/ADT/foo.h",
                StringRef("header guard does not follow preferred style")));

  // Allow trailing underscores.
  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H_\n"
            "#define LLVM_ADT_FOO_H_\n"
            "#endif\n",
            runHeaderGuardCheck("#ifndef LLVM_ADT_FOO_H_\n"
                                "#define LLVM_ADT_FOO_H_\n"
                                "#endif\n",
                                "include/llvm/ADT/foo.h", std::nullopt));

  EXPECT_EQ("#ifndef LLVM_CLANG_C_BAR_H\n"
            "#define LLVM_CLANG_C_BAR_H\n"
            "\n"
            "\n"
            "#endif\n",
            runHeaderGuardCheck("", "./include/clang-c/bar.h",
                                StringRef("header is missing header guard")));

  EXPECT_EQ("#ifndef LLVM_CLANG_LIB_CODEGEN_C_H\n"
            "#define LLVM_CLANG_LIB_CODEGEN_C_H\n"
            "\n"
            "\n"
            "#endif\n",
            runHeaderGuardCheck("", "tools/clang/lib/CodeGen/c.h",
                                StringRef("header is missing header guard")));

  EXPECT_EQ("#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_X_H\n"
            "#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_X_H\n"
            "\n"
            "\n"
            "#endif\n",
            runHeaderGuardCheck("", "tools/clang/tools/extra/clang-tidy/x.h",
                                StringRef("header is missing header guard")));

  EXPECT_EQ(
      "int foo;\n"
      "#ifndef LLVM_CLANG_BAR_H\n"
      "#define LLVM_CLANG_BAR_H\n"
      "#endif\n",
      runHeaderGuardCheck("int foo;\n"
                          "#ifndef LLVM_CLANG_BAR_H\n"
                          "#define LLVM_CLANG_BAR_H\n"
                          "#endif\n",
                          "include/clang/bar.h",
                          StringRef("code/includes outside of area guarded by "
                                    "header guard; consider moving it")));

  EXPECT_EQ(
      "#ifndef LLVM_CLANG_BAR_H\n"
      "#define LLVM_CLANG_BAR_H\n"
      "#endif\n"
      "int foo;\n",
      runHeaderGuardCheck("#ifndef LLVM_CLANG_BAR_H\n"
                          "#define LLVM_CLANG_BAR_H\n"
                          "#endif\n"
                          "int foo;\n",
                          "include/clang/bar.h",
                          StringRef("code/includes outside of area guarded by "
                                    "header guard; consider moving it")));

  EXPECT_EQ("#ifndef LLVM_CLANG_BAR_H\n"
            "#define LLVM_CLANG_BAR_H\n"
            "\n"
            "int foo;\n"
            "#ifndef FOOLOLO\n"
            "#define FOOLOLO\n"
            "#endif\n"
            "\n"
            "#endif\n",
            runHeaderGuardCheck("int foo;\n"
                                "#ifndef FOOLOLO\n"
                                "#define FOOLOLO\n"
                                "#endif\n",
                                "include/clang/bar.h",
                                StringRef("header is missing header guard")));

  // Fix incorrect #endif comments even if we shouldn't add new ones.
  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H\n"
            "#define LLVM_ADT_FOO_H\n"
            "#endif // LLVM_ADT_FOO_H\n",
            runHeaderGuardCheck(
                "#ifndef FOO\n"
                "#define FOO\n"
                "#endif // FOO\n",
                "include/llvm/ADT/foo.h",
                StringRef("header guard does not follow preferred style")));

  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H\n"
            "#define LLVM_ADT_FOO_H\n"
            "#endif // LLVM_ADT_FOO_H\n",
            runHeaderGuardCheckWithEndif(
                "#ifndef FOO\n"
                "#define FOO\n"
                "#endif\n",
                "include/llvm/ADT/foo.h",
                StringRef("header guard does not follow preferred style")));

  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H\n"
            "#define LLVM_ADT_FOO_H\n"
            "#endif // LLVM_ADT_FOO_H\n",
            runHeaderGuardCheckWithEndif(
                "#ifndef LLVM_ADT_FOO_H\n"
                "#define LLVM_ADT_FOO_H\n"
                "#endif // LLVM_H\n",
                "include/llvm/ADT/foo.h",
                StringRef("#endif for a header guard should reference the "
                          "guard macro in a comment")));

  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H\n"
            "#define LLVM_ADT_FOO_H\n"
            "#endif /* LLVM_ADT_FOO_H */\n",
            runHeaderGuardCheckWithEndif("#ifndef LLVM_ADT_FOO_H\n"
                                         "#define LLVM_ADT_FOO_H\n"
                                         "#endif /* LLVM_ADT_FOO_H */\n",
                                         "include/llvm/ADT/foo.h",
                                         std::nullopt));

  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H_\n"
            "#define LLVM_ADT_FOO_H_\n"
            "#endif // LLVM_ADT_FOO_H_\n",
            runHeaderGuardCheckWithEndif("#ifndef LLVM_ADT_FOO_H_\n"
                                         "#define LLVM_ADT_FOO_H_\n"
                                         "#endif // LLVM_ADT_FOO_H_\n",
                                         "include/llvm/ADT/foo.h",
                                         std::nullopt));

  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H\n"
            "#define LLVM_ADT_FOO_H\n"
            "#endif // LLVM_ADT_FOO_H\n",
            runHeaderGuardCheckWithEndif(
                "#ifndef LLVM_ADT_FOO_H_\n"
                "#define LLVM_ADT_FOO_H_\n"
                "#endif // LLVM\n",
                "include/llvm/ADT/foo.h",
                StringRef("header guard does not follow preferred style")));

  // An extra space inside the comment is OK.
  llvm::StringRef WithExtraSpace = "#ifndef LLVM_ADT_FOO_H\n"
                                   "#define LLVM_ADT_FOO_H\n"
                                   "#endif //  LLVM_ADT_FOO_H\n";
  EXPECT_EQ(WithExtraSpace,
            runHeaderGuardCheckWithEndif(
                WithExtraSpace, "include/llvm/ADT/foo.h", std::nullopt));

  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H\n"
            "#define LLVM_ADT_FOO_H\n"
            "#endif \\ \n"
            "// LLVM_ADT_FOO_H\n",
            runHeaderGuardCheckWithEndif(
                "#ifndef LLVM_ADT_FOO_H\n"
                "#define LLVM_ADT_FOO_H\n"
                "#endif \\ \n"
                "// LLVM_ADT_FOO_H\n",
                "include/llvm/ADT/foo.h",
                StringRef("backslash and newline separated by space")));

  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H\n"
            "#define LLVM_ADT_FOO_H\n"
            "#endif  /* LLVM_ADT_FOO_H\\ \n"
            " FOO */",
            runHeaderGuardCheckWithEndif("#ifndef LLVM_ADT_FOO_H\n"
                                         "#define LLVM_ADT_FOO_H\n"
                                         "#endif  /* LLVM_ADT_FOO_H\\ \n"
                                         " FOO */",
                                         "include/llvm/ADT/foo.h",
                                         std::nullopt));

  EXPECT_EQ("#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_FOO_H\n"
            "#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_FOO_H\n"
            "\n"
            "\n"
            "#endif\n",
            runHeaderGuardCheck(
                "", "/llvm-project/clang-tools-extra/clangd/foo.h",
                StringRef("header is missing header guard")));

  // Substitution of characters should not result in a header guard starting
  // with "_".
  EXPECT_EQ("#ifndef BAR_H\n"
            "#define BAR_H\n"
            "\n"
            "\n"
            "#endif\n",
            runHeaderGuardCheck("", "include/--bar.h",
                                StringRef("header is missing header guard")));

#ifdef WIN32
  // Check interaction with Windows-style path separators (\).
  EXPECT_EQ(
      "#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_FOO_H\n"
      "#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_FOO_H\n"
      "\n"
      "\n"
      "#endif\n",
      runHeaderGuardCheck("", "llvm-project\\clang-tools-extra\\clangd\\foo.h",
                          StringRef("header is missing header guard")));

  EXPECT_EQ("#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_FOO_H\n"
            "#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_FOO_H\n"
            "\n"
            "\n"
            "#endif\n",
            runHeaderGuardCheck(
                "", "C:\\llvm-project\\clang-tools-extra\\clangd\\foo.h",
                StringRef("header is missing header guard")));

  EXPECT_EQ("#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_FOO_H\n"
            "#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_FOO_H\n"
            "\n"
            "\n"
            "#endif\n",
            runHeaderGuardCheck(
                "",
                "\\\\SMBShare\\llvm-project\\clang-tools-extra\\clangd\\foo.h",
                StringRef("header is missing header guard")));

  EXPECT_EQ("#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_FOO_H\n"
            "#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_FOO_H\n"
            "\n"
            "\n"
            "#endif\n",
            runHeaderGuardCheck(
                "", "\\\\?\\C:\\llvm-project\\clang-tools-extra\\clangd\\foo.h",
                StringRef("header is missing header guard")));
#endif
}

TEST(IncludeOrderCheck, GTestHeaders) {
  EXPECT_EQ(
      R"cpp(
  #include "foo.h"
  #include "llvm/foo.h"
  #include "gtest/foo.h"
  #include <algorithm>)cpp",
      runIncludeOrderCheck(
          R"cpp(
  #include "foo.h"
  #include "llvm/foo.h"
  #include <algorithm>
  #include "gtest/foo.h")cpp",
          "foo.cc", StringRef("#includes are not sorted properly"),
          {"foo.h", "algorithm", "gtest/foo.h", "llvm/foo.h"}));
}

} // namespace test
} // namespace tidy
} // namespace clang
