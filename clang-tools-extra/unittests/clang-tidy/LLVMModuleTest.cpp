#include "ClangTidyTest.h"
#include "llvm/HeaderGuardCheck.h"
#include "llvm/IncludeOrderCheck.h"
#include "gtest/gtest.h"

using namespace clang::tidy::llvm;

namespace clang {
namespace tidy {
namespace test {

// FIXME: It seems this might be incompatible to dos path. Investigating.
#if !defined(_WIN32)
static std::string runHeaderGuardCheck(StringRef Code, const Twine &Filename,
                                       Optional<StringRef> ExpectedWarning) {
  std::vector<ClangTidyError> Errors;
  std::string Result = test::runCheckOnCode<LLVMHeaderGuardCheck>(
      Code, &Errors, Filename, std::string("-xc++-header"));
  if (Errors.size() != (size_t)ExpectedWarning.hasValue())
    return "invalid error count";
  if (ExpectedWarning && *ExpectedWarning != Errors.back().Message.Message)
    return "expected: '" + ExpectedWarning->str() + "', saw: '" +
           Errors.back().Message.Message + "'";
  return Result;
}

namespace {
struct WithEndifComment : public LLVMHeaderGuardCheck {
  WithEndifComment(StringRef Name, ClangTidyContext *Context)
      : LLVMHeaderGuardCheck(Name, Context) {}
  bool shouldSuggestEndifComment(StringRef Filename) override { return true; }
};
} // namespace

static std::string
runHeaderGuardCheckWithEndif(StringRef Code, const Twine &Filename,
                             Optional<StringRef> ExpectedWarning) {
  std::vector<ClangTidyError> Errors;
  std::string Result = test::runCheckOnCode<WithEndifComment>(
      Code, &Errors, Filename, std::string("-xc++-header"));
  if (Errors.size() != (size_t)ExpectedWarning.hasValue())
    return "invalid error count";
  if (ExpectedWarning && *ExpectedWarning != Errors.back().Message.Message)
    return "expected: '" + ExpectedWarning->str() + "', saw: '" +
           Errors.back().Message.Message + "'";
  return Result;
}

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
                                "include/llvm/ADT/foo.h", None));

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
                                         "include/llvm/ADT/foo.h", None));

  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H_\n"
            "#define LLVM_ADT_FOO_H_\n"
            "#endif // LLVM_ADT_FOO_H_\n",
            runHeaderGuardCheckWithEndif("#ifndef LLVM_ADT_FOO_H_\n"
                                         "#define LLVM_ADT_FOO_H_\n"
                                         "#endif // LLVM_ADT_FOO_H_\n",
                                         "include/llvm/ADT/foo.h", None));

  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H\n"
            "#define LLVM_ADT_FOO_H\n"
            "#endif // LLVM_ADT_FOO_H\n",
            runHeaderGuardCheckWithEndif(
                "#ifndef LLVM_ADT_FOO_H_\n"
                "#define LLVM_ADT_FOO_H_\n"
                "#endif // LLVM\n",
                "include/llvm/ADT/foo.h",
                StringRef("header guard does not follow preferred style")));

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
                                         "include/llvm/ADT/foo.h", None));
}
#endif

} // namespace test
} // namespace tidy
} // namespace clang
