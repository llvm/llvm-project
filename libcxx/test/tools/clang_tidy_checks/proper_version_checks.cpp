//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "proper_version_checks.hpp"

#include <clang/Lex/Lexer.h>
#include <clang/Lex/PPCallbacks.h>
#include <clang/Lex/Preprocessor.h>

namespace libcpp {
namespace {
class proper_version_checks_callbacks : public clang::PPCallbacks {
public:
  proper_version_checks_callbacks(clang::Preprocessor& preprocessor, clang::tidy::ClangTidyCheck& check)
      : preprocessor_(preprocessor), check_(check) {}

  void If(clang::SourceLocation location, clang::SourceRange condition_range, ConditionValueKind) override {
    check_condition(location, condition_range);
  }

  void Elif(clang::SourceLocation location,
            clang::SourceRange condition_range,
            ConditionValueKind,
            clang::SourceLocation if_location) override {
    check_condition(location, condition_range);
  }

private:
  void check_condition(clang::SourceLocation location, clang::SourceRange condition_range) {
    std::string_view condition = clang::Lexer::getSourceText(
        clang::CharSourceRange::getTokenRange(condition_range),
        preprocessor_.getSourceManager(),
        preprocessor_.getLangOpts());

    if (condition == "__cplusplus < 201103L && defined(_LIBCPP_USE_FROZEN_CXX03_HEADERS)")
      return;

    if (condition.starts_with("_LIBCPP_STD_VER") && condition.find(">") != std::string_view::npos &&
        condition.find(">=") == std::string_view::npos)
      check_.diag(location, "_LIBCPP_STD_VER >= version should be used instead of _LIBCPP_STD_VER > prev_version");

    else if (condition.starts_with("__cplusplus"))
      check_.diag(location, "Use _LIBCPP_STD_VER instead of __cplusplus to constrain based on the C++ version");

    else if (condition == "_LIBCPP_STD_VER >= 11")
      check_.diag(location, "_LIBCPP_STD_VER >= 11 is always true. Did you mean '#ifndef _LIBCPP_CXX03_LANG'?");

    else if (condition.starts_with("_LIBCPP_STD_VER >= ") &&
             std::ranges::none_of(std::array{"14", "17", "20", "23", "26"}, [&](auto val) {
               return condition.find(val) != std::string_view::npos;
             }))
      check_.diag(location, "Not a valid value for _LIBCPP_STD_VER. Use 14, 17, 20, 23, or 26");
  }

  clang::Preprocessor& preprocessor_;
  clang::tidy::ClangTidyCheck& check_;
};
} // namespace

proper_version_checks::proper_version_checks(llvm::StringRef name, clang::tidy::ClangTidyContext* context)
    : clang::tidy::ClangTidyCheck(name, context) {}

void proper_version_checks::registerPPCallbacks(
    const clang::SourceManager& source_manager,
    clang::Preprocessor* preprocessor,
    clang::Preprocessor* module_expander) {
  preprocessor->addPPCallbacks(std::make_unique<proper_version_checks_callbacks>(*preprocessor, *this));
}

} // namespace libcpp
