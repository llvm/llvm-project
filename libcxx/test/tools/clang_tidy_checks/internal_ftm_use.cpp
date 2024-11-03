//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This clang-tidy check ensures that we don't use any _LIBCPP_HAS_FOO macro
// inside `#ifdef`, `#ifndef` & friends, since the intent is to always use `#if` instead.

#include "internal_ftm_use.hpp"

#include <clang/Lex/Lexer.h>
#include <clang/Lex/PPCallbacks.h>
#include <clang/Lex/Preprocessor.h>

#include <string>

namespace libcpp {
namespace {
std::array valid_macros{
    // Public API macros
    "_LIBCPP_HAS_ASAN_CONTAINER_ANNOTATIONS_FOR_ALL_ALLOCATORS",

    // Testing macros
    "_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER",

    // TODO: Why does this macro even exist?
    "_LIBCPP_HAS_NO_TREE_BARRIER",

    // Configuration macros
    "_LIBCPP_HAS_NO_TIME_ZONE_DATABASE",
    "_LIBCPP_HAS_NO_FILESYSTEM",
    "_LIBCPP_HAS_NO_LOCALIZATION",
    "_LIBCPP_HAS_NO_THREADS",
    "_LIBCPP_HAS_NO_MONOTONIC_CLOCK",
    "_LIBCPP_HAS_MUSL_LIBC",
    "_LIBCPP_HAS_NO_WIDE_CHARACTERS",
    "_LIBCPP_HAS_NO_VENDOR_AVAILABILITY_ANNOTATIONS",
    "_LIBCPP_HAS_NO_RANDOM_DEVICE",
    "_LIBCPP_HAS_NO_UNICODE",
    "_LIBCPP_HAS_NO_TERMINAL",

    // Atomic API macros
    "_LIBCPP_HAS_EXTERNAL_ATOMIC_IMP",
    "_LIBCPP_HAS_GCC_ATOMIC_IMP",
    "_LIBCPP_HAS_C_ATOMIC_IMP",

    // Thread API macros
    "_LIBCPP_HAS_THREAD_API_PTHREAD",
    "_LIBCPP_HAS_THREAD_API_WIN32",
    "_LIBCPP_HAS_THREAD_API_EXTERNAL",
    "_LIBCPP_HAS_THREAD_API_C11",

    // Experimental features
    "_LIBCPP_HAS_NO_EXPERIMENTAL_TZDB",
    "_LIBCPP_HAS_NO_EXPERIMENTAL_SYNCSTREAM",
    "_LIBCPP_HAS_NO_EXPERIMENTAL_STOP_TOKEN",
    "_LIBCPP_HAS_NO_INCOMPLETE_PSTL",
};

class internal_ftm_use_callbacks : public clang::PPCallbacks {
public:
  internal_ftm_use_callbacks(clang::tidy::ClangTidyCheck& check) : check_(check) {}

  void Defined(const clang::Token& token,
               const clang::MacroDefinition& macro_definition,
               clang::SourceRange location) override {
    check_macro(token.getIdentifierInfo()->getName(), location.getBegin());
  }

  void Ifdef(clang::SourceLocation location, const clang::Token& token, const clang::MacroDefinition&) override {
    check_macro(token.getIdentifierInfo()->getName(), location);
  }

  void Elifdef(clang::SourceLocation location, const clang::Token& token, const clang::MacroDefinition&) override {
    check_macro(token.getIdentifierInfo()->getName(), location);
  }

  void Ifndef(clang::SourceLocation location, const clang::Token& token, const clang::MacroDefinition&) override {
    check_macro(token.getIdentifierInfo()->getName(), location);
  }

  void Elifndef(clang::SourceLocation location, const clang::Token& token, const clang::MacroDefinition&) override {
    check_macro(token.getIdentifierInfo()->getName(), location);
  }

private:
  void check_macro(std::string_view macro, clang::SourceLocation location) {
    if (macro.starts_with("_LIBCPP_HAS_") && std::ranges::find(valid_macros, macro) == valid_macros.end()) {
      check_.diag(location, std::string("\'") + std::string{macro} + "' is always defined to 1 or 0.");
    }
  }

  clang::tidy::ClangTidyCheck& check_;
};
} // namespace

internal_ftm_use::internal_ftm_use(llvm::StringRef name, clang::tidy::ClangTidyContext* context)
    : clang::tidy::ClangTidyCheck(name, context) {}

void internal_ftm_use::registerPPCallbacks(const clang::SourceManager& source_manager,
                                           clang::Preprocessor* preprocessor,
                                           clang::Preprocessor* module_expander) {
  preprocessor->addPPCallbacks(std::make_unique<internal_ftm_use_callbacks>(*this));
}

} // namespace libcpp
