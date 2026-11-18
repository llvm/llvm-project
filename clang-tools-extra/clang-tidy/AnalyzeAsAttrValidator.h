//===--- AnalyzeAsAttrValidator.h - clang-tidy ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANALYZEASATTRVALIDATOR_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANALYZEASATTRVALIDATOR_H

#include <memory>

namespace clang {
class ASTConsumer;
class Preprocessor;
} // namespace clang

namespace clang::tidy {

class ClangTidyContext;

/// Validates `analyze_as_class` / `analyze_as_method` attributes against a
/// throw-away AST of the standard library header they mimic. The header is
/// given by `#pragma clang_tidy_analyze_as_header sysroot("...") header("...") std("...")`
/// (no defaults, nothing inferred from the analysed TU); its AST is built at
/// most once per process per distinct config.
///
/// Registers the pragma on `PP` and returns a consumer that, at the end of the
/// TU, reports a `clang-tidy-config` diagnostic for each malformed pragma and
/// for each attribute that matches nothing in its target class.
std::unique_ptr<ASTConsumer>
createAnalyzeAsAttrValidationConsumer(ClangTidyContext &Context,
                                      Preprocessor &PP);

} // namespace clang::tidy

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANALYZEASATTRVALIDATOR_H
