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
class CompilerInstance;
} // namespace clang

namespace clang::tidy {

class ClangTidyContext;

/// Validates `analyze_as_class` / `analyze_as_method` attributes against a
/// throw-away AST of the standard library header they mimic (e.g.
/// `<optional>` for `std::optional`). That AST is built with the analysed
/// TU's own compiler settings, so the header is found wherever the project's
/// build finds it; nothing has to be configured. It is built only if the TU
/// contains annotations, and at most once per process per distinct setting.
///
/// Returns a consumer that, at the end of the TU, reports a
/// `clang-tidy-config` diagnostic for each attribute that matches nothing in
/// its target class.
std::unique_ptr<ASTConsumer>
createAnalyzeAsAttrValidationConsumer(ClangTidyContext &Context,
                                      CompilerInstance &Compiler);

} // namespace clang::tidy

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANALYZEASATTRVALIDATOR_H
