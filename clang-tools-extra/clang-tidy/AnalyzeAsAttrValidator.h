//===--- AnalyzeAsAttrValidator.h - clang-tidy ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANALYZEASATTRVALIDATOR_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANALYZEASATTRVALIDATOR_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace clang {
class ASTConsumer;
class CompilerInstance;
} // namespace clang

namespace clang::tidy {

class ClangTidyContext;

/// A class that `analyze_as_class` may name, the standard library header that
/// declares it, and the check that gives the annotation a meaning.
struct AnalyzeAsTarget {
  llvm::StringRef Name;
  llvm::StringRef Header;
  llvm::StringRef Check;
};

/// What the validator does with an `analyze_as_*` attribute.
struct AnalyzeAsDecision {
  enum ActionKind {
    /// Check it against the target class in the header AST.
    Validate,
    /// Say nothing: the class is supported but its check is disabled, or the
    /// class is unsupported and no check that uses analyze_as is enabled.
    Silent,
    /// The class is not supported. Reported for `analyze_as_class` only; the
    /// methods of such a class are not validated.
    ReportUnknown,
  };
  ActionKind Action;
  /// For Validate: the check that uses the target.
  /// For ReportUnknown: the enabled check to report the warning under.
  llvm::StringRef Check;
};

/// Decides what to do with an attribute whose target class is `Target`, given
/// the supported classes `Table` and which checks are enabled.
AnalyzeAsDecision
classifyAnalyzeAsTarget(llvm::StringRef Target,
                        llvm::ArrayRef<AnalyzeAsTarget> Table,
                        llvm::function_ref<bool(llvm::StringRef)> IsEnabled);

/// Validates `analyze_as_class` / `analyze_as_method` attributes against a
/// throw-away AST of the standard library header they mimic (e.g.
/// `<optional>` for `std::optional`). That AST is built with the analysed
/// TU's own compiler settings, so the header is found wherever the project's
/// build finds it; nothing has to be configured. It is built only if the TU
/// contains annotations, and at most once per process per distinct setting.
///
/// Returns a consumer that, at the end of the TU, reports attributes naming an
/// unsupported class (under the check that uses analyze_as) and, for
/// supported classes whose check is enabled, a `clang-tidy-config` diagnostic
/// for each attribute that matches nothing in its target class.
std::unique_ptr<ASTConsumer>
createAnalyzeAsAttrValidationConsumer(ClangTidyContext &Context,
                                      CompilerInstance &Compiler);

} // namespace clang::tidy

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANALYZEASATTRVALIDATOR_H
