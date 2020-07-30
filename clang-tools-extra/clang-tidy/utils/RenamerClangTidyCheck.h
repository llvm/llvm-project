//===--- RenamderClangTidyCheck.h - clang-tidy ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_RENAMERCLANGTIDYCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_RENAMERCLANGTIDYCHECK_H

#include "../ClangTidyCheck.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/Optional.h"
#include <string>
#include <utility>

namespace clang {

class MacroInfo;

namespace tidy {

/// Base class for clang-tidy checks that want to flag declarations and/or
/// macros for renaming based on customizable criteria.
class RenamerClangTidyCheck : public ClangTidyCheck {
public:
  RenamerClangTidyCheck(StringRef CheckName, ClangTidyContext *Context);
  ~RenamerClangTidyCheck();

  /// Derived classes should not implement any matching logic themselves; this
  /// class will do the matching and call the derived class'
  /// GetDeclFailureInfo() and GetMacroFailureInfo() for determining whether a
  /// given identifier passes or fails the check.
  void registerMatchers(ast_matchers::MatchFinder *Finder) override final;
  void
  check(const ast_matchers::MatchFinder::MatchResult &Result) override final;
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override final;
  void onEndOfTranslationUnit() override final;

  /// Derived classes that override this function should call this method from
  /// the overridden method.
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

  /// This enum will be used in %select of the diagnostic message.
  /// Each value below IgnoreFailureThreshold should have an error message.
  enum class ShouldFixStatus {
    ShouldFix,

    /// The fixup will conflict with a language keyword,
    /// so we can't fix it automatically.
    ConflictsWithKeyword,

    /// The fixup will conflict with a macro
    /// definition, so we can't fix it
    /// automatically.
    ConflictsWithMacroDefinition,

    /// Values pass this threshold will be ignored completely
    /// i.e no message, no fixup.
    IgnoreFailureThreshold,

    /// If the identifier was used or declared within a macro we
    /// won't offer a fixup for safety reasons.
    InsideMacro,
  };

  /// Information describing a failed check
  struct FailureInfo {
    std::string KindName; // Tag or misc info to be used as derived classes need
    std::string Fixup;    // The name that will be proposed as a fix-it hint
  };

  /// Holds an identifier name check failure, tracking the kind of the
  /// identifier, its possible fixup and the starting locations of all the
  /// identifier usages.
  struct NamingCheckFailure {
    FailureInfo Info;

    /// Whether the failure should be fixed or not.
    ///
    /// e.g.: if the identifier was used or declared within a macro we won't
    /// offer a fixup for safety reasons.
    bool ShouldFix() const {
      return FixStatus == ShouldFixStatus::ShouldFix && !Info.Fixup.empty();
    }

    bool ShouldNotify() const {
      return FixStatus < ShouldFixStatus::IgnoreFailureThreshold;
    }

    ShouldFixStatus FixStatus = ShouldFixStatus::ShouldFix;

    /// A set of all the identifier usages starting SourceLocation, in
    /// their encoded form.
    llvm::DenseSet<unsigned> RawUsageLocs;

    NamingCheckFailure() = default;
  };

  using NamingCheckId = std::pair<SourceLocation, std::string>;

  using NamingCheckFailureMap =
      llvm::DenseMap<NamingCheckId, NamingCheckFailure>;

  /// Check Macros for style violations.
  void checkMacro(SourceManager &sourceMgr, const Token &MacroNameTok,
                  const MacroInfo *MI);

  /// Add a usage of a macro if it already has a violation.
  void expandMacro(const Token &MacroNameTok, const MacroInfo *MI);

  void addUsage(const RenamerClangTidyCheck::NamingCheckId &Decl,
                SourceRange Range, SourceManager *SourceMgr = nullptr);

  /// Convenience method when the usage to be added is a NamedDecl.
  void addUsage(const NamedDecl *Decl, SourceRange Range,
                SourceManager *SourceMgr = nullptr);

protected:
  /// Overridden by derived classes, returns information about if and how a Decl
  /// failed the check. A 'None' result means the Decl did not fail the check.
  virtual llvm::Optional<FailureInfo>
  GetDeclFailureInfo(const NamedDecl *Decl, const SourceManager &SM) const = 0;

  /// Overridden by derived classes, returns information about if and how a
  /// macro failed the check. A 'None' result means the macro did not fail the
  /// check.
  virtual llvm::Optional<FailureInfo>
  GetMacroFailureInfo(const Token &MacroNameTok,
                      const SourceManager &SM) const = 0;

  /// Represents customized diagnostic text and how arguments should be applied.
  /// Example usage:
  ///
  /// return DiagInfo{"my %1 very %2 special %3 text",
  ///                  [=](DiagnosticBuilder &diag) {
  ///                    diag << arg1 << arg2 << arg3;
  ///                  }};
  struct DiagInfo {
    std::string Text;
    llvm::unique_function<void(DiagnosticBuilder &)> ApplyArgs;
  };

  /// Overridden by derived classes, returns a description of the diagnostic
  /// that should be emitted for the given failure. The base class will then
  /// further customize the diagnostic by adding info about whether the fix-it
  /// can be automatically applied or not.
  virtual DiagInfo GetDiagInfo(const NamingCheckId &ID,
                               const NamingCheckFailure &Failure) const = 0;

private:
  NamingCheckFailureMap NamingCheckFailures;
  const bool AggressiveDependentMemberLookup;
};

} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_RENAMERCLANGTIDYCHECK_H
