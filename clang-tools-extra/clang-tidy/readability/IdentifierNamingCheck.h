//===--- IdentifierNamingCheck.h - clang-tidy -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_IDENTIFIERNAMINGCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_IDENTIFIERNAMINGCHECK_H

#include "../utils/RenamerClangTidyCheck.h"
#include "llvm/ADT/Optional.h"
namespace clang {

class MacroInfo;

namespace tidy {
namespace readability {

/// Checks for identifiers naming style mismatch.
///
/// This check will try to enforce coding guidelines on the identifiers naming.
/// It supports `lower_case`, `UPPER_CASE`, `camelBack` and `CamelCase` casing
/// and tries to convert from one to another if a mismatch is detected.
///
/// It also supports a fixed prefix and suffix that will be prepended or
/// appended to the identifiers, regardless of the casing.
///
/// Many configuration options are available, in order to be able to create
/// different rules for different kind of identifier. In general, the
/// rules are falling back to a more generic rule if the specific case is not
/// configured.
class IdentifierNamingCheck final : public RenamerClangTidyCheck {
public:
  IdentifierNamingCheck(StringRef Name, ClangTidyContext *Context);
  ~IdentifierNamingCheck();

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

  enum CaseType {
    CT_AnyCase = 0,
    CT_LowerCase,
    CT_CamelBack,
    CT_UpperCase,
    CT_CamelCase,
    CT_CamelSnakeCase,
    CT_CamelSnakeBack
  };

  struct NamingStyle {
    NamingStyle() = default;

    NamingStyle(llvm::Optional<CaseType> Case, const std::string &Prefix,
                const std::string &Suffix)
        : Case(Case), Prefix(Prefix), Suffix(Suffix) {}

    llvm::Optional<CaseType> Case;
    std::string Prefix;
    std::string Suffix;
  };

  struct FileStyle {
    FileStyle() : IsActive(false), IgnoreMainLikeFunctions(false) {}
    FileStyle(SmallVectorImpl<Optional<NamingStyle>> &&Styles,
              bool IgnoreMainLike)
        : Styles(std::move(Styles)), IsActive(true),
          IgnoreMainLikeFunctions(IgnoreMainLike) {}

    ArrayRef<Optional<NamingStyle>> getStyles() const {
      assert(IsActive);
      return Styles;
    }
    bool isActive() const { return IsActive; }
    bool isIgnoringMainLikeFunction() const { return IgnoreMainLikeFunctions; }

  private:
    SmallVector<Optional<NamingStyle>, 0> Styles;
    bool IsActive;
    bool IgnoreMainLikeFunctions;
  };

private:
  llvm::Optional<FailureInfo>
  GetDeclFailureInfo(const NamedDecl *Decl,
                     const SourceManager &SM) const override;
  llvm::Optional<FailureInfo>
  GetMacroFailureInfo(const Token &MacroNameTok,
                      const SourceManager &SM) const override;
  DiagInfo GetDiagInfo(const NamingCheckId &ID,
                       const NamingCheckFailure &Failure) const override;

  const FileStyle &getStyleForFile(StringRef FileName) const;

  /// Stores the style options as a vector, indexed by the specified \ref
  /// StyleKind, for a given directory.
  mutable llvm::StringMap<FileStyle> NamingStylesCache;
  FileStyle *MainFileStyle;
  ClangTidyContext *const Context;
  const std::string CheckName;
  const bool GetConfigPerFile;
  const bool IgnoreFailedSplit;
};

} // namespace readability
template <>
struct OptionEnumMapping<readability::IdentifierNamingCheck::CaseType> {
  static llvm::ArrayRef<
      std::pair<readability::IdentifierNamingCheck::CaseType, StringRef>>
  getEnumMapping();
};
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_IDENTIFIERNAMINGCHECK_H
