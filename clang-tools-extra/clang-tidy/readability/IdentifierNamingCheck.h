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
#include <optional>
#include <string>
namespace clang::tidy {
namespace readability {

enum StyleKind : int;

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
    CT_CamelSnakeBack,
    CT_LeadingUpperSnakeCase
  };

  enum HungarianPrefixType {
    HPT_Off = 0,
    HPT_On,
    HPT_LowerCase,
    HPT_CamelCase,
  };

  struct HungarianNotationOption {
    HungarianNotationOption() = default;

    std::optional<CaseType> Case;
    HungarianPrefixType HPType = HungarianPrefixType::HPT_Off;
    llvm::StringMap<std::string> General;
    llvm::StringMap<std::string> CString;
    llvm::StringMap<std::string> PrimitiveType;
    llvm::StringMap<std::string> UserDefinedType;
    llvm::StringMap<std::string> DerivedType;
  };

  struct NamingStyle {
    NamingStyle() = default;

    NamingStyle(std::optional<CaseType> Case, StringRef Prefix,
                StringRef Suffix, StringRef IgnoredRegexpStr,
                HungarianPrefixType HPType);
    NamingStyle(const NamingStyle &O) = delete;
    NamingStyle &operator=(NamingStyle &&O) = default;
    NamingStyle(NamingStyle &&O) = default;

    std::optional<CaseType> Case;
    std::string Prefix;
    std::string Suffix;
    // Store both compiled and non-compiled forms so original value can be
    // serialized
    llvm::Regex IgnoredRegexp;
    std::string IgnoredRegexpStr;

    HungarianPrefixType HPType;
  };

  struct HungarianNotation {
  public:
    bool checkOptionValid(int StyleKindIndex) const;
    bool isOptionEnabled(StringRef OptionKey,
                         const llvm::StringMap<std::string> &StrMap) const;

    size_t getAsteriskCount(const std::string &TypeName) const;
    size_t getAsteriskCount(const std::string &TypeName,
                            const NamedDecl *ND) const;

    void loadDefaultConfig(
        IdentifierNamingCheck::HungarianNotationOption &HNOption) const;
    void loadFileConfig(
        const ClangTidyCheck::OptionsView &Options,
        IdentifierNamingCheck::HungarianNotationOption &HNOption) const;

    bool removeDuplicatedPrefix(
        SmallVector<StringRef, 8> &Words,
        const IdentifierNamingCheck::HungarianNotationOption &HNOption) const;

    std::string getPrefix(
        const Decl *D,
        const IdentifierNamingCheck::HungarianNotationOption &HNOption) const;

    std::string getDataTypePrefix(
        StringRef TypeName, const NamedDecl *ND,
        const IdentifierNamingCheck::HungarianNotationOption &HNOption) const;

    std::string getClassPrefix(
        const CXXRecordDecl *CRD,
        const IdentifierNamingCheck::HungarianNotationOption &HNOption) const;

    std::string getEnumPrefix(const EnumConstantDecl *ECD) const;
    std::string getDeclTypeName(const NamedDecl *ND) const;
  };

  struct FileStyle {
    FileStyle() : IsActive(false), IgnoreMainLikeFunctions(false) {}
    FileStyle(SmallVectorImpl<std::optional<NamingStyle>> &&Styles,
              HungarianNotationOption HNOption, bool IgnoreMainLike)
        : Styles(std::move(Styles)), HNOption(std::move(HNOption)),
          IsActive(true), IgnoreMainLikeFunctions(IgnoreMainLike) {}

    ArrayRef<std::optional<NamingStyle>> getStyles() const {
      assert(IsActive);
      return Styles;
    }

    const HungarianNotationOption &getHNOption() const {
      assert(IsActive);
      return HNOption;
    }

    bool isActive() const { return IsActive; }
    bool isIgnoringMainLikeFunction() const { return IgnoreMainLikeFunctions; }

  private:
    SmallVector<std::optional<NamingStyle>, 0> Styles;
    HungarianNotationOption HNOption;
    bool IsActive;
    bool IgnoreMainLikeFunctions;
  };

  IdentifierNamingCheck::FileStyle
  getFileStyleFromOptions(const ClangTidyCheck::OptionsView &Options) const;

  bool
  matchesStyle(StringRef Type, StringRef Name,
               const IdentifierNamingCheck::NamingStyle &Style,
               const IdentifierNamingCheck::HungarianNotationOption &HNOption,
               const NamedDecl *Decl) const;

  std::string
  fixupWithCase(StringRef Type, StringRef Name, const Decl *D,
                const IdentifierNamingCheck::NamingStyle &Style,
                const IdentifierNamingCheck::HungarianNotationOption &HNOption,
                IdentifierNamingCheck::CaseType Case) const;

  std::string
  fixupWithStyle(StringRef Type, StringRef Name,
                 const IdentifierNamingCheck::NamingStyle &Style,
                 const IdentifierNamingCheck::HungarianNotationOption &HNOption,
                 const Decl *D) const;

  StyleKind findStyleKind(
      const NamedDecl *D,
      ArrayRef<std::optional<IdentifierNamingCheck::NamingStyle>> NamingStyles,
      bool IgnoreMainLikeFunctions) const;

  std::optional<RenamerClangTidyCheck::FailureInfo> getFailureInfo(
      StringRef Type, StringRef Name, const NamedDecl *ND,
      SourceLocation Location,
      ArrayRef<std::optional<IdentifierNamingCheck::NamingStyle>> NamingStyles,
      const IdentifierNamingCheck::HungarianNotationOption &HNOption,
      StyleKind SK, const SourceManager &SM, bool IgnoreFailedSplit) const;

  bool isParamInMainLikeFunction(const ParmVarDecl &ParmDecl,
                                 bool IncludeMainLike) const;

private:
  std::optional<FailureInfo>
  getDeclFailureInfo(const NamedDecl *Decl,
                     const SourceManager &SM) const override;
  std::optional<FailureInfo>
  getMacroFailureInfo(const Token &MacroNameTok,
                      const SourceManager &SM) const override;
  DiagInfo getDiagInfo(const NamingCheckId &ID,
                       const NamingCheckFailure &Failure) const override;

  const FileStyle &getStyleForFile(StringRef FileName) const;

  /// Stores the style options as a vector, indexed by the specified \ref
  /// StyleKind, for a given directory.
  mutable llvm::StringMap<FileStyle> NamingStylesCache;
  FileStyle *MainFileStyle;
  ClangTidyContext *Context;
  const bool GetConfigPerFile;
  const bool IgnoreFailedSplit;
  HungarianNotation HungarianNotation;
};

} // namespace readability
template <>
struct OptionEnumMapping<readability::IdentifierNamingCheck::CaseType> {
  static llvm::ArrayRef<
      std::pair<readability::IdentifierNamingCheck::CaseType, StringRef>>
  getEnumMapping();
};
} // namespace clang::tidy

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_IDENTIFIERNAMINGCHECK_H
