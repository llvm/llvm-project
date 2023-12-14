//===--- ReservedIdentifierCheck.cpp - clang-tidy -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ReservedIdentifierCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Token.h"
#include <algorithm>
#include <cctype>
#include <optional>

// FixItHint

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

static const char DoubleUnderscoreTag[] = "du";
static const char UnderscoreCapitalTag[] = "uc";
static const char GlobalUnderscoreTag[] = "global-under";
static const char NonReservedTag[] = "non-reserved";

static const char Message[] =
    "declaration uses identifier '%0', which is %select{a reserved "
    "identifier|not a reserved identifier|reserved in the global namespace}1";

static int getMessageSelectIndex(StringRef Tag) {
  if (Tag == NonReservedTag)
    return 1;
  if (Tag == GlobalUnderscoreTag)
    return 2;
  return 0;
}

llvm::SmallVector<llvm::Regex>
ReservedIdentifierCheck::parseAllowedIdentifiers() const {
  llvm::SmallVector<llvm::Regex> AllowedIdentifiers;
  AllowedIdentifiers.reserve(AllowedIdentifiersRaw.size());

  for (const auto &Identifier : AllowedIdentifiersRaw) {
    AllowedIdentifiers.emplace_back(Identifier.str());
    if (!AllowedIdentifiers.back().isValid()) {
      configurationDiag("Invalid allowed identifier regex '%0'") << Identifier;
      AllowedIdentifiers.pop_back();
    }
  }

  return AllowedIdentifiers;
}

ReservedIdentifierCheck::ReservedIdentifierCheck(StringRef Name,
                                                 ClangTidyContext *Context)
    : RenamerClangTidyCheck(Name, Context),
      Invert(Options.get("Invert", false)),
      AllowedIdentifiersRaw(utils::options::parseStringList(
          Options.get("AllowedIdentifiers", ""))),
      AllowedIdentifiers(parseAllowedIdentifiers()) {}

void ReservedIdentifierCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  RenamerClangTidyCheck::storeOptions(Opts);
  Options.store(Opts, "Invert", Invert);
  Options.store(Opts, "AllowedIdentifiers",
                utils::options::serializeStringList(AllowedIdentifiersRaw));
}

static std::string collapseConsecutive(StringRef Str, char C) {
  std::string Result;
  std::unique_copy(Str.begin(), Str.end(), std::back_inserter(Result),
                   [C](char A, char B) { return A == C && B == C; });
  return Result;
}

static bool hasReservedDoubleUnderscore(StringRef Name,
                                        const LangOptions &LangOpts) {
  if (LangOpts.CPlusPlus)
    return Name.contains("__");
  return Name.starts_with("__");
}

static std::optional<std::string>
getDoubleUnderscoreFixup(StringRef Name, const LangOptions &LangOpts) {
  if (hasReservedDoubleUnderscore(Name, LangOpts))
    return collapseConsecutive(Name, '_');
  return std::nullopt;
}

static bool startsWithUnderscoreCapital(StringRef Name) {
  return Name.size() >= 2 && Name[0] == '_' && std::isupper(Name[1]);
}

static std::optional<std::string> getUnderscoreCapitalFixup(StringRef Name) {
  if (startsWithUnderscoreCapital(Name))
    return std::string(Name.drop_front(1));
  return std::nullopt;
}

static bool startsWithUnderscoreInGlobalNamespace(StringRef Name,
                                                  bool IsInGlobalNamespace,
                                                  bool IsMacro) {
  return !IsMacro && IsInGlobalNamespace && !Name.empty() && Name[0] == '_';
}

static std::optional<std::string>
getUnderscoreGlobalNamespaceFixup(StringRef Name, bool IsInGlobalNamespace,
                                  bool IsMacro) {
  if (startsWithUnderscoreInGlobalNamespace(Name, IsInGlobalNamespace, IsMacro))
    return std::string(Name.drop_front(1));
  return std::nullopt;
}

static std::string getNonReservedFixup(std::string Name) {
  assert(!Name.empty());
  if (Name[0] == '_' || std::isupper(Name[0]))
    Name.insert(Name.begin(), '_');
  else
    Name.insert(Name.begin(), 2, '_');
  return Name;
}

static std::optional<RenamerClangTidyCheck::FailureInfo>
getFailureInfoImpl(StringRef Name, bool IsInGlobalNamespace, bool IsMacro,
                   const LangOptions &LangOpts, bool Invert,
                   ArrayRef<llvm::Regex> AllowedIdentifiers) {
  assert(!Name.empty());

  if (llvm::any_of(AllowedIdentifiers, [&](const llvm::Regex &Regex) {
        return Regex.match(Name);
      })) {
    return std::nullopt;
  }
  // TODO: Check for names identical to language keywords, and other names
  // specifically reserved by language standards, e.g. C++ 'zombie names' and C
  // future library directions

  using FailureInfo = RenamerClangTidyCheck::FailureInfo;
  if (!Invert) {
    std::optional<FailureInfo> Info;
    auto AppendFailure = [&](StringRef Kind, std::string &&Fixup) {
      if (!Info) {
        Info = FailureInfo{std::string(Kind), std::move(Fixup)};
      } else {
        Info->KindName += Kind;
        Info->Fixup = std::move(Fixup);
      }
    };
    auto InProgressFixup = [&] {
      return llvm::transformOptional(
                 Info,
                 [](const FailureInfo &Info) { return StringRef(Info.Fixup); })
          .value_or(Name);
    };
    if (auto Fixup = getDoubleUnderscoreFixup(InProgressFixup(), LangOpts))
      AppendFailure(DoubleUnderscoreTag, std::move(*Fixup));
    if (auto Fixup = getUnderscoreCapitalFixup(InProgressFixup()))
      AppendFailure(UnderscoreCapitalTag, std::move(*Fixup));
    if (auto Fixup = getUnderscoreGlobalNamespaceFixup(
            InProgressFixup(), IsInGlobalNamespace, IsMacro))
      AppendFailure(GlobalUnderscoreTag, std::move(*Fixup));

    return Info;
  }
  if (!(hasReservedDoubleUnderscore(Name, LangOpts) ||
        startsWithUnderscoreCapital(Name) ||
        startsWithUnderscoreInGlobalNamespace(Name, IsInGlobalNamespace,
                                              IsMacro)))
    return FailureInfo{NonReservedTag, getNonReservedFixup(std::string(Name))};
  return std::nullopt;
}

std::optional<RenamerClangTidyCheck::FailureInfo>
ReservedIdentifierCheck::getDeclFailureInfo(const NamedDecl *Decl,
                                            const SourceManager &) const {
  assert(Decl && Decl->getIdentifier() && !Decl->getName().empty() &&
         !Decl->isImplicit() &&
         "Decl must be an explicit identifier with a name.");
  return getFailureInfoImpl(
      Decl->getName(), isa<TranslationUnitDecl>(Decl->getDeclContext()),
      /*IsMacro = */ false, getLangOpts(), Invert, AllowedIdentifiers);
}

std::optional<RenamerClangTidyCheck::FailureInfo>
ReservedIdentifierCheck::getMacroFailureInfo(const Token &MacroNameTok,
                                             const SourceManager &) const {
  return getFailureInfoImpl(MacroNameTok.getIdentifierInfo()->getName(), true,
                            /*IsMacro = */ true, getLangOpts(), Invert,
                            AllowedIdentifiers);
}

RenamerClangTidyCheck::DiagInfo
ReservedIdentifierCheck::getDiagInfo(const NamingCheckId &ID,
                                     const NamingCheckFailure &Failure) const {
  return DiagInfo{Message, [&](DiagnosticBuilder &Diag) {
                    Diag << ID.second
                         << getMessageSelectIndex(Failure.Info.KindName);
                  }};
}

} // namespace clang::tidy::bugprone
