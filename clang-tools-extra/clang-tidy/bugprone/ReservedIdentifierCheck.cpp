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

// FixItHint

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

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

ReservedIdentifierCheck::ReservedIdentifierCheck(StringRef Name,
                                                 ClangTidyContext *Context)
    : RenamerClangTidyCheck(Name, Context),
      Invert(Options.get("Invert", false)),
      AllowedIdentifiers(utils::options::parseStringList(
          Options.get("AllowedIdentifiers", ""))) {}

void ReservedIdentifierCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  RenamerClangTidyCheck::storeOptions(Opts);
  Options.store(Opts, "Invert", Invert);
  Options.store(Opts, "AllowedIdentifiers",
                utils::options::serializeStringList(AllowedIdentifiers));
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
  return Name.startswith("__");
}

static Optional<std::string>
getDoubleUnderscoreFixup(StringRef Name, const LangOptions &LangOpts) {
  if (hasReservedDoubleUnderscore(Name, LangOpts))
    return collapseConsecutive(Name, '_');
  return None;
}

static bool startsWithUnderscoreCapital(StringRef Name) {
  return Name.size() >= 2 && Name[0] == '_' && std::isupper(Name[1]);
}

static Optional<std::string> getUnderscoreCapitalFixup(StringRef Name) {
  if (startsWithUnderscoreCapital(Name))
    return std::string(Name.drop_front(1));
  return None;
}

static bool startsWithUnderscoreInGlobalNamespace(StringRef Name,
                                                  bool IsInGlobalNamespace) {
  return IsInGlobalNamespace && Name.size() >= 1 && Name[0] == '_';
}

static Optional<std::string>
getUnderscoreGlobalNamespaceFixup(StringRef Name, bool IsInGlobalNamespace) {
  if (startsWithUnderscoreInGlobalNamespace(Name, IsInGlobalNamespace))
    return std::string(Name.drop_front(1));
  return None;
}

static std::string getNonReservedFixup(std::string Name) {
  assert(!Name.empty());
  if (Name[0] == '_' || std::isupper(Name[0]))
    Name.insert(Name.begin(), '_');
  else
    Name.insert(Name.begin(), 2, '_');
  return Name;
}

static Optional<RenamerClangTidyCheck::FailureInfo>
getFailureInfoImpl(StringRef Name, bool IsInGlobalNamespace,
                   const LangOptions &LangOpts, bool Invert,
                   ArrayRef<std::string> AllowedIdentifiers) {
  assert(!Name.empty());
  if (llvm::is_contained(AllowedIdentifiers, Name))
    return None;

  // TODO: Check for names identical to language keywords, and other names
  // specifically reserved by language standards, e.g. C++ 'zombie names' and C
  // future library directions

  using FailureInfo = RenamerClangTidyCheck::FailureInfo;
  if (!Invert) {
    Optional<FailureInfo> Info;
    auto AppendFailure = [&](StringRef Kind, std::string &&Fixup) {
      if (!Info) {
        Info = FailureInfo{std::string(Kind), std::move(Fixup)};
      } else {
        Info->KindName += Kind;
        Info->Fixup = std::move(Fixup);
      }
    };
    auto InProgressFixup = [&] {
      return Info
          .map([](const FailureInfo &Info) { return StringRef(Info.Fixup); })
          .getValueOr(Name);
    };
    if (auto Fixup = getDoubleUnderscoreFixup(InProgressFixup(), LangOpts))
      AppendFailure(DoubleUnderscoreTag, std::move(*Fixup));
    if (auto Fixup = getUnderscoreCapitalFixup(InProgressFixup()))
      AppendFailure(UnderscoreCapitalTag, std::move(*Fixup));
    if (auto Fixup = getUnderscoreGlobalNamespaceFixup(InProgressFixup(),
                                                       IsInGlobalNamespace))
      AppendFailure(GlobalUnderscoreTag, std::move(*Fixup));

    return Info;
  }
  if (!(hasReservedDoubleUnderscore(Name, LangOpts) ||
        startsWithUnderscoreCapital(Name) ||
        startsWithUnderscoreInGlobalNamespace(Name, IsInGlobalNamespace)))
    return FailureInfo{NonReservedTag, getNonReservedFixup(std::string(Name))};
  return None;
}

Optional<RenamerClangTidyCheck::FailureInfo>
ReservedIdentifierCheck::getDeclFailureInfo(const NamedDecl *Decl,
                                            const SourceManager &) const {
  assert(Decl && Decl->getIdentifier() && !Decl->getName().empty() &&
         !Decl->isImplicit() &&
         "Decl must be an explicit identifier with a name.");
  return getFailureInfoImpl(Decl->getName(),
                            isa<TranslationUnitDecl>(Decl->getDeclContext()),
                            getLangOpts(), Invert, AllowedIdentifiers);
}

Optional<RenamerClangTidyCheck::FailureInfo>
ReservedIdentifierCheck::getMacroFailureInfo(const Token &MacroNameTok,
                                             const SourceManager &) const {
  return getFailureInfoImpl(MacroNameTok.getIdentifierInfo()->getName(), true,
                            getLangOpts(), Invert, AllowedIdentifiers);
}

RenamerClangTidyCheck::DiagInfo
ReservedIdentifierCheck::getDiagInfo(const NamingCheckId &ID,
                                     const NamingCheckFailure &Failure) const {
  return DiagInfo{Message, [&](DiagnosticBuilder &Diag) {
                    Diag << ID.second
                         << getMessageSelectIndex(Failure.Info.KindName);
                  }};
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
