//===--- IntegerTypesCheck.cpp - clang-tidy -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "IntegerTypesCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/Lexer.h"

namespace clang {

using namespace ast_matchers;

static Token getTokenAtLoc(SourceLocation Loc,
                           const MatchFinder::MatchResult &MatchResult,
                           IdentifierTable &IdentTable) {
  Token Tok;
  if (Lexer::getRawToken(Loc, Tok, *MatchResult.SourceManager,
                         MatchResult.Context->getLangOpts(), false))
    return Tok;

  if (Tok.is(tok::raw_identifier)) {
    IdentifierInfo &Info = IdentTable.get(Tok.getRawIdentifier());
    Tok.setIdentifierInfo(&Info);
    Tok.setKind(Info.getTokenID());
  }
  return Tok;
}

namespace tidy {
namespace google {
namespace runtime {

IntegerTypesCheck::IntegerTypesCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      UnsignedTypePrefix(Options.get("UnsignedTypePrefix", "uint")),
      SignedTypePrefix(Options.get("SignedTypePrefix", "int")),
      TypeSuffix(Options.get("TypeSuffix", "")) {}

void IntegerTypesCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "UnsignedTypePrefix", UnsignedTypePrefix);
  Options.store(Opts, "SignedTypePrefix", SignedTypePrefix);
  Options.store(Opts, "TypeSuffix", TypeSuffix);
}

void IntegerTypesCheck::registerMatchers(MatchFinder *Finder) {
  // Find all TypeLocs. The relevant Style Guide rule only applies to C++.
  if (!getLangOpts().CPlusPlus)
    return;
  Finder->addMatcher(typeLoc(loc(isInteger())).bind("tl"), this);
  IdentTable = llvm::make_unique<IdentifierTable>(getLangOpts());
}

void IntegerTypesCheck::check(const MatchFinder::MatchResult &Result) {
  auto TL = *Result.Nodes.getNodeAs<TypeLoc>("tl");
  SourceLocation Loc = TL.getLocStart();

  if (Loc.isInvalid() || Loc.isMacroID())
    return;

  // Look through qualification.
  if (auto QualLoc = TL.getAs<QualifiedTypeLoc>())
    TL = QualLoc.getUnqualifiedLoc();

  auto BuiltinLoc = TL.getAs<BuiltinTypeLoc>();
  if (!BuiltinLoc)
    return;

  Token Tok = getTokenAtLoc(Loc, Result, *IdentTable);
  // Ensure the location actually points to one of the builting integral type
  // names we're interested in. Otherwise, we might be getting this match from
  // implicit code (e.g. an implicit assignment operator of a class containing
  // an array of non-POD types).
  if (!Tok.isOneOf(tok::kw_short, tok::kw_long, tok::kw_unsigned,
                   tok::kw_signed))
    return;

  bool IsSigned;
  unsigned Width;
  const TargetInfo &TargetInfo = Result.Context->getTargetInfo();

  // Look for uses of short, long, long long and their unsigned versions.
  switch (BuiltinLoc.getTypePtr()->getKind()) {
  case BuiltinType::Short:
    Width = TargetInfo.getShortWidth();
    IsSigned = true;
    break;
  case BuiltinType::Long:
    Width = TargetInfo.getLongWidth();
    IsSigned = true;
    break;
  case BuiltinType::LongLong:
    Width = TargetInfo.getLongLongWidth();
    IsSigned = true;
    break;
  case BuiltinType::UShort:
    Width = TargetInfo.getShortWidth();
    IsSigned = false;
    break;
  case BuiltinType::ULong:
    Width = TargetInfo.getLongWidth();
    IsSigned = false;
    break;
  case BuiltinType::ULongLong:
    Width = TargetInfo.getLongLongWidth();
    IsSigned = false;
    break;
  default:
    return;
  }

  // We allow "unsigned short port" as that's reasonably common and required by
  // the sockets API.
  const StringRef Port = "unsigned short port";
  const char *Data = Result.SourceManager->getCharacterData(Loc);
  if (!std::strncmp(Data, Port.data(), Port.size()) &&
      !isIdentifierBody(Data[Port.size()]))
    return;

  std::string Replacement =
      ((IsSigned ? SignedTypePrefix : UnsignedTypePrefix) + Twine(Width) +
       TypeSuffix)
          .str();

  // We don't add a fix-it as changing the type can easily break code,
  // e.g. when a function requires a 'long' argument on all platforms.
  // QualTypes are printed with implicit quotes.
  diag(Loc, "consider replacing %0 with '%1'") << BuiltinLoc.getType()
                                               << Replacement;
}

} // namespace runtime
} // namespace google
} // namespace tidy
} // namespace clang
