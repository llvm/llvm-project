//===--- AvoidBindCheck.cpp - clang-tidy-----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AvoidBindCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstddef>
#include <string>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

namespace {

enum BindArgumentKind { BK_Temporary, BK_Placeholder, BK_CallExpr, BK_Other };

struct BindArgument {
  StringRef Tokens;
  BindArgumentKind Kind = BK_Other;
  size_t PlaceHolderIndex = 0;
};

} // end namespace

static SmallVector<BindArgument, 4>
buildBindArguments(const MatchFinder::MatchResult &Result, const CallExpr *C) {
  SmallVector<BindArgument, 4> BindArguments;
  llvm::Regex MatchPlaceholder("^_([0-9]+)$");

  // Start at index 1 as first argument to bind is the function name.
  for (size_t I = 1, ArgCount = C->getNumArgs(); I < ArgCount; ++I) {
    const Expr *E = C->getArg(I);
    BindArgument B;
    if (const auto *M = dyn_cast<MaterializeTemporaryExpr>(E)) {
      const auto *TE = M->GetTemporaryExpr();
      B.Kind = isa<CallExpr>(TE) ? BK_CallExpr : BK_Temporary;
    }

    B.Tokens = Lexer::getSourceText(
        CharSourceRange::getTokenRange(E->getBeginLoc(), E->getEndLoc()),
        *Result.SourceManager, Result.Context->getLangOpts());

    SmallVector<StringRef, 2> Matches;
    if (B.Kind == BK_Other && MatchPlaceholder.match(B.Tokens, &Matches)) {
      B.Kind = BK_Placeholder;
      B.PlaceHolderIndex = std::stoi(Matches[1]);
    }
    BindArguments.push_back(B);
  }
  return BindArguments;
}

static void addPlaceholderArgs(const ArrayRef<BindArgument> Args,
                               llvm::raw_ostream &Stream) {
  auto MaxPlaceholderIt =
      std::max_element(Args.begin(), Args.end(),
                       [](const BindArgument &B1, const BindArgument &B2) {
                         return B1.PlaceHolderIndex < B2.PlaceHolderIndex;
                       });

  // Placeholders (if present) have index 1 or greater.
  if (MaxPlaceholderIt == Args.end() || MaxPlaceholderIt->PlaceHolderIndex == 0)
    return;

  size_t PlaceholderCount = MaxPlaceholderIt->PlaceHolderIndex;
  Stream << "(";
  StringRef Delimiter = "";
  for (size_t I = 1; I <= PlaceholderCount; ++I) {
    Stream << Delimiter << "auto && arg" << I;
    Delimiter = ", ";
  }
  Stream << ")";
}

static void addFunctionCallArgs(const ArrayRef<BindArgument> Args,
                                llvm::raw_ostream &Stream) {
  StringRef Delimiter = "";
  for (const auto &B : Args) {
    if (B.PlaceHolderIndex)
      Stream << Delimiter << "arg" << B.PlaceHolderIndex;
    else
      Stream << Delimiter << B.Tokens;
    Delimiter = ", ";
  }
}

static bool isPlaceHolderIndexRepeated(const ArrayRef<BindArgument> Args) {
  llvm::SmallSet<size_t, 4> PlaceHolderIndices;
  for (const BindArgument &B : Args) {
    if (B.PlaceHolderIndex) {
      if (!PlaceHolderIndices.insert(B.PlaceHolderIndex).second)
        return true;
    }
  }
  return false;
}

void AvoidBindCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus14) // Need C++14 for generic lambdas.
    return;

  Finder->addMatcher(
      callExpr(
          callee(namedDecl(hasName("::std::bind"))),
          hasArgument(0, declRefExpr(to(functionDecl().bind("f"))).bind("ref")))
          .bind("bind"),
      this);
}

void AvoidBindCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<CallExpr>("bind");
  auto Diag = diag(MatchedDecl->getBeginLoc(), "prefer a lambda to std::bind");

  const auto Args = buildBindArguments(Result, MatchedDecl);

  // Do not attempt to create fixits for nested call expressions.
  // FIXME: Create lambda capture variables to capture output of calls.
  // NOTE: Supporting nested std::bind will be more difficult due to placeholder
  // sharing between outer and inner std:bind invocations.
  if (llvm::any_of(Args,
                   [](const BindArgument &B) { return B.Kind == BK_CallExpr; }))
    return;

  // Do not attempt to create fixits when placeholders are reused.
  // Unused placeholders are supported by requiring C++14 generic lambdas.
  // FIXME: Support this case by deducing the common type.
  if (isPlaceHolderIndexRepeated(Args))
    return;

  const auto *F = Result.Nodes.getNodeAs<FunctionDecl>("f");

  // std::bind can support argument count mismatch between its arguments and the
  // bound function's arguments. Do not attempt to generate a fixit for such
  // cases.
  // FIXME: Support this case by creating unused lambda capture variables.
  if (F->getNumParams() != Args.size())
    return;

  std::string Buffer;
  llvm::raw_string_ostream Stream(Buffer);

  bool HasCapturedArgument = llvm::any_of(
      Args, [](const BindArgument &B) { return B.Kind == BK_Other; });
  const auto *Ref = Result.Nodes.getNodeAs<DeclRefExpr>("ref");
  Stream << "[" << (HasCapturedArgument ? "=" : "") << "]";
  addPlaceholderArgs(Args, Stream);
  Stream << " { return ";
  Ref->printPretty(Stream, nullptr, Result.Context->getPrintingPolicy());
  Stream << "(";
  addFunctionCallArgs(Args, Stream);
  Stream << "); };";

  Diag << FixItHint::CreateReplacement(MatchedDecl->getSourceRange(),
                                       Stream.str());
}

} // namespace modernize
} // namespace tidy
} // namespace clang
