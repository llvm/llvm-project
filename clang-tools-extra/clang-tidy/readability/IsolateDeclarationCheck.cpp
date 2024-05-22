//===--- IsolateDeclarationCheck.cpp - clang-tidy -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IsolateDeclarationCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <optional>

using namespace clang::ast_matchers;
using namespace clang::tidy::utils::lexer;

namespace clang::tidy::readability {

namespace {
AST_MATCHER(DeclStmt, isSingleDecl) { return Node.isSingleDecl(); }
AST_MATCHER(DeclStmt, onlyDeclaresVariables) {
  return llvm::all_of(Node.decls(), [](Decl *D) { return isa<VarDecl>(D); });
}
} // namespace

void IsolateDeclarationCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(declStmt(onlyDeclaresVariables(), unless(isSingleDecl()),
                              hasParent(compoundStmt()))
                         .bind("decl_stmt"),
                     this);
}

static SourceLocation findStartOfIndirection(SourceLocation Start,
                                             int Indirections,
                                             const SourceManager &SM,
                                             const LangOptions &LangOpts) {
  assert(Indirections >= 0 && "Indirections must be non-negative");
  if (Indirections == 0)
    return Start;

  // Note that the post-fix decrement is necessary to perform the correct
  // number of transformations.
  while (Indirections-- != 0) {
    Start = findPreviousAnyTokenKind(Start, SM, LangOpts, tok::star, tok::amp);
    if (Start.isInvalid() || Start.isMacroID())
      return {};
  }
  return Start;
}

static bool isMacroID(SourceRange R) {
  return R.getBegin().isMacroID() || R.getEnd().isMacroID();
}

/// This function counts the number of written indirections for the given
/// Type \p T. It does \b NOT resolve typedefs as it's a helper for lexing
/// the source code.
/// \see declRanges
static int countIndirections(const Type *T, int Indirections = 0) {
  if (T->isFunctionPointerType()) {
    const auto *Pointee = T->getPointeeType()->castAs<FunctionType>();
    return countIndirections(
        Pointee->getReturnType().IgnoreParens().getTypePtr(), ++Indirections);
  }

  // Note: Do not increment the 'Indirections' because it is not yet clear
  // if there is an indirection added in the source code of the array
  // declaration.
  if (const auto *AT = dyn_cast<ArrayType>(T))
    return countIndirections(AT->getElementType().IgnoreParens().getTypePtr(),
                             Indirections);

  if (isa<PointerType>(T) || isa<ReferenceType>(T))
    return countIndirections(T->getPointeeType().IgnoreParens().getTypePtr(),
                             ++Indirections);

  return Indirections;
}

static bool typeIsMemberPointer(const Type *T) {
  if (isa<ArrayType>(T))
    return typeIsMemberPointer(T->getArrayElementTypeNoTypeQual());

  if ((isa<PointerType>(T) || isa<ReferenceType>(T)) &&
      isa<PointerType>(T->getPointeeType()))
    return typeIsMemberPointer(T->getPointeeType().getTypePtr());

  return isa<MemberPointerType>(T);
}

/// This function tries to extract the SourceRanges that make up all
/// declarations in this \c DeclStmt.
///
/// The resulting vector has the structure {UnderlyingType, Decl1, Decl2, ...}.
/// Each \c SourceRange is of the form [Begin, End).
/// If any of the create ranges is invalid or in a macro the result will be
/// \c None.
/// If the \c DeclStmt contains only one declaration, the result is \c None.
/// If the \c DeclStmt contains declarations other than \c VarDecl the result
/// is \c None.
///
/// \code
///    int * ptr1 = nullptr, value = 42;
/// // [  ][              ] [         ] - The ranges here are inclusive
/// \endcode
/// \todo Generalize this function to take other declarations than \c VarDecl.
static std::optional<std::vector<SourceRange>>
declRanges(const DeclStmt *DS, const SourceManager &SM,
           const LangOptions &LangOpts) {
  std::size_t DeclCount = std::distance(DS->decl_begin(), DS->decl_end());
  if (DeclCount < 2)
    return std::nullopt;

  if (rangeContainsExpansionsOrDirectives(DS->getSourceRange(), SM, LangOpts))
    return std::nullopt;

  // The initial type of the declaration and each declaration has it's own
  // slice. This is necessary, because pointers and references bind only
  // to the local variable and not to all variables in the declaration.
  // Example: 'int *pointer, value = 42;'
  std::vector<SourceRange> Slices;
  Slices.reserve(DeclCount + 1);

  // Calculate the first slice, for now only variables are handled but in the
  // future this should be relaxed and support various kinds of declarations.
  const auto *FirstDecl = dyn_cast<VarDecl>(*DS->decl_begin());

  if (FirstDecl == nullptr)
    return std::nullopt;

  // FIXME: Member pointers are not transformed correctly right now, that's
  // why they are treated as problematic here.
  if (typeIsMemberPointer(FirstDecl->getType().IgnoreParens().getTypePtr()))
    return std::nullopt;

  // Consider the following case: 'int * pointer, value = 42;'
  // Created slices (inclusive)    [  ][       ] [         ]
  // Because 'getBeginLoc' points to the start of the variable *name*, the
  // location of the pointer must be determined separately.
  SourceLocation Start = findStartOfIndirection(
      FirstDecl->getLocation(),
      countIndirections(FirstDecl->getType().IgnoreParens().getTypePtr()), SM,
      LangOpts);

  // Fix function-pointer declarations that have a '(' in front of the
  // pointer.
  // Example: 'void (*f2)(int), (*g2)(int, float) = gg;'
  // Slices:   [   ][        ] [                     ]
  if (FirstDecl->getType()->isFunctionPointerType())
    Start = findPreviousTokenKind(Start, SM, LangOpts, tok::l_paren);

  // It is possible that a declarator is wrapped with parens.
  // Example: 'float (((*f_ptr2)))[42], *f_ptr3, ((f_value2)) = 42.f;'
  // The slice for the type-part must not contain these parens. Consequently
  // 'Start' is moved to the most left paren if there are parens.
  while (true) {
    if (Start.isInvalid() || Start.isMacroID())
      break;

    Token T = getPreviousToken(Start, SM, LangOpts);
    if (T.is(tok::l_paren)) {
      Start = findPreviousTokenStart(Start, SM, LangOpts);
      continue;
    }
    break;
  }

  SourceRange DeclRange(DS->getBeginLoc(), Start);
  if (DeclRange.isInvalid() || isMacroID(DeclRange))
    return std::nullopt;

  // The first slice, that is prepended to every isolated declaration, is
  // created.
  Slices.emplace_back(DeclRange);

  // Create all following slices that each declare a variable.
  SourceLocation DeclBegin = Start;
  for (const auto &Decl : DS->decls()) {
    const auto *CurrentDecl = cast<VarDecl>(Decl);

    // FIXME: Member pointers are not transformed correctly right now, that's
    // why they are treated as problematic here.
    if (typeIsMemberPointer(CurrentDecl->getType().IgnoreParens().getTypePtr()))
      return std::nullopt;

    SourceLocation DeclEnd =
        CurrentDecl->hasInit()
            ? findNextTerminator(CurrentDecl->getInit()->getEndLoc(), SM,
                                 LangOpts)
            : findNextTerminator(CurrentDecl->getEndLoc(), SM, LangOpts);

    SourceRange VarNameRange(DeclBegin, DeclEnd);
    if (VarNameRange.isInvalid() || isMacroID(VarNameRange))
      return std::nullopt;

    Slices.emplace_back(VarNameRange);
    DeclBegin = DeclEnd.getLocWithOffset(1);
  }
  return Slices;
}

static std::optional<std::vector<StringRef>>
collectSourceRanges(llvm::ArrayRef<SourceRange> Ranges, const SourceManager &SM,
                    const LangOptions &LangOpts) {
  std::vector<StringRef> Snippets;
  Snippets.reserve(Ranges.size());

  for (const auto &Range : Ranges) {
    CharSourceRange CharRange = Lexer::getAsCharRange(
        CharSourceRange::getCharRange(Range.getBegin(), Range.getEnd()), SM,
        LangOpts);

    if (CharRange.isInvalid())
      return std::nullopt;

    bool InvalidText = false;
    StringRef Snippet =
        Lexer::getSourceText(CharRange, SM, LangOpts, &InvalidText);

    if (InvalidText)
      return std::nullopt;

    Snippets.emplace_back(Snippet);
  }

  return Snippets;
}

/// Expects a vector {TypeSnippet, Firstdecl, SecondDecl, ...}.
static std::vector<std::string>
createIsolatedDecls(llvm::ArrayRef<StringRef> Snippets) {
  // The first section is the type snippet, which does not make a decl itself.
  assert(Snippets.size() > 2 && "Not enough snippets to create isolated decls");
  std::vector<std::string> Decls(Snippets.size() - 1);

  for (std::size_t I = 1; I < Snippets.size(); ++I)
    Decls[I - 1] = Twine(Snippets[0])
                       .concat(Snippets[0].ends_with(" ") ? "" : " ")
                       .concat(Snippets[I].ltrim())
                       .concat(";")
                       .str();

  return Decls;
}

void IsolateDeclarationCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *WholeDecl = Result.Nodes.getNodeAs<DeclStmt>("decl_stmt");

  auto Diag =
      diag(WholeDecl->getBeginLoc(),
           "multiple declarations in a single statement reduces readability");

  std::optional<std::vector<SourceRange>> PotentialRanges =
      declRanges(WholeDecl, *Result.SourceManager, getLangOpts());
  if (!PotentialRanges)
    return;

  std::optional<std::vector<StringRef>> PotentialSnippets = collectSourceRanges(
      *PotentialRanges, *Result.SourceManager, getLangOpts());

  if (!PotentialSnippets)
    return;

  std::vector<std::string> NewDecls = createIsolatedDecls(*PotentialSnippets);
  std::string Replacement = llvm::join(
      NewDecls,
      (Twine("\n") + Lexer::getIndentationForLine(WholeDecl->getBeginLoc(),
                                                  *Result.SourceManager))
          .str());

  Diag << FixItHint::CreateReplacement(WholeDecl->getSourceRange(),
                                       Replacement);
}
} // namespace clang::tidy::readability
