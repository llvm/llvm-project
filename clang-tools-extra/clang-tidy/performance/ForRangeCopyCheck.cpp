//===--- ForRangeCopyCheck.cpp - clang-tidy--------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ForRangeCopyCheck.h"
#include "../utils/LexerUtils.h"
#include "../utils/TypeTraits.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace clang {
namespace tidy {
namespace performance {

using namespace ::clang::ast_matchers;
using llvm::SmallPtrSet;

namespace {

template <typename S> bool isSetDifferenceEmpty(const S &S1, const S &S2) {
  for (const auto &E : S1)
    if (S2.count(E) == 0)
      return false;
  return true;
}

// Extracts all Nodes keyed by ID from Matches and inserts them into Nodes.
template <typename Node>
void extractNodesByIdTo(ArrayRef<BoundNodes> Matches, StringRef ID,
                        SmallPtrSet<const Node *, 16> &Nodes) {
  for (const auto &Match : Matches)
    Nodes.insert(Match.getNodeAs<Node>(ID));
}

// Finds all DeclRefExprs to VarDecl in Stmt.
SmallPtrSet<const DeclRefExpr *, 16>
declRefExprs(const VarDecl &VarDecl, const Stmt &Stmt, ASTContext &Context) {
  auto Matches = match(
      findAll(declRefExpr(to(varDecl(equalsNode(&VarDecl)))).bind("declRef")),
      Stmt, Context);
  SmallPtrSet<const DeclRefExpr *, 16> DeclRefs;
  extractNodesByIdTo(Matches, "declRef", DeclRefs);
  return DeclRefs;
}

// Finds all DeclRefExprs where a const method is called on VarDecl or VarDecl
// is the a const reference or value argument to a CallExpr or CXXConstructExpr.
SmallPtrSet<const DeclRefExpr *, 16>
constReferenceDeclRefExprs(const VarDecl &VarDecl, const Stmt &Stmt,
                           ASTContext &Context) {
  auto DeclRefToVar =
      declRefExpr(to(varDecl(equalsNode(&VarDecl)))).bind("declRef");
  auto ConstMethodCallee = callee(cxxMethodDecl(isConst()));
  // Match method call expressions where the variable is referenced as the this
  // implicit object argument and opertor call expression for member operators
  // where the variable is the 0-th argument.
  auto Matches = match(
      findAll(expr(anyOf(cxxMemberCallExpr(ConstMethodCallee, on(DeclRefToVar)),
                         cxxOperatorCallExpr(ConstMethodCallee,
                                             hasArgument(0, DeclRefToVar))))),
      Stmt, Context);
  SmallPtrSet<const DeclRefExpr *, 16> DeclRefs;
  extractNodesByIdTo(Matches, "declRef", DeclRefs);
  auto ConstReferenceOrValue =
      qualType(anyOf(referenceType(pointee(qualType(isConstQualified()))),
                     unless(anyOf(referenceType(), pointerType()))));
  auto UsedAsConstRefOrValueArg = forEachArgumentWithParam(
      DeclRefToVar, parmVarDecl(hasType(ConstReferenceOrValue)));
  Matches = match(findAll(callExpr(UsedAsConstRefOrValueArg)), Stmt, Context);
  extractNodesByIdTo(Matches, "declRef", DeclRefs);
  Matches =
      match(findAll(cxxConstructExpr(UsedAsConstRefOrValueArg)), Stmt, Context);
  extractNodesByIdTo(Matches, "declRef", DeclRefs);
  return DeclRefs;
}

// Modifies VarDecl to be a reference.
FixItHint createAmpersandFix(const VarDecl &VarDecl, ASTContext &Context) {
  SourceLocation AmpLocation = VarDecl.getLocation();
  auto Token = lexer_utils::getPreviousNonCommentToken(Context, AmpLocation);
  if (!Token.is(tok::unknown))
    AmpLocation = Token.getLocation().getLocWithOffset(Token.getLength());
  return FixItHint::CreateInsertion(AmpLocation, "&");
}

// Modifies VarDecl to be const.
FixItHint createConstFix(const VarDecl &VarDecl) {
  return FixItHint::CreateInsertion(VarDecl.getTypeSpecStartLoc(), "const ");
}

} // namespace

ForRangeCopyCheck::ForRangeCopyCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      WarnOnAllAutoCopies(Options.get("WarnOnAllAutoCopies", 0)) {}

void ForRangeCopyCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "WarnOnAllAutoCopies", WarnOnAllAutoCopies);
}

void ForRangeCopyCheck::registerMatchers(MatchFinder *Finder) {
  // Match loop variables that are not references or pointers or are already
  // initialized through MaterializeTemporaryExpr which indicates a type
  // conversion.
  auto LoopVar = varDecl(
      hasType(hasCanonicalType(unless(anyOf(referenceType(), pointerType())))),
      unless(hasInitializer(expr(hasDescendant(materializeTemporaryExpr())))));
  Finder->addMatcher(cxxForRangeStmt(hasLoopVariable(LoopVar.bind("loopVar")))
                         .bind("forRange"),
                     this);
}

void ForRangeCopyCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Var = Result.Nodes.getNodeAs<VarDecl>("loopVar");
  // Ignore code in macros since we can't place the fixes correctly.
  if (Var->getLocStart().isMacroID())
    return;
  if (handleConstValueCopy(*Var, *Result.Context))
    return;
  const auto *ForRange = Result.Nodes.getNodeAs<CXXForRangeStmt>("forRange");
  handleCopyIsOnlyConstReferenced(*Var, *ForRange, *Result.Context);
}

bool ForRangeCopyCheck::handleConstValueCopy(const VarDecl &LoopVar,
                                             ASTContext &Context) {
  if (WarnOnAllAutoCopies) {
    // For aggressive check just test that loop variable has auto type.
    if (!isa<AutoType>(LoopVar.getType()))
      return false;
  } else if (!LoopVar.getType().isConstQualified()) {
    return false;
  }
  if (!type_traits::isExpensiveToCopy(LoopVar.getType(), Context))
    return false;
  auto Diagnostic =
      diag(LoopVar.getLocation(),
           "the loop variable's type is not a reference type; this creates a "
           "copy in each iteration; consider making this a reference")
      << createAmpersandFix(LoopVar, Context);
  if (!LoopVar.getType().isConstQualified())
    Diagnostic << createConstFix(LoopVar);
  return true;
}

bool ForRangeCopyCheck::handleCopyIsOnlyConstReferenced(
    const VarDecl &LoopVar, const CXXForRangeStmt &ForRange,
    ASTContext &Context) {
  if (LoopVar.getType().isConstQualified() ||
      !type_traits::isExpensiveToCopy(LoopVar.getType(), Context)) {
    return false;
  }
  // Collect all DeclRefExprs to the loop variable and all CallExprs and
  // CXXConstructExprs where the loop variable is used as argument to a const
  // reference parameter.
  // If the difference is empty it is safe for the loop variable to be a const
  // reference.
  auto AllDeclRefs = declRefExprs(LoopVar, *ForRange.getBody(), Context);
  auto ConstReferenceDeclRefs =
      constReferenceDeclRefExprs(LoopVar, *ForRange.getBody(), Context);
  if (AllDeclRefs.empty() ||
      !isSetDifferenceEmpty(AllDeclRefs, ConstReferenceDeclRefs))
    return false;
  diag(LoopVar.getLocation(),
       "loop variable is copied but only used as const reference; consider "
       "making it a const reference")
      << createConstFix(LoopVar) << createAmpersandFix(LoopVar, Context);
  return true;
}

} // namespace performance
} // namespace tidy
} // namespace clang
