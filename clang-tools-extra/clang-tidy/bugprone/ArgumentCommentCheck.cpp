//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ArgumentCommentCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Token.h"

#include "../utils/LexerUtils.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

using utils::lexer::CommentToken;
namespace {
AST_MATCHER(Decl, isFromStdNamespaceOrSystemHeader) {
  if (const auto *D = Node.getDeclContext()->getEnclosingNamespaceContext())
    if (D->isStdNamespace())
      return true;
  if (Node.getLocation().isInvalid())
    return false;
  return Node.getASTContext().getSourceManager().isInSystemHeader(
      Node.getLocation());
}
} // namespace

ArgumentCommentCheck::ArgumentCommentCheck(StringRef Name,
                                           ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      StrictMode(Options.get("StrictMode", false)),
      IgnoreSingleArgument(Options.get("IgnoreSingleArgument", false)),
      CommentBoolLiterals(Options.get("CommentBoolLiterals", false)),
      CommentIntegerLiterals(Options.get("CommentIntegerLiterals", false)),
      CommentFloatLiterals(Options.get("CommentFloatLiterals", false)),
      CommentStringLiterals(Options.get("CommentStringLiterals", false)),
      CommentUserDefinedLiterals(
          Options.get("CommentUserDefinedLiterals", false)),
      CommentCharacterLiterals(Options.get("CommentCharacterLiterals", false)),
      CommentNullPtrs(Options.get("CommentNullPtrs", false)),
      IdentRE("^(/\\* *)([_A-Za-z][_A-Za-z0-9]*)( *= *\\*/)$") {}

void ArgumentCommentCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "StrictMode", StrictMode);
  Options.store(Opts, "IgnoreSingleArgument", IgnoreSingleArgument);
  Options.store(Opts, "CommentBoolLiterals", CommentBoolLiterals);
  Options.store(Opts, "CommentIntegerLiterals", CommentIntegerLiterals);
  Options.store(Opts, "CommentFloatLiterals", CommentFloatLiterals);
  Options.store(Opts, "CommentStringLiterals", CommentStringLiterals);
  Options.store(Opts, "CommentUserDefinedLiterals", CommentUserDefinedLiterals);
  Options.store(Opts, "CommentCharacterLiterals", CommentCharacterLiterals);
  Options.store(Opts, "CommentNullPtrs", CommentNullPtrs);
}

void ArgumentCommentCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(unless(cxxOperatorCallExpr()), unless(userDefinedLiteral()),
               // NewCallback's arguments relate to the pointed function,
               // don't check them against NewCallback's parameter names.
               // FIXME: Make this configurable.
               unless(hasDeclaration(functionDecl(
                   hasAnyName("NewCallback", "NewPermanentCallback")))),
               // Ignore APIs from the standard library, since their names are
               // not specified by the standard, and standard library
               // implementations in practice have to use reserved names to
               // avoid conflicts with same-named macros.
               unless(hasDeclaration(isFromStdNamespaceOrSystemHeader())))
          .bind("expr"),
      this);
  Finder->addMatcher(cxxConstructExpr(unless(hasDeclaration(
                                          isFromStdNamespaceOrSystemHeader())))
                         .bind("expr"),
                     this);
}

static std::vector<CommentToken> getCommentsBeforeLoc(ASTContext *Ctx,
                                                      SourceLocation Loc) {
  std::vector<CommentToken> Comments;
  while (Loc.isValid()) {
    const std::optional<Token> Tok = utils::lexer::getPreviousToken(
        Loc, Ctx->getSourceManager(), Ctx->getLangOpts(),
        /*SkipComments=*/false);
    if (!Tok || Tok->isNot(tok::comment))
      break;
    Loc = Tok->getLocation();
    Comments.emplace_back(CommentToken{
        Loc,
        Lexer::getSourceText(CharSourceRange::getCharRange(
                                 Loc, Loc.getLocWithOffset(Tok->getLength())),
                             Ctx->getSourceManager(), Ctx->getLangOpts()),
    });
  }
  return Comments;
}

template <typename NamedDeclRange>
static bool isLikelyTypo(const NamedDeclRange &Candidates, StringRef ArgName,
                         StringRef TargetName) {
  const std::string ArgNameLowerStr = ArgName.lower();
  const StringRef ArgNameLower = ArgNameLowerStr;
  // The threshold is arbitrary.
  const unsigned UpperBound = ((ArgName.size() + 2) / 3) + 1;
  const unsigned ThisED =
      ArgNameLower.edit_distance(TargetName.lower(),
                                 /*AllowReplacements=*/true, UpperBound);
  if (ThisED >= UpperBound)
    return false;

  return llvm::all_of(Candidates, [&](const auto &Candidate) {
    const IdentifierInfo *II = Candidate->getIdentifier();
    if (!II)
      return true;

    // Skip the target itself.
    if (II->getName() == TargetName)
      return true;

    const unsigned Threshold = 2;
    // Other candidates must be an edit distance at least Threshold more away
    // from this candidate. This gives us greater confidence that this is a
    // typo of this candidate and not one with a similar name.
    const unsigned OtherED = ArgNameLower.edit_distance(
        II->getName().lower(),
        /*AllowReplacements=*/true, ThisED + Threshold);
    return OtherED >= ThisED + Threshold;
  });
}

static bool sameName(StringRef InComment, StringRef InDecl, bool StrictMode) {
  if (StrictMode)
    return InComment == InDecl;
  InComment = InComment.trim('_');
  InDecl = InDecl.trim('_');
  // FIXME: compare_insensitive only works for ASCII.
  return InComment.compare_insensitive(InDecl) == 0;
}

static bool looksLikeExpectMethod(const CXXMethodDecl *Expect) {
  return Expect != nullptr && Expect->getLocation().isMacroID() &&
         Expect->getNameInfo().getName().isIdentifier() &&
         Expect->getName().starts_with("gmock_");
}
static bool areMockAndExpectMethods(const CXXMethodDecl *Mock,
                                    const CXXMethodDecl *Expect) {
  assert(looksLikeExpectMethod(Expect));
  return Mock != nullptr && Mock->getNextDeclInContext() == Expect &&
         Mock->getNumParams() == Expect->getNumParams() &&
         Mock->getLocation().isMacroID() &&
         Mock->getNameInfo().getName().isIdentifier() &&
         Mock->getName() == Expect->getName().substr(strlen("gmock_"));
}

// This uses implementation details of MOCK_METHODx_ macros: for each mocked
// method M it defines M() with appropriate signature and a method used to set
// up expectations - gmock_M() - with each argument's type changed the
// corresponding matcher. This function returns M when given either M or
// gmock_M.
static const CXXMethodDecl *findMockedMethod(const CXXMethodDecl *Method) {
  if (looksLikeExpectMethod(Method)) {
    const DeclContext *Ctx = Method->getDeclContext();
    if (Ctx == nullptr || !Ctx->isRecord())
      return nullptr;
    for (const auto *D : Ctx->decls()) {
      if (D->getNextDeclInContext() == Method) {
        const auto *Previous = dyn_cast<CXXMethodDecl>(D);
        return areMockAndExpectMethods(Previous, Method) ? Previous : nullptr;
      }
    }
    return nullptr;
  }
  if (const auto *Next =
          dyn_cast_or_null<CXXMethodDecl>(Method->getNextDeclInContext())) {
    if (looksLikeExpectMethod(Next) && areMockAndExpectMethods(Method, Next))
      return Method;
  }
  return nullptr;
}

// For gmock expectation builder method (the target of the call generated by
// `EXPECT_CALL(obj, Method(...))`) tries to find the real method being mocked
// (returns nullptr, if the mock method doesn't override anything). For other
// functions returns the function itself.
static const FunctionDecl *resolveMocks(const FunctionDecl *Func) {
  if (const auto *Method = dyn_cast<CXXMethodDecl>(Func)) {
    if (const auto *MockedMethod = findMockedMethod(Method)) {
      // If mocked method overrides the real one, we can use its parameter
      // names, otherwise we're out of luck.
      if (MockedMethod->size_overridden_methods() > 0)
        return *MockedMethod->begin_overridden_methods();
      return nullptr;
    }
  }
  return Func;
}

// Given the argument type and the options determine if we should
// be adding an argument comment.
bool ArgumentCommentCheck::shouldAddComment(const Expr *Arg) const {
  Arg = Arg->IgnoreImpCasts();
  if (isa<UnaryOperator>(Arg))
    Arg = cast<UnaryOperator>(Arg)->getSubExpr();
  if (Arg->getExprLoc().isMacroID())
    return false;
  return (CommentBoolLiterals && isa<CXXBoolLiteralExpr>(Arg)) ||
         (CommentIntegerLiterals && isa<IntegerLiteral>(Arg)) ||
         (CommentFloatLiterals && isa<FloatingLiteral>(Arg)) ||
         (CommentUserDefinedLiterals && isa<UserDefinedLiteral>(Arg)) ||
         (CommentCharacterLiterals && isa<CharacterLiteral>(Arg)) ||
         (CommentStringLiterals && isa<StringLiteral>(Arg)) ||
         (CommentNullPtrs && isa<CXXNullPtrLiteralExpr>(Arg));
}

void ArgumentCommentCheck::checkCallArgs(ASTContext *Ctx,
                                         const FunctionDecl *OriginalCallee,
                                         SourceLocation ArgBeginLoc,
                                         llvm::ArrayRef<const Expr *> Args) {
  const FunctionDecl *Callee = resolveMocks(OriginalCallee);
  if (!Callee)
    return;

  Callee = Callee->getFirstDecl();
  if (const auto *Ctor = dyn_cast<CXXConstructorDecl>(Callee);
      Ctor && Ctor->isInheritingConstructor()) {
    if (const auto *BaseCtor = Ctor->getInheritedConstructor().getConstructor())
      Callee = BaseCtor->getFirstDecl();
  }
  const unsigned NumArgs =
      std::min<unsigned>(Args.size(), Callee->getNumParams());
  if ((NumArgs == 0) || (IgnoreSingleArgument && NumArgs == 1))
    return;

  auto MakeFileCharRange = [Ctx](SourceLocation Begin, SourceLocation End) {
    return Lexer::makeFileCharRange(CharSourceRange::getCharRange(Begin, End),
                                    Ctx->getSourceManager(),
                                    Ctx->getLangOpts());
  };

  for (unsigned I = 0; I < NumArgs; ++I) {
    const ParmVarDecl *PVD = Callee->getParamDecl(I);
    const IdentifierInfo *II = PVD->getIdentifier();
    if (!II)
      continue;
    if (FunctionDecl *Template = Callee->getTemplateInstantiationPattern()) {
      // Don't warn on arguments for parameters instantiated from template
      // parameter packs. If we find more arguments than the template
      // definition has, it also means that they correspond to a parameter
      // pack.
      if (Template->getNumParams() <= I ||
          Template->getParamDecl(I)->isParameterPack()) {
        continue;
      }
    }

    const CharSourceRange BeforeArgument =
        MakeFileCharRange(ArgBeginLoc, Args[I]->getBeginLoc());
    ArgBeginLoc = Args[I]->getEndLoc();

    std::vector<CommentToken> Comments;
    if (BeforeArgument.isValid()) {
      Comments = utils::lexer::getTrailingCommentsInRange(
          BeforeArgument, Ctx->getSourceManager(), Ctx->getLangOpts());
    } else {
      // Fall back to parsing back from the start of the argument.
      const CharSourceRange ArgsRange =
          MakeFileCharRange(Args[I]->getBeginLoc(), Args[I]->getEndLoc());
      Comments = getCommentsBeforeLoc(Ctx, ArgsRange.getBegin());
    }

    for (const auto &Comment : Comments) {
      llvm::SmallVector<StringRef, 2> Matches;
      if (IdentRE.match(Comment.Text, &Matches) &&
          !sameName(Matches[2], II->getName(), StrictMode)) {
        {
          const DiagnosticBuilder Diag =
              diag(Comment.Loc, "argument name '%0' in comment does not "
                                "match parameter name %1")
              << Matches[2] << II;
          if (isLikelyTypo(Callee->parameters(), Matches[2], II->getName())) {
            Diag << FixItHint::CreateReplacement(
                Comment.Loc, (Matches[1] + II->getName() + Matches[3]).str());
          }
        }
        diag(PVD->getLocation(), "%0 declared here", DiagnosticIDs::Note) << II;
        if (OriginalCallee != Callee) {
          diag(OriginalCallee->getLocation(),
               "actual callee (%0) is declared here", DiagnosticIDs::Note)
              << OriginalCallee;
        }
      }
    }

    // If the argument comments are missing for literals add them.
    if (Comments.empty() && shouldAddComment(Args[I])) {
      llvm::SmallString<32> ArgComment;
      (llvm::Twine("/*") + II->getName() + "=*/").toStringRef(ArgComment);
      const DiagnosticBuilder Diag =
          diag(Args[I]->getBeginLoc(),
               "argument comment missing for literal argument %0")
          << II
          << FixItHint::CreateInsertion(Args[I]->getBeginLoc(), ArgComment);
    }
  }
}

void ArgumentCommentCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *E = Result.Nodes.getNodeAs<Expr>("expr");
  if (const auto *Call = dyn_cast<CallExpr>(E)) {
    const FunctionDecl *Callee = Call->getDirectCallee();
    if (!Callee)
      return;

    checkCallArgs(Result.Context, Callee, Call->getCallee()->getEndLoc(),
                  llvm::ArrayRef(Call->getArgs(), Call->getNumArgs()));
  } else {
    const auto *Construct = cast<CXXConstructExpr>(E);
    if (Construct->getNumArgs() > 0 &&
        Construct->getArg(0)->getSourceRange() == Construct->getSourceRange()) {
      // Ignore implicit construction.
      return;
    }
    checkCallArgs(
        Result.Context, Construct->getConstructor(),
        Construct->getParenOrBraceRange().getBegin(),
        llvm::ArrayRef(Construct->getArgs(), Construct->getNumArgs()));
  }
}

} // namespace clang::tidy::bugprone
