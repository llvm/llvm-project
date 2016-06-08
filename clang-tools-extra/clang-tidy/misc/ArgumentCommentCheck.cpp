//===--- ArgumentCommentCheck.cpp - clang-tidy ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ArgumentCommentCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Token.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

ArgumentCommentCheck::ArgumentCommentCheck(StringRef Name,
                                           ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IdentRE("^(/\\* *)([_A-Za-z][_A-Za-z0-9]*)( *= *\\*/)$") {}

void ArgumentCommentCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(unless(cxxOperatorCallExpr()),
               // NewCallback's arguments relate to the pointed function, don't
               // check them against NewCallback's parameter names.
               // FIXME: Make this configurable.
               unless(hasDeclaration(functionDecl(anyOf(
                   hasName("NewCallback"), hasName("NewPermanentCallback"))))))
          .bind("expr"),
      this);
  Finder->addMatcher(cxxConstructExpr().bind("expr"), this);
}

static std::vector<std::pair<SourceLocation, StringRef>>
getCommentsInRange(ASTContext *Ctx, CharSourceRange Range) {
  std::vector<std::pair<SourceLocation, StringRef>> Comments;
  auto &SM = Ctx->getSourceManager();
  std::pair<FileID, unsigned> BeginLoc = SM.getDecomposedLoc(Range.getBegin()),
                              EndLoc = SM.getDecomposedLoc(Range.getEnd());

  if (BeginLoc.first != EndLoc.first)
    return Comments;

  bool Invalid = false;
  StringRef Buffer = SM.getBufferData(BeginLoc.first, &Invalid);
  if (Invalid)
    return Comments;

  const char *StrData = Buffer.data() + BeginLoc.second;

  Lexer TheLexer(SM.getLocForStartOfFile(BeginLoc.first), Ctx->getLangOpts(),
                 Buffer.begin(), StrData, Buffer.end());
  TheLexer.SetCommentRetentionState(true);

  while (true) {
    Token Tok;
    if (TheLexer.LexFromRawLexer(Tok))
      break;
    if (Tok.getLocation() == Range.getEnd() || Tok.getKind() == tok::eof)
      break;

    if (Tok.getKind() == tok::comment) {
      std::pair<FileID, unsigned> CommentLoc =
          SM.getDecomposedLoc(Tok.getLocation());
      assert(CommentLoc.first == BeginLoc.first);
      Comments.emplace_back(
          Tok.getLocation(),
          StringRef(Buffer.begin() + CommentLoc.second, Tok.getLength()));
    }
  }

  return Comments;
}

bool
ArgumentCommentCheck::isLikelyTypo(llvm::ArrayRef<ParmVarDecl *> Params,
                                   StringRef ArgName, unsigned ArgIndex) {
  std::string ArgNameLower = ArgName.lower();
  unsigned UpperBound = (ArgName.size() + 2) / 3 + 1;
  unsigned ThisED = StringRef(ArgNameLower).edit_distance(
      Params[ArgIndex]->getIdentifier()->getName().lower(),
      /*AllowReplacements=*/true, UpperBound);
  if (ThisED >= UpperBound)
    return false;

  for (unsigned I = 0, E = Params.size(); I != E; ++I) {
    if (I == ArgIndex)
      continue;
    IdentifierInfo *II = Params[I]->getIdentifier();
    if (!II)
      continue;

    const unsigned Threshold = 2;
    // Other parameters must be an edit distance at least Threshold more away
    // from this parameter. This gives us greater confidence that this is a typo
    // of this parameter and not one with a similar name.
    unsigned OtherED = StringRef(ArgNameLower).edit_distance(
        II->getName().lower(),
        /*AllowReplacements=*/true, ThisED + Threshold);
    if (OtherED < ThisED + Threshold)
      return false;
  }

  return true;
}

void ArgumentCommentCheck::checkCallArgs(ASTContext *Ctx,
                                         const FunctionDecl *Callee,
                                         SourceLocation ArgBeginLoc,
                                         llvm::ArrayRef<const Expr *> Args) {
  Callee = Callee->getFirstDecl();
  for (unsigned i = 0,
                e = std::min<unsigned>(Args.size(), Callee->getNumParams());
       i != e; ++i) {
    const ParmVarDecl *PVD = Callee->getParamDecl(i);
    IdentifierInfo *II = PVD->getIdentifier();
    if (!II)
      continue;
    if (auto Template = Callee->getTemplateInstantiationPattern()) {
      // Don't warn on arguments for parameters instantiated from template
      // parameter packs. If we find more arguments than the template definition
      // has, it also means that they correspond to a parameter pack.
      if (Template->getNumParams() <= i ||
          Template->getParamDecl(i)->isParameterPack()) {
        continue;
      }
    }

    CharSourceRange BeforeArgument = CharSourceRange::getCharRange(
        i == 0 ? ArgBeginLoc : Args[i - 1]->getLocEnd(),
        Args[i]->getLocStart());
    BeforeArgument = Lexer::makeFileCharRange(
        BeforeArgument, Ctx->getSourceManager(), Ctx->getLangOpts());

    for (auto Comment : getCommentsInRange(Ctx, BeforeArgument)) {
      llvm::SmallVector<StringRef, 2> Matches;
      if (IdentRE.match(Comment.second, &Matches)) {
        if (Matches[2] != II->getName()) {
          {
            DiagnosticBuilder Diag =
                diag(Comment.first, "argument name '%0' in comment does not "
                                    "match parameter name %1")
                << Matches[2] << II;
            if (isLikelyTypo(Callee->parameters(), Matches[2], i)) {
              Diag << FixItHint::CreateReplacement(
                          Comment.first,
                          (Matches[1] + II->getName() + Matches[3]).str());
            }
          }
          diag(PVD->getLocation(), "%0 declared here", DiagnosticIDs::Note)
              << II;
        }
      }
    }
  }
}

void ArgumentCommentCheck::check(const MatchFinder::MatchResult &Result) {
  const Expr *E = Result.Nodes.getNodeAs<Expr>("expr");
  if (auto Call = dyn_cast<CallExpr>(E)) {
    const FunctionDecl *Callee = Call->getDirectCallee();
    if (!Callee)
      return;

    checkCallArgs(Result.Context, Callee, Call->getCallee()->getLocEnd(),
                  llvm::makeArrayRef(Call->getArgs(), Call->getNumArgs()));
  } else {
    auto Construct = cast<CXXConstructExpr>(E);
    checkCallArgs(
        Result.Context, Construct->getConstructor(),
        Construct->getParenOrBraceRange().getBegin(),
        llvm::makeArrayRef(Construct->getArgs(), Construct->getNumArgs()));
  }
}

} // namespace misc
} // namespace tidy
} // namespace clang
