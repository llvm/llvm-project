#include "TestingSupport.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/FlowSensitive/NoopAnalysis.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Annotations/Annotations.h"
#include "gtest/gtest.h"
#include <cassert>
#include <functional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace clang;
using namespace dataflow;
using namespace ast_matchers;

static bool
isAnnotationDirectlyAfterStatement(const Stmt *Stmt, unsigned AnnotationBegin,
                                   const SourceManager &SourceManager,
                                   const LangOptions &LangOptions) {
  auto NextToken =
      Lexer::findNextToken(Stmt->getEndLoc(), SourceManager, LangOptions);

  while (NextToken && SourceManager.getFileOffset(NextToken->getLocation()) <
                          AnnotationBegin) {
    if (NextToken->isNot(tok::semi))
      return false;

    NextToken = Lexer::findNextToken(NextToken->getEndLoc(), SourceManager,
                                     LangOptions);
  }

  return true;
}

llvm::DenseMap<unsigned, std::string> test::buildLineToAnnotationMapping(
    const SourceManager &SM, const LangOptions &LangOpts,
    SourceRange BoundingRange, llvm::Annotations AnnotatedCode) {
  CharSourceRange CharBoundingRange =
      Lexer::getAsCharRange(BoundingRange, SM, LangOpts);

  llvm::DenseMap<unsigned, std::string> LineNumberToContent;
  auto Code = AnnotatedCode.code();
  auto Annotations = AnnotatedCode.ranges();
  for (auto &AnnotationRange : Annotations) {
    SourceLocation Loc = SM.getLocForStartOfFile(SM.getMainFileID())
                             .getLocWithOffset(AnnotationRange.Begin);
    if (SM.isPointWithin(Loc, CharBoundingRange.getBegin(),
                         CharBoundingRange.getEnd())) {
      LineNumberToContent[SM.getPresumedLineNumber(Loc)] =
          Code.slice(AnnotationRange.Begin, AnnotationRange.End).str();
    }
  }
  return LineNumberToContent;
}

llvm::Expected<llvm::DenseMap<const Stmt *, std::string>>
test::buildStatementToAnnotationMapping(const FunctionDecl *Func,
                                        llvm::Annotations AnnotatedCode) {
  llvm::DenseMap<const Stmt *, std::string> Result;
  llvm::StringSet<> ExistingAnnotations;

  auto StmtMatcher =
      findAll(stmt(unless(anyOf(hasParent(expr()), hasParent(returnStmt()))))
                  .bind("stmt"));

  // This map should stay sorted because the binding algorithm relies on the
  // ordering of statement offsets
  std::map<unsigned, const Stmt *> Stmts;
  auto &Context = Func->getASTContext();
  auto &SourceManager = Context.getSourceManager();

  for (auto &Match : match(StmtMatcher, *Func->getBody(), Context)) {
    const auto *S = Match.getNodeAs<Stmt>("stmt");
    unsigned Offset = SourceManager.getFileOffset(S->getEndLoc());
    Stmts[Offset] = S;
  }

  unsigned FunctionBeginOffset =
      SourceManager.getFileOffset(Func->getBeginLoc());
  unsigned FunctionEndOffset = SourceManager.getFileOffset(Func->getEndLoc());

  std::vector<llvm::Annotations::Range> Annotations = AnnotatedCode.ranges();
  llvm::erase_if(Annotations, [=](llvm::Annotations::Range R) {
    return R.Begin < FunctionBeginOffset || R.End >= FunctionEndOffset;
  });
  std::reverse(Annotations.begin(), Annotations.end());
  auto Code = AnnotatedCode.code();

  unsigned I = 0;
  for (auto OffsetAndStmt = Stmts.rbegin(); OffsetAndStmt != Stmts.rend();
       OffsetAndStmt++) {
    unsigned Offset = OffsetAndStmt->first;
    const Stmt *Stmt = OffsetAndStmt->second;

    if (I < Annotations.size() && Annotations[I].Begin >= Offset) {
      auto Range = Annotations[I];

      if (!isAnnotationDirectlyAfterStatement(Stmt, Range.Begin, SourceManager,
                                              Context.getLangOpts())) {
        return llvm::createStringError(
            std::make_error_code(std::errc::invalid_argument),
            "Annotation is not placed after a statement: %s",
            SourceManager.getLocForStartOfFile(SourceManager.getMainFileID())
                .getLocWithOffset(Offset)
                .printToString(SourceManager)
                .data());
      }

      auto Annotation = Code.slice(Range.Begin, Range.End).str();
      if (!ExistingAnnotations.insert(Annotation).second) {
        return llvm::createStringError(
            std::make_error_code(std::errc::invalid_argument),
            "Repeated use of annotation: %s", Annotation.data());
      }
      Result[Stmt] = std::move(Annotation);

      I++;

      if (I < Annotations.size() && Annotations[I].Begin >= Offset) {
        return llvm::createStringError(
            std::make_error_code(std::errc::invalid_argument),
            "Multiple annotations bound to the statement at the location: %s",
            Stmt->getBeginLoc().printToString(SourceManager).data());
      }
    }
  }

  if (I < Annotations.size()) {
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "Not all annotations were bound to statements. Unbound annotation at: "
        "%s",
        SourceManager.getLocForStartOfFile(SourceManager.getMainFileID())
            .getLocWithOffset(Annotations[I].Begin)
            .printToString(SourceManager)
            .data());
  }

  return Result;
}

llvm::Error test::checkDataflowWithNoopAnalysis(
    llvm::StringRef Code,
    std::function<
        void(const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &,
             ASTContext &)>
        VerifyResults,
    DataflowAnalysisOptions Options, LangStandard::Kind Std,
    llvm::StringRef TargetFun) {
  return checkDataflowWithNoopAnalysis(Code, ast_matchers::hasName(TargetFun),
                                       VerifyResults, Options, Std);
}

llvm::Error test::checkDataflowWithNoopAnalysis(
    llvm::StringRef Code,
    ast_matchers::internal::Matcher<FunctionDecl> TargetFuncMatcher,
    std::function<
        void(const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &,
             ASTContext &)>
        VerifyResults,
    DataflowAnalysisOptions Options, LangStandard::Kind Std,
    std::function<llvm::StringMap<QualType>(QualType)> SyntheticFieldCallback) {
  llvm::SmallVector<std::string, 3> ASTBuildArgs = {
      // -fnodelayed-template-parsing is the default everywhere but on Windows.
      // Set it explicitly so that tests behave the same on Windows as on other
      // platforms.
      "-fsyntax-only", "-fno-delayed-template-parsing",
      "-std=" +
          std::string(LangStandard::getLangStandardForKind(Std).getName())};
  AnalysisInputs<NoopAnalysis> AI(
      Code, TargetFuncMatcher,
      [UseBuiltinModel = Options.BuiltinOpts.has_value(),
       &SyntheticFieldCallback](ASTContext &C, Environment &Env) {
        Env.getDataflowAnalysisContext().setSyntheticFieldCallback(
            std::move(SyntheticFieldCallback));
        return NoopAnalysis(
            C,
            DataflowAnalysisOptions{
                UseBuiltinModel ? Env.getDataflowAnalysisContext().getOptions()
                                : std::optional<BuiltinOptions>()});
      });
  AI.ASTBuildArgs = ASTBuildArgs;
  if (Options.BuiltinOpts)
    AI.BuiltinOptions = *Options.BuiltinOpts;
  return checkDataflow<NoopAnalysis>(
      std::move(AI),
      /*VerifyResults=*/
      [&VerifyResults](
          const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
          const AnalysisOutputs &AO) { VerifyResults(Results, AO.ASTCtx); });
}

const ValueDecl *test::findValueDecl(ASTContext &ASTCtx, llvm::StringRef Name) {
  auto TargetNodes = match(
      valueDecl(unless(indirectFieldDecl()), hasName(Name)).bind("v"), ASTCtx);
  assert(TargetNodes.size() == 1 && "Name must be unique");
  auto *const Result = selectFirst<ValueDecl>("v", TargetNodes);
  assert(Result != nullptr);
  return Result;
}

const IndirectFieldDecl *test::findIndirectFieldDecl(ASTContext &ASTCtx,
                                                     llvm::StringRef Name) {
  auto TargetNodes = match(indirectFieldDecl(hasName(Name)).bind("i"), ASTCtx);
  assert(TargetNodes.size() == 1 && "Name must be unique");
  const auto *Result = selectFirst<IndirectFieldDecl>("i", TargetNodes);
  assert(Result != nullptr);
  return Result;
}

std::vector<const Formula *> test::parseFormulas(Arena &A, StringRef Lines) {
  std::vector<const Formula *> Result;
  while (!Lines.empty()) {
    auto [First, Rest] = Lines.split('\n');
    Lines = Rest;
    if (First.trim().empty())
      continue;
    if (auto F = A.parseFormula(First))
      Result.push_back(&*F);
    else
      ADD_FAILURE() << llvm::toString(F.takeError());
  }
  return Result;
}
