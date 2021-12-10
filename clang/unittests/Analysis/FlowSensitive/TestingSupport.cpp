#include "TestingSupport.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/Lexer.h"
#include "clang/Serialization/PCHContainerOperations.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Annotations.h"
#include "gtest/gtest.h"
#include <functional>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace clang;
using namespace dataflow;

namespace {
using ast_matchers::MatchFinder;

class FindTranslationUnitCallback : public MatchFinder::MatchCallback {
public:
  explicit FindTranslationUnitCallback(
      std::function<void(ASTContext &)> Operation)
      : Operation{Operation} {}

  void run(const MatchFinder::MatchResult &Result) override {
    const auto *TU = Result.Nodes.getNodeAs<TranslationUnitDecl>("tu");
    if (TU->getASTContext().getDiagnostics().getClient()->getNumErrors() != 0) {
      FAIL() << "Source file has syntax or type errors, they were printed to "
                "the test log";
    }
    Operation(TU->getASTContext());
  }

  std::function<void(ASTContext &)> Operation;
};
} // namespace

static bool
isAnnotationDirectlyAfterStatement(const Stmt *Stmt, unsigned AnnotationBegin,
                                   const SourceManager &SourceManager,
                                   const LangOptions &LangOptions) {
  auto NextToken =
      Lexer::findNextToken(Stmt->getEndLoc(), SourceManager, LangOptions);

  while (NextToken.hasValue() &&
         SourceManager.getFileOffset(NextToken->getLocation()) <
             AnnotationBegin) {
    if (NextToken->isNot(tok::semi))
      return false;

    NextToken = Lexer::findNextToken(NextToken->getEndLoc(), SourceManager,
                                     LangOptions);
  }

  return true;
}

llvm::Expected<llvm::DenseMap<const Stmt *, std::string>>
test::buildStatementToAnnotationMapping(const FunctionDecl *Func,
                                        llvm::Annotations AnnotatedCode) {
  llvm::DenseMap<const Stmt *, std::string> Result;

  using namespace ast_matchers; // NOLINT: Too many names
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

  unsigned I = 0;
  auto Annotations = AnnotatedCode.ranges();
  std::reverse(Annotations.begin(), Annotations.end());
  auto Code = AnnotatedCode.code();

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

      Result[Stmt] = Code.slice(Range.Begin, Range.End).str();
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

std::pair<const FunctionDecl *, std::unique_ptr<CFG>>
test::buildCFG(ASTContext &Context,
               ast_matchers::internal::Matcher<FunctionDecl> FuncMatcher) {
  CFG::BuildOptions Options;
  Options.PruneTriviallyFalseEdges = false;
  Options.AddInitializers = true;
  Options.AddImplicitDtors = true;
  Options.AddTemporaryDtors = true;
  Options.setAllAlwaysAdd();

  const FunctionDecl *F = ast_matchers::selectFirst<FunctionDecl>(
      "target",
      ast_matchers::match(
          ast_matchers::functionDecl(ast_matchers::isDefinition(), FuncMatcher)
              .bind("target"),
          Context));
  if (F == nullptr)
    return std::make_pair(nullptr, nullptr);

  return std::make_pair(
      F, clang::CFG::buildCFG(F, F->getBody(), &Context, Options));
}
