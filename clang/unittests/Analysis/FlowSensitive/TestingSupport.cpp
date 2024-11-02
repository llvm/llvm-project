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
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Annotations.h"
#include <cassert>
#include <functional>
#include <memory>
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

llvm::DenseMap<unsigned, std::string>
test::buildLineToAnnotationMapping(SourceManager &SM,
                                   llvm::Annotations AnnotatedCode) {
  llvm::DenseMap<unsigned, std::string> LineNumberToContent;
  auto Code = AnnotatedCode.code();
  auto Annotations = AnnotatedCode.ranges();
  for (auto &AnnotationRange : Annotations) {
    auto LineNumber =
        SM.getPresumedLineNumber(SM.getLocForStartOfFile(SM.getMainFileID())
                                     .getLocWithOffset(AnnotationRange.Begin));
    auto Content = Code.slice(AnnotationRange.Begin, AnnotationRange.End).str();
    LineNumberToContent[LineNumber] = Content;
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

const ValueDecl *test::findValueDecl(ASTContext &ASTCtx, llvm::StringRef Name) {
  auto TargetNodes = match(valueDecl(hasName(Name)).bind("v"), ASTCtx);
  assert(TargetNodes.size() == 1 && "Name must be unique");
  auto *const Result = selectFirst<ValueDecl>("v", TargetNodes);
  assert(Result != nullptr);
  return Result;
}
