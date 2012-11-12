//===-- loop-convert/LoopActions.h - C++11 For loop migration ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares matchers and callbacks for use in migrating C++ for loops.
//
//===----------------------------------------------------------------------===//
#ifndef _LLVM_TOOLS_CLANG_TOOLS_EXTRA_LOOP_CONVERT_LOOPACTIONS_H_
#define _LLVM_TOOLS_CLANG_TOOLS_EXTRA_LOOP_CONVERT_LOOPACTIONS_H_

#include "StmtAncestor.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace clang {
namespace loop_migrate {

struct Usage;
class Confidence;
// The main computational result of ForLoopIndexUseVisitor.
typedef llvm::SmallVector<Usage, 8> UsageResult;

/// \brief The level of safety to require of transformations.
enum TranslationConfidenceKind {
  TCK_Risky,
  TCK_Reasonable,
  TCK_Safe
};

enum LoopFixerKind {
  LFK_Array,
  LFK_Iterator,
  LFK_PseudoArray
};

/// \brief The callback to be used for loop migration matchers.
///
/// The callback does extra checking not possible in matchers, and attempts to
/// convert the for loop, if possible.
class LoopFixer : public ast_matchers::MatchFinder::MatchCallback {
 public:
  LoopFixer(StmtAncestorASTVisitor *ParentFinder,
            tooling::Replacements *Replace,
            StmtGeneratedVarNameMap *GeneratedDecls,
            ReplacedVarsMap *ReplacedVarRanges,
            unsigned *AcceptedChanges, unsigned *DeferredChanges,
            unsigned *RejectedChanges, bool CountOnly,
            TranslationConfidenceKind RequiredConfidenceLevel,
            LoopFixerKind FixerKind) :
  ParentFinder(ParentFinder), Replace(Replace),
  GeneratedDecls(GeneratedDecls), ReplacedVarRanges(ReplacedVarRanges),
  AcceptedChanges(AcceptedChanges), DeferredChanges(DeferredChanges),
  RejectedChanges(RejectedChanges), CountOnly(CountOnly),
  RequiredConfidenceLevel(RequiredConfidenceLevel), FixerKind(FixerKind)  { }
  virtual void run(const ast_matchers::MatchFinder::MatchResult &Result);

 private:
  StmtAncestorASTVisitor *ParentFinder;
  tooling::Replacements *Replace;
  StmtGeneratedVarNameMap *GeneratedDecls;
  ReplacedVarsMap *ReplacedVarRanges;
  unsigned *AcceptedChanges;
  unsigned *DeferredChanges;
  unsigned *RejectedChanges;
  bool CountOnly;
  TranslationConfidenceKind RequiredConfidenceLevel;
  LoopFixerKind FixerKind;

  /// \brief Computes the changes needed to convert a given for loop, and
  /// applies it if this->CountOnly is false.
  void doConversion(ASTContext *Context,
                    const VarDecl *IndexVar,
                    const VarDecl *MaybeContainer,
                    StringRef ContainerString,
                    const UsageResult &Usages,
                    const DeclStmt *AliasDecl, const ForStmt *TheLoop,
                    bool ContainerNeedsDereference);

  /// \brief Given a loop header that would be convertible, discover all usages
  /// of the index variable and convert the loop if possible.
  void findAndVerifyUsages(ASTContext *Context,
                           const VarDecl *LoopVar,
                           const VarDecl *EndVar,
                           const Expr *ContainerExpr,
                           const Expr *BoundExpr,
                           bool ContainerNeedsDereference,
                           const ForStmt *TheLoop,
                           Confidence ConfidenceLevel);

  /// \brief Determine if the change should be deferred or rejected, returning
  /// text which refers to the container iterated over if the change should
  /// proceed.
  StringRef checkDeferralsAndRejections(ASTContext *Context,
                                        const Expr *ContainerExpr,
                                        Confidence ConfidenceLevel,
                                        const ForStmt *TheLoop);
};

} // namespace loop_migrate
} // namespace clang
#endif  // _LLVM_TOOLS_CLANG_TOOLS_EXTRA_LOOP_CONVERT_LOOPACTIONS_H_
