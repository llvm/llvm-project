//===--- UseScopedLockCheck.cpp - clang-tidy ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseScopedLockCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include <iterator>

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

namespace {

bool isLockGuard(const QualType &Type) {
  if (const auto *Record = Type->getAs<RecordType>()) {
    if (const RecordDecl *Decl = Record->getDecl()) {
      return Decl->getName() == "lock_guard" && Decl->isInStdNamespace();
    }
  }

  if (const auto *TemplateSpecType =
          Type->getAs<TemplateSpecializationType>()) {
    if (const TemplateDecl *Decl =
            TemplateSpecType->getTemplateName().getAsTemplateDecl()) {
      return Decl->getName() == "lock_guard" && Decl->isInStdNamespace();
    }
  }

  return false;
}

llvm::SmallVector<const VarDecl *> getLockGuardsFromDecl(const DeclStmt *DS) {
  llvm::SmallVector<const VarDecl *> LockGuards;

  for (const Decl *Decl : DS->decls()) {
    if (const auto *VD = dyn_cast<VarDecl>(Decl)) {
      const QualType Type = VD->getType().getCanonicalType();
      if (isLockGuard(Type)) {
        LockGuards.push_back(VD);
      }
    }
  }

  return LockGuards;
}

// Scans through the statements in a block and groups consecutive
// 'std::lock_guard' variable declarations together.
llvm::SmallVector<llvm::SmallVector<const VarDecl *>>
findLocksInCompoundStmt(const CompoundStmt *Block,
                        const ast_matchers::MatchFinder::MatchResult &Result) {
  // store groups of consecutive 'std::lock_guard' declarations
  llvm::SmallVector<llvm::SmallVector<const VarDecl *>> LockGuardGroups;
  llvm::SmallVector<const VarDecl *> CurrentLockGuardGroup;

  auto AddAndClearCurrentGroup = [&]() {
    if (!CurrentLockGuardGroup.empty()) {
      LockGuardGroups.push_back(CurrentLockGuardGroup);
      CurrentLockGuardGroup.clear();
    }
  };

  for (const Stmt *Stmt : Block->body()) {
    if (const auto *DS = dyn_cast<DeclStmt>(Stmt)) {
      llvm::SmallVector<const VarDecl *> LockGuards = getLockGuardsFromDecl(DS);

      if (!LockGuards.empty()) {
        CurrentLockGuardGroup.insert(
            CurrentLockGuardGroup.end(),
            std::make_move_iterator(LockGuards.begin()),
            std::make_move_iterator(LockGuards.end()));
        continue;
      }
    }
    AddAndClearCurrentGroup();
  }

  AddAndClearCurrentGroup();

  return LockGuardGroups;
}

TemplateSpecializationTypeLoc
getTemplateLockGuardTypeLoc(const TypeSourceInfo *SourceInfo) {
  const TypeLoc Loc = SourceInfo->getTypeLoc();

  const auto ElaboratedLoc = Loc.getAs<ElaboratedTypeLoc>();
  if (!ElaboratedLoc)
    return {};

  return ElaboratedLoc.getNamedTypeLoc().getAs<TemplateSpecializationTypeLoc>();
}

// Find the exact source range of the 'lock_guard<...>' token
SourceRange getLockGuardTemplateRange(const TypeSourceInfo *SourceInfo) {
  const TemplateSpecializationTypeLoc TemplateLoc =
      getTemplateLockGuardTypeLoc(SourceInfo);
  if (!TemplateLoc)
    return {};

  return SourceRange(TemplateLoc.getTemplateNameLoc(),
                     TemplateLoc.getRAngleLoc());
}

// Find the exact source range of the 'lock_guard' token
SourceRange getLockGuardRange(const TypeSourceInfo *SourceInfo) {
  const TemplateSpecializationTypeLoc TemplateLoc =
      getTemplateLockGuardTypeLoc(SourceInfo);
  if (!TemplateLoc)
    return {};

  return SourceRange(TemplateLoc.getTemplateNameLoc(),
                     TemplateLoc.getLAngleLoc().getLocWithOffset(-1));
}

const StringRef UseScopedLockMessage =
    "use 'std::scoped_lock' instead of 'std::lock_guard'";

} // namespace

UseScopedLockCheck::UseScopedLockCheck(StringRef Name,
                                       ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      WarnOnSingleLocks(Options.get("WarnOnSingleLocks", true)),
      WarnOnUsingAndTypedef(Options.get("WarnOnUsingAndTypedef", true)) {}

void UseScopedLockCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "WarnOnSingleLocks", WarnOnSingleLocks);
  Options.store(Opts, "WarnOnUsingAndTypedef", WarnOnUsingAndTypedef);
}

void UseScopedLockCheck::registerMatchers(MatchFinder *Finder) {
  auto LockGuardClassDecl =
      namedDecl(anyOf(classTemplateDecl(), classTemplateSpecializationDecl()),
                hasName("lock_guard"), isInStdNamespace());
  auto LockGuardType = qualType(hasDeclaration(LockGuardClassDecl));
  auto LockVarDecl = varDecl(hasType(LockGuardType));

  if (WarnOnSingleLocks) {
    // Match CompoundStmt with only one 'std::lock_guard'
    Finder->addMatcher(
        compoundStmt(unless(isExpansionInSystemHeader()),
                     has(declStmt(has(LockVarDecl.bind("lock-decl-single")))),
                     unless(hasDescendant(declStmt(has(varDecl(
                         hasType(LockGuardType),
                         unless(equalsBoundNode("lock-decl-single")))))))),
        this);
  }

  // Match CompoundStmt with multiple 'std::lock_guard'
  Finder->addMatcher(
      compoundStmt(unless(isExpansionInSystemHeader()),
                   has(declStmt(has(LockVarDecl.bind("lock-decl-multiple")))),
                   hasDescendant(declStmt(has(varDecl(
                       hasType(LockGuardType),
                       unless(equalsBoundNode("lock-decl-multiple")))))))
          .bind("block-multiple"),
      this);

  if (WarnOnUsingAndTypedef) {
    // Match 'typedef std::lock_guard<std::mutex> Lock'
    Finder->addMatcher(typedefDecl(unless(isExpansionInSystemHeader()),
                                   hasUnderlyingType(LockGuardType))
                           .bind("lock-guard-typedef"),
                       this);

    // Match 'using Lock = std::lock_guard<std::mutex>'
    Finder->addMatcher(
        typeAliasDecl(
            unless(isExpansionInSystemHeader()),
            hasType(elaboratedType(namesType(templateSpecializationType(
                hasDeclaration(LockGuardClassDecl))))))
            .bind("lock-guard-using-alias"),
        this);

    // Match 'using std::lock_guard'
    Finder->addMatcher(
        usingDecl(unless(isExpansionInSystemHeader()),
                  hasAnyUsingShadowDecl(hasTargetDecl(LockGuardClassDecl)))
            .bind("lock-guard-using-decl"),
        this);
  }
}

void UseScopedLockCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Decl = Result.Nodes.getNodeAs<VarDecl>("lock-decl-single")) {
    emitDiag(Decl, Result);
    return;
  }

  if (const auto *Compound =
          Result.Nodes.getNodeAs<CompoundStmt>("block-multiple")) {
    emitDiag(findLocksInCompoundStmt(Compound, Result), Result);
    return;
  }

  if (const auto *Typedef =
          Result.Nodes.getNodeAs<TypedefDecl>("lock-guard-typedef")) {
    emitDiag(Typedef->getTypeSourceInfo(), Result);
    return;
  }

  if (const auto *UsingAlias =
          Result.Nodes.getNodeAs<TypeAliasDecl>("lock-guard-using-alias")) {
    emitDiag(UsingAlias->getTypeSourceInfo(), Result);
    return;
  }

  if (const auto *Using =
          Result.Nodes.getNodeAs<UsingDecl>("lock-guard-using-decl")) {
    emitDiag(Using, Result);
  }
}

void UseScopedLockCheck::emitDiag(const VarDecl *LockGuard,
                                  const MatchFinder::MatchResult &Result) {
  auto Diag = diag(LockGuard->getBeginLoc(), UseScopedLockMessage);

  const SourceRange LockGuardTypeRange =
      getLockGuardTemplateRange(LockGuard->getTypeSourceInfo());

  if (LockGuardTypeRange.isInvalid()) {
    return;
  }

  // Create Fix-its only if we can find the constructor call to properly handle
  // 'std::lock_guard l(m, std::adopt_lock)' case.
  const auto *CtorCall = dyn_cast<CXXConstructExpr>(LockGuard->getInit());
  if (!CtorCall) {
    return;
  }

  if (CtorCall->getNumArgs() == 1) {
    Diag << FixItHint::CreateReplacement(LockGuardTypeRange, "scoped_lock");
    return;
  }

  if (CtorCall->getNumArgs() == 2) {
    const Expr *const *CtorArgs = CtorCall->getArgs();

    const Expr *MutexArg = CtorArgs[0];
    const Expr *AdoptLockArg = CtorArgs[1];

    const StringRef MutexSourceText = Lexer::getSourceText(
        CharSourceRange::getTokenRange(MutexArg->getSourceRange()),
        *Result.SourceManager, Result.Context->getLangOpts());
    const StringRef AdoptLockSourceText = Lexer::getSourceText(
        CharSourceRange::getTokenRange(AdoptLockArg->getSourceRange()),
        *Result.SourceManager, Result.Context->getLangOpts());

    Diag << FixItHint::CreateReplacement(LockGuardTypeRange, "scoped_lock")
         << FixItHint::CreateReplacement(
                SourceRange(MutexArg->getBeginLoc(), AdoptLockArg->getEndLoc()),
                (llvm::Twine(AdoptLockSourceText) + ", " + MutexSourceText)
                    .str());
    return;
  }

  llvm_unreachable("Invalid argument number of std::lock_guard constructor");
}

void UseScopedLockCheck::emitDiag(
    const llvm::SmallVector<llvm::SmallVector<const VarDecl *>> &LockGroups,
    const ast_matchers::MatchFinder::MatchResult &Result) {
  for (const llvm::SmallVector<const VarDecl *> &Group : LockGroups) {
    if (Group.size() == 1 && WarnOnSingleLocks) {
      emitDiag(Group[0], Result);
    } else {
      diag(Group[0]->getBeginLoc(),
           "use single 'std::scoped_lock' instead of multiple "
           "'std::lock_guard'");

      for (const VarDecl *Lock : llvm::drop_begin(Group)) {
        diag(Lock->getLocation(), "additional 'std::lock_guard' declared here",
             DiagnosticIDs::Note);
      }
    }
  }
}

void UseScopedLockCheck::emitDiag(
    const TypeSourceInfo *LockGuardSourceInfo,
    const ast_matchers::MatchFinder::MatchResult &Result) {
  const TypeLoc TL = LockGuardSourceInfo->getTypeLoc();

  if (const auto ElaboratedTL = TL.getAs<ElaboratedTypeLoc>()) {
    auto Diag = diag(ElaboratedTL.getBeginLoc(), UseScopedLockMessage);

    const SourceRange LockGuardRange = getLockGuardRange(LockGuardSourceInfo);
    if (LockGuardRange.isInvalid()) {
      return;
    }

    Diag << FixItHint::CreateReplacement(LockGuardRange, "scoped_lock");
  }
}

void UseScopedLockCheck::emitDiag(
    const UsingDecl *UsingDecl,
    const ast_matchers::MatchFinder::MatchResult &Result) {
  diag(UsingDecl->getLocation(), UseScopedLockMessage)
      << FixItHint::CreateReplacement(UsingDecl->getLocation(), "scoped_lock");
}

} // namespace clang::tidy::modernize
