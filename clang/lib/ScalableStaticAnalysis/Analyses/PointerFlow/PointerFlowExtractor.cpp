//===- PointerFlowExtractor.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SSAFAnalysesCommon.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/TypeBase.h"
#include "clang/ScalableStaticAnalysis/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysis/Analyses/PointerFlow/PointerFlow.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/ExtractorRegistry.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryBuilder.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryExtractor.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Error.h"
#include <memory>

namespace clang::ssaf {
extern PointerFlowEntitySummary buildPointerFlowEntitySummary(EdgeSet Edges);
} // namespace  clang::ssaf

namespace {
using namespace clang;
using namespace ssaf;

class PointerFlowMatcher {
public:
  EdgeSet Results;
  ASTContext &Ctx;
  TUSummaryExtractor &Extractor;

  PointerFlowMatcher(ASTContext &Ctx, TUSummaryExtractor &Extractor)
      : Ctx(Ctx), Extractor(Extractor) {}

  llvm::Error matches(const DynTypedNode &DynNode, const NamedDecl *RootDecl);

  llvm::Error matchesInitializerList(const ValueDecl *Base,
                                     const Expr *InitExpr,
                                     unsigned ArrayElementIndirectLevel = 0);

  llvm::Error matchesStmt(const Stmt *S, const NamedDecl *RootDecl);

  llvm::Error matchesDecl(const Decl *D, const NamedDecl *RootDecl);

private:
  llvm::Error addEdges(Expected<DeclPointerLevels> &&LHS,
                       Expected<DeclPointerLevels> &&RHS);

  Expected<DeclPointerLevels> toDPL(const Expr *N) const {
    return translateDeclPointerLevel(N, Ctx, Extractor);
  }

  static DeclPointerLevel toDPL(const NamedDecl *N, bool IsRet = false) {
    return createDeclPointerLevel(N, IsRet);
  }

  template <typename ParmsProvider, typename ArgsProvider>
  llvm::Error matchesArgsWithParams(unsigned ArgIdxStart, ParmsProvider *PP,
                                    ArgsProvider *AP) {
    unsigned ArgIdx = ArgIdxStart;

    for (unsigned ParmIdx = 0;
         ParmIdx < PP->getNumParams() && ArgIdx < AP->getNumArgs();
         ++ArgIdx, ++ParmIdx) {
      if (const ParmVarDecl *PD = PP->getParamDecl(ParmIdx);
          PD && hasPtrOrArrType(PD)) {
        if (auto Err = addEdges(DeclPointerLevels{toDPL(PD)},
                                toDPL(AP->getArg(ArgIdx))))
          return Err;
      }
    }
    return llvm::Error::success();
  }
};

llvm::Error PointerFlowMatcher::addEdges(Expected<DeclPointerLevels> &&LHS,
                                         Expected<DeclPointerLevels> &&RHS) {
  if (!LHS && !RHS)
    return llvm::joinErrors(LHS.takeError(), RHS.takeError());
  if (!LHS)
    return LHS.takeError();
  if (!RHS)
    return RHS.takeError();
  if (RHS->empty())
    return llvm::Error::success();

  std::vector<DeclPointerLevels> LVecs, RVecs;

  for (const auto &L : *LHS)
    LVecs.push_back(elaborateHigherDeclPointerLevels(L));
  for (const auto &R : *RHS)
    RVecs.push_back(elaborateHigherDeclPointerLevels(R));

  // Imagine an assignment from pointer q to p: 'p = q'.  It encodes that if 'p'
  // has some property, so must 'q'; moreover, if '*p/p[i]' has some property,
  // so must '*q/q[i]' and so on.  Therefore, for each edge '(a, n) -> (b, m)'
  // that represents an explicitly spelled place in the source code, we also add
  // '(a, n + 1) -> (b, m + 1)',
  // '(a, n + 2) -> (b, m + 2)', ... continuing until either 'a' or 'b' reaches
  // its maximum pointer level, whichever happens first.
  //
  // Note that type checking ensures that 'p' and 'q' have
  // identical pointer levels, but '(a, n)' and '(b, m)' may have different
  // upper bounds on their pointer levels, when, for example, 'q' is a
  // cast-expression.

  for (const DeclPointerLevels &L : LVecs)
    for (const DeclPointerLevels &R : RVecs)
      for (const auto &[LDPL, RDPL] : llvm::zip(L, R)) {
        auto LEPL = toEntityPointerLevel(LDPL, Ctx, Extractor);
        if (!LEPL)
          return LEPL.takeError();
        auto REPL = toEntityPointerLevel(RDPL, Ctx, Extractor);
        if (!REPL)
          return REPL.takeError();
        Results[*LEPL].insert(*REPL);
      }
  return llvm::Error::success();
}

/// Match and extract pointer flow.
/// The extraction function 'XF' can be described by the following rules:
///
/// XF(l = r)               := addEdges(toDPL(l), toDPL(r))
/// XF(foo(a, b, ...))      := XF(Param_1 = a), XF(Param_2 = b), ...
/// XF(return e;)           := XF(FunRet = e), where 'FunRet' is the return
///                                            entity of the enclosing
///                                            function
/// XF(ctor(a, ...) : x1(y1), ... {...})
///                         := XF(Param_1 = a), ...,
///                            XF(x1 = y1), ...,
///                            ctor's body will be visited separately.
/// XF(T var = e)           := XF(var = e)
/// XF(T var = init-list)   := see \ref
///                            PointerFlowMatcher::matchesInitializerList
llvm::Error PointerFlowMatcher::matches(const DynTypedNode &DynNode,
                                        const NamedDecl *RootDecl) {
  if (const Stmt *S = DynNode.get<Stmt>())
    return matchesStmt(S, RootDecl);
  if (const Decl *D = DynNode.get<Decl>())
    return matchesDecl(D, RootDecl);
  return llvm::Error::success();
}

llvm::Error PointerFlowMatcher::matchesStmt(const Stmt *S,
                                            const NamedDecl *RootDecl) {
  // Match 'p = q' whenever it has pointer or array type:
  if (const auto *BO = dyn_cast<BinaryOperator>(S);
      BO && BO->getOpcode() == BO_Assign && hasPtrOrArrType(BO)) {
    return addEdges(toDPL(BO->getLHS()), toDPL(BO->getRHS()));
  }

  // Match arg-to-param passing (in CallExpr) for any pointer type argument:
  if (const auto *CE = dyn_cast<CallExpr>(S)) {
    const FunctionDecl *FD = CE->getDirectCallee();

    if (!FD)
      return llvm::Error::success();

    unsigned ArgIdx = 0;

    if (isa<CXXOperatorCallExpr>(CE))
      if (auto *MD = dyn_cast<CXXMethodDecl>(FD);
          MD && !MD->isExplicitObjectMemberFunction())
        ArgIdx = 1;
    return matchesArgsWithParams(ArgIdx, FD, CE);
  }
  // Match arg-to-param passing (in CXXConstructExpr) for any pointer type
  // argument:
  if (const auto *CCE = dyn_cast<CXXConstructExpr>(S)) {
    return matchesArgsWithParams(/*ArgIdxStart=*/0, CCE->getConstructor(), CCE);
  }
  if (const auto *RS = dyn_cast<ReturnStmt>(S)) {
    const Expr *RetExpr = RS->getRetValue();
    if (!RetExpr || !hasPtrOrArrType(RetExpr))
      return llvm::Error::success();
    return addEdges(DeclPointerLevels{toDPL(RootDecl, true)}, toDPL(RetExpr));
  }
  return llvm::Error::success();
}

llvm::Error PointerFlowMatcher::matchesDecl(const Decl *D,
                                            const NamedDecl *RootDecl) {
  const Expr *InitExpr = nullptr;

  if (const auto *VD = dyn_cast<ValueDecl>(D)) {
    if (const auto *Var = dyn_cast<VarDecl>(VD))
      InitExpr = Var->getInit();
    if (const auto *Fd = dyn_cast<FieldDecl>(VD))
      InitExpr = Fd->getInClassInitializer();

    // Match initializer-list:
    if (auto *InitLst = dyn_cast_or_null<InitListExpr>(InitExpr))
      return matchesInitializerList(VD, InitLst);

    // Match initializers to variables/fields of a pointer type:
    if (InitExpr && hasPtrOrArrType(VD))
      return addEdges(DeclPointerLevels{toDPL(VD)}, toDPL(InitExpr));
  }

  // Match C++ constructor member-initializers:
  if (const auto *CtorD = dyn_cast<CXXConstructorDecl>(D)) {
    for (auto *E : CtorD->inits()) {
      if (E->isDelegatingInitializer())
        return matches(DynTypedNode::create(*E->getInit()), RootDecl);
      if (const FieldDecl *FD = E->getMember(); FD && hasPtrOrArrType(FD)) {
        if (auto Err = addEdges(DeclPointerLevels{toDPL(E->getMember())},
                                toDPL(E->getInit())))
          return Err;
      }
    }
  }
  return llvm::Error::success();
}

// Helper function for matchesInitializerList that handles record:
llvm::Error matchInitializerListForRecordDecl(PointerFlowMatcher &Matcher,
                                              const RecordDecl *RecordTy,
                                              const InitListExpr *ILE) {
  if (auto *CXXRD = dyn_cast<CXXRecordDecl>(RecordTy))
    if (CXXRD->getNumBases() != 0) {
      // FIXME: support this:
      return makeErrAtNode(
          Matcher.Ctx, ILE,
          "attempt to create pointer assignment edges between "
          "CXXRecordDecls with base classes and initializer-lists");
    }
  // Handle union:
  if (RecordTy->isUnion()) {
    auto *InitField = ILE->getInitializedFieldInUnion();

    if (!InitField || ILE->inits().empty())
      return llvm::Error::success();
    return Matcher.matchesInitializerList(InitField, ILE->getInit(0));
  }
  // Handle struct/class:
  ILE = ILE->isSemanticForm() ? ILE : ILE->getSemanticForm();

  auto FieldIter = RecordTy->field_begin();

  assert(RecordTy->getNumFields() >= ILE->getNumInits());
  for (auto *Init : ILE->inits())
    if (auto Err = Matcher.matchesInitializerList(*(FieldIter++), Init))
      return Err;
  return llvm::Error::success();
}

// Helper function for matchesInitializerList that handles array:
llvm::Error matchInitializerListForArray(PointerFlowMatcher &Matcher,
                                         const ValueDecl *Array,
                                         const InitListExpr *ILE,
                                         unsigned ArrayIndirectLevel = 0) {
  for (auto *E : ILE->inits())
    if (auto Err =
            Matcher.matchesInitializerList(Array, E, ArrayIndirectLevel + 1))
      return Err;
  return llvm::Error::success();
}

/// Match initializer lists of the form 'Var = {a, b, c, ...}':
///
///   If 'Var' is a struct/union:
///     XF(Var = {a, b, c, ...})  :=   XF(Var.field_1 = a)
///                                    XF(Var.field_2 = b)
///                                    ...
///   If 'Var' is an array:
///     XF(Var = {a, b, c, ...})  :=   XF(*Var = a)
///                                    XF(*Var = b)
///                                    ...
///
/// The process is recursive: 'a', 'b', 'c', ...  may themselves be
/// initializer lists.  We therefore use \p ArrayElementIndirectLevel to keep
/// track of the pointer level of the left-hand side.
llvm::Error
PointerFlowMatcher::matchesInitializerList(const ValueDecl *Base,
                                           const Expr *InitExpr,
                                           unsigned ArrayElementIndirectLevel) {
  const InitListExpr *ILE = dyn_cast<InitListExpr>(InitExpr);

  if (!ILE) {
    if (!hasPtrOrArrType(InitExpr))
      return llvm::Error::success();

    auto BaseDPL = toDPL(Base);
    // Apply ArrayElementIndirectLevel to BaseDPL
    BaseDPL.PointerLevel += ArrayElementIndirectLevel;
    return addEdges(DeclPointerLevels{BaseDPL}, toDPL(InitExpr));
  }
  // Note that `Base`'s type is NOT the real LHS type when
  // ArrayElementIndirectLevel > 0:
  QualType Type = InitExpr->getType();

  if (auto *RD = Type->getAsRecordDecl())
    return matchInitializerListForRecordDecl(*this, RD, ILE);
  if (Type->isArrayType())
    return matchInitializerListForArray(*this, Base, ILE,
                                        ArrayElementIndirectLevel);

  // Must be the case of using a initializer-list for a scalar.
  // The initializer-list can be either singleton or empty:
  if (ILE->getNumInits() == 0)
    return llvm::Error::success();
  return matchesInitializerList(Base, ILE->getInit(0));
}

class PointerFlowTUSummaryExtractor : public TUSummaryExtractor {
public:
  using TUSummaryExtractor::TUSummaryExtractor;

  /// \return a non-null unique pointer to a PointerFlowEntitySummary
  std::unique_ptr<PointerFlowEntitySummary>
  extractEntitySummary(const std::vector<const NamedDecl *> &ContributorDecls,
                       ASTContext &Ctx, TUSummaryExtractor &Extractor) {
    PointerFlowMatcher Matcher(Ctx, Extractor);

    for (const auto *Contrib : ContributorDecls) {
      auto MatchAction = [&Matcher, Contrib](const DynTypedNode &Node) {
        if (auto Err = Matcher.matches(Node, Contrib))
          logWarningFromError(std::move(Err));
      };

      findMatchesIn(Contrib, MatchAction);
    }
    return std::make_unique<PointerFlowEntitySummary>(
        buildPointerFlowEntitySummary(std::move(Matcher.Results)));
  }

  void HandleTranslationUnit(ASTContext &Ctx) override {
    extractAndAddSummaries(
        *this, SummaryBuilder, Ctx,
        [&](const std::vector<const NamedDecl *> &Decls) {
          return extractEntitySummary(Decls, Ctx, *this);
        },
        "PointerFlow");
  }
};
} // namespace

namespace clang::ssaf {
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int PointerFlowExtractorAnchorSource = 0;
} // namespace clang::ssaf

static TUSummaryExtractorRegistry::Add<PointerFlowTUSummaryExtractor>
    RegisterExtractor(PointerFlowEntitySummary::Name,
                      "Extract pointer flow information");
