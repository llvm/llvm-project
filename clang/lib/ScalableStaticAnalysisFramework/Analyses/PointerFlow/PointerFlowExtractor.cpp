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
#include "clang/ScalableStaticAnalysisFramework/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/PointerFlow/PointerFlow.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/ExtractorRegistry.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummaryBuilder.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummaryExtractor.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Error.h"
#include <memory>
#include <optional>

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
  std::function<EntityId(const EntityName &)> AddEntity;

  Expected<EntityPointerLevelSet> toEPL(const NamedDecl *N,
                                        bool IsRet = false) const;

  Expected<EntityPointerLevelSet> toEPL(const Expr *N) const;

  llvm::Error addEdges(Expected<EntityPointerLevelSet> &&LHS,
                       Expected<EntityPointerLevelSet> &&RHS);

  template <typename ParmsProvider, typename ArgsProvider>
  llvm::Error matchesArgsWithParams(unsigned ArgIdxStart, ParmsProvider *PP,
                                    ArgsProvider *AP) {
    unsigned ArgIdx = ArgIdxStart;

    for (unsigned ParmIdx = 0;
         ParmIdx < PP->getNumParams() && ArgIdx < AP->getNumArgs();
         ++ArgIdx, ++ParmIdx) {
      if (const ParmVarDecl *PD = PP->getParamDecl(ParmIdx);
          PD && hasPtrOrArrType(PD)) {
        if (auto Err = addEdges(toEPL(PD), toEPL(AP->getArg(ArgIdx))))
          return Err;
      }
    }
    return llvm::Error::success();
  }
};

Expected<EntityPointerLevelSet> PointerFlowMatcher::toEPL(const NamedDecl *N,
                                                          bool IsRet) const {
  auto Ret = createEntityPointerLevel(N, Extractor, IsRet);

  if (Ret)
    return EntityPointerLevelSet{*Ret};
  return Ret.takeError();
}

Expected<EntityPointerLevelSet> PointerFlowMatcher::toEPL(const Expr *N) const {
  return translateEntityPointerLevel(N, Ctx, Extractor);
}

llvm::Error
PointerFlowMatcher::addEdges(Expected<EntityPointerLevelSet> &&LHS,
                             Expected<EntityPointerLevelSet> &&RHS) {
  if (!LHS && !RHS)
    return llvm::joinErrors(LHS.takeError(), RHS.takeError());
  if (!LHS)
    return LHS.takeError();
  if (!RHS)
    return RHS.takeError();
  for (auto L : *LHS)
    Results[L].insert(RHS->begin(), RHS->end());
  return llvm::Error::success();
}

/// Match and extract pointer flow.
/// The extraction function 'XF' can be described by the following rules:
///
/// XF(l = r)               := add edge "toEPL(l) -> toEPL(r))"
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
///                            PointerFlowMatcher::matchInitializerList
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
    return addEdges(toEPL(BO->getLHS()), toEPL(BO->getRHS()));
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
    return addEdges(toEPL(RootDecl, true), toEPL(RetExpr));
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
      return addEdges(toEPL(VD), toEPL(InitExpr));
  }

  // Match C++ constructor member-initializers:
  if (const auto *CtorD = dyn_cast<CXXConstructorDecl>(D)) {
    for (auto *E : CtorD->inits()) {
      if (E->isDelegatingInitializer())
        return matches(DynTypedNode::create(*E->getInit()), RootDecl);
      if (const FieldDecl *FD = E->getMember(); FD && hasPtrOrArrType(FD)) {
        if (auto Err = addEdges(toEPL(E->getMember()), toEPL(E->getInit())))
          return Err;
      }
    }
  }
  return llvm::Error::success();
}

// Helper function for matchInitializerList that handles record:
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

    if (!InitField)
      return llvm::Error::success();
    assert(!ILE->inits().empty());
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

// Helper function for matchInitializerList that handles array:
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
/// track of the pointer level the left-hand side.
llvm::Error
PointerFlowMatcher::matchesInitializerList(const ValueDecl *Base,
                                           const Expr *InitExpr,
                                           unsigned ArrayElementIndirectLevel) {
  const InitListExpr *ILE = dyn_cast<InitListExpr>(InitExpr);

  if (!ILE) {
    if (!hasPtrOrArrType(InitExpr))
      return llvm::Error::success();

    auto BaseEPL = toEPL(Base);

    if (!BaseEPL)
      return BaseEPL.takeError();

    // Apply ArrayElementIndirectLevel to BaseEPL
    auto R = llvm::map_range(*BaseEPL, [&ArrayElementIndirectLevel](
                                           const EntityPointerLevel &EPL) {
      EntityPointerLevel Result = EPL;
      for ([[maybe_unused]] auto Ignored : llvm::seq(ArrayElementIndirectLevel))
        Result = incrementPointerLevel(Result);
      return Result;
    });
    return addEdges(EntityPointerLevelSet{R.begin(), R.end()}, toEPL(InitExpr));
  }
  // Note that `Base`'s type is NOT the real LHS type when
  // ArrayElementIndirectLevel > 0:
  QualType Type = InitExpr->getType();

  if (auto *RD = Type->getAsRecordDecl())
    return matchInitializerListForRecordDecl(*this, RD, ILE);
  if (Type->isArrayType())
    return matchInitializerListForArray(*this, Base, ILE,
                                        ArrayElementIndirectLevel);
  // Must be the case of using a initializer-list for a scalar:
  return matchesInitializerList(Base, ILE->getInit(0));
}

class PointerFlowTUSummaryExtractor : public TUSummaryExtractor {
public:
  PointerFlowTUSummaryExtractor(TUSummaryBuilder &Builder)
      : TUSummaryExtractor(Builder) {}

  Expected<std::unique_ptr<PointerFlowEntitySummary>>
  extractEntitySummary(const NamedDecl *Contributor, ASTContext &Ctx,
                       TUSummaryExtractor &Extractor) {
    PointerFlowMatcher Matcher(Ctx, Extractor);
    auto MatchAction = [&Matcher, &Contributor](const DynTypedNode &Node) {
      auto Err = Matcher.matches(Node, Contributor);

      if (Err)
        llvm::report_fatal_error(std::move(Err));
    };

    findMatchesIn(Contributor, MatchAction);
    return std::make_unique<PointerFlowEntitySummary>(
        buildPointerFlowEntitySummary(std::move(Matcher.Results)));
  }

  void HandleTranslationUnit(ASTContext &Ctx) override {
    std::vector<const NamedDecl *> Contributors;

    findContributors(Ctx, Contributors);
    for (auto *CD : Contributors) {
      auto EntitySummary = extractEntitySummary(CD, Ctx, *this);

      if (!EntitySummary)
        llvm::reportFatalInternalError(EntitySummary.takeError());
      assert(*EntitySummary);
      if ((*EntitySummary)->empty())
        continue;

      std::optional<EntityId> ContributorId = addEntity(CD);
      if (!ContributorId)
        llvm::reportFatalInternalError(makeEntityNameErr(Ctx, CD));

      [[maybe_unused]] auto [_, InsertionSucceeded] =
          SummaryBuilder.addSummary(*ContributorId, std::move(*EntitySummary));

      assert(InsertionSucceeded && "duplicated contributor extraction");
    }
  }
};
} // namespace

// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int PointerFlowTUSummaryExtractorAnchorSource = 0;

static TUSummaryExtractorRegistry::Add<PointerFlowTUSummaryExtractor>
    RegisterExtractor(PointerFlowEntitySummary::Name,
                      "Extract pointer flow information");
