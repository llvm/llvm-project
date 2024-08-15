//=== SemaFunctionEffects.cpp - Sema warnings for function effects --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements caller/callee analysis for function effects.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Decl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Sema/SemaInternal.h"

#define DEBUG_TYPE "effectanalysis"

using namespace clang;

namespace {

enum class ViolationID : uint8_t {
  None = 0, // Sentinel for an empty Violation.
  Throws,
  Catches,
  CallsObjC,
  AllocatesMemory,
  HasStaticLocal,
  AccessesThreadLocal,

  // These only apply to callees, where the analysis stops at the Decl.
  DeclDisallowsInference,

  CallsDeclWithoutEffect,
  CallsExprWithoutEffect,
};

// Represents a violation of the rules, potentially for the entire duration of
// the analysis phase, in order to refer to it when explaining why a caller has
// been made unsafe by a callee. Can be transformed into either a Diagnostic
// (warning or a note), depending on whether the violation pertains to a
// function failing to be verifed as holding an effect vs. a function failing to
// be inferred as holding that effect.
struct Violation {
  FunctionEffect Effect;
  FunctionEffect CalleeEffectPreventingInference; // Only for certain IDs.
  ViolationID ID = ViolationID::None;
  SourceLocation Loc;
  const Decl *Callee = nullptr; // Only valid for Calls*.

  Violation() = default;

  Violation(FunctionEffect Effect, ViolationID ID, SourceLocation Loc,
            const Decl *Callee = nullptr,
            const FunctionEffect *CalleeEffect = nullptr)
      : Effect(Effect), ID(ID), Loc(Loc), Callee(Callee) {
    if (CalleeEffect != nullptr)
      CalleeEffectPreventingInference = *CalleeEffect;
  }
};

enum class SpecialFuncType : uint8_t { None, OperatorNew, OperatorDelete };
enum class CallableType : uint8_t {
  // Unknown: probably function pointer
  Unknown,
  Function,
  Virtual,
  Block
};

// Return whether a function's effects CAN be verified.
// The question of whether it SHOULD be verified is independent.
static bool functionIsVerifiable(const FunctionDecl *FD) {
  if (FD->isTrivial()) {
    // Otherwise `struct x { int a; };` would have an unverifiable default
    // constructor.
    return true;
  }
  return FD->hasBody();
}

static bool isNoexcept(const FunctionDecl *FD) {
  const auto *FPT = FD->getType()->castAs<FunctionProtoType>();
  if (FPT->isNothrow() || FD->hasAttr<NoThrowAttr>())
    return true;
  return false;
}

// Transitory, more extended information about a callable, which can be a
// function, block, or function pointer.
struct CallableInfo {
  // CDecl holds the function's definition, if any.
  // FunctionDecl if CallableType::Function or Virtual
  // BlockDecl if CallableType::Block
  const Decl *CDecl;

  // Remember whether the callable is a function, block, virtual method,
  // or (presumed) function pointer.
  CallableType CType = CallableType::Unknown;

  // Remember whether the callable is an operator new or delete function,
  // so that calls to them are reported more meaningfully, as memory
  // allocations.
  SpecialFuncType FuncType = SpecialFuncType::None;

  // We inevitably want to know the callable's declared effects, so cache them.
  FunctionEffectKindSet Effects;

  CallableInfo(const Decl &CD, SpecialFuncType FT = SpecialFuncType::None)
      : CDecl(&CD), FuncType(FT) {
    FunctionEffectsRef DeclEffects;
    if (auto *FD = dyn_cast<FunctionDecl>(CDecl)) {
      // Use the function's definition, if any.
      if (const FunctionDecl *Def = FD->getDefinition())
        CDecl = FD = Def;
      CType = CallableType::Function;
      if (auto *Method = dyn_cast<CXXMethodDecl>(FD);
          Method && Method->isVirtual())
        CType = CallableType::Virtual;
      DeclEffects = FD->getFunctionEffects();
    } else if (auto *BD = dyn_cast<BlockDecl>(CDecl)) {
      CType = CallableType::Block;
      DeclEffects = BD->getFunctionEffects();
    } else if (auto *VD = dyn_cast<ValueDecl>(CDecl)) {
      // ValueDecl is function, enum, or variable, so just look at its type.
      DeclEffects = FunctionEffectsRef::get(VD->getType());
    }
    Effects = FunctionEffectKindSet(DeclEffects);
  }

  CallableType type() const { return CType; }

  bool isCalledDirectly() const {
    return CType == CallableType::Function || CType == CallableType::Block;
  }

  bool isVerifiable() const {
    switch (CType) {
    case CallableType::Unknown:
    case CallableType::Virtual:
      return false;
    case CallableType::Block:
      return true;
    case CallableType::Function:
      return functionIsVerifiable(dyn_cast<FunctionDecl>(CDecl));
    }
    llvm_unreachable("undefined CallableType");
  }

  /// Generate a name for logging and diagnostics.
  std::string name(Sema &S) const {
    std::string Name;
    llvm::raw_string_ostream OS(Name);

    if (auto *FD = dyn_cast<FunctionDecl>(CDecl))
      FD->getNameForDiagnostic(OS, S.getPrintingPolicy(),
                               /*Qualified=*/true);
    else if (auto *BD = dyn_cast<BlockDecl>(CDecl))
      OS << "(block " << BD->getBlockManglingNumber() << ")";
    else if (auto *VD = dyn_cast<NamedDecl>(CDecl))
      VD->printQualifiedName(OS);
    return Name;
  }
};

// ----------
// Map effects to single Violations, to hold the first (of potentially many)
// violations pertaining to an effect, per function.
class EffectToViolationMap {
  // Since we currently only have a tiny number of effects (typically no more
  // than 1), use a SmallVector with an inline capacity of 1. Since it
  // is often empty, use a unique_ptr to the SmallVector.
  // Note that Violation itself contains a FunctionEffect which is the key.
  using ImplVec = llvm::SmallVector<Violation, 1>;
  std::unique_ptr<ImplVec> Impl;

public:
  // Insert a new Violation if we do not already have one for its effect.
  void maybeInsert(const Violation &Viol) {
    if (Impl == nullptr)
      Impl = std::make_unique<ImplVec>();
    else if (lookup(Viol.Effect) != nullptr)
      return;

    Impl->push_back(Viol);
  }

  const Violation *lookup(FunctionEffect Key) {
    if (Impl == nullptr)
      return nullptr;

    auto *Iter =
        std::find_if(Impl->begin(), Impl->end(),
                     [&](const auto &Item) { return Item.Effect == Key; });
    return Iter != Impl->end() ? &*Iter : nullptr;
  }

  size_t size() const { return Impl ? Impl->size() : 0; }
};

// ----------
// State pertaining to a function whose AST is walked and whose effect analysis
// is dependent on a subsequent analysis of other functions.
class PendingFunctionAnalysis {
  friend class CompleteFunctionAnalysis;

public:
  struct DirectCall {
    const Decl *Callee;
    SourceLocation CallLoc;
    // Not all recursive calls are detected, just enough
    // to break cycles.
    bool Recursed = false;

    DirectCall(const Decl *D, SourceLocation CallLoc)
        : Callee(D), CallLoc(CallLoc) {}
  };

  // We always have two disjoint sets of effects to verify:
  // 1. Effects declared explicitly by this function.
  // 2. All other inferrable effects needing verification.
  FunctionEffectKindSet DeclaredVerifiableEffects;
  FunctionEffectKindSet EffectsToInfer;

private:
  // Violations pertaining to the function's explicit effects.
  SmallVector<Violation, 0> ViolationsForExplicitEffects;

  // Violations pertaining to other, non-explicit, inferrable effects.
  EffectToViolationMap InferrableEffectToFirstViolation;

  // These unverified direct calls are what keeps the analysis "pending",
  // until the callees can be verified.
  SmallVector<DirectCall, 0> UnverifiedDirectCalls;

public:
  PendingFunctionAnalysis(Sema &S, const CallableInfo &CInfo,
                          FunctionEffectKindSet AllInferrableEffectsToVerify)
      : DeclaredVerifiableEffects(CInfo.Effects) {
    // Check for effects we are not allowed to infer
    FunctionEffectKindSet InferrableEffects;

    for (FunctionEffect effect : AllInferrableEffectsToVerify) {
      std::optional<FunctionEffect> ProblemCalleeEffect =
          effect.effectProhibitingInference(*CInfo.CDecl, CInfo.Effects);
      if (!ProblemCalleeEffect)
        InferrableEffects.insert(effect);
      else {
        // Add a Violation for this effect if a caller were to
        // try to infer it.
        InferrableEffectToFirstViolation.maybeInsert(Violation(
            effect, ViolationID::DeclDisallowsInference,
            CInfo.CDecl->getLocation(), nullptr, &*ProblemCalleeEffect));
      }
    }
    // InferrableEffects is now the set of inferrable effects which are not
    // prohibited
    EffectsToInfer = FunctionEffectKindSet::difference(
        InferrableEffects, DeclaredVerifiableEffects);
  }

  // Hide the way that Violations for explicitly required effects vs. inferred
  // ones are handled differently.
  void checkAddViolation(bool Inferring, const Violation &NewViol) {
    if (!Inferring)
      ViolationsForExplicitEffects.push_back(NewViol);
    else
      InferrableEffectToFirstViolation.maybeInsert(NewViol);
  }

  void addUnverifiedDirectCall(const Decl *D, SourceLocation CallLoc) {
    UnverifiedDirectCalls.emplace_back(D, CallLoc);
  }

  // Analysis is complete when there are no unverified direct calls.
  bool isComplete() const { return UnverifiedDirectCalls.empty(); }

  const Violation *violationForInferrableEffect(FunctionEffect effect) {
    return InferrableEffectToFirstViolation.lookup(effect);
  }

  // Mutable because caller may need to set a DirectCall's Recursing flag.
  MutableArrayRef<DirectCall> unverifiedCalls() {
    assert(!isComplete());
    return UnverifiedDirectCalls;
  }

  ArrayRef<Violation> getSortedViolationsForExplicitEffects(SourceManager &SM) {
    if (!ViolationsForExplicitEffects.empty())
      std::sort(ViolationsForExplicitEffects.begin(),
                ViolationsForExplicitEffects.end(),
                [&SM](const Violation &LHS, const Violation &RHS) {
                  return SM.isBeforeInTranslationUnit(LHS.Loc, RHS.Loc);
                });
    return ViolationsForExplicitEffects;
  }

  void dump(Sema &SemaRef, llvm::raw_ostream &OS) const {
    OS << "Pending: Declared ";
    DeclaredVerifiableEffects.dump(OS);
    OS << ", " << ViolationsForExplicitEffects.size() << " violations; ";
    OS << " Infer ";
    EffectsToInfer.dump(OS);
    OS << ", " << InferrableEffectToFirstViolation.size() << " violations";
    if (!UnverifiedDirectCalls.empty()) {
      OS << "; Calls: ";
      for (const DirectCall &Call : UnverifiedDirectCalls) {
        CallableInfo CI(*Call.Callee);
        OS << " " << CI.name(SemaRef);
      }
    }
    OS << "\n";
  }
};

// ----------
class CompleteFunctionAnalysis {
  // Current size: 2 pointers
public:
  // Has effects which are both the declared ones -- not to be inferred -- plus
  // ones which have been successfully inferred. These are all considered
  // "verified" for the purposes of callers; any issue with verifying declared
  // effects has already been reported and is not the problem of any caller.
  FunctionEffectKindSet VerifiedEffects;

private:
  // This is used to generate notes about failed inference.
  EffectToViolationMap InferrableEffectToFirstViolation;

public:
  // The incoming Pending analysis is consumed (member(s) are moved-from).
  CompleteFunctionAnalysis(ASTContext &Ctx, PendingFunctionAnalysis &Pending,
                           FunctionEffectKindSet DeclaredEffects,
                           FunctionEffectKindSet AllInferrableEffectsToVerify)
      : VerifiedEffects(DeclaredEffects) {
    for (FunctionEffect effect : AllInferrableEffectsToVerify)
      if (Pending.violationForInferrableEffect(effect) == nullptr)
        VerifiedEffects.insert(effect);

    InferrableEffectToFirstViolation =
        std::move(Pending.InferrableEffectToFirstViolation);
  }

  const Violation *firstViolationForEffect(FunctionEffect Effect) {
    return InferrableEffectToFirstViolation.lookup(Effect);
  }

  void dump(llvm::raw_ostream &OS) const {
    OS << "Complete: Verified ";
    VerifiedEffects.dump(OS);
    OS << "; Infer ";
    OS << InferrableEffectToFirstViolation.size() << " violations\n";
  }
};

// ==========
class Analyzer {
  Sema &S;

  // Subset of Sema.AllEffectsToVerify
  FunctionEffectKindSet AllInferrableEffectsToVerify;

  using FuncAnalysisPtr =
      llvm::PointerUnion<PendingFunctionAnalysis *, CompleteFunctionAnalysis *>;

  // Map all Decls analyzed to FuncAnalysisPtr. Pending state is larger
  // than complete state, so use different objects to represent them.
  // The state pointers are owned by the container.
  class AnalysisMap : llvm::DenseMap<const Decl *, FuncAnalysisPtr> {
    using Base = llvm::DenseMap<const Decl *, FuncAnalysisPtr>;

  public:
    ~AnalysisMap();

    // Use non-public inheritance in order to maintain the invariant
    // that lookups and insertions are via the canonical Decls.

    FuncAnalysisPtr lookup(const Decl *Key) const {
      return Base::lookup(Key->getCanonicalDecl());
    }

    FuncAnalysisPtr &operator[](const Decl *Key) {
      return Base::operator[](Key->getCanonicalDecl());
    }

    /// Shortcut for the case where we only care about completed analysis.
    CompleteFunctionAnalysis *completedAnalysisForDecl(const Decl *D) const {
      if (FuncAnalysisPtr AP = lookup(D);
          isa_and_nonnull<CompleteFunctionAnalysis *>(AP))
        return AP.get<CompleteFunctionAnalysis *>();
      return nullptr;
    }

    void dump(Sema &SemaRef, llvm::raw_ostream &OS) {
      OS << "\nAnalysisMap:\n";
      for (const auto &item : *this) {
        CallableInfo CI(*item.first);
        const auto AP = item.second;
        OS << item.first << " " << CI.name(SemaRef) << " : ";
        if (AP.isNull())
          OS << "null\n";
        else if (isa<CompleteFunctionAnalysis *>(AP)) {
          auto *CFA = AP.get<CompleteFunctionAnalysis *>();
          OS << CFA << " ";
          CFA->dump(OS);
        } else if (isa<PendingFunctionAnalysis *>(AP)) {
          auto *PFA = AP.get<PendingFunctionAnalysis *>();
          OS << PFA << " ";
          PFA->dump(SemaRef, OS);
        } else
          llvm_unreachable("never");
      }
      OS << "---\n";
    }
  };
  AnalysisMap DeclAnalysis;

public:
  Analyzer(Sema &S) : S(S) {}

  void run(const TranslationUnitDecl &TU) {
    // Gather all of the effects to be verified to see what operations need to
    // be checked, and to see which ones are inferrable.
    for (FunctionEffect Effect : S.AllEffectsToVerify) {
      const FunctionEffect::Flags Flags = Effect.flags();
      if (Flags & FunctionEffect::FE_InferrableOnCallees)
        AllInferrableEffectsToVerify.insert(Effect);
    }
    LLVM_DEBUG(llvm::dbgs() << "AllInferrableEffectsToVerify: ";
               AllInferrableEffectsToVerify.dump(llvm::dbgs());
               llvm::dbgs() << "\n";);

    // We can use DeclsWithEffectsToVerify as a stack for a
    // depth-first traversal; there's no need for a second container. But first,
    // reverse it, so when working from the end, Decls are verified in the order
    // they are declared.
    SmallVector<const Decl *> &VerificationQueue = S.DeclsWithEffectsToVerify;
    std::reverse(VerificationQueue.begin(), VerificationQueue.end());

    while (!VerificationQueue.empty()) {
      const Decl *D = VerificationQueue.back();
      if (FuncAnalysisPtr AP = DeclAnalysis.lookup(D)) {
        if (auto *Pending = AP.dyn_cast<PendingFunctionAnalysis *>()) {
          // All children have been traversed; finish analysis.
          finishPendingAnalysis(D, Pending);
        }
        VerificationQueue.pop_back();
        continue;
      }

      // Not previously visited; begin a new analysis for this Decl.
      PendingFunctionAnalysis *Pending = verifyDecl(D);
      if (Pending == nullptr) {
        // Completed now.
        VerificationQueue.pop_back();
        continue;
      }

      // Analysis remains pending because there are direct callees to be
      // verified first. Push them onto the queue.
      for (PendingFunctionAnalysis::DirectCall &Call :
           Pending->unverifiedCalls()) {
        FuncAnalysisPtr AP = DeclAnalysis.lookup(Call.Callee);
        if (AP.isNull()) {
          VerificationQueue.push_back(Call.Callee);
          continue;
        }

        // This indicates recursion (not necessarily direct). For the
        // purposes of effect analysis, we can just ignore it since
        // no effects forbid recursion.
        assert(isa<PendingFunctionAnalysis *>(AP));
        Call.Recursed = true;
      }
    }
  }

private:
  // Verify a single Decl. Return the pending structure if that was the result,
  // else null. This method must not recurse.
  PendingFunctionAnalysis *verifyDecl(const Decl *D) {
    CallableInfo CInfo(*D);
    bool isExternC = false;

    if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
      isExternC = FD->getCanonicalDecl()->isExternCContext();

    // For C++, with non-extern "C" linkage only - if any of the Decl's declared
    // effects forbid throwing (e.g. nonblocking) then the function should also
    // be declared noexcept.
    if (S.getLangOpts().CPlusPlus && !isExternC) {
      for (FunctionEffect Effect : CInfo.Effects) {
        if (!(Effect.flags() & FunctionEffect::FE_ExcludeThrow))
          continue;

        bool IsNoexcept = false;
        if (auto *FD = D->getAsFunction()) {
          IsNoexcept = isNoexcept(FD);
        } else if (auto *BD = dyn_cast<BlockDecl>(D)) {
          if (auto *TSI = BD->getSignatureAsWritten()) {
            auto *FPT = TSI->getType()->getAs<FunctionProtoType>();
            IsNoexcept = FPT->isNothrow() || BD->hasAttr<NoThrowAttr>();
          }
        }
        if (!IsNoexcept)
          S.Diag(D->getBeginLoc(), diag::warn_perf_constraint_implies_noexcept)
              << Effect.name();
        break;
      }
    }

    // Build a PendingFunctionAnalysis on the stack. If it turns out to be
    // complete, we'll have avoided a heap allocation; if it's incomplete, it's
    // a fairly trivial move to a heap-allocated object.
    PendingFunctionAnalysis FAnalysis(S, CInfo, AllInferrableEffectsToVerify);

    LLVM_DEBUG(llvm::dbgs() << "\nVerifying " << CInfo.name(S) << " ";
               FAnalysis.dump(S, llvm::dbgs()););

    FunctionBodyASTVisitor Visitor(*this, FAnalysis, CInfo);

    Visitor.run();
    if (FAnalysis.isComplete()) {
      completeAnalysis(CInfo, FAnalysis);
      return nullptr;
    }
    // Move the pending analysis to the heap and save it in the map.
    PendingFunctionAnalysis *PendingPtr =
        new PendingFunctionAnalysis(std::move(FAnalysis));
    DeclAnalysis[D] = PendingPtr;
    LLVM_DEBUG(llvm::dbgs() << "inserted pending " << PendingPtr << "\n";
               DeclAnalysis.dump(S, llvm::dbgs()););
    return PendingPtr;
  }

  // Consume PendingFunctionAnalysis, create with it a CompleteFunctionAnalysis,
  // inserted in the container.
  void completeAnalysis(const CallableInfo &CInfo,
                        PendingFunctionAnalysis &Pending) {
    if (ArrayRef<Violation> Viols =
            Pending.getSortedViolationsForExplicitEffects(S.getSourceManager());
        !Viols.empty())
      emitDiagnostics(Viols, CInfo, S);

    CompleteFunctionAnalysis *CompletePtr =
        new CompleteFunctionAnalysis(S.getASTContext(), Pending, CInfo.Effects,
                                     AllInferrableEffectsToVerify);
    DeclAnalysis[CInfo.CDecl] = CompletePtr;
    LLVM_DEBUG(llvm::dbgs() << "inserted complete " << CompletePtr << "\n";
               DeclAnalysis.dump(S, llvm::dbgs()););
  }

  // Called after all direct calls requiring inference have been found -- or
  // not. Repeats calls to FunctionBodyASTVisitor::followCall() but without
  // the possibility of inference. Deletes Pending.
  void finishPendingAnalysis(const Decl *D, PendingFunctionAnalysis *Pending) {
    CallableInfo Caller(*D);
    LLVM_DEBUG(llvm::dbgs()
                   << "finishPendingAnalysis for " << Caller.name(S) << " : ";
               Pending->dump(S, llvm::dbgs()); llvm::dbgs() << "\n";);
    for (const PendingFunctionAnalysis::DirectCall &Call :
         Pending->unverifiedCalls()) {
      if (Call.Recursed)
        continue;

      CallableInfo Callee(*Call.Callee);
      followCall(Caller, *Pending, Callee, Call.CallLoc,
                 /*AssertNoFurtherInference=*/true);
    }
    completeAnalysis(Caller, *Pending);
    delete Pending;
  }

  // Here we have a call to a Decl, either explicitly via a CallExpr or some
  // other AST construct. PFA pertains to the caller.
  void followCall(const CallableInfo &Caller, PendingFunctionAnalysis &PFA,
                  const CallableInfo &Callee, SourceLocation CallLoc,
                  bool AssertNoFurtherInference) {
    const bool DirectCall = Callee.isCalledDirectly();

    // Initially, the declared effects; inferred effects will be added.
    FunctionEffectKindSet CalleeEffects = Callee.Effects;

    bool IsInferencePossible = DirectCall;

    if (DirectCall)
      if (CompleteFunctionAnalysis *CFA =
              DeclAnalysis.completedAnalysisForDecl(Callee.CDecl)) {
        // Combine declared effects with those which may have been inferred.
        CalleeEffects.insert(CFA->VerifiedEffects);
        IsInferencePossible = false; // We've already traversed it.
      }

    if (AssertNoFurtherInference) {
      assert(!IsInferencePossible);
    }

    if (!Callee.isVerifiable())
      IsInferencePossible = false;

    LLVM_DEBUG(llvm::dbgs() << "followCall from " << Caller.name(S) << " to "
                            << Callee.name(S) << "; verifiable: "
                            << Callee.isVerifiable() << "; callee ";
               CalleeEffects.dump(llvm::dbgs()); llvm::dbgs() << "\n";
               llvm::dbgs() << "  callee " << Callee.CDecl << " canonical "
                            << Callee.CDecl->getCanonicalDecl() << "\n";);

    auto Check1Effect = [&](FunctionEffect Effect, bool Inferring) {
      if (!Effect.shouldDiagnoseFunctionCall(DirectCall, CalleeEffects))
        return;

      // If inference is not allowed, or the target is indirect (virtual
      // method/function ptr?), generate a Violation now.
      if (!IsInferencePossible ||
          !(Effect.flags() & FunctionEffect::FE_InferrableOnCallees)) {
        if (Callee.FuncType == SpecialFuncType::None)
          PFA.checkAddViolation(Inferring,
                                {Effect, ViolationID::CallsDeclWithoutEffect,
                                 CallLoc, Callee.CDecl});
        else
          PFA.checkAddViolation(
              Inferring, {Effect, ViolationID::AllocatesMemory, CallLoc});
      } else {
        // Inference is allowed and necessary; defer it.
        PFA.addUnverifiedDirectCall(Callee.CDecl, CallLoc);
      }
    };

    for (FunctionEffect Effect : PFA.DeclaredVerifiableEffects)
      Check1Effect(Effect, false);

    for (FunctionEffect Effect : PFA.EffectsToInfer)
      Check1Effect(Effect, true);
  }

  // Should only be called when function's analysis is determined to be
  // complete.
  void emitDiagnostics(ArrayRef<Violation> Viols, const CallableInfo &CInfo,
                       Sema &S) {
    if (Viols.empty())
      return;

    auto MaybeAddTemplateNote = [&](const Decl *D) {
      if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
        while (FD != nullptr && FD->isTemplateInstantiation()) {
          S.Diag(FD->getPointOfInstantiation(),
                 diag::note_func_effect_from_template);
          FD = FD->getTemplateInstantiationPattern();
        }
      }
    };

    // Top-level violations are warnings.
    for (const Violation &Viol1 : Viols) {
      StringRef effectName = Viol1.Effect.name();
      switch (Viol1.ID) {
      case ViolationID::None:
      case ViolationID::DeclDisallowsInference: // Shouldn't happen
                                                // here.
        llvm_unreachable("Unexpected violation kind");
        break;
      case ViolationID::AllocatesMemory:
        S.Diag(Viol1.Loc, diag::warn_func_effect_allocates) << effectName;
        MaybeAddTemplateNote(CInfo.CDecl);
        break;
      case ViolationID::Throws:
      case ViolationID::Catches:
        S.Diag(Viol1.Loc, diag::warn_func_effect_throws_or_catches)
            << effectName;
        MaybeAddTemplateNote(CInfo.CDecl);
        break;
      case ViolationID::HasStaticLocal:
        S.Diag(Viol1.Loc, diag::warn_func_effect_has_static_local)
            << effectName;
        MaybeAddTemplateNote(CInfo.CDecl);
        break;
      case ViolationID::AccessesThreadLocal:
        S.Diag(Viol1.Loc, diag::warn_func_effect_uses_thread_local)
            << effectName;
        MaybeAddTemplateNote(CInfo.CDecl);
        break;
      case ViolationID::CallsObjC:
        S.Diag(Viol1.Loc, diag::warn_func_effect_calls_objc) << effectName;
        MaybeAddTemplateNote(CInfo.CDecl);
        break;
      case ViolationID::CallsExprWithoutEffect:
        S.Diag(Viol1.Loc, diag::warn_func_effect_calls_expr_without_effect)
            << effectName;
        MaybeAddTemplateNote(CInfo.CDecl);
        break;

      case ViolationID::CallsDeclWithoutEffect: {
        CallableInfo CalleeInfo(*Viol1.Callee);
        std::string CalleeName = CalleeInfo.name(S);

        S.Diag(Viol1.Loc, diag::warn_func_effect_calls_func_without_effect)
            << effectName << CalleeName;
        MaybeAddTemplateNote(CInfo.CDecl);

        // Emit notes explaining the transitive chain of inferences: Why isn't
        // the callee safe?
        for (const Decl *Callee = Viol1.Callee; Callee != nullptr;) {
          std::optional<CallableInfo> MaybeNextCallee;
          CompleteFunctionAnalysis *Completed =
              DeclAnalysis.completedAnalysisForDecl(CalleeInfo.CDecl);
          if (Completed == nullptr) {
            // No result - could be
            // - non-inline and extern
            // - indirect (virtual or through function pointer)
            // - effect has been explicitly disclaimed (e.g. "blocking")

            CallableType CType = CalleeInfo.type();
            if (CType == CallableType::Virtual)
              S.Diag(Callee->getLocation(), diag::note_func_effect_call_virtual)
                  << effectName;
            else if (CType == CallableType::Unknown)
              S.Diag(Callee->getLocation(),
                     diag::note_func_effect_call_func_ptr)
                  << effectName;
            else if (CalleeInfo.Effects.contains(Viol1.Effect.oppositeKind()))
              S.Diag(Callee->getLocation(),
                     diag::note_func_effect_call_disallows_inference)
                  << effectName
                  << FunctionEffect(Viol1.Effect.oppositeKind()).name();
            else if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(Callee);
                     FD == nullptr || FD->getBuiltinID() == 0) {
              // A builtin callee generally doesn't have a useful source
              // location at which to insert a note.
              S.Diag(Callee->getLocation(), diag::note_func_effect_call_extern)
                  << effectName;
            }
            break;
          }
          const Violation *PtrViol2 =
              Completed->firstViolationForEffect(Viol1.Effect);
          if (PtrViol2 == nullptr)
            break;

          const Violation &Viol2 = *PtrViol2;
          switch (Viol2.ID) {
          case ViolationID::None:
            llvm_unreachable("Unexpected violation kind");
            break;
          case ViolationID::DeclDisallowsInference:
            S.Diag(Viol2.Loc, diag::note_func_effect_call_disallows_inference)
                << effectName << Viol2.CalleeEffectPreventingInference.name();
            break;
          case ViolationID::CallsExprWithoutEffect:
            S.Diag(Viol2.Loc, diag::note_func_effect_call_func_ptr)
                << effectName;
            break;
          case ViolationID::AllocatesMemory:
            S.Diag(Viol2.Loc, diag::note_func_effect_allocates) << effectName;
            break;
          case ViolationID::Throws:
          case ViolationID::Catches:
            S.Diag(Viol2.Loc, diag::note_func_effect_throws_or_catches)
                << effectName;
            break;
          case ViolationID::HasStaticLocal:
            S.Diag(Viol2.Loc, diag::note_func_effect_has_static_local)
                << effectName;
            break;
          case ViolationID::AccessesThreadLocal:
            S.Diag(Viol2.Loc, diag::note_func_effect_uses_thread_local)
                << effectName;
            break;
          case ViolationID::CallsObjC:
            S.Diag(Viol2.Loc, diag::note_func_effect_calls_objc) << effectName;
            break;
          case ViolationID::CallsDeclWithoutEffect:
            MaybeNextCallee.emplace(*Viol2.Callee);
            S.Diag(Viol2.Loc, diag::note_func_effect_calls_func_without_effect)
                << effectName << MaybeNextCallee->name(S);
            break;
          }
          MaybeAddTemplateNote(Callee);
          Callee = Viol2.Callee;
          if (MaybeNextCallee) {
            CalleeInfo = *MaybeNextCallee;
            CalleeName = CalleeInfo.name(S);
          }
        }
      } break;
      }
    }
  }

  // ----------
  // This AST visitor is used to traverse the body of a function during effect
  // verification. This happens in 2 situations:
  //  [1] The function has declared effects which need to be validated.
  //  [2] The function has not explicitly declared an effect in question, and is
  //      being checked for implicit conformance.
  //
  // Violations are always routed to a PendingFunctionAnalysis.
  struct FunctionBodyASTVisitor : RecursiveASTVisitor<FunctionBodyASTVisitor> {

    Analyzer &Outer;
    PendingFunctionAnalysis &CurrentFunction;
    CallableInfo &CurrentCaller;

    FunctionBodyASTVisitor(Analyzer &Outer,
                           PendingFunctionAnalysis &CurrentFunction,
                           CallableInfo &CurrentCaller)
        : Outer(Outer), CurrentFunction(CurrentFunction),
          CurrentCaller(CurrentCaller) {}

    // -- Entry point --
    void run() {
      // The target function may have implicit code paths beyond the
      // body: member and base destructors. Visit these first.
      if (auto *Dtor = dyn_cast<CXXDestructorDecl>(CurrentCaller.CDecl))
        followDestructor(dyn_cast<CXXRecordDecl>(Dtor->getParent()), Dtor);

      // Do an AST traversal of the function/block body
      TraverseDecl(const_cast<Decl *>(CurrentCaller.CDecl));
    }

    // -- Methods implementing common logic --

    // Handle a language construct forbidden by some effects. Only effects whose
    // flags include the specified flag receive a violation. \p Flag describes
    // the construct.
    void diagnoseLanguageConstruct(FunctionEffect::FlagBit Flag,
                                   ViolationID VID, SourceLocation Loc,
                                   const Decl *Callee = nullptr) {
      // If there are any declared verifiable effects which forbid the construct
      // represented by the flag, store just one violation.
      for (FunctionEffect Effect : CurrentFunction.DeclaredVerifiableEffects) {
        if (Effect.flags() & Flag) {
          addViolation(/*inferring=*/false, Effect, VID, Loc, Callee);
          break;
        }
      }
      // For each inferred effect which forbids the construct, store a
      // violation, if we don't already have a violation for that effect.
      for (FunctionEffect Effect : CurrentFunction.EffectsToInfer)
        if (Effect.flags() & Flag)
          addViolation(/*inferring=*/true, Effect, VID, Loc, Callee);
    }

    void addViolation(bool Inferring, FunctionEffect Effect, ViolationID D,
                      SourceLocation Loc, const Decl *Callee = nullptr) {
      CurrentFunction.checkAddViolation(Inferring,
                                        Violation(Effect, D, Loc, Callee));
    }

    // Here we have a call to a Decl, either explicitly via a CallExpr or some
    // other AST construct. CallableInfo pertains to the callee.
    void followCall(const CallableInfo &CI, SourceLocation CallLoc) {
      if (const auto *FD = dyn_cast<FunctionDecl>(CI.CDecl);
          FD && isSafeBuiltinFunction(FD))
        return;

      Outer.followCall(CurrentCaller, CurrentFunction, CI, CallLoc,
                       /*AssertNoFurtherInference=*/false);
    }

    // FIXME: This is currently specific to the `nonblocking` and
    // `nonallocating` effects. More ideally, the builtin functions themselves
    // would have the `allocating` attribute.
    static bool isSafeBuiltinFunction(const FunctionDecl *FD) {
      unsigned BuiltinID = FD->getBuiltinID();
      switch (BuiltinID) {
      case 0: // Not builtin.
        return false;
      default: // Not disallowed via cases below.
        return true;

      // Disallow list
      case Builtin::ID::BIaligned_alloc:
      case Builtin::ID::BI__builtin_calloc:
      case Builtin::ID::BI__builtin_malloc:
      case Builtin::ID::BI__builtin_realloc:
      case Builtin::ID::BI__builtin_free:
      case Builtin::ID::BI__builtin_operator_delete:
      case Builtin::ID::BI__builtin_operator_new:
      case Builtin::ID::BIcalloc:
      case Builtin::ID::BImalloc:
      case Builtin::ID::BImemalign:
      case Builtin::ID::BIrealloc:
      case Builtin::ID::BIfree:
        return false;
      }
    }

    void checkIndirectCall(CallExpr *Call, QualType CalleeType) {
      auto *FPT =
          CalleeType->getAs<FunctionProtoType>(); // Null if FunctionType.
      FunctionEffectKindSet CalleeEffects;
      if (FPT)
        CalleeEffects.insert(FPT->getFunctionEffects());

      auto Check1Effect = [&](FunctionEffect Effect, bool Inferring) {
        if (FPT == nullptr || Effect.shouldDiagnoseFunctionCall(
                                  /*direct=*/false, CalleeEffects))
          addViolation(Inferring, Effect, ViolationID::CallsExprWithoutEffect,
                       Call->getBeginLoc());
      };

      for (FunctionEffect Effect : CurrentFunction.DeclaredVerifiableEffects)
        Check1Effect(Effect, false);

      for (FunctionEffect Effect : CurrentFunction.EffectsToInfer)
        Check1Effect(Effect, true);
    }

    // This destructor's body should be followed by the caller, but here we
    // follow the field and base destructors.
    void followDestructor(const CXXRecordDecl *Rec,
                          const CXXDestructorDecl *Dtor) {
      for (const FieldDecl *Field : Rec->fields())
        followTypeDtor(Field->getType(), Dtor);

      if (const auto *Class = dyn_cast<CXXRecordDecl>(Rec)) {
        for (const CXXBaseSpecifier &Base : Class->bases())
          followTypeDtor(Base.getType(), Dtor);

        for (const CXXBaseSpecifier &Base : Class->vbases())
          followTypeDtor(Base.getType(), Dtor);
      }
    }

    void followTypeDtor(QualType QT, const CXXDestructorDecl *OuterDtor) {
      const Type *Ty = QT.getTypePtr();
      while (Ty->isArrayType()) {
        const ArrayType *Arr = Ty->getAsArrayTypeUnsafe();
        QT = Arr->getElementType();
        Ty = QT.getTypePtr();
      }

      if (Ty->isRecordType()) {
        if (const CXXRecordDecl *Class = Ty->getAsCXXRecordDecl()) {
          if (CXXDestructorDecl *Dtor = Class->getDestructor()) {
            CallableInfo CI(*Dtor);
            followCall(CI, OuterDtor->getLocation());
          }
        }
      }
    }

    // -- Methods for use of RecursiveASTVisitor --

    bool shouldVisitImplicitCode() const { return true; }

    bool shouldWalkTypesOfTypeLocs() const { return false; }

    bool VisitCXXThrowExpr(CXXThrowExpr *Throw) {
      diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeThrow,
                                ViolationID::Throws, Throw->getThrowLoc());
      return true;
    }

    bool VisitCXXCatchStmt(CXXCatchStmt *Catch) {
      diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeCatch,
                                ViolationID::Catches, Catch->getCatchLoc());
      return true;
    }

    bool VisitObjCAtThrowStmt(ObjCAtThrowStmt *Throw) {
      diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeThrow,
                                ViolationID::Throws, Throw->getThrowLoc());
      return true;
    }

    bool VisitObjCAtCatchStmt(ObjCAtCatchStmt *Catch) {
      diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeCatch,
                                ViolationID::Catches, Catch->getAtCatchLoc());
      return true;
    }

    bool VisitObjCMessageExpr(ObjCMessageExpr *Msg) {
      diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeObjCMessageSend,
                                ViolationID::CallsObjC, Msg->getBeginLoc());
      return true;
    }

    bool VisitSEHExceptStmt(SEHExceptStmt *Exc) {
      diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeCatch,
                                ViolationID::Catches, Exc->getExceptLoc());
      return true;
    }

    bool VisitCallExpr(CallExpr *Call) {
      LLVM_DEBUG(llvm::dbgs()
                     << "VisitCallExpr : "
                     << Call->getBeginLoc().printToString(Outer.S.SourceMgr)
                     << "\n";);

      Expr *CalleeExpr = Call->getCallee();
      if (const Decl *Callee = CalleeExpr->getReferencedDeclOfCallee()) {
        CallableInfo CI(*Callee);
        followCall(CI, Call->getBeginLoc());
        return true;
      }

      if (isa<CXXPseudoDestructorExpr>(CalleeExpr)) {
        // Just destroying a scalar, fine.
        return true;
      }

      // No Decl, just an Expr. Just check based on its type.
      checkIndirectCall(Call, CalleeExpr->getType());

      return true;
    }

    bool VisitVarDecl(VarDecl *Var) {
      LLVM_DEBUG(llvm::dbgs()
                     << "VisitVarDecl : "
                     << Var->getBeginLoc().printToString(Outer.S.SourceMgr)
                     << "\n";);

      if (Var->isStaticLocal())
        diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeStaticLocalVars,
                                  ViolationID::HasStaticLocal,
                                  Var->getLocation());

      const QualType::DestructionKind DK =
          Var->needsDestruction(Outer.S.getASTContext());
      if (DK == QualType::DK_cxx_destructor) {
        QualType QT = Var->getType();
        if (const auto *ClsType = QT.getTypePtr()->getAs<RecordType>()) {
          if (const auto *CxxRec =
                  dyn_cast<CXXRecordDecl>(ClsType->getDecl())) {
            if (const CXXDestructorDecl *Dtor = CxxRec->getDestructor()) {
              CallableInfo CI(*Dtor);
              followCall(CI, Var->getLocation());
            }
          }
        }
      }
      return true;
    }

    bool VisitCXXNewExpr(CXXNewExpr *New) {
      // RecursiveASTVisitor does not visit the implicit call to operator new.
      if (FunctionDecl *FD = New->getOperatorNew()) {
        CallableInfo CI(*FD, SpecialFuncType::OperatorNew);
        followCall(CI, New->getBeginLoc());
      }

      // It's a bit excessive to check operator delete here, since it's
      // just a fallback for operator new followed by a failed constructor.
      // We could check it via New->getOperatorDelete().

      // It DOES however visit the called constructor
      return true;
    }

    bool VisitCXXDeleteExpr(CXXDeleteExpr *Delete) {
      // RecursiveASTVisitor does not visit the implicit call to operator
      // delete.
      if (FunctionDecl *FD = Delete->getOperatorDelete()) {
        CallableInfo CI(*FD, SpecialFuncType::OperatorDelete);
        followCall(CI, Delete->getBeginLoc());
      }

      // It DOES however visit the called destructor

      return true;
    }

    bool VisitCXXConstructExpr(CXXConstructExpr *Construct) {
      LLVM_DEBUG(llvm::dbgs() << "VisitCXXConstructExpr : "
                              << Construct->getBeginLoc().printToString(
                                     Outer.S.SourceMgr)
                              << "\n";);

      // RecursiveASTVisitor does not visit the implicit call to the
      // constructor.
      const CXXConstructorDecl *Ctor = Construct->getConstructor();
      CallableInfo CI(*Ctor);
      followCall(CI, Construct->getLocation());

      return true;
    }

    bool TraverseLambdaExpr(LambdaExpr *Lambda) {
      // We override this so as the be able to skip traversal of the lambda's
      // body. We have to explicitly traverse the captures. Why not return
      // false from shouldVisitLambdaBody()? Because we need to visit a lambda's
      // body when we are verifying the lambda itself; we only want to skip it
      // in the context of the outer function.
      for (unsigned I = 0, N = Lambda->capture_size(); I < N; ++I)
        TraverseLambdaCapture(Lambda, Lambda->capture_begin() + I,
                              Lambda->capture_init_begin()[I]);

      return true;
    }

    bool TraverseBlockExpr(BlockExpr * /*unused*/) {
      // TODO: are the capture expressions (ctor call?) safe?
      return true;
    }

    bool VisitDeclRefExpr(const DeclRefExpr *E) {
      const ValueDecl *Val = E->getDecl();
      if (const auto *Var = dyn_cast<VarDecl>(Val)) {
        if (Var->getTLSKind() != VarDecl::TLS_None) {
          // At least on macOS, thread-local variables are initialized on
          // first access, including a heap allocation.
          diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeThreadLocalVars,
                                    ViolationID::AccessesThreadLocal,
                                    E->getLocation());
        }
      }
      return true;
    }

    bool TraverseGenericSelectionExpr(GenericSelectionExpr *Node) {
      return TraverseStmt(Node->getResultExpr());
    }
    bool TraverseUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *Node) {
      return true;
    }

    bool TraverseTypeOfExprTypeLoc(TypeOfExprTypeLoc Node) { return true; }

    bool TraverseDecltypeTypeLoc(DecltypeTypeLoc Node) { return true; }

    bool TraverseCXXNoexceptExpr(CXXNoexceptExpr *Node) { return true; }

    bool TraverseCXXTypeidExpr(CXXTypeidExpr *Node) { return true; }
  };
};

Analyzer::AnalysisMap::~AnalysisMap() {
  for (const auto &Item : *this) {
    FuncAnalysisPtr AP = Item.second;
    if (isa<PendingFunctionAnalysis *>(AP))
      delete AP.get<PendingFunctionAnalysis *>();
    else
      delete AP.get<CompleteFunctionAnalysis *>();
  }
}

} // anonymous namespace

namespace clang {

void performEffectAnalysis(Sema &S, TranslationUnitDecl *TU) {
  if (S.hasUncompilableErrorOccurred() || S.Diags.getIgnoreAllWarnings())
    return;
  if (TU == nullptr)
    return;
  Analyzer{S}.run(*TU);
}

} // namespace clang
