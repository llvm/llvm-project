//=== EffectAnalysis.cpp - Sema warnings for function effects -------------===//
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
#include "clang/AST/Type.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Sema/SemaInternal.h"

#define DEBUG_TYPE "fxanalysis"

using namespace clang;

namespace {

enum class DiagnosticID : uint8_t {
  None = 0, // sentinel for an empty Diagnostic
  Throws,
  Catches,
  CallsObjC,
  AllocatesMemory,
  HasStaticLocal,
  AccessesThreadLocal,

  // These only apply to callees, where the analysis stops at the Decl
  DeclDisallowsInference,

  CallsDeclWithoutEffect,
  CallsExprWithoutEffect,
};

// Holds an effect diagnosis, potentially for the entire duration of the
// analysis phase, in order to refer to it when explaining why a caller has been
// made unsafe by a callee.
struct Diagnostic {
  FunctionEffect Effect;
  DiagnosticID ID = DiagnosticID::None;
  SourceLocation Loc;
  const Decl *Callee = nullptr; // only valid for Calls*

  Diagnostic() = default;

  Diagnostic(const FunctionEffect &Effect, DiagnosticID ID, SourceLocation Loc,
             const Decl *Callee = nullptr)
      : Effect(Effect), ID(ID), Loc(Loc), Callee(Callee) {}
};

enum class SpecialFuncType : uint8_t { None, OperatorNew, OperatorDelete };
enum class CallType {
  // unknown: probably function pointer
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

/// A mutable set of FunctionEffect, for use in places where any conditions
/// have been resolved or can be ignored.
class EffectSet {
  // This implementation optimizes footprint, since we hold one of these for
  // every function visited, which, due to inference, can be many more functions
  // than have declared effects.

  template <typename T, typename SizeT, SizeT Capacity> struct FixedVector {
    SizeT Count = 0;
    T Items[Capacity] = {};

    using value_type = T;

    using iterator = T *;
    using const_iterator = const T *;
    iterator begin() { return &Items[0]; }
    iterator end() { return &Items[Count]; }
    const_iterator begin() const { return &Items[0]; }
    const_iterator end() const { return &Items[Count]; }
    const_iterator cbegin() const { return &Items[0]; }
    const_iterator cend() const { return &Items[Count]; }

    void insert(iterator I, const T &Value) {
      assert(Count < Capacity);
      iterator E = end();
      if (I != E)
        std::copy_backward(I, E, E + 1);
      *I = Value;
      ++Count;
    }

    void push_back(const T &Value) {
      assert(Count < Capacity);
      Items[Count++] = Value;
    }
  };

  // As long as FunctionEffect is only 1 byte, and there are only 2 verifiable
  // effects, this fixed-size vector with a capacity of 7 is more than
  // sufficient and is only 8 bytes.
  FixedVector<FunctionEffect, uint8_t, 7> Impl;

public:
  EffectSet() = default;
  explicit EffectSet(FunctionEffectsRef FX) { insert(FX); }

  operator ArrayRef<FunctionEffect>() const {
    return ArrayRef(Impl.cbegin(), Impl.cend());
  }

  using iterator = const FunctionEffect *;
  iterator begin() const { return Impl.cbegin(); }
  iterator end() const { return Impl.cend(); }

  void insert(const FunctionEffect &Effect) {
    FunctionEffect *Iter = Impl.begin();
    FunctionEffect *End = Impl.end();
    // linear search; lower_bound is overkill for a tiny vector like this
    for (; Iter != End; ++Iter) {
      if (*Iter == Effect)
        return;
      if (Effect < *Iter)
        break;
    }
    Impl.insert(Iter, Effect);
  }
  void insert(const EffectSet &Set) {
    for (const FunctionEffect &Item : Set) {
      // push_back because set is already sorted
      Impl.push_back(Item);
    }
  }
  void insert(FunctionEffectsRef FX) {
    for (const FunctionEffectWithCondition &EC : FX) {
      assert(EC.Cond.getCondition() ==
             nullptr); // should be resolved by now, right?
      // push_back because set is already sorted
      Impl.push_back(EC.Effect);
    }
  }
  bool contains(const FunctionEffect::Kind EK) const {
    for (const FunctionEffect &E : Impl)
      if (E.kind() == EK)
        return true;
    return false;
  }

  void dump(llvm::raw_ostream &OS) const;

  static EffectSet difference(ArrayRef<FunctionEffect> LHS,
                              ArrayRef<FunctionEffect> RHS) {
    EffectSet Result;
    std::set_difference(LHS.begin(), LHS.end(), RHS.begin(), RHS.end(),
                        std::back_inserter(Result.Impl));
    return Result;
  }
};

LLVM_DUMP_METHOD void EffectSet::dump(llvm::raw_ostream &OS) const {
  OS << "Effects{";
  bool First = true;
  for (const FunctionEffect &Effect : *this) {
    if (!First)
      OS << ", ";
    else
      First = false;
    OS << Effect.name();
  }
  OS << "}";
}

// Transitory, more extended information about a callable, which can be a
// function, block, function pointer, etc.
struct CallableInfo {
  // CDecl holds the function's definition, if any.
  // FunctionDecl if CallType::Function or Virtual
  // BlockDecl if CallType::Block
  const Decl *CDecl;
  SpecialFuncType FuncType = SpecialFuncType::None;
  EffectSet Effects;
  CallType CType = CallType::Unknown;

  CallableInfo(Sema &SemaRef, const Decl &CD,
               SpecialFuncType FT = SpecialFuncType::None)
      : CDecl(&CD), FuncType(FT) {
    FunctionEffectsRef FXRef;

    if (auto *FD = dyn_cast<FunctionDecl>(CDecl)) {
      // Use the function's definition, if any.
      if (const FunctionDecl *Def = FD->getDefinition())
        CDecl = FD = Def;
      CType = CallType::Function;
      if (auto *Method = dyn_cast<CXXMethodDecl>(FD);
          Method && Method->isVirtual())
        CType = CallType::Virtual;
      FXRef = FD->getFunctionEffects();
    } else if (auto *BD = dyn_cast<BlockDecl>(CDecl)) {
      CType = CallType::Block;
      FXRef = BD->getFunctionEffects();
    } else if (auto *VD = dyn_cast<ValueDecl>(CDecl)) {
      // ValueDecl is function, enum, or variable, so just look at its type.
      FXRef = FunctionEffectsRef::get(VD->getType());
    }
    Effects = EffectSet(FXRef);
  }

  bool isDirectCall() const {
    return CType == CallType::Function || CType == CallType::Block;
  }

  bool isVerifiable() const {
    switch (CType) {
    case CallType::Unknown:
    case CallType::Virtual:
      return false;
    case CallType::Block:
      return true;
    case CallType::Function:
      return functionIsVerifiable(dyn_cast<FunctionDecl>(CDecl));
    }
    llvm_unreachable("undefined CallType");
  }

  /// Generate a name for logging and diagnostics.
  std::string name(Sema &Sem) const {
    std::string Name;
    llvm::raw_string_ostream OS(Name);

    if (auto *FD = dyn_cast<FunctionDecl>(CDecl))
      FD->getNameForDiagnostic(OS, Sem.getPrintingPolicy(),
                                /*Qualified=*/true);
    else if (auto *BD = dyn_cast<BlockDecl>(CDecl))
      OS << "(block " << BD->getBlockManglingNumber() << ")";
    else if (auto *VD = dyn_cast<NamedDecl>(CDecl))
      VD->printQualifiedName(OS);
    return Name;
  }
};

// ----------
// Map effects to single diagnostics, to hold the first (of potentially many)
// diagnostics pertaining to an effect, per function.
class EffectToDiagnosticMap {
  // Since we currently only have a tiny number of effects (typically no more
  // than 1), use a sorted SmallVector with an inline capacity of 1. Since it
  // is often empty, use a unique_ptr to the SmallVector.
  // Note that Diagnostic itself contains a FunctionEffect which is the key.
  using ImplVec = llvm::SmallVector<Diagnostic, 1>;
  std::unique_ptr<ImplVec> Impl;

public:
  // Insert a new diagnostic if we do not already have one for its effect.
  void maybeInsert(const Diagnostic &Diag) {
    if (Impl == nullptr)
      Impl = std::make_unique<ImplVec>();
    auto *Iter = _find(Diag.Effect);
    if (Iter != Impl->end() && Iter->Effect == Diag.Effect)
      return;

    Impl->insert(Iter, Diag);
  }

  const Diagnostic *lookup(FunctionEffect Key) {
    if (Impl == nullptr)
      return nullptr;

    auto *Iter = _find(Key);
    if (Iter != Impl->end() && Iter->Effect == Key)
      return &*Iter;

    return nullptr;
  }

  size_t size() const { return Impl ? Impl->size() : 0; }

private:
  ImplVec::iterator _find(const FunctionEffect &key) {
    // A linear search suffices for a tiny number of possible effects.
    auto *End = Impl->end();
    for (auto *Iter = Impl->begin(); Iter != End; ++Iter)
      if (!(Iter->Effect < key))
        return Iter;
    return End;
  }
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
  EffectSet DeclaredVerifiableEffects;
  EffectSet FXToInfer;

private:
  // Diagnostics pertaining to the function's explicit effects.
  SmallVector<Diagnostic, 0> DiagnosticsForExplicitFX;

  // Diagnostics pertaining to other, non-explicit, inferrable effects.
  EffectToDiagnosticMap InferrableEffectToFirstDiagnostic;

  // These unverified direct calls are what keeps the analysis "pending",
  // until the callees can be verified.
  SmallVector<DirectCall, 0> UnverifiedDirectCalls;

public:
  PendingFunctionAnalysis(
      Sema &Sem, const CallableInfo &CInfo,
      ArrayRef<FunctionEffect> AllInferrableEffectsToVerify) {
    DeclaredVerifiableEffects = CInfo.Effects;

    // Check for effects we are not allowed to infer
    EffectSet InferrableFX;

    for (const FunctionEffect &effect : AllInferrableEffectsToVerify) {
      if (effect.canInferOnFunction(*CInfo.CDecl))
        InferrableFX.insert(effect);
      else {
        // Add a diagnostic for this effect if a caller were to
        // try to infer it.
        InferrableEffectToFirstDiagnostic.maybeInsert(
            Diagnostic(effect, DiagnosticID::DeclDisallowsInference,
                       CInfo.CDecl->getLocation()));
      }
    }
    // InferrableFX is now the set of inferrable effects which are not
    // prohibited
    FXToInfer = EffectSet::difference(InferrableFX, DeclaredVerifiableEffects);
  }

  // Hide the way that diagnostics for explicitly required effects vs. inferred
  // ones are handled differently.
  void checkAddDiagnostic(bool Inferring, const Diagnostic &NewDiag) {
    if (!Inferring)
      DiagnosticsForExplicitFX.push_back(NewDiag);
    else
      InferrableEffectToFirstDiagnostic.maybeInsert(NewDiag);
  }

  void addUnverifiedDirectCall(const Decl *D, SourceLocation CallLoc) {
    UnverifiedDirectCalls.emplace_back(D, CallLoc);
  }

  // Analysis is complete when there are no unverified direct calls.
  bool isComplete() const { return UnverifiedDirectCalls.empty(); }

  const Diagnostic *diagnosticForInferrableEffect(FunctionEffect effect) {
    return InferrableEffectToFirstDiagnostic.lookup(effect);
  }

  SmallVector<DirectCall, 0> &unverifiedCalls() {
    assert(!isComplete());
    return UnverifiedDirectCalls;
  }

  SmallVector<Diagnostic, 0> &getDiagnosticsForExplicitFX() {
    return DiagnosticsForExplicitFX;
  }

  void dump(Sema &SemaRef, llvm::raw_ostream &OS) const {
    OS << "Pending: Declared ";
    DeclaredVerifiableEffects.dump(OS);
    OS << ", " << DiagnosticsForExplicitFX.size() << " diags; ";
    OS << " Infer ";
    FXToInfer.dump(OS);
    OS << ", " << InferrableEffectToFirstDiagnostic.size() << " diags";
    if (!UnverifiedDirectCalls.empty()) {
      OS << "; Calls: ";
      for (const DirectCall &Call : UnverifiedDirectCalls) {
        CallableInfo CI(SemaRef, *Call.Callee);
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
  EffectSet VerifiedEffects;

private:
  // This is used to generate notes about failed inference.
  EffectToDiagnosticMap InferrableEffectToFirstDiagnostic;

public:
  // The incoming Pending analysis is consumed (member(s) are moved-from).
  CompleteFunctionAnalysis(
      ASTContext &Ctx, PendingFunctionAnalysis &Pending,
      const EffectSet &DeclaredEffects,
      ArrayRef<FunctionEffect> AllInferrableEffectsToVerify) {
    VerifiedEffects.insert(DeclaredEffects);
    for (const FunctionEffect &effect : AllInferrableEffectsToVerify)
      if (Pending.diagnosticForInferrableEffect(effect) == nullptr)
        VerifiedEffects.insert(effect);

    InferrableEffectToFirstDiagnostic =
        std::move(Pending.InferrableEffectToFirstDiagnostic);
  }

  const Diagnostic *firstDiagnosticForEffect(const FunctionEffect &Effect) {
    return InferrableEffectToFirstDiagnostic.lookup(Effect);
  }

  void dump(llvm::raw_ostream &OS) const {
    OS << "Complete: Verified ";
    VerifiedEffects.dump(OS);
    OS << "; Infer ";
    OS << InferrableEffectToFirstDiagnostic.size() << " diags\n";
  }
};

const Decl *CanonicalFunctionDecl(const Decl *D) {
  if (auto *FD = dyn_cast<FunctionDecl>(D)) {
    FD = FD->getCanonicalDecl();
    assert(FD != nullptr);
    return FD;
  }
  return D;
}

// ==========
class Analyzer {
  Sema &Sem;

  // Subset of Sema.AllEffectsToVerify
  EffectSet AllInferrableEffectsToVerify;

  using FuncAnalysisPtr =
      llvm::PointerUnion<PendingFunctionAnalysis *, CompleteFunctionAnalysis *>;

  // Map all Decls analyzed to FuncAnalysisPtr. Pending state is larger
  // than complete state, so use different objects to represent them.
  // The state pointers are owned by the container.
  class AnalysisMap : protected llvm::DenseMap<const Decl *, FuncAnalysisPtr> {
    using Base = llvm::DenseMap<const Decl *, FuncAnalysisPtr>;

  public:
    ~AnalysisMap();

    // Use non-public inheritance in order to maintain the invariant
    // that lookups and insertions are via the canonical Decls.

    FuncAnalysisPtr lookup(const Decl *Key) const {
      return Base::lookup(CanonicalFunctionDecl(Key));
    }

    FuncAnalysisPtr &operator[](const Decl *Key) {
      return Base::operator[](CanonicalFunctionDecl(Key));
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
        CallableInfo CI(SemaRef, *item.first);
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
  Analyzer(Sema &S) : Sem(S) {}

  void run(const TranslationUnitDecl &TU) {
    // Gather all of the effects to be verified to see what operations need to
    // be checked, and to see which ones are inferrable.
    for (const FunctionEffectWithCondition &CFE : Sem.AllEffectsToVerify) {
      const FunctionEffect &Effect = CFE.Effect;
      const FunctionEffect::Flags Flags = Effect.flags();
      if (Flags & FunctionEffect::FE_InferrableOnCallees)
        AllInferrableEffectsToVerify.insert(Effect);
    }
    LLVM_DEBUG(
      llvm::dbgs() << "AllInferrableEffectsToVerify: ";
      AllInferrableEffectsToVerify.dump(llvm::dbgs());
      llvm::dbgs() << "\n";
    );

    // We can use DeclsWithEffectsToVerify as a stack for a
    // depth-first traversal; there's no need for a second container. But first,
    // reverse it, so when working from the end, Decls are verified in the order
    // they are declared.
    SmallVector<const Decl *> &VerificationQueue = Sem.DeclsWithEffectsToVerify;
    std::reverse(VerificationQueue.begin(), VerificationQueue.end());

    while (!VerificationQueue.empty()) {
      const Decl *D = VerificationQueue.back();
      if (FuncAnalysisPtr AP = DeclAnalysis.lookup(D)) {
        if (isa<CompleteFunctionAnalysis *>(AP)) {
          // already done
          VerificationQueue.pop_back();
          continue;
        }
        if (isa<PendingFunctionAnalysis *>(AP)) {
          // All children have been traversed; finish analysis.
          auto *Pending = AP.get<PendingFunctionAnalysis *>();
          finishPendingAnalysis(D, Pending);
          VerificationQueue.pop_back();
          continue;
        }
        llvm_unreachable("unexpected DeclAnalysis item");
      }

      // Not previously visited; begin a new analysis for this Decl.
      PendingFunctionAnalysis *Pending = verifyDecl(D);
      if (Pending == nullptr) {
        // completed now
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
        if (isa<PendingFunctionAnalysis *>(AP)) {
          // This indicates recursion (not necessarily direct). For the
          // purposes of effect analysis, we can just ignore it since
          // no effects forbid recursion.
          Call.Recursed = true;
          continue;
        }
        llvm_unreachable("unexpected DeclAnalysis item");
      }
    }
  }

private:
  // Verify a single Decl. Return the pending structure if that was the result,
  // else null. This method must not recurse.
  PendingFunctionAnalysis *verifyDecl(const Decl *D) {
    CallableInfo CInfo(Sem, *D);
    bool isExternC = false;

    if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
      assert(FD->getBuiltinID() == 0);
      isExternC = FD->getCanonicalDecl()->isExternCContext();
    }

    // For C++, with non-extern "C" linkage only - if any of the Decl's declared
    // effects forbid throwing (e.g. nonblocking) then the function should also
    // be declared noexcept.
    if (Sem.getLangOpts().CPlusPlus && !isExternC) {
      for (const FunctionEffect &Effect : CInfo.Effects) {
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
          Sem.Diag(D->getBeginLoc(),
                   diag::warn_perf_constraint_implies_noexcept)
              << Effect.name();
        break;
      }
    }

    // Build a PendingFunctionAnalysis on the stack. If it turns out to be
    // complete, we'll have avoided a heap allocation; if it's incomplete, it's
    // a fairly trivial move to a heap-allocated object.
    PendingFunctionAnalysis FAnalysis(Sem, CInfo, AllInferrableEffectsToVerify);

    LLVM_DEBUG(
      llvm::dbgs() << "\nVerifying " << CInfo.name(Sem) << " ";
      FAnalysis.dump(Sem, llvm::dbgs());
    );

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
    LLVM_DEBUG(
      llvm::dbgs() << "inserted pending " << PendingPtr << "\n";
      DeclAnalysis.dump(Sem, llvm::dbgs());
    );
    return PendingPtr;
  }

  // Consume PendingFunctionAnalysis, create with it a CompleteFunctionAnalysis,
  // inserted in the container.
  void completeAnalysis(const CallableInfo &CInfo,
                        PendingFunctionAnalysis &Pending) {
    if (SmallVector<Diagnostic, 0> &Diags =
            Pending.getDiagnosticsForExplicitFX();
        !Diags.empty())
      emitDiagnostics(Diags, CInfo, Sem);

    CompleteFunctionAnalysis *CompletePtr = new CompleteFunctionAnalysis(
        Sem.getASTContext(), Pending, CInfo.Effects,
        AllInferrableEffectsToVerify);
    DeclAnalysis[CInfo.CDecl] = CompletePtr;
    LLVM_DEBUG(
      llvm::dbgs() << "inserted complete " << CompletePtr << "\n";
      DeclAnalysis.dump(Sem, llvm::dbgs());
    );
  }

  // Called after all direct calls requiring inference have been found -- or
  // not. Repeats calls to FunctionBodyASTVisitor::followCall() but without
  // the possibility of inference. Deletes Pending.
  void finishPendingAnalysis(const Decl *D, PendingFunctionAnalysis *Pending) {
    CallableInfo Caller(Sem, *D);
    LLVM_DEBUG(
      llvm::dbgs() << "finishPendingAnalysis for " << Caller.name(Sem) << " : ";
      Pending->dump(Sem, llvm::dbgs());
      llvm::dbgs() << "\n";
    );
    for (const PendingFunctionAnalysis::DirectCall &Call :
         Pending->unverifiedCalls()) {
      if (Call.Recursed)
        continue;

      CallableInfo Callee(Sem, *Call.Callee);
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
    const bool DirectCall = Callee.isDirectCall();

    // Initially, the declared effects; inferred effects will be added.
    EffectSet CalleeEffects = Callee.Effects;

    bool IsInferencePossible = DirectCall;

    if (DirectCall) {
      if (CompleteFunctionAnalysis *CFA =
              DeclAnalysis.completedAnalysisForDecl(Callee.CDecl)) {
        // Combine declared effects with those which may have been inferred.
        CalleeEffects.insert(CFA->VerifiedEffects);
        IsInferencePossible = false; // we've already traversed it
      }
    }

    if (AssertNoFurtherInference) {
      assert(!IsInferencePossible);
    }

    if (!Callee.isVerifiable())
      IsInferencePossible = false;

    LLVM_DEBUG(
      llvm::dbgs() << "followCall from " << Caller.name(Sem) << " to "
                   << Callee.name(Sem)
                   << "; verifiable: " << Callee.isVerifiable() << "; callee ";
      CalleeEffects.dump(llvm::dbgs());
      llvm::dbgs() << "\n";
      llvm::dbgs() << "  callee " << Callee.CDecl << " canonical "
                   << CanonicalFunctionDecl(Callee.CDecl) << " redecls";
      for (Decl *D : Callee.CDecl->redecls())
        llvm::dbgs() << " " << D;

      llvm::dbgs() << "\n";
    );

    auto check1Effect = [&](const FunctionEffect &Effect, bool Inferring) {
      FunctionEffect::Flags Flags = Effect.flags();
      bool Diagnose =
          Effect.shouldDiagnoseFunctionCall(DirectCall, CalleeEffects);
      if (Diagnose) {
        // If inference is not allowed, or the target is indirect (virtual
        // method/function ptr?), generate a diagnostic now.
        if (!IsInferencePossible ||
            !(Flags & FunctionEffect::FE_InferrableOnCallees)) {
          if (Callee.FuncType == SpecialFuncType::None)
            PFA.checkAddDiagnostic(
                Inferring, {Effect, DiagnosticID::CallsDeclWithoutEffect,
                            CallLoc, Callee.CDecl});
          else
            PFA.checkAddDiagnostic(
                Inferring, {Effect, DiagnosticID::AllocatesMemory, CallLoc});
        } else {
          // Inference is allowed and necessary; defer it.
          PFA.addUnverifiedDirectCall(Callee.CDecl, CallLoc);
        }
      }
    };

    for (const FunctionEffect &Effect : PFA.DeclaredVerifiableEffects)
      check1Effect(Effect, false);

    for (const FunctionEffect &Effect : PFA.FXToInfer)
      check1Effect(Effect, true);
  }

  // Should only be called when determined to be complete.
  void emitDiagnostics(SmallVector<Diagnostic, 0> &Diags,
                       const CallableInfo &CInfo, Sema &S) {
    if (Diags.empty())
      return;
    const SourceManager &SM = S.getSourceManager();
    std::sort(Diags.begin(), Diags.end(),
              [&SM](const Diagnostic &LHS, const Diagnostic &RHS) {
                return SM.isBeforeInTranslationUnit(LHS.Loc, RHS.Loc);
              });

    auto checkAddTemplateNote = [&](const Decl *D) {
      if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
        while (FD != nullptr && FD->isTemplateInstantiation()) {
          S.Diag(FD->getPointOfInstantiation(),
                 diag::note_func_effect_from_template);
          FD = FD->getTemplateInstantiationPattern();
        }
      }
    };

    // Top-level diagnostics are warnings.
    for (const Diagnostic &Diag : Diags) {
      StringRef effectName = Diag.Effect.name();
      switch (Diag.ID) {
      case DiagnosticID::None:
      case DiagnosticID::DeclDisallowsInference: // shouldn't happen
                                                 // here
        llvm_unreachable("Unexpected diagnostic kind");
        break;
      case DiagnosticID::AllocatesMemory:
        S.Diag(Diag.Loc, diag::warn_func_effect_allocates) << effectName;
        checkAddTemplateNote(CInfo.CDecl);
        break;
      case DiagnosticID::Throws:
      case DiagnosticID::Catches:
        S.Diag(Diag.Loc, diag::warn_func_effect_throws_or_catches)
            << effectName;
        checkAddTemplateNote(CInfo.CDecl);
        break;
      case DiagnosticID::HasStaticLocal:
        S.Diag(Diag.Loc, diag::warn_func_effect_has_static_local) << effectName;
        checkAddTemplateNote(CInfo.CDecl);
        break;
      case DiagnosticID::AccessesThreadLocal:
        S.Diag(Diag.Loc, diag::warn_func_effect_uses_thread_local)
            << effectName;
        checkAddTemplateNote(CInfo.CDecl);
        break;
      case DiagnosticID::CallsObjC:
        S.Diag(Diag.Loc, diag::warn_func_effect_calls_objc) << effectName;
        checkAddTemplateNote(CInfo.CDecl);
        break;
      case DiagnosticID::CallsExprWithoutEffect:
        S.Diag(Diag.Loc, diag::warn_func_effect_calls_expr_without_effect)
            << effectName;
        checkAddTemplateNote(CInfo.CDecl);
        break;

      case DiagnosticID::CallsDeclWithoutEffect: {
        CallableInfo CalleeInfo(S, *Diag.Callee);
        std::string CalleeName = CalleeInfo.name(S);

        S.Diag(Diag.Loc, diag::warn_func_effect_calls_func_without_effect)
            << effectName << CalleeName;
        checkAddTemplateNote(CInfo.CDecl);

        // Emit notes explaining the transitive chain of inferences: Why isn't
        // the callee safe?
        for (const Decl *Callee = Diag.Callee; Callee != nullptr;) {
          std::optional<CallableInfo> MaybeNextCallee;
          CompleteFunctionAnalysis *Completed =
              DeclAnalysis.completedAnalysisForDecl(CalleeInfo.CDecl);
          if (Completed == nullptr) {
            // No result - could be
            // - non-inline
            // - indirect (virtual or through function pointer)
            // - effect has been explicitly disclaimed (e.g. "blocking")
            if (CalleeInfo.CType == CallType::Virtual)
              S.Diag(Callee->getLocation(), diag::note_func_effect_call_virtual)
                  << effectName;
            else if (CalleeInfo.CType == CallType::Unknown)
              S.Diag(Callee->getLocation(),
                     diag::note_func_effect_call_func_ptr)
                  << effectName;
            else if (CalleeInfo.Effects.contains(Diag.Effect.oppositeKind()))
              S.Diag(Callee->getLocation(),
                     diag::note_func_effect_call_disallows_inference)
                  << effectName;
            else
              S.Diag(Callee->getLocation(), diag::note_func_effect_call_extern)
                  << effectName;

            break;
          }
          const Diagnostic *PtrDiag2 =
              Completed->firstDiagnosticForEffect(Diag.Effect);
          if (PtrDiag2 == nullptr)
            break;

          const Diagnostic &Diag2 = *PtrDiag2;
          switch (Diag2.ID) {
          case DiagnosticID::None:
            llvm_unreachable("Unexpected diagnostic kind");
            break;
          case DiagnosticID::DeclDisallowsInference:
            S.Diag(Diag2.Loc, diag::note_func_effect_call_disallows_inference)
                << effectName;
            break;
          case DiagnosticID::CallsExprWithoutEffect:
            S.Diag(Diag2.Loc, diag::note_func_effect_call_func_ptr)
                << effectName;
            break;
          case DiagnosticID::AllocatesMemory:
            S.Diag(Diag2.Loc, diag::note_func_effect_allocates) << effectName;
            break;
          case DiagnosticID::Throws:
          case DiagnosticID::Catches:
            S.Diag(Diag2.Loc, diag::note_func_effect_throws_or_catches)
                << effectName;
            break;
          case DiagnosticID::HasStaticLocal:
            S.Diag(Diag2.Loc, diag::note_func_effect_has_static_local)
                << effectName;
            break;
          case DiagnosticID::AccessesThreadLocal:
            S.Diag(Diag2.Loc, diag::note_func_effect_uses_thread_local)
                << effectName;
            break;
          case DiagnosticID::CallsObjC:
            S.Diag(Diag2.Loc, diag::note_func_effect_calls_objc) << effectName;
            break;
          case DiagnosticID::CallsDeclWithoutEffect:
            MaybeNextCallee.emplace(S, *Diag2.Callee);
            S.Diag(Diag2.Loc, diag::note_func_effect_calls_func_without_effect)
                << effectName << MaybeNextCallee->name(S);
            break;
          }
          checkAddTemplateNote(Callee);
          Callee = Diag2.Callee;
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
  // Diagnostics are always routed to a PendingFunctionAnalysis, which holds
  // all diagnostic output.
  struct FunctionBodyASTVisitor
      : public RecursiveASTVisitor<FunctionBodyASTVisitor> {

    Analyzer &Outer;
    PendingFunctionAnalysis &CurrentFunction;
    CallableInfo &CurrentCaller;

    FunctionBodyASTVisitor(Analyzer &outer,
                           PendingFunctionAnalysis &CurrentFunction,
                           CallableInfo &CurrentCaller)
        : Outer(outer), CurrentFunction(CurrentFunction),
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
    // flags include the specified flag receive a diagnostic. \p Flag describes
    // the construct.
    void diagnoseLanguageConstruct(FunctionEffect::FlagBit Flag, DiagnosticID D,
                                   SourceLocation Loc,
                                   const Decl *Callee = nullptr) {
      // If there are any declared verifiable effects which forbid the construct
      // represented by the flag, store just one diagnostic.
      for (const FunctionEffect &Effect :
           CurrentFunction.DeclaredVerifiableEffects) {
        if (Effect.flags() & Flag) {
          addDiagnostic(/*inferring=*/false, Effect, D, Loc, Callee);
          break;
        }
      }
      // For each inferred effect which forbids the construct, store a
      // diagnostic, if we don't already have a diagnostic for that effect.
      for (const FunctionEffect &Effect : CurrentFunction.FXToInfer)
        if (Effect.flags() & Flag)
          addDiagnostic(/*inferring=*/true, Effect, D, Loc, Callee);
    }

    void addDiagnostic(bool Inferring, const FunctionEffect &Effect,
                       DiagnosticID D, SourceLocation Loc,
                       const Decl *Callee = nullptr) {
      CurrentFunction.checkAddDiagnostic(Inferring,
                                         Diagnostic(Effect, D, Loc, Callee));
    }

    // Here we have a call to a Decl, either explicitly via a CallExpr or some
    // other AST construct. CallableInfo pertains to the callee.
    void followCall(const CallableInfo &CI, SourceLocation CallLoc) {
      // Currently, built-in functions are always considered safe.
      // FIXME: Some are not.
      if (const auto *FD = dyn_cast<FunctionDecl>(CI.CDecl);
          FD && FD->getBuiltinID() != 0)
        return;

      Outer.followCall(CurrentCaller, CurrentFunction, CI, CallLoc,
                       /*AssertNoFurtherInference=*/false);
    }

    void checkIndirectCall(CallExpr *Call, Expr *CalleeExpr) {
      const QualType CalleeType = CalleeExpr->getType();
      auto *FPT =
          CalleeType->getAs<FunctionProtoType>(); // null if FunctionType
      EffectSet CalleeFX;
      if (FPT)
        CalleeFX.insert(FPT->getFunctionEffects());

      auto check1Effect = [&](const FunctionEffect &Effect, bool Inferring) {
        if (FPT == nullptr || Effect.shouldDiagnoseFunctionCall(
                                  /*direct=*/false, CalleeFX))
          addDiagnostic(Inferring, Effect, DiagnosticID::CallsExprWithoutEffect,
                        Call->getBeginLoc());
      };

      for (const FunctionEffect &Effect :
           CurrentFunction.DeclaredVerifiableEffects)
        check1Effect(Effect, false);

      for (const FunctionEffect &Effect : CurrentFunction.FXToInfer)
        check1Effect(Effect, true);
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
            CallableInfo CI(Outer.Sem, *Dtor);
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
                                DiagnosticID::Throws, Throw->getThrowLoc());
      return true;
    }

    bool VisitCXXCatchStmt(CXXCatchStmt *Catch) {
      diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeCatch,
                                DiagnosticID::Catches, Catch->getCatchLoc());
      return true;
    }

    bool VisitObjCAtThrowStmt(ObjCAtThrowStmt *Throw) {
      diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeThrow,
                                DiagnosticID::Throws, Throw->getThrowLoc());
      return true;
    }

    bool VisitObjCAtCatchStmt(ObjCAtCatchStmt *Catch) {
      diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeCatch,
                                DiagnosticID::Catches, Catch->getAtCatchLoc());
      return true;
    }

    bool VisitObjCMessageExpr(ObjCMessageExpr *Msg) {
      diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeObjCMessageSend,
                                DiagnosticID::CallsObjC, Msg->getBeginLoc());
      return true;
    }

    bool VisitCallExpr(CallExpr *Call) {
      LLVM_DEBUG(
        llvm::dbgs() << "VisitCallExpr : "
                     << Call->getBeginLoc().printToString(Outer.Sem.SourceMgr)
                     << "\n";
      );

      Expr *CalleeExpr = Call->getCallee();
      if (const Decl *Callee = CalleeExpr->getReferencedDeclOfCallee()) {
        CallableInfo CI(Outer.Sem, *Callee);
        followCall(CI, Call->getBeginLoc());
        return true;
      }

      if (isa<CXXPseudoDestructorExpr>(CalleeExpr))
        // just destroying a scalar, fine.
        return true;

      // No Decl, just an Expr. Just check based on its type.
      checkIndirectCall(Call, CalleeExpr);

      return true;
    }

    bool VisitVarDecl(VarDecl *Var) {
      LLVM_DEBUG(
        llvm::dbgs() << "VisitVarDecl : "
                     << Var->getBeginLoc().printToString(Outer.Sem.SourceMgr)
                     << "\n";
      );

      if (Var->isStaticLocal())
        diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeStaticLocalVars,
                                  DiagnosticID::HasStaticLocal,
                                  Var->getLocation());

      const QualType::DestructionKind DK =
          Var->needsDestruction(Outer.Sem.getASTContext());
      if (DK == QualType::DK_cxx_destructor) {
        QualType QT = Var->getType();
        if (const auto *ClsType = QT.getTypePtr()->getAs<RecordType>()) {
          if (const auto *CxxRec =
                  dyn_cast<CXXRecordDecl>(ClsType->getDecl())) {
            if (const CXXDestructorDecl *Dtor = CxxRec->getDestructor()) {
              CallableInfo CI(Outer.Sem, *Dtor);
              followCall(CI, Var->getLocation());
            }
          }
        }
      }
      return true;
    }

    bool VisitCXXNewExpr(CXXNewExpr *New) {
      // BUG? It seems incorrect that RecursiveASTVisitor does not
      // visit the call to operator new.
      if (FunctionDecl *FD = New->getOperatorNew()) {
        CallableInfo CI(Outer.Sem, *FD, SpecialFuncType::OperatorNew);
        followCall(CI, New->getBeginLoc());
      }

      // It's a bit excessive to check operator delete here, since it's
      // just a fallback for operator new followed by a failed constructor.
      // We could check it via New->getOperatorDelete().

      // It DOES however visit the called constructor
      return true;
    }

    bool VisitCXXDeleteExpr(CXXDeleteExpr *Delete) {
      // BUG? It seems incorrect that RecursiveASTVisitor does not
      // visit the call to operator delete.
      if (FunctionDecl *FD = Delete->getOperatorDelete()) {
        CallableInfo CI(Outer.Sem, *FD, SpecialFuncType::OperatorDelete);
        followCall(CI, Delete->getBeginLoc());
      }

      // It DOES however visit the called destructor

      return true;
    }

    bool VisitCXXConstructExpr(CXXConstructExpr *Construct) {
      LLVM_DEBUG(
        llvm::dbgs() << "VisitCXXConstructExpr : "
                     << Construct->getBeginLoc().printToString(
                            Outer.Sem.SourceMgr)
                     << "\n";
      );

      // BUG? It seems incorrect that RecursiveASTVisitor does not
      // visit the call to the constructor.
      const CXXConstructorDecl *Ctor = Construct->getConstructor();
      CallableInfo CI(Outer.Sem, *Ctor);
      followCall(CI, Construct->getLocation());

      return true;
    }

    bool VisitCXXDefaultInitExpr(CXXDefaultInitExpr *DEI) {
      if (Expr *E = DEI->getExpr())
        TraverseStmt(E);

      return true;
    }

    bool TraverseLambdaExpr(LambdaExpr *Lambda) {
      // We override this so as the be able to skip traversal of the lambda's
      // body. We have to explicitly traverse the captures.
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
      if (isa<VarDecl>(Val)) {
        const VarDecl *Var = cast<VarDecl>(Val);
        VarDecl::TLSKind TLSK = Var->getTLSKind();
        if (TLSK != VarDecl::TLS_None) {
          // At least on macOS, thread-local variables are initialized on
          // first access, including a heap allocation.
          diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeThreadLocalVars,
                                    DiagnosticID::AccessesThreadLocal,
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

void performEffectAnalysis(Sema &S, TranslationUnitDecl *TU)
{
	if (S.hasUncompilableErrorOccurred() || S.Diags.getIgnoreAllWarnings())
	  // exit if having uncompilable errors or ignoring all warnings:
	  return;
	if (TU == nullptr)
	  return;
	Analyzer{S}.run(*TU);
}

} // namespace clang
