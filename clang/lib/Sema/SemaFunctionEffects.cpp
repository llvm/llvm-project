//=== SemaFunctionEffects.cpp - Sema handling of function effects ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Sema handling of function effects.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/TypeBase.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Sema/SemaInternal.h"

#define DEBUG_TYPE "effectanalysis"

using namespace clang;

namespace {

enum class ViolationID : uint8_t {
  None = 0, // Sentinel for an empty Violation.
  // These first 5 map to a %select{} in one of several FunctionEffects
  // diagnostics, e.g. warn_func_effect_violation.
  BaseDiagnosticIndex,
  AllocatesMemory = BaseDiagnosticIndex,
  ThrowsOrCatchesExceptions,
  HasStaticLocalVariable,
  AccessesThreadLocalVariable,
  AccessesObjCMethodOrProperty,

  // These only apply to callees, where the analysis stops at the Decl.
  DeclDisallowsInference,

  // These both apply to indirect calls. The difference is that sometimes
  // we have an actual Decl (generally a variable) which is the function
  // pointer being called, and sometimes, typically due to a cast, we only
  // have an expression.
  CallsDeclWithoutEffect,
  CallsExprWithoutEffect,
};

// Information about the AST context in which a violation was found, so
// that diagnostics can point to the correct source.
class ViolationSite {
public:
  enum class Kind : uint8_t {
    Default, // Function body.
    MemberInitializer,
    DefaultArgExpr
  };

private:
  llvm::PointerIntPair<CXXDefaultArgExpr *, 2, Kind> Impl;

public:
  ViolationSite() = default;

  explicit ViolationSite(CXXDefaultArgExpr *E)
      : Impl(E, Kind::DefaultArgExpr) {}

  Kind kind() const { return static_cast<Kind>(Impl.getInt()); }
  CXXDefaultArgExpr *defaultArgExpr() const { return Impl.getPointer(); }

  void setKind(Kind K) { Impl.setPointerAndInt(nullptr, K); }
};

// Represents a violation of the rules, potentially for the entire duration of
// the analysis phase, in order to refer to it when explaining why a caller has
// been made unsafe by a callee. Can be transformed into either a Diagnostic
// (warning or a note), depending on whether the violation pertains to a
// function failing to be verifed as holding an effect vs. a function failing to
// be inferred as holding that effect.
struct Violation {
  FunctionEffect Effect;
  std::optional<FunctionEffect>
      CalleeEffectPreventingInference; // Only for certain IDs; can be nullopt.
  ViolationID ID = ViolationID::None;
  ViolationSite Site;
  SourceLocation Loc;
  const Decl *Callee =
      nullptr; // Only valid for ViolationIDs Calls{Decl,Expr}WithoutEffect.

  Violation(FunctionEffect Effect, ViolationID ID, ViolationSite VS,
            SourceLocation Loc, const Decl *Callee = nullptr,
            std::optional<FunctionEffect> CalleeEffect = std::nullopt)
      : Effect(Effect), CalleeEffectPreventingInference(CalleeEffect), ID(ID),
        Site(VS), Loc(Loc), Callee(Callee) {}

  unsigned diagnosticSelectIndex() const {
    return unsigned(ID) - unsigned(ViolationID::BaseDiagnosticIndex);
  }
};

enum class SpecialFuncType : uint8_t { None, OperatorNew, OperatorDelete };
enum class CallableType : uint8_t {
  // Unknown: probably function pointer.
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
  const auto *FPT = FD->getType()->getAs<FunctionProtoType>();
  return FPT && (FPT->isNothrow() || FD->hasAttr<NoThrowAttr>());
}

// This list is probably incomplete.
// FIXME: Investigate:
// __builtin_eh_return?
// __builtin_allow_runtime_check?
// __builtin_unwind_init and other similar things that sound exception-related.
// va_copy?
// coroutines?
static FunctionEffectKindSet getBuiltinFunctionEffects(unsigned BuiltinID) {
  FunctionEffectKindSet Result;

  switch (BuiltinID) {
  case 0:  // Not builtin.
  default: // By default, builtins have no known effects.
    break;

  // These allocate/deallocate heap memory.
  case Builtin::ID::BI__builtin_calloc:
  case Builtin::ID::BI__builtin_malloc:
  case Builtin::ID::BI__builtin_realloc:
  case Builtin::ID::BI__builtin_free:
  case Builtin::ID::BI__builtin_operator_delete:
  case Builtin::ID::BI__builtin_operator_new:
  case Builtin::ID::BIaligned_alloc:
  case Builtin::ID::BIcalloc:
  case Builtin::ID::BImalloc:
  case Builtin::ID::BImemalign:
  case Builtin::ID::BIrealloc:
  case Builtin::ID::BIfree:

  case Builtin::ID::BIfopen:
  case Builtin::ID::BIpthread_create:
  case Builtin::ID::BI_Block_object_dispose:
    Result.insert(FunctionEffect(FunctionEffect::Kind::Allocating));
    break;

  // These block in some other way than allocating memory.
  // longjmp() and friends are presumed unsafe because they are the moral
  // equivalent of throwing a C++ exception, which is unsafe.
  case Builtin::ID::BIlongjmp:
  case Builtin::ID::BI_longjmp:
  case Builtin::ID::BIsiglongjmp:
  case Builtin::ID::BI__builtin_longjmp:
  case Builtin::ID::BIobjc_exception_throw:

  // Objective-C runtime.
  case Builtin::ID::BIobjc_msgSend:
  case Builtin::ID::BIobjc_msgSend_fpret:
  case Builtin::ID::BIobjc_msgSend_fp2ret:
  case Builtin::ID::BIobjc_msgSend_stret:
  case Builtin::ID::BIobjc_msgSendSuper:
  case Builtin::ID::BIobjc_getClass:
  case Builtin::ID::BIobjc_getMetaClass:
  case Builtin::ID::BIobjc_enumerationMutation:
  case Builtin::ID::BIobjc_assign_ivar:
  case Builtin::ID::BIobjc_assign_global:
  case Builtin::ID::BIobjc_sync_enter:
  case Builtin::ID::BIobjc_sync_exit:
  case Builtin::ID::BINSLog:
  case Builtin::ID::BINSLogv:

  // stdio.h
  case Builtin::ID::BIfread:
  case Builtin::ID::BIfwrite:

  // stdio.h: printf family.
  case Builtin::ID::BIprintf:
  case Builtin::ID::BI__builtin_printf:
  case Builtin::ID::BIfprintf:
  case Builtin::ID::BIsnprintf:
  case Builtin::ID::BIsprintf:
  case Builtin::ID::BIvprintf:
  case Builtin::ID::BIvfprintf:
  case Builtin::ID::BIvsnprintf:
  case Builtin::ID::BIvsprintf:

  // stdio.h: scanf family.
  case Builtin::ID::BIscanf:
  case Builtin::ID::BIfscanf:
  case Builtin::ID::BIsscanf:
  case Builtin::ID::BIvscanf:
  case Builtin::ID::BIvfscanf:
  case Builtin::ID::BIvsscanf:
    Result.insert(FunctionEffect(FunctionEffect::Kind::Blocking));
    break;
  }

  return Result;
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
  std::string getNameForDiagnostic(Sema &S) const {
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
  // FIXME: Is there a way to simplify this using existing data structures?
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

    auto *Iter = llvm::find_if(
        *Impl, [&](const auto &Item) { return Item.Effect == Key; });
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
    ViolationSite VSite;

    DirectCall(const Decl *D, SourceLocation CallLoc, ViolationSite VSite)
        : Callee(D), CallLoc(CallLoc), VSite(VSite) {}
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
    // Check for effects we are not allowed to infer.
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
            effect, ViolationID::DeclDisallowsInference, ViolationSite{},
            CInfo.CDecl->getLocation(), nullptr, ProblemCalleeEffect));
      }
    }
    // InferrableEffects is now the set of inferrable effects which are not
    // prohibited.
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

  void addUnverifiedDirectCall(const Decl *D, SourceLocation CallLoc,
                               ViolationSite VSite) {
    UnverifiedDirectCalls.emplace_back(D, CallLoc, VSite);
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
      llvm::sort(ViolationsForExplicitEffects,
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
        OS << " " << CI.getNameForDiagnostic(SemaRef);
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
  CompleteFunctionAnalysis(ASTContext &Ctx, PendingFunctionAnalysis &&Pending,
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
        return cast<CompleteFunctionAnalysis *>(AP);
      return nullptr;
    }

    void dump(Sema &SemaRef, llvm::raw_ostream &OS) {
      OS << "\nAnalysisMap:\n";
      for (const auto &item : *this) {
        CallableInfo CI(*item.first);
        const auto AP = item.second;
        OS << item.first << " " << CI.getNameForDiagnostic(SemaRef) << " : ";
        if (AP.isNull()) {
          OS << "null\n";
        } else if (auto *CFA = dyn_cast<CompleteFunctionAnalysis *>(AP)) {
          OS << CFA << " ";
          CFA->dump(OS);
        } else if (auto *PFA = dyn_cast<PendingFunctionAnalysis *>(AP)) {
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
        if (auto *Pending = dyn_cast<PendingFunctionAnalysis *>(AP)) {
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
            auto *FPT = TSI->getType()->castAs<FunctionProtoType>();
            IsNoexcept = FPT->isNothrow() || BD->hasAttr<NoThrowAttr>();
          }
        }
        if (!IsNoexcept)
          S.Diag(D->getBeginLoc(), diag::warn_perf_constraint_implies_noexcept)
              << GetCallableDeclKind(D, nullptr) << Effect.name();
        break;
      }
    }

    // Build a PendingFunctionAnalysis on the stack. If it turns out to be
    // complete, we'll have avoided a heap allocation; if it's incomplete, it's
    // a fairly trivial move to a heap-allocated object.
    PendingFunctionAnalysis FAnalysis(S, CInfo, AllInferrableEffectsToVerify);

    LLVM_DEBUG(llvm::dbgs()
                   << "\nVerifying " << CInfo.getNameForDiagnostic(S) << " ";
               FAnalysis.dump(S, llvm::dbgs()););

    FunctionBodyASTVisitor Visitor(*this, FAnalysis, CInfo);

    Visitor.run();
    if (FAnalysis.isComplete()) {
      completeAnalysis(CInfo, std::move(FAnalysis));
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
                        PendingFunctionAnalysis &&Pending) {
    if (ArrayRef<Violation> Viols =
            Pending.getSortedViolationsForExplicitEffects(S.getSourceManager());
        !Viols.empty())
      emitDiagnostics(Viols, CInfo);

    CompleteFunctionAnalysis *CompletePtr = new CompleteFunctionAnalysis(
        S.getASTContext(), std::move(Pending), CInfo.Effects,
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
    LLVM_DEBUG(llvm::dbgs() << "finishPendingAnalysis for "
                            << Caller.getNameForDiagnostic(S) << " : ";
               Pending->dump(S, llvm::dbgs()); llvm::dbgs() << "\n";);
    for (const PendingFunctionAnalysis::DirectCall &Call :
         Pending->unverifiedCalls()) {
      if (Call.Recursed)
        continue;

      CallableInfo Callee(*Call.Callee);
      followCall(Caller, *Pending, Callee, Call.CallLoc,
                 /*AssertNoFurtherInference=*/true, Call.VSite);
    }
    completeAnalysis(Caller, std::move(*Pending));
    delete Pending;
  }

  // Here we have a call to a Decl, either explicitly via a CallExpr or some
  // other AST construct. PFA pertains to the caller.
  void followCall(const CallableInfo &Caller, PendingFunctionAnalysis &PFA,
                  const CallableInfo &Callee, SourceLocation CallLoc,
                  bool AssertNoFurtherInference, ViolationSite VSite) {
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

    LLVM_DEBUG(llvm::dbgs()
                   << "followCall from " << Caller.getNameForDiagnostic(S)
                   << " to " << Callee.getNameForDiagnostic(S)
                   << "; verifiable: " << Callee.isVerifiable() << "; callee ";
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
                                 VSite, CallLoc, Callee.CDecl});
        else
          PFA.checkAddViolation(
              Inferring,
              {Effect, ViolationID::AllocatesMemory, VSite, CallLoc});
      } else {
        // Inference is allowed and necessary; defer it.
        PFA.addUnverifiedDirectCall(Callee.CDecl, CallLoc, VSite);
      }
    };

    for (FunctionEffect Effect : PFA.DeclaredVerifiableEffects)
      Check1Effect(Effect, false);

    for (FunctionEffect Effect : PFA.EffectsToInfer)
      Check1Effect(Effect, true);
  }

  // Describe a callable Decl for a diagnostic.
  // (Not an enum class because the value is always converted to an integer for
  // use in a diagnostic.)
  enum CallableDeclKind {
    CDK_Function,
    CDK_Constructor,
    CDK_Destructor,
    CDK_Lambda,
    CDK_Block,
    CDK_MemberInitializer,
  };

  // Describe a call site or target using an enum mapping to a %select{}
  // in a diagnostic, e.g. warn_func_effect_violation,
  // warn_perf_constraint_implies_noexcept, and others.
  static CallableDeclKind GetCallableDeclKind(const Decl *D,
                                              const Violation *V) {
    if (V != nullptr &&
        V->Site.kind() == ViolationSite::Kind::MemberInitializer)
      return CDK_MemberInitializer;
    if (isa<BlockDecl>(D))
      return CDK_Block;
    if (auto *Method = dyn_cast<CXXMethodDecl>(D)) {
      if (isa<CXXConstructorDecl>(D))
        return CDK_Constructor;
      if (isa<CXXDestructorDecl>(D))
        return CDK_Destructor;
      const CXXRecordDecl *Rec = Method->getParent();
      if (Rec->isLambda())
        return CDK_Lambda;
    }
    return CDK_Function;
  };

  // Should only be called when function's analysis is determined to be
  // complete.
  void emitDiagnostics(ArrayRef<Violation> Viols, const CallableInfo &CInfo) {
    if (Viols.empty())
      return;

    auto MaybeAddTemplateNote = [&](const Decl *D) {
      if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
        while (FD != nullptr && FD->isTemplateInstantiation() &&
               FD->getPointOfInstantiation().isValid()) {
          S.Diag(FD->getPointOfInstantiation(),
                 diag::note_func_effect_from_template);
          FD = FD->getTemplateInstantiationPattern();
        }
      }
    };

    // For note_func_effect_call_indirect.
    enum { Indirect_VirtualMethod, Indirect_FunctionPtr };

    auto MaybeAddSiteContext = [&](const Decl *D, const Violation &V) {
      // If a violation site is a member initializer, add a note pointing to
      // the constructor which invoked it.
      if (V.Site.kind() == ViolationSite::Kind::MemberInitializer) {
        unsigned ImplicitCtor = 0;
        if (auto *Ctor = dyn_cast<CXXConstructorDecl>(D);
            Ctor && Ctor->isImplicit())
          ImplicitCtor = 1;
        S.Diag(D->getLocation(), diag::note_func_effect_in_constructor)
            << ImplicitCtor;
      }

      // If a violation site is a default argument expression, add a note
      // pointing to the call site using the default argument.
      else if (V.Site.kind() == ViolationSite::Kind::DefaultArgExpr)
        S.Diag(V.Site.defaultArgExpr()->getUsedLocation(),
               diag::note_in_evaluating_default_argument);
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
      case ViolationID::ThrowsOrCatchesExceptions:
      case ViolationID::HasStaticLocalVariable:
      case ViolationID::AccessesThreadLocalVariable:
      case ViolationID::AccessesObjCMethodOrProperty:
        S.Diag(Viol1.Loc, diag::warn_func_effect_violation)
            << GetCallableDeclKind(CInfo.CDecl, &Viol1) << effectName
            << Viol1.diagnosticSelectIndex();
        MaybeAddSiteContext(CInfo.CDecl, Viol1);
        MaybeAddTemplateNote(CInfo.CDecl);
        break;
      case ViolationID::CallsExprWithoutEffect:
        S.Diag(Viol1.Loc, diag::warn_func_effect_calls_expr_without_effect)
            << GetCallableDeclKind(CInfo.CDecl, &Viol1) << effectName;
        MaybeAddSiteContext(CInfo.CDecl, Viol1);
        MaybeAddTemplateNote(CInfo.CDecl);
        break;

      case ViolationID::CallsDeclWithoutEffect: {
        CallableInfo CalleeInfo(*Viol1.Callee);
        std::string CalleeName = CalleeInfo.getNameForDiagnostic(S);

        S.Diag(Viol1.Loc, diag::warn_func_effect_calls_func_without_effect)
            << GetCallableDeclKind(CInfo.CDecl, &Viol1) << effectName
            << GetCallableDeclKind(CalleeInfo.CDecl, nullptr) << CalleeName;
        MaybeAddSiteContext(CInfo.CDecl, Viol1);
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
              S.Diag(Callee->getLocation(),
                     diag::note_func_effect_call_indirect)
                  << Indirect_VirtualMethod << effectName;
            else if (CType == CallableType::Unknown)
              S.Diag(Callee->getLocation(),
                     diag::note_func_effect_call_indirect)
                  << Indirect_FunctionPtr << effectName;
            else if (CalleeInfo.Effects.contains(Viol1.Effect.oppositeKind()))
              S.Diag(Callee->getLocation(),
                     diag::note_func_effect_call_disallows_inference)
                  << GetCallableDeclKind(CInfo.CDecl, nullptr) << effectName
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
                << GetCallableDeclKind(CalleeInfo.CDecl, nullptr) << effectName
                << Viol2.CalleeEffectPreventingInference->name();
            break;
          case ViolationID::CallsExprWithoutEffect:
            S.Diag(Viol2.Loc, diag::note_func_effect_call_indirect)
                << Indirect_FunctionPtr << effectName;
            break;
          case ViolationID::AllocatesMemory:
          case ViolationID::ThrowsOrCatchesExceptions:
          case ViolationID::HasStaticLocalVariable:
          case ViolationID::AccessesThreadLocalVariable:
          case ViolationID::AccessesObjCMethodOrProperty:
            S.Diag(Viol2.Loc, diag::note_func_effect_violation)
                << GetCallableDeclKind(CalleeInfo.CDecl, &Viol2) << effectName
                << Viol2.diagnosticSelectIndex();
            MaybeAddSiteContext(CalleeInfo.CDecl, Viol2);
            break;
          case ViolationID::CallsDeclWithoutEffect:
            MaybeNextCallee.emplace(*Viol2.Callee);
            S.Diag(Viol2.Loc, diag::note_func_effect_calls_func_without_effect)
                << GetCallableDeclKind(CalleeInfo.CDecl, &Viol2) << effectName
                << GetCallableDeclKind(Viol2.Callee, nullptr)
                << MaybeNextCallee->getNameForDiagnostic(S);
            break;
          }
          MaybeAddTemplateNote(Callee);
          Callee = Viol2.Callee;
          if (MaybeNextCallee) {
            CalleeInfo = *MaybeNextCallee;
            CalleeName = CalleeInfo.getNameForDiagnostic(S);
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
  struct FunctionBodyASTVisitor : DynamicRecursiveASTVisitor {
    Analyzer &Outer;
    PendingFunctionAnalysis &CurrentFunction;
    CallableInfo &CurrentCaller;
    ViolationSite VSite;
    const Expr *TrailingRequiresClause = nullptr;
    const Expr *NoexceptExpr = nullptr;

    FunctionBodyASTVisitor(Analyzer &Outer,
                           PendingFunctionAnalysis &CurrentFunction,
                           CallableInfo &CurrentCaller)
        : Outer(Outer), CurrentFunction(CurrentFunction),
          CurrentCaller(CurrentCaller) {
      ShouldVisitImplicitCode = true;
      ShouldWalkTypesOfTypeLocs = false;
    }

    // -- Entry point --
    void run() {
      // The target function may have implicit code paths beyond the
      // body: member and base destructors. Visit these first.
      if (auto *Dtor = dyn_cast<CXXDestructorDecl>(CurrentCaller.CDecl))
        followDestructor(dyn_cast<CXXRecordDecl>(Dtor->getParent()), Dtor);

      if (auto *FD = dyn_cast<FunctionDecl>(CurrentCaller.CDecl)) {
        TrailingRequiresClause = FD->getTrailingRequiresClause().ConstraintExpr;

        // Note that FD->getType->getAs<FunctionProtoType>() can yield a
        // noexcept Expr which has been boiled down to a constant expression.
        // Going through the TypeSourceInfo obtains the actual expression which
        // will be traversed as part of the function -- unless we capture it
        // here and have TraverseStmt skip it.
        if (TypeSourceInfo *TSI = FD->getTypeSourceInfo()) {
          if (FunctionProtoTypeLoc TL =
                  TSI->getTypeLoc().getAs<FunctionProtoTypeLoc>())
            if (const FunctionProtoType *FPT = TL.getTypePtr())
              NoexceptExpr = FPT->getNoexceptExpr();
        }
      }

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

    void addViolation(bool Inferring, FunctionEffect Effect, ViolationID VID,
                      SourceLocation Loc, const Decl *Callee = nullptr) {
      CurrentFunction.checkAddViolation(
          Inferring, Violation(Effect, VID, VSite, Loc, Callee));
    }

    // Here we have a call to a Decl, either explicitly via a CallExpr or some
    // other AST construct. CallableInfo pertains to the callee.
    void followCall(CallableInfo &CI, SourceLocation CallLoc) {
      // Check for a call to a builtin function, whose effects are
      // handled specially.
      if (const auto *FD = dyn_cast<FunctionDecl>(CI.CDecl)) {
        if (unsigned BuiltinID = FD->getBuiltinID()) {
          CI.Effects = getBuiltinFunctionEffects(BuiltinID);
          if (CI.Effects.empty()) {
            // A builtin with no known effects is assumed safe.
            return;
          }
          // A builtin WITH effects doesn't get any special treatment for
          // being noreturn/noexcept, e.g. longjmp(), so we skip the check
          // below.
        } else {
          // If the callee is both `noreturn` and `noexcept`, it presumably
          // terminates. Ignore it for the purposes of effect analysis.
          // If not C++, `noreturn` alone is sufficient.
          if (FD->isNoReturn() &&
              (!Outer.S.getLangOpts().CPlusPlus || isNoexcept(FD)))
            return;
        }
      }

      Outer.followCall(CurrentCaller, CurrentFunction, CI, CallLoc,
                       /*AssertNoFurtherInference=*/false, VSite);
    }

    void checkIndirectCall(CallExpr *Call, QualType CalleeType) {
      FunctionEffectKindSet CalleeEffects;
      if (FunctionEffectsRef Effects = FunctionEffectsRef::get(CalleeType);
          !Effects.empty())
        CalleeEffects.insert(Effects);

      auto Check1Effect = [&](FunctionEffect Effect, bool Inferring) {
        if (Effect.shouldDiagnoseFunctionCall(
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
      SourceLocation DtorLoc = Dtor->getLocation();
      for (const FieldDecl *Field : Rec->fields())
        followTypeDtor(Field->getType(), DtorLoc);

      if (const auto *Class = dyn_cast<CXXRecordDecl>(Rec))
        for (const CXXBaseSpecifier &Base : Class->bases())
          followTypeDtor(Base.getType(), DtorLoc);
    }

    void followTypeDtor(QualType QT, SourceLocation CallSite) {
      const Type *Ty = QT.getTypePtr();
      while (Ty->isArrayType()) {
        const ArrayType *Arr = Ty->getAsArrayTypeUnsafe();
        QT = Arr->getElementType();
        Ty = QT.getTypePtr();
      }

      if (Ty->isRecordType()) {
        if (const CXXRecordDecl *Class = Ty->getAsCXXRecordDecl()) {
          if (CXXDestructorDecl *Dtor = Class->getDestructor();
              Dtor && !Dtor->isDeleted()) {
            CallableInfo CI(*Dtor);
            followCall(CI, CallSite);
          }
        }
      }
    }

    // -- Methods for use of RecursiveASTVisitor --

    bool VisitCXXThrowExpr(CXXThrowExpr *Throw) override {
      diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeThrow,
                                ViolationID::ThrowsOrCatchesExceptions,
                                Throw->getThrowLoc());
      return true;
    }

    bool VisitCXXCatchStmt(CXXCatchStmt *Catch) override {
      diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeCatch,
                                ViolationID::ThrowsOrCatchesExceptions,
                                Catch->getCatchLoc());
      return true;
    }

    bool VisitObjCAtThrowStmt(ObjCAtThrowStmt *Throw) override {
      diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeThrow,
                                ViolationID::ThrowsOrCatchesExceptions,
                                Throw->getThrowLoc());
      return true;
    }

    bool VisitObjCAtCatchStmt(ObjCAtCatchStmt *Catch) override {
      diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeCatch,
                                ViolationID::ThrowsOrCatchesExceptions,
                                Catch->getAtCatchLoc());
      return true;
    }

    bool VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *Finally) override {
      diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeCatch,
                                ViolationID::ThrowsOrCatchesExceptions,
                                Finally->getAtFinallyLoc());
      return true;
    }

    bool VisitObjCMessageExpr(ObjCMessageExpr *Msg) override {
      diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeObjCMessageSend,
                                ViolationID::AccessesObjCMethodOrProperty,
                                Msg->getBeginLoc());
      return true;
    }

    bool VisitObjCAutoreleasePoolStmt(ObjCAutoreleasePoolStmt *ARP) override {
      // Under the hood, @autorelease (potentially?) allocates memory and
      // invokes ObjC methods. We don't currently have memory allocation as
      // a "language construct" but we do have ObjC messaging, so diagnose that.
      diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeObjCMessageSend,
                                ViolationID::AccessesObjCMethodOrProperty,
                                ARP->getBeginLoc());
      return true;
    }

    bool VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt *Sync) override {
      // Under the hood, this calls objc_sync_enter and objc_sync_exit, wrapped
      // in a @try/@finally block. Diagnose this generically as "ObjC
      // messaging".
      diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeObjCMessageSend,
                                ViolationID::AccessesObjCMethodOrProperty,
                                Sync->getBeginLoc());
      return true;
    }

    bool VisitSEHExceptStmt(SEHExceptStmt *Exc) override {
      diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeCatch,
                                ViolationID::ThrowsOrCatchesExceptions,
                                Exc->getExceptLoc());
      return true;
    }

    bool VisitCallExpr(CallExpr *Call) override {
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

    bool VisitVarDecl(VarDecl *Var) override {
      LLVM_DEBUG(llvm::dbgs()
                     << "VisitVarDecl : "
                     << Var->getBeginLoc().printToString(Outer.S.SourceMgr)
                     << "\n";);

      if (Var->isStaticLocal())
        diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeStaticLocalVars,
                                  ViolationID::HasStaticLocalVariable,
                                  Var->getLocation());

      const QualType::DestructionKind DK =
          Var->needsDestruction(Outer.S.getASTContext());
      if (DK == QualType::DK_cxx_destructor)
        followTypeDtor(Var->getType(), Var->getLocation());
      return true;
    }

    bool VisitCXXNewExpr(CXXNewExpr *New) override {
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

    bool VisitCXXDeleteExpr(CXXDeleteExpr *Delete) override {
      // RecursiveASTVisitor does not visit the implicit call to operator
      // delete.
      if (FunctionDecl *FD = Delete->getOperatorDelete()) {
        CallableInfo CI(*FD, SpecialFuncType::OperatorDelete);
        followCall(CI, Delete->getBeginLoc());
      }

      // It DOES however visit the called destructor

      return true;
    }

    bool VisitCXXConstructExpr(CXXConstructExpr *Construct) override {
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

    bool TraverseStmt(Stmt *Statement) override {
      // If this statement is a `requires` clause from the top-level function
      // being traversed, ignore it, since it's not generating runtime code.
      // We skip the traversal of lambdas (beyond their captures, see
      // TraverseLambdaExpr below), so just caching this from our constructor
      // should suffice.
      if (Statement != TrailingRequiresClause && Statement != NoexceptExpr)
        return DynamicRecursiveASTVisitor::TraverseStmt(Statement);
      return true;
    }

    bool TraverseConstructorInitializer(CXXCtorInitializer *Init) override {
      ViolationSite PrevVS = VSite;
      if (Init->isAnyMemberInitializer())
        VSite.setKind(ViolationSite::Kind::MemberInitializer);
      bool Result =
          DynamicRecursiveASTVisitor::TraverseConstructorInitializer(Init);
      VSite = PrevVS;
      return Result;
    }

    bool TraverseCXXDefaultArgExpr(CXXDefaultArgExpr *E) override {
      LLVM_DEBUG(llvm::dbgs()
                     << "TraverseCXXDefaultArgExpr : "
                     << E->getUsedLocation().printToString(Outer.S.SourceMgr)
                     << "\n";);

      ViolationSite PrevVS = VSite;
      if (VSite.kind() == ViolationSite::Kind::Default)
        VSite = ViolationSite{E};

      bool Result = DynamicRecursiveASTVisitor::TraverseCXXDefaultArgExpr(E);
      VSite = PrevVS;
      return Result;
    }

    bool TraverseLambdaExpr(LambdaExpr *Lambda) override {
      // We override this so as to be able to skip traversal of the lambda's
      // body. We have to explicitly traverse the captures. Why not return
      // false from shouldVisitLambdaBody()? Because we need to visit a lambda's
      // body when we are verifying the lambda itself; we only want to skip it
      // in the context of the outer function.
      for (unsigned I = 0, N = Lambda->capture_size(); I < N; ++I)
        TraverseLambdaCapture(Lambda, Lambda->capture_begin() + I,
                              Lambda->capture_init_begin()[I]);

      return true;
    }

    bool TraverseBlockExpr(BlockExpr * /*unused*/) override {
      // As with lambdas, don't traverse the block's body.
      // TODO: are the capture expressions (ctor call?) safe?
      return true;
    }

    bool VisitDeclRefExpr(DeclRefExpr *E) override {
      const ValueDecl *Val = E->getDecl();
      if (const auto *Var = dyn_cast<VarDecl>(Val)) {
        if (Var->getTLSKind() != VarDecl::TLS_None) {
          // At least on macOS, thread-local variables are initialized on
          // first access, including a heap allocation.
          diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeThreadLocalVars,
                                    ViolationID::AccessesThreadLocalVariable,
                                    E->getLocation());
        }
      }
      return true;
    }

    bool TraverseGenericSelectionExpr(GenericSelectionExpr *Node) override {
      return TraverseStmt(Node->getResultExpr());
    }
    bool
    TraverseUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *Node) override {
      return true;
    }

    bool TraverseTypeOfExprTypeLoc(TypeOfExprTypeLoc Node,
                                   bool TraverseQualifier) override {
      return true;
    }

    bool TraverseDecltypeTypeLoc(DecltypeTypeLoc Node,
                                 bool TraverseQualifier) override {
      return true;
    }

    bool TraverseCXXNoexceptExpr(CXXNoexceptExpr *Node) override {
      return true;
    }

    bool TraverseCXXTypeidExpr(CXXTypeidExpr *Node) override { return true; }

    // Skip concept requirements since they don't generate code.
    bool TraverseConceptRequirement(concepts::Requirement *R) override {
      return true;
    }
  };
};

Analyzer::AnalysisMap::~AnalysisMap() {
  for (const auto &Item : *this) {
    FuncAnalysisPtr AP = Item.second;
    if (auto *PFA = dyn_cast<PendingFunctionAnalysis *>(AP))
      delete PFA;
    else
      delete cast<CompleteFunctionAnalysis *>(AP);
  }
}

} // anonymous namespace

namespace clang {

bool Sema::diagnoseConflictingFunctionEffect(
    const FunctionEffectsRef &FX, const FunctionEffectWithCondition &NewEC,
    SourceLocation NewAttrLoc) {
  // If the new effect has a condition, we can't detect conflicts until the
  // condition is resolved.
  if (NewEC.Cond.getCondition() != nullptr)
    return false;

  // Diagnose the new attribute as incompatible with a previous one.
  auto Incompatible = [&](const FunctionEffectWithCondition &PrevEC) {
    Diag(NewAttrLoc, diag::err_attributes_are_not_compatible)
        << ("'" + NewEC.description() + "'")
        << ("'" + PrevEC.description() + "'") << false;
    // We don't necessarily have the location of the previous attribute,
    // so no note.
    return true;
  };

  // Compare against previous attributes.
  FunctionEffect::Kind NewKind = NewEC.Effect.kind();

  for (const FunctionEffectWithCondition &PrevEC : FX) {
    // Again, can't check yet when the effect is conditional.
    if (PrevEC.Cond.getCondition() != nullptr)
      continue;

    FunctionEffect::Kind PrevKind = PrevEC.Effect.kind();
    // Note that we allow PrevKind == NewKind; it's redundant and ignored.

    if (PrevEC.Effect.oppositeKind() == NewKind)
      return Incompatible(PrevEC);

    // A new allocating is incompatible with a previous nonblocking.
    if (PrevKind == FunctionEffect::Kind::NonBlocking &&
        NewKind == FunctionEffect::Kind::Allocating)
      return Incompatible(PrevEC);

    // A new nonblocking is incompatible with a previous allocating.
    if (PrevKind == FunctionEffect::Kind::Allocating &&
        NewKind == FunctionEffect::Kind::NonBlocking)
      return Incompatible(PrevEC);
  }

  return false;
}

void Sema::diagnoseFunctionEffectMergeConflicts(
    const FunctionEffectSet::Conflicts &Errs, SourceLocation NewLoc,
    SourceLocation OldLoc) {
  for (const FunctionEffectSet::Conflict &Conflict : Errs) {
    Diag(NewLoc, diag::warn_conflicting_func_effects)
        << Conflict.Kept.description() << Conflict.Rejected.description();
    Diag(OldLoc, diag::note_previous_declaration);
  }
}

// Decl should be a FunctionDecl or BlockDecl.
void Sema::maybeAddDeclWithEffects(const Decl *D,
                                   const FunctionEffectsRef &FX) {
  if (!D->hasBody()) {
    if (const auto *FD = D->getAsFunction(); FD && !FD->willHaveBody())
      return;
  }

  if (Diags.getIgnoreAllWarnings() ||
      (Diags.getSuppressSystemWarnings() &&
       SourceMgr.isInSystemHeader(D->getLocation())))
    return;

  if (hasUncompilableErrorOccurred())
    return;

  // For code in dependent contexts, we'll do this at instantiation time.
  // Without this check, we would analyze the function based on placeholder
  // template parameters, and potentially generate spurious diagnostics.
  if (cast<DeclContext>(D)->isDependentContext())
    return;

  addDeclWithEffects(D, FX);
}

void Sema::addDeclWithEffects(const Decl *D, const FunctionEffectsRef &FX) {
  // To avoid the possibility of conflict, don't add effects which are
  // not FE_InferrableOnCallees and therefore not verified; this removes
  // blocking/allocating but keeps nonblocking/nonallocating.
  // Also, ignore any conditions when building the list of effects.
  bool AnyVerifiable = false;
  for (const FunctionEffectWithCondition &EC : FX)
    if (EC.Effect.flags() & FunctionEffect::FE_InferrableOnCallees) {
      AllEffectsToVerify.insert(EC.Effect);
      AnyVerifiable = true;
    }

  // Record the declaration for later analysis.
  if (AnyVerifiable)
    DeclsWithEffectsToVerify.push_back(D);
}

void Sema::performFunctionEffectAnalysis(TranslationUnitDecl *TU) {
  if (hasUncompilableErrorOccurred() || Diags.getIgnoreAllWarnings())
    return;
  if (TU == nullptr)
    return;
  Analyzer{*this}.run(*TU);
}

Sema::FunctionEffectDiffVector::FunctionEffectDiffVector(
    const FunctionEffectsRef &Old, const FunctionEffectsRef &New) {

  FunctionEffectsRef::iterator POld = Old.begin();
  FunctionEffectsRef::iterator OldEnd = Old.end();
  FunctionEffectsRef::iterator PNew = New.begin();
  FunctionEffectsRef::iterator NewEnd = New.end();

  while (true) {
    int cmp = 0;
    if (POld == OldEnd) {
      if (PNew == NewEnd)
        break;
      cmp = 1;
    } else if (PNew == NewEnd)
      cmp = -1;
    else {
      FunctionEffectWithCondition Old = *POld;
      FunctionEffectWithCondition New = *PNew;
      if (Old.Effect.kind() < New.Effect.kind())
        cmp = -1;
      else if (New.Effect.kind() < Old.Effect.kind())
        cmp = 1;
      else {
        cmp = 0;
        if (Old.Cond.getCondition() != New.Cond.getCondition()) {
          // FIXME: Cases where the expressions are equivalent but
          // don't have the same identity.
          push_back(FunctionEffectDiff{
              Old.Effect.kind(), FunctionEffectDiff::Kind::ConditionMismatch,
              Old, New});
        }
      }
    }

    if (cmp < 0) {
      // removal
      FunctionEffectWithCondition Old = *POld;
      push_back(FunctionEffectDiff{Old.Effect.kind(),
                                   FunctionEffectDiff::Kind::Removed, Old,
                                   std::nullopt});
      ++POld;
    } else if (cmp > 0) {
      // addition
      FunctionEffectWithCondition New = *PNew;
      push_back(FunctionEffectDiff{New.Effect.kind(),
                                   FunctionEffectDiff::Kind::Added,
                                   std::nullopt, New});
      ++PNew;
    } else {
      ++POld;
      ++PNew;
    }
  }
}

bool Sema::FunctionEffectDiff::shouldDiagnoseConversion(
    QualType SrcType, const FunctionEffectsRef &SrcFX, QualType DstType,
    const FunctionEffectsRef &DstFX) const {

  switch (EffectKind) {
  case FunctionEffect::Kind::NonAllocating:
    // nonallocating can't be added (spoofed) during a conversion, unless we
    // have nonblocking.
    if (DiffKind == Kind::Added) {
      for (const auto &CFE : SrcFX) {
        if (CFE.Effect.kind() == FunctionEffect::Kind::NonBlocking)
          return false;
      }
    }
    [[fallthrough]];
  case FunctionEffect::Kind::NonBlocking:
    // nonblocking can't be added (spoofed) during a conversion.
    switch (DiffKind) {
    case Kind::Added:
      return true;
    case Kind::Removed:
      return false;
    case Kind::ConditionMismatch:
      // FIXME: Condition mismatches are too coarse right now -- expressions
      // which are equivalent but don't have the same identity are detected as
      // mismatches. We're going to diagnose those anyhow until expression
      // matching is better.
      return true;
    }
    break;
  case FunctionEffect::Kind::Blocking:
  case FunctionEffect::Kind::Allocating:
    return false;
  }
  llvm_unreachable("unknown effect kind");
}

bool Sema::FunctionEffectDiff::shouldDiagnoseRedeclaration(
    const FunctionDecl &OldFunction, const FunctionEffectsRef &OldFX,
    const FunctionDecl &NewFunction, const FunctionEffectsRef &NewFX) const {
  switch (EffectKind) {
  case FunctionEffect::Kind::NonAllocating:
  case FunctionEffect::Kind::NonBlocking:
    // nonblocking/nonallocating can't be removed in a redeclaration.
    switch (DiffKind) {
    case Kind::Added:
      return false; // No diagnostic.
    case Kind::Removed:
      return true; // Issue diagnostic.
    case Kind::ConditionMismatch:
      // All these forms of mismatches are diagnosed.
      return true;
    }
    break;
  case FunctionEffect::Kind::Blocking:
  case FunctionEffect::Kind::Allocating:
    return false;
  }
  llvm_unreachable("unknown effect kind");
}

Sema::FunctionEffectDiff::OverrideResult
Sema::FunctionEffectDiff::shouldDiagnoseMethodOverride(
    const CXXMethodDecl &OldMethod, const FunctionEffectsRef &OldFX,
    const CXXMethodDecl &NewMethod, const FunctionEffectsRef &NewFX) const {
  switch (EffectKind) {
  case FunctionEffect::Kind::NonAllocating:
  case FunctionEffect::Kind::NonBlocking:
    switch (DiffKind) {

    // If added on an override, that's fine and not diagnosed.
    case Kind::Added:
      return OverrideResult::NoAction;

    // If missing from an override (removed), propagate from base to derived.
    case Kind::Removed:
      return OverrideResult::Merge;

    // If there's a mismatch involving the effect's polarity or condition,
    // issue a warning.
    case Kind::ConditionMismatch:
      return OverrideResult::Warn;
    }
    break;
  case FunctionEffect::Kind::Blocking:
  case FunctionEffect::Kind::Allocating:
    return OverrideResult::NoAction;
  }
  llvm_unreachable("unknown effect kind");
}

} // namespace clang
