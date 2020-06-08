//===--- SemaCUDA.cpp - Semantic Analysis for CUDA constructs -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements semantic analysis for CUDA constructs.
///
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Basic/Cuda.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Template.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
using namespace clang;

void Sema::PushForceCUDAHostDevice() {
  assert(getLangOpts().CUDA && "Should only be called during CUDA compilation");
  ForceCUDAHostDeviceDepth++;
}

bool Sema::PopForceCUDAHostDevice() {
  assert(getLangOpts().CUDA && "Should only be called during CUDA compilation");
  if (ForceCUDAHostDeviceDepth == 0)
    return false;
  ForceCUDAHostDeviceDepth--;
  return true;
}

ExprResult Sema::ActOnCUDAExecConfigExpr(Scope *S, SourceLocation LLLLoc,
                                         MultiExprArg ExecConfig,
                                         SourceLocation GGGLoc) {
  FunctionDecl *ConfigDecl = Context.getcudaConfigureCallDecl();
  if (!ConfigDecl)
    return ExprError(Diag(LLLLoc, diag::err_undeclared_var_use)
                     << getCudaConfigureFuncName());
  QualType ConfigQTy = ConfigDecl->getType();

  DeclRefExpr *ConfigDR = new (Context)
      DeclRefExpr(Context, ConfigDecl, false, ConfigQTy, VK_LValue, LLLLoc);
  MarkFunctionReferenced(LLLLoc, ConfigDecl);

  return BuildCallExpr(S, ConfigDR, LLLLoc, ExecConfig, GGGLoc, nullptr,
                       /*IsExecConfig=*/true);
}

Sema::CUDAFunctionTarget
Sema::IdentifyCUDATarget(const ParsedAttributesView &Attrs) {
  bool HasHostAttr = false;
  bool HasDeviceAttr = false;
  bool HasGlobalAttr = false;
  bool HasInvalidTargetAttr = false;
  for (const ParsedAttr &AL : Attrs) {
    switch (AL.getKind()) {
    case ParsedAttr::AT_CUDAGlobal:
      HasGlobalAttr = true;
      break;
    case ParsedAttr::AT_CUDAHost:
      HasHostAttr = true;
      break;
    case ParsedAttr::AT_CUDADevice:
      HasDeviceAttr = true;
      break;
    case ParsedAttr::AT_CUDAInvalidTarget:
      HasInvalidTargetAttr = true;
      break;
    default:
      break;
    }
  }

  if (HasInvalidTargetAttr)
    return CFT_InvalidTarget;

  if (HasGlobalAttr)
    return CFT_Global;

  if (HasHostAttr && HasDeviceAttr)
    return CFT_HostDevice;

  if (HasDeviceAttr)
    return CFT_Device;

  return CFT_Host;
}

template <typename A>
static bool hasAttr(const FunctionDecl *D, bool IgnoreImplicitAttr) {
  return D->hasAttrs() && llvm::any_of(D->getAttrs(), [&](Attr *Attribute) {
           return isa<A>(Attribute) &&
                  !(IgnoreImplicitAttr && Attribute->isImplicit());
         });
}

/// IdentifyCUDATarget - Determine the CUDA compilation target for this function
Sema::CUDAFunctionTarget Sema::IdentifyCUDATarget(const FunctionDecl *D,
                                                  bool IgnoreImplicitHDAttr) {
  // Code that lives outside a function is run on the host.
  if (D == nullptr)
    return CFT_Host;

  if (D->hasAttr<CUDAInvalidTargetAttr>())
    return CFT_InvalidTarget;

  if (D->hasAttr<CUDAGlobalAttr>())
    return CFT_Global;

  if (hasAttr<CUDADeviceAttr>(D, IgnoreImplicitHDAttr)) {
    if (hasAttr<CUDAHostAttr>(D, IgnoreImplicitHDAttr))
      return CFT_HostDevice;
    return CFT_Device;
  } else if (hasAttr<CUDAHostAttr>(D, IgnoreImplicitHDAttr)) {
    return CFT_Host;
  } else if (D->isImplicit() && !IgnoreImplicitHDAttr) {
    // Some implicit declarations (like intrinsic functions) are not marked.
    // Set the most lenient target on them for maximal flexibility.
    return CFT_HostDevice;
  }

  return CFT_Host;
}

// * CUDA Call preference table
//
// F - from,
// T - to
// Ph - preference in host mode
// Pd - preference in device mode
// H  - handled in (x)
// Preferences: N:native, SS:same side, HD:host-device, WS:wrong side, --:never.
//
// | F  | T  | Ph  | Pd  |  H  |
// |----+----+-----+-----+-----+
// | d  | d  | N   | N   | (c) |
// | d  | g  | --  | --  | (a) |
// | d  | h  | --  | --  | (e) |
// | d  | hd | HD  | HD  | (b) |
// | g  | d  | N   | N   | (c) |
// | g  | g  | --  | --  | (a) |
// | g  | h  | --  | --  | (e) |
// | g  | hd | HD  | HD  | (b) |
// | h  | d  | --  | --  | (e) |
// | h  | g  | N   | N   | (c) |
// | h  | h  | N   | N   | (c) |
// | h  | hd | HD  | HD  | (b) |
// | hd | d  | WS  | SS  | (d) |
// | hd | g  | SS  | --  |(d/a)|
// | hd | h  | SS  | WS  | (d) |
// | hd | hd | HD  | HD  | (b) |

Sema::CUDAFunctionPreference
Sema::IdentifyCUDAPreference(const FunctionDecl *Caller,
                             const FunctionDecl *Callee) {
  assert(Callee && "Callee must be valid.");
  CUDAFunctionTarget CallerTarget = IdentifyCUDATarget(Caller);
  CUDAFunctionTarget CalleeTarget = IdentifyCUDATarget(Callee);

  // If one of the targets is invalid, the check always fails, no matter what
  // the other target is.
  if (CallerTarget == CFT_InvalidTarget || CalleeTarget == CFT_InvalidTarget)
    return CFP_Never;

  // (a) Can't call global from some contexts until we support CUDA's
  // dynamic parallelism.
  if (CalleeTarget == CFT_Global &&
      (CallerTarget == CFT_Global || CallerTarget == CFT_Device))
    return CFP_Never;

  // (b) Calling HostDevice is OK for everyone.
  if (CalleeTarget == CFT_HostDevice)
    return CFP_HostDevice;

  // (c) Best case scenarios
  if (CalleeTarget == CallerTarget ||
      (CallerTarget == CFT_Host && CalleeTarget == CFT_Global) ||
      (CallerTarget == CFT_Global && CalleeTarget == CFT_Device))
    return CFP_Native;

  // (d) HostDevice behavior depends on compilation mode.
  if (CallerTarget == CFT_HostDevice) {
    // It's OK to call a compilation-mode matching function from an HD one.
    if ((getLangOpts().CUDAIsDevice && CalleeTarget == CFT_Device) ||
        (!getLangOpts().CUDAIsDevice &&
         (CalleeTarget == CFT_Host || CalleeTarget == CFT_Global)))
      return CFP_SameSide;

    // Calls from HD to non-mode-matching functions (i.e., to host functions
    // when compiling in device mode or to device functions when compiling in
    // host mode) are allowed at the sema level, but eventually rejected if
    // they're ever codegened.  TODO: Reject said calls earlier.
    return CFP_WrongSide;
  }

  // (e) Calling across device/host boundary is not something you should do.
  if ((CallerTarget == CFT_Host && CalleeTarget == CFT_Device) ||
      (CallerTarget == CFT_Device && CalleeTarget == CFT_Host) ||
      (CallerTarget == CFT_Global && CalleeTarget == CFT_Host))
    return CFP_Never;

  llvm_unreachable("All cases should've been handled by now.");
}

template <typename AttrT> static bool hasImplicitAttr(const FunctionDecl *D) {
  if (!D)
    return false;
  if (auto *A = D->getAttr<AttrT>())
    return A->isImplicit();
  return D->isImplicit();
}

bool Sema::IsCUDAImplicitHostDeviceFunction(const FunctionDecl *D) {
  bool IsImplicitDevAttr = hasImplicitAttr<CUDADeviceAttr>(D);
  bool IsImplicitHostAttr = hasImplicitAttr<CUDAHostAttr>(D);
  return IsImplicitDevAttr && IsImplicitHostAttr;
}

void Sema::EraseUnwantedCUDAMatches(
    const FunctionDecl *Caller,
    SmallVectorImpl<std::pair<DeclAccessPair, FunctionDecl *>> &Matches) {
  if (Matches.size() <= 1)
    return;

  using Pair = std::pair<DeclAccessPair, FunctionDecl*>;

  // Gets the CUDA function preference for a call from Caller to Match.
  auto GetCFP = [&](const Pair &Match) {
    return IdentifyCUDAPreference(Caller, Match.second);
  };

  // Find the best call preference among the functions in Matches.
  CUDAFunctionPreference BestCFP = GetCFP(*std::max_element(
      Matches.begin(), Matches.end(),
      [&](const Pair &M1, const Pair &M2) { return GetCFP(M1) < GetCFP(M2); }));

  // Erase all functions with lower priority.
  llvm::erase_if(Matches,
                 [&](const Pair &Match) { return GetCFP(Match) < BestCFP; });
}

/// When an implicitly-declared special member has to invoke more than one
/// base/field special member, conflicts may occur in the targets of these
/// members. For example, if one base's member __host__ and another's is
/// __device__, it's a conflict.
/// This function figures out if the given targets \param Target1 and
/// \param Target2 conflict, and if they do not it fills in
/// \param ResolvedTarget with a target that resolves for both calls.
/// \return true if there's a conflict, false otherwise.
static bool
resolveCalleeCUDATargetConflict(Sema::CUDAFunctionTarget Target1,
                                Sema::CUDAFunctionTarget Target2,
                                Sema::CUDAFunctionTarget *ResolvedTarget) {
  // Only free functions and static member functions may be global.
  assert(Target1 != Sema::CFT_Global);
  assert(Target2 != Sema::CFT_Global);

  if (Target1 == Sema::CFT_HostDevice) {
    *ResolvedTarget = Target2;
  } else if (Target2 == Sema::CFT_HostDevice) {
    *ResolvedTarget = Target1;
  } else if (Target1 != Target2) {
    return true;
  } else {
    *ResolvedTarget = Target1;
  }

  return false;
}

bool Sema::inferCUDATargetForImplicitSpecialMember(CXXRecordDecl *ClassDecl,
                                                   CXXSpecialMember CSM,
                                                   CXXMethodDecl *MemberDecl,
                                                   bool ConstRHS,
                                                   bool Diagnose) {
  // If the defaulted special member is defined lexically outside of its
  // owning class, or the special member already has explicit device or host
  // attributes, do not infer.
  bool InClass = MemberDecl->getLexicalParent() == MemberDecl->getParent();
  bool HasH = MemberDecl->hasAttr<CUDAHostAttr>();
  bool HasD = MemberDecl->hasAttr<CUDADeviceAttr>();
  bool HasExplicitAttr =
      (HasD && !MemberDecl->getAttr<CUDADeviceAttr>()->isImplicit()) ||
      (HasH && !MemberDecl->getAttr<CUDAHostAttr>()->isImplicit());
  if (!InClass || HasExplicitAttr)
    return false;

  llvm::Optional<CUDAFunctionTarget> InferredTarget;

  // We're going to invoke special member lookup; mark that these special
  // members are called from this one, and not from its caller.
  ContextRAII MethodContext(*this, MemberDecl);

  // Look for special members in base classes that should be invoked from here.
  // Infer the target of this member base on the ones it should call.
  // Skip direct and indirect virtual bases for abstract classes.
  llvm::SmallVector<const CXXBaseSpecifier *, 16> Bases;
  for (const auto &B : ClassDecl->bases()) {
    if (!B.isVirtual()) {
      Bases.push_back(&B);
    }
  }

  if (!ClassDecl->isAbstract()) {
    for (const auto &VB : ClassDecl->vbases()) {
      Bases.push_back(&VB);
    }
  }

  for (const auto *B : Bases) {
    const RecordType *BaseType = B->getType()->getAs<RecordType>();
    if (!BaseType) {
      continue;
    }

    CXXRecordDecl *BaseClassDecl = cast<CXXRecordDecl>(BaseType->getDecl());
    Sema::SpecialMemberOverloadResult SMOR =
        LookupSpecialMember(BaseClassDecl, CSM,
                            /* ConstArg */ ConstRHS,
                            /* VolatileArg */ false,
                            /* RValueThis */ false,
                            /* ConstThis */ false,
                            /* VolatileThis */ false);

    if (!SMOR.getMethod())
      continue;

    CUDAFunctionTarget BaseMethodTarget = IdentifyCUDATarget(SMOR.getMethod());
    if (!InferredTarget.hasValue()) {
      InferredTarget = BaseMethodTarget;
    } else {
      bool ResolutionError = resolveCalleeCUDATargetConflict(
          InferredTarget.getValue(), BaseMethodTarget,
          InferredTarget.getPointer());
      if (ResolutionError) {
        if (Diagnose) {
          Diag(ClassDecl->getLocation(),
               diag::note_implicit_member_target_infer_collision)
              << (unsigned)CSM << InferredTarget.getValue() << BaseMethodTarget;
        }
        MemberDecl->addAttr(CUDAInvalidTargetAttr::CreateImplicit(Context));
        return true;
      }
    }
  }

  // Same as for bases, but now for special members of fields.
  for (const auto *F : ClassDecl->fields()) {
    if (F->isInvalidDecl()) {
      continue;
    }

    const RecordType *FieldType =
        Context.getBaseElementType(F->getType())->getAs<RecordType>();
    if (!FieldType) {
      continue;
    }

    CXXRecordDecl *FieldRecDecl = cast<CXXRecordDecl>(FieldType->getDecl());
    Sema::SpecialMemberOverloadResult SMOR =
        LookupSpecialMember(FieldRecDecl, CSM,
                            /* ConstArg */ ConstRHS && !F->isMutable(),
                            /* VolatileArg */ false,
                            /* RValueThis */ false,
                            /* ConstThis */ false,
                            /* VolatileThis */ false);

    if (!SMOR.getMethod())
      continue;

    CUDAFunctionTarget FieldMethodTarget =
        IdentifyCUDATarget(SMOR.getMethod());
    if (!InferredTarget.hasValue()) {
      InferredTarget = FieldMethodTarget;
    } else {
      bool ResolutionError = resolveCalleeCUDATargetConflict(
          InferredTarget.getValue(), FieldMethodTarget,
          InferredTarget.getPointer());
      if (ResolutionError) {
        if (Diagnose) {
          Diag(ClassDecl->getLocation(),
               diag::note_implicit_member_target_infer_collision)
              << (unsigned)CSM << InferredTarget.getValue()
              << FieldMethodTarget;
        }
        MemberDecl->addAttr(CUDAInvalidTargetAttr::CreateImplicit(Context));
        return true;
      }
    }
  }


  // If no target was inferred, mark this member as __host__ __device__;
  // it's the least restrictive option that can be invoked from any target.
  bool NeedsH = true, NeedsD = true;
  if (InferredTarget.hasValue()) {
    if (InferredTarget.getValue() == CFT_Device)
      NeedsH = false;
    else if (InferredTarget.getValue() == CFT_Host)
      NeedsD = false;
  }

  // We either setting attributes first time, or the inferred ones must match
  // previously set ones.
  if (NeedsD && !HasD)
    MemberDecl->addAttr(CUDADeviceAttr::CreateImplicit(Context));
  if (NeedsH && !HasH)
    MemberDecl->addAttr(CUDAHostAttr::CreateImplicit(Context));

  return false;
}

bool Sema::isEmptyCudaConstructor(SourceLocation Loc, CXXConstructorDecl *CD) {
  if (!CD->isDefined() && CD->isTemplateInstantiation())
    InstantiateFunctionDefinition(Loc, CD->getFirstDecl());

  // (E.2.3.1, CUDA 7.5) A constructor for a class type is considered
  // empty at a point in the translation unit, if it is either a
  // trivial constructor
  if (CD->isTrivial())
    return true;

  // ... or it satisfies all of the following conditions:
  // The constructor function has been defined.
  // The constructor function has no parameters,
  // and the function body is an empty compound statement.
  if (!(CD->hasTrivialBody() && CD->getNumParams() == 0))
    return false;

  // Its class has no virtual functions and no virtual base classes.
  if (CD->getParent()->isDynamicClass())
    return false;

  // Union ctor does not call ctors of its data members.
  if (CD->getParent()->isUnion())
    return true;

  // The only form of initializer allowed is an empty constructor.
  // This will recursively check all base classes and member initializers
  if (!llvm::all_of(CD->inits(), [&](const CXXCtorInitializer *CI) {
        if (const CXXConstructExpr *CE =
                dyn_cast<CXXConstructExpr>(CI->getInit()))
          return isEmptyCudaConstructor(Loc, CE->getConstructor());
        return false;
      }))
    return false;

  return true;
}

bool Sema::isEmptyCudaDestructor(SourceLocation Loc, CXXDestructorDecl *DD) {
  // No destructor -> no problem.
  if (!DD)
    return true;

  if (!DD->isDefined() && DD->isTemplateInstantiation())
    InstantiateFunctionDefinition(Loc, DD->getFirstDecl());

  // (E.2.3.1, CUDA 7.5) A destructor for a class type is considered
  // empty at a point in the translation unit, if it is either a
  // trivial constructor
  if (DD->isTrivial())
    return true;

  // ... or it satisfies all of the following conditions:
  // The destructor function has been defined.
  // and the function body is an empty compound statement.
  if (!DD->hasTrivialBody())
    return false;

  const CXXRecordDecl *ClassDecl = DD->getParent();

  // Its class has no virtual functions and no virtual base classes.
  if (ClassDecl->isDynamicClass())
    return false;

  // Union does not have base class and union dtor does not call dtors of its
  // data members.
  if (DD->getParent()->isUnion())
    return true;

  // Only empty destructors are allowed. This will recursively check
  // destructors for all base classes...
  if (!llvm::all_of(ClassDecl->bases(), [&](const CXXBaseSpecifier &BS) {
        if (CXXRecordDecl *RD = BS.getType()->getAsCXXRecordDecl())
          return isEmptyCudaDestructor(Loc, RD->getDestructor());
        return true;
      }))
    return false;

  // ... and member fields.
  if (!llvm::all_of(ClassDecl->fields(), [&](const FieldDecl *Field) {
        if (CXXRecordDecl *RD = Field->getType()
                                    ->getBaseElementTypeUnsafe()
                                    ->getAsCXXRecordDecl())
          return isEmptyCudaDestructor(Loc, RD->getDestructor());
        return true;
      }))
    return false;

  return true;
}

void Sema::checkAllowedCUDAInitializer(VarDecl *VD) {
  if (VD->isInvalidDecl() || !VD->hasInit() || !VD->hasGlobalStorage())
    return;
  const Expr *Init = VD->getInit();
  if (VD->hasAttr<CUDADeviceAttr>() || VD->hasAttr<CUDAConstantAttr>() ||
      VD->hasAttr<CUDASharedAttr>()) {
    if (LangOpts.GPUAllowDeviceInit)
      return;
    assert(!VD->isStaticLocal() || VD->hasAttr<CUDASharedAttr>());
    bool AllowedInit = false;
    if (const CXXConstructExpr *CE = dyn_cast<CXXConstructExpr>(Init))
      AllowedInit =
          isEmptyCudaConstructor(VD->getLocation(), CE->getConstructor());
    // We'll allow constant initializers even if it's a non-empty
    // constructor according to CUDA rules. This deviates from NVCC,
    // but allows us to handle things like constexpr constructors.
    if (!AllowedInit &&
        (VD->hasAttr<CUDADeviceAttr>() || VD->hasAttr<CUDAConstantAttr>())) {
      auto *Init = VD->getInit();
      AllowedInit =
          ((VD->getType()->isDependentType() || Init->isValueDependent()) &&
           VD->isConstexpr()) ||
          Init->isConstantInitializer(Context,
                                      VD->getType()->isReferenceType());
    }

    // Also make sure that destructor, if there is one, is empty.
    if (AllowedInit)
      if (CXXRecordDecl *RD = VD->getType()->getAsCXXRecordDecl())
        AllowedInit =
            isEmptyCudaDestructor(VD->getLocation(), RD->getDestructor());

    if (!AllowedInit) {
      Diag(VD->getLocation(), VD->hasAttr<CUDASharedAttr>()
                                  ? diag::err_shared_var_init
                                  : diag::err_dynamic_var_init)
          << Init->getSourceRange();
      VD->setInvalidDecl();
    }
  } else {
    // This is a host-side global variable.  Check that the initializer is
    // callable from the host side.
    const FunctionDecl *InitFn = nullptr;
    if (const CXXConstructExpr *CE = dyn_cast<CXXConstructExpr>(Init)) {
      InitFn = CE->getConstructor();
    } else if (const CallExpr *CE = dyn_cast<CallExpr>(Init)) {
      InitFn = CE->getDirectCallee();
    }
    if (InitFn) {
      CUDAFunctionTarget InitFnTarget = IdentifyCUDATarget(InitFn);
      if (InitFnTarget != CFT_Host && InitFnTarget != CFT_HostDevice) {
        Diag(VD->getLocation(), diag::err_ref_bad_target_global_initializer)
            << InitFnTarget << InitFn;
        Diag(InitFn->getLocation(), diag::note_previous_decl) << InitFn;
        VD->setInvalidDecl();
      }
    }
  }
}

// With -fcuda-host-device-constexpr, an unattributed constexpr function is
// treated as implicitly __host__ __device__, unless:
//  * it is a variadic function (device-side variadic functions are not
//    allowed), or
//  * a __device__ function with this signature was already declared, in which
//    case in which case we output an error, unless the __device__ decl is in a
//    system header, in which case we leave the constexpr function unattributed.
//
// In addition, all function decls are treated as __host__ __device__ when
// ForceCUDAHostDeviceDepth > 0 (corresponding to code within a
//   #pragma clang force_cuda_host_device_begin/end
// pair).
void Sema::maybeAddCUDAHostDeviceAttrs(FunctionDecl *NewD,
                                       const LookupResult &Previous) {
  assert(getLangOpts().CUDA && "Should only be called during CUDA compilation");

  if (ForceCUDAHostDeviceDepth > 0) {
    if (!NewD->hasAttr<CUDAHostAttr>())
      NewD->addAttr(CUDAHostAttr::CreateImplicit(Context));
    if (!NewD->hasAttr<CUDADeviceAttr>())
      NewD->addAttr(CUDADeviceAttr::CreateImplicit(Context));
    return;
  }

  if (!getLangOpts().CUDAHostDeviceConstexpr || !NewD->isConstexpr() ||
      NewD->isVariadic() || NewD->hasAttr<CUDAHostAttr>() ||
      NewD->hasAttr<CUDADeviceAttr>() || NewD->hasAttr<CUDAGlobalAttr>())
    return;

  // Is D a __device__ function with the same signature as NewD, ignoring CUDA
  // attributes?
  auto IsMatchingDeviceFn = [&](NamedDecl *D) {
    if (UsingShadowDecl *Using = dyn_cast<UsingShadowDecl>(D))
      D = Using->getTargetDecl();
    FunctionDecl *OldD = D->getAsFunction();
    return OldD && OldD->hasAttr<CUDADeviceAttr>() &&
           !OldD->hasAttr<CUDAHostAttr>() &&
           !IsOverload(NewD, OldD, /* UseMemberUsingDeclRules = */ false,
                       /* ConsiderCudaAttrs = */ false);
  };
  auto It = llvm::find_if(Previous, IsMatchingDeviceFn);
  if (It != Previous.end()) {
    // We found a __device__ function with the same name and signature as NewD
    // (ignoring CUDA attrs).  This is an error unless that function is defined
    // in a system header, in which case we simply return without making NewD
    // host+device.
    NamedDecl *Match = *It;
    if (!getSourceManager().isInSystemHeader(Match->getLocation())) {
      Diag(NewD->getLocation(),
           diag::err_cuda_unattributed_constexpr_cannot_overload_device)
          << NewD;
      Diag(Match->getLocation(),
           diag::note_cuda_conflicting_device_function_declared_here);
    }
    return;
  }

  NewD->addAttr(CUDAHostAttr::CreateImplicit(Context));
  NewD->addAttr(CUDADeviceAttr::CreateImplicit(Context));
}

void Sema::MaybeAddCUDAConstantAttr(VarDecl *VD) {
  if (getLangOpts().CUDAIsDevice && VD->isConstexpr() &&
      (VD->isFileVarDecl() || VD->isStaticDataMember())) {
    VD->addAttr(CUDAConstantAttr::CreateImplicit(getASTContext()));
  }
}

Sema::DeviceDiagBuilder Sema::CUDADiagIfDeviceCode(SourceLocation Loc,
                                                   unsigned DiagID) {
  assert(getLangOpts().CUDA && "Should only be called during CUDA compilation");
  DeviceDiagBuilder::Kind DiagKind = [this] {
    switch (CurrentCUDATarget()) {
    case CFT_Global:
    case CFT_Device:
      return DeviceDiagBuilder::K_Immediate;
    case CFT_HostDevice:
      // An HD function counts as host code if we're compiling for host, and
      // device code if we're compiling for device.  Defer any errors in device
      // mode until the function is known-emitted.
      if (getLangOpts().CUDAIsDevice) {
        return (getEmissionStatus(cast<FunctionDecl>(CurContext)) ==
                FunctionEmissionStatus::Emitted)
                   ? DeviceDiagBuilder::K_ImmediateWithCallStack
                   : DeviceDiagBuilder::K_Deferred;
      }
      return DeviceDiagBuilder::K_Nop;

    default:
      return DeviceDiagBuilder::K_Nop;
    }
  }();
  return DeviceDiagBuilder(DiagKind, Loc, DiagID,
                           dyn_cast<FunctionDecl>(CurContext), *this);
}

Sema::DeviceDiagBuilder Sema::CUDADiagIfHostCode(SourceLocation Loc,
                                                 unsigned DiagID) {
  assert(getLangOpts().CUDA && "Should only be called during CUDA compilation");
  DeviceDiagBuilder::Kind DiagKind = [this] {
    switch (CurrentCUDATarget()) {
    case CFT_Host:
      return DeviceDiagBuilder::K_Immediate;
    case CFT_HostDevice:
      // An HD function counts as host code if we're compiling for host, and
      // device code if we're compiling for device.  Defer any errors in device
      // mode until the function is known-emitted.
      if (getLangOpts().CUDAIsDevice)
        return DeviceDiagBuilder::K_Nop;

      return (getEmissionStatus(cast<FunctionDecl>(CurContext)) ==
              FunctionEmissionStatus::Emitted)
                 ? DeviceDiagBuilder::K_ImmediateWithCallStack
                 : DeviceDiagBuilder::K_Deferred;
    default:
      return DeviceDiagBuilder::K_Nop;
    }
  }();
  return DeviceDiagBuilder(DiagKind, Loc, DiagID,
                           dyn_cast<FunctionDecl>(CurContext), *this);
}

bool Sema::CheckCUDACall(SourceLocation Loc, FunctionDecl *Callee) {
  assert(getLangOpts().CUDA && "Should only be called during CUDA compilation");
  assert(Callee && "Callee may not be null.");

  auto &ExprEvalCtx = ExprEvalContexts.back();
  if (ExprEvalCtx.isUnevaluated() || ExprEvalCtx.isConstantEvaluated())
    return true;

  // FIXME: Is bailing out early correct here?  Should we instead assume that
  // the caller is a global initializer?
  FunctionDecl *Caller = dyn_cast<FunctionDecl>(CurContext);
  if (!Caller)
    return true;

  // If the caller is known-emitted, mark the callee as known-emitted.
  // Otherwise, mark the call in our call graph so we can traverse it later.
  bool CallerKnownEmitted =
      getEmissionStatus(Caller) == FunctionEmissionStatus::Emitted;
  DeviceDiagBuilder::Kind DiagKind = [this, Caller, Callee,
                                      CallerKnownEmitted] {
    switch (IdentifyCUDAPreference(Caller, Callee)) {
    case CFP_Never:
      return DeviceDiagBuilder::K_Immediate;
    case CFP_WrongSide:
      assert(Caller && "WrongSide calls require a non-null caller");
      // If we know the caller will be emitted, we know this wrong-side call
      // will be emitted, so it's an immediate error.  Otherwise, defer the
      // error until we know the caller is emitted.
      return CallerKnownEmitted ? DeviceDiagBuilder::K_ImmediateWithCallStack
                                : DeviceDiagBuilder::K_Deferred;
    default:
      return DeviceDiagBuilder::K_Nop;
    }
  }();

  if (DiagKind == DeviceDiagBuilder::K_Nop)
    return true;

  // Avoid emitting this error twice for the same location.  Using a hashtable
  // like this is unfortunate, but because we must continue parsing as normal
  // after encountering a deferred error, it's otherwise very tricky for us to
  // ensure that we only emit this deferred error once.
  if (!LocsWithCUDACallDiags.insert({Caller, Loc}).second)
    return true;

  DeviceDiagBuilder(DiagKind, Loc, diag::err_ref_bad_target, Caller, *this)
      << IdentifyCUDATarget(Callee) << Callee << IdentifyCUDATarget(Caller);
  DeviceDiagBuilder(DiagKind, Callee->getLocation(), diag::note_previous_decl,
                    Caller, *this)
      << Callee;
  return DiagKind != DeviceDiagBuilder::K_Immediate &&
         DiagKind != DeviceDiagBuilder::K_ImmediateWithCallStack;
}

void Sema::CUDASetLambdaAttrs(CXXMethodDecl *Method) {
  assert(getLangOpts().CUDA && "Should only be called during CUDA compilation");
  if (Method->hasAttr<CUDAHostAttr>() || Method->hasAttr<CUDADeviceAttr>())
    return;
  FunctionDecl *CurFn = dyn_cast<FunctionDecl>(CurContext);
  if (!CurFn)
    return;
  CUDAFunctionTarget Target = IdentifyCUDATarget(CurFn);
  if (Target == CFT_Global || Target == CFT_Device) {
    Method->addAttr(CUDADeviceAttr::CreateImplicit(Context));
  } else if (Target == CFT_HostDevice) {
    Method->addAttr(CUDADeviceAttr::CreateImplicit(Context));
    Method->addAttr(CUDAHostAttr::CreateImplicit(Context));
  }
}

void Sema::checkCUDATargetOverload(FunctionDecl *NewFD,
                                   const LookupResult &Previous) {
  assert(getLangOpts().CUDA && "Should only be called during CUDA compilation");
  CUDAFunctionTarget NewTarget = IdentifyCUDATarget(NewFD);
  for (NamedDecl *OldND : Previous) {
    FunctionDecl *OldFD = OldND->getAsFunction();
    if (!OldFD)
      continue;

    CUDAFunctionTarget OldTarget = IdentifyCUDATarget(OldFD);
    // Don't allow HD and global functions to overload other functions with the
    // same signature.  We allow overloading based on CUDA attributes so that
    // functions can have different implementations on the host and device, but
    // HD/global functions "exist" in some sense on both the host and device, so
    // should have the same implementation on both sides.
    if (NewTarget != OldTarget &&
        ((NewTarget == CFT_HostDevice) || (OldTarget == CFT_HostDevice) ||
         (NewTarget == CFT_Global) || (OldTarget == CFT_Global)) &&
        !IsOverload(NewFD, OldFD, /* UseMemberUsingDeclRules = */ false,
                    /* ConsiderCudaAttrs = */ false)) {
      Diag(NewFD->getLocation(), diag::err_cuda_ovl_target)
          << NewTarget << NewFD->getDeclName() << OldTarget << OldFD;
      Diag(OldFD->getLocation(), diag::note_previous_declaration);
      NewFD->setInvalidDecl();
      break;
    }
  }
}

template <typename AttrTy>
static void copyAttrIfPresent(Sema &S, FunctionDecl *FD,
                              const FunctionDecl &TemplateFD) {
  if (AttrTy *Attribute = TemplateFD.getAttr<AttrTy>()) {
    AttrTy *Clone = Attribute->clone(S.Context);
    Clone->setInherited(true);
    FD->addAttr(Clone);
  }
}

void Sema::inheritCUDATargetAttrs(FunctionDecl *FD,
                                  const FunctionTemplateDecl &TD) {
  const FunctionDecl &TemplateFD = *TD.getTemplatedDecl();
  copyAttrIfPresent<CUDAGlobalAttr>(*this, FD, TemplateFD);
  copyAttrIfPresent<CUDAHostAttr>(*this, FD, TemplateFD);
  copyAttrIfPresent<CUDADeviceAttr>(*this, FD, TemplateFD);
}

std::string Sema::getCudaConfigureFuncName() const {
  if (getLangOpts().HIP)
    return getLangOpts().HIPUseNewLaunchAPI ? "__hipPushCallConfiguration"
                                            : "hipConfigureCall";

  // New CUDA kernel launch sequence.
  if (CudaFeatureEnabled(Context.getTargetInfo().getSDKVersion(),
                         CudaFeature::CUDA_USES_NEW_LAUNCH))
    return "__cudaPushCallConfiguration";

  // Legacy CUDA kernel configuration call
  return "cudaConfigureCall";
}
