//===--- SemaCUDA.cpp - Semantic Analysis for CUDA constructs -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// \brief This file implements semantic analysis for CUDA constructs.
///
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Sema/Template.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
using namespace clang;

ExprResult Sema::ActOnCUDAExecConfigExpr(Scope *S, SourceLocation LLLLoc,
                                         MultiExprArg ExecConfig,
                                         SourceLocation GGGLoc) {
  FunctionDecl *ConfigDecl = Context.getcudaConfigureCallDecl();
  if (!ConfigDecl)
    return ExprError(Diag(LLLLoc, diag::err_undeclared_var_use)
                     << "cudaConfigureCall");
  QualType ConfigQTy = ConfigDecl->getType();

  DeclRefExpr *ConfigDR = new (Context)
      DeclRefExpr(ConfigDecl, false, ConfigQTy, VK_LValue, LLLLoc);
  MarkFunctionReferenced(LLLLoc, ConfigDecl);

  return ActOnCallExpr(S, ConfigDR, LLLLoc, ExecConfig, GGGLoc, nullptr,
                       /*IsExecConfig=*/true);
}

/// IdentifyCUDATarget - Determine the CUDA compilation target for this function
Sema::CUDAFunctionTarget Sema::IdentifyCUDATarget(const FunctionDecl *D) {
  if (D->hasAttr<CUDAInvalidTargetAttr>())
    return CFT_InvalidTarget;

  if (D->hasAttr<CUDAGlobalAttr>())
    return CFT_Global;

  if (D->hasAttr<CUDADeviceAttr>()) {
    if (D->hasAttr<CUDAHostAttr>())
      return CFT_HostDevice;
    return CFT_Device;
  } else if (D->hasAttr<CUDAHostAttr>()) {
    return CFT_Host;
  } else if (D->isImplicit()) {
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
  CUDAFunctionTarget CalleeTarget = IdentifyCUDATarget(Callee);
  CUDAFunctionTarget CallerTarget =
      (Caller != nullptr) ? IdentifyCUDATarget(Caller) : Sema::CFT_Host;

  // If one of the targets is invalid, the check always fails, no matter what
  // the other target is.
  if (CallerTarget == CFT_InvalidTarget || CalleeTarget == CFT_InvalidTarget)
    return CFP_Never;

  // (a) Can't call global from some contexts until we support CUDA's
  // dynamic parallelism.
  if (CalleeTarget == CFT_Global &&
      (CallerTarget == CFT_Global || CallerTarget == CFT_Device ||
       (CallerTarget == CFT_HostDevice && getLangOpts().CUDAIsDevice)))
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

template <typename T>
static void EraseUnwantedCUDAMatchesImpl(
    Sema &S, const FunctionDecl *Caller, llvm::SmallVectorImpl<T> &Matches,
    std::function<const FunctionDecl *(const T &)> FetchDecl) {
  if (Matches.size() <= 1)
    return;

  // Gets the CUDA function preference for a call from Caller to Match.
  auto GetCFP = [&](const T &Match) {
    return S.IdentifyCUDAPreference(Caller, FetchDecl(Match));
  };

  // Find the best call preference among the functions in Matches.
  Sema::CUDAFunctionPreference BestCFP = GetCFP(*std::max_element(
      Matches.begin(), Matches.end(),
      [&](const T &M1, const T &M2) { return GetCFP(M1) < GetCFP(M2); }));

  // Erase all functions with lower priority.
  Matches.erase(
      llvm::remove_if(Matches,
                      [&](const T &Match) { return GetCFP(Match) < BestCFP; }),
      Matches.end());
}

void Sema::EraseUnwantedCUDAMatches(const FunctionDecl *Caller,
                                    SmallVectorImpl<FunctionDecl *> &Matches){
  EraseUnwantedCUDAMatchesImpl<FunctionDecl *>(
      *this, Caller, Matches, [](const FunctionDecl *item) { return item; });
}

void Sema::EraseUnwantedCUDAMatches(const FunctionDecl *Caller,
                                    SmallVectorImpl<DeclAccessPair> &Matches) {
  EraseUnwantedCUDAMatchesImpl<DeclAccessPair>(
      *this, Caller, Matches, [](const DeclAccessPair &item) {
        return dyn_cast<FunctionDecl>(item.getDecl());
      });
}

void Sema::EraseUnwantedCUDAMatches(
    const FunctionDecl *Caller,
    SmallVectorImpl<std::pair<DeclAccessPair, FunctionDecl *>> &Matches){
  EraseUnwantedCUDAMatchesImpl<std::pair<DeclAccessPair, FunctionDecl *>>(
      *this, Caller, Matches,
      [](const std::pair<DeclAccessPair, FunctionDecl *> &item) {
        return dyn_cast<FunctionDecl>(item.second);
      });
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
    Sema::SpecialMemberOverloadResult *SMOR =
        LookupSpecialMember(BaseClassDecl, CSM,
                            /* ConstArg */ ConstRHS,
                            /* VolatileArg */ false,
                            /* RValueThis */ false,
                            /* ConstThis */ false,
                            /* VolatileThis */ false);

    if (!SMOR || !SMOR->getMethod()) {
      continue;
    }

    CUDAFunctionTarget BaseMethodTarget = IdentifyCUDATarget(SMOR->getMethod());
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
    Sema::SpecialMemberOverloadResult *SMOR =
        LookupSpecialMember(FieldRecDecl, CSM,
                            /* ConstArg */ ConstRHS && !F->isMutable(),
                            /* VolatileArg */ false,
                            /* RValueThis */ false,
                            /* ConstThis */ false,
                            /* VolatileThis */ false);

    if (!SMOR || !SMOR->getMethod()) {
      continue;
    }

    CUDAFunctionTarget FieldMethodTarget =
        IdentifyCUDATarget(SMOR->getMethod());
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

  if (InferredTarget.hasValue()) {
    if (InferredTarget.getValue() == CFT_Device) {
      MemberDecl->addAttr(CUDADeviceAttr::CreateImplicit(Context));
    } else if (InferredTarget.getValue() == CFT_Host) {
      MemberDecl->addAttr(CUDAHostAttr::CreateImplicit(Context));
    } else {
      MemberDecl->addAttr(CUDADeviceAttr::CreateImplicit(Context));
      MemberDecl->addAttr(CUDAHostAttr::CreateImplicit(Context));
    }
  } else {
    // If no target was inferred, mark this member as __host__ __device__;
    // it's the least restrictive option that can be invoked from any target.
    MemberDecl->addAttr(CUDADeviceAttr::CreateImplicit(Context));
    MemberDecl->addAttr(CUDAHostAttr::CreateImplicit(Context));
  }

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

// With -fcuda-host-device-constexpr, an unattributed constexpr function is
// treated as implicitly __host__ __device__, unless:
//  * it is a variadic function (device-side variadic functions are not
//    allowed), or
//  * a __device__ function with this signature was already declared, in which
//    case in which case we output an error, unless the __device__ decl is in a
//    system header, in which case we leave the constexpr function unattributed.
void Sema::maybeAddCUDAHostDeviceAttrs(Scope *S, FunctionDecl *NewD,
                                       const LookupResult &Previous) {
  assert(getLangOpts().CUDA && "May be called only for CUDA compilations.");
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
          << NewD->getName();
      Diag(Match->getLocation(),
           diag::note_cuda_conflicting_device_function_declared_here);
    }
    return;
  }

  NewD->addAttr(CUDAHostAttr::CreateImplicit(Context));
  NewD->addAttr(CUDADeviceAttr::CreateImplicit(Context));
}

bool Sema::CheckCUDACall(SourceLocation Loc, FunctionDecl *Callee) {
  assert(getLangOpts().CUDA &&
         "Should only be called during CUDA compilation.");
  assert(Callee && "Callee may not be null.");
  FunctionDecl *Caller = dyn_cast<FunctionDecl>(CurContext);
  if (!Caller)
    return true;

  Sema::CUDAFunctionPreference Pref = IdentifyCUDAPreference(Caller, Callee);
  if (Pref == Sema::CFP_Never) {
    Diag(Loc, diag::err_ref_bad_target) << IdentifyCUDATarget(Callee) << Callee
                                        << IdentifyCUDATarget(Caller);
    Diag(Callee->getLocation(), diag::note_previous_decl) << Callee;
    return false;
  }
  if (Pref == Sema::CFP_WrongSide) {
    // We have to do this odd dance to create our PartialDiagnostic because we
    // want its storage to be allocated with operator new, not in an arena.
    PartialDiagnostic ErrPD{PartialDiagnostic::NullDiagnostic()};
    ErrPD.Reset(diag::err_ref_bad_target);
    ErrPD << IdentifyCUDATarget(Callee) << Callee << IdentifyCUDATarget(Caller);
    Caller->addDeferredDiag({Loc, std::move(ErrPD)});

    PartialDiagnostic NotePD{PartialDiagnostic::NullDiagnostic()};
    NotePD.Reset(diag::note_previous_decl);
    NotePD << Callee;
    Caller->addDeferredDiag({Callee->getLocation(), std::move(NotePD)});

    // This is not immediately an error, so return true.  The deferred errors
    // will be emitted if and when Caller is codegen'ed.
    return true;
  }
  return true;
}
