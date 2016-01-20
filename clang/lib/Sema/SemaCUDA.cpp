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

#include "clang/Sema/Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/SemaDiagnostic.h"
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
// Preferences: b-best, f-fallback, l-last resort, n-never.
//
// | F  | T  | Ph | Pd |  H  |
// |----+----+----+----+-----+
// | d  | d  | b  | b  | (b) |
// | d  | g  | n  | n  | (a) |
// | d  | h  | l  | l  | (e) |
// | d  | hd | f  | f  | (c) |
// | g  | d  | b  | b  | (b) |
// | g  | g  | n  | n  | (a) |
// | g  | h  | l  | l  | (e) |
// | g  | hd | f  | f  | (c) |
// | h  | d  | l  | l  | (e) |
// | h  | g  | b  | b  | (b) |
// | h  | h  | b  | b  | (b) |
// | h  | hd | f  | f  | (c) |
// | hd | d  | l  | f  | (d) |
// | hd | g  | f  | n  |(d/a)|
// | hd | h  | f  | l  | (d) |
// | hd | hd | b  | b  | (b) |

Sema::CUDAFunctionPreference
Sema::IdentifyCUDAPreference(const FunctionDecl *Caller,
                             const FunctionDecl *Callee) {
  assert(getLangOpts().CUDATargetOverloads &&
         "Should not be called w/o enabled target overloads.");

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

  // (b) Best case scenarios
  if (CalleeTarget == CallerTarget ||
      (CallerTarget == CFT_Host && CalleeTarget == CFT_Global) ||
      (CallerTarget == CFT_Global && CalleeTarget == CFT_Device))
    return CFP_Best;

  // (c) Calling HostDevice is OK as a fallback that works for everyone.
  if (CalleeTarget == CFT_HostDevice)
    return CFP_Fallback;

  // Figure out what should be returned 'last resort' cases. Normally
  // those would not be allowed, but we'll consider them if
  // CUDADisableTargetCallChecks is true.
  CUDAFunctionPreference QuestionableResult =
      getLangOpts().CUDADisableTargetCallChecks ? CFP_LastResort : CFP_Never;

  // (d) HostDevice behavior depends on compilation mode.
  if (CallerTarget == CFT_HostDevice) {
    // Calling a function that matches compilation mode is OK.
    // Calling a function from the other side is frowned upon.
    if (getLangOpts().CUDAIsDevice)
      return CalleeTarget == CFT_Device ? CFP_Fallback : QuestionableResult;
    else
      return (CalleeTarget == CFT_Host || CalleeTarget == CFT_Global)
                 ? CFP_Fallback
                 : QuestionableResult;
  }

  // (e) Calling across device/host boundary is not something you should do.
  if ((CallerTarget == CFT_Host && CalleeTarget == CFT_Device) ||
      (CallerTarget == CFT_Device && CalleeTarget == CFT_Host) ||
      (CallerTarget == CFT_Global && CalleeTarget == CFT_Host))
    return QuestionableResult;

  llvm_unreachable("All cases should've been handled by now.");
}

bool Sema::CheckCUDATarget(const FunctionDecl *Caller,
                           const FunctionDecl *Callee) {
  // With target overloads enabled, we only disallow calling
  // combinations with CFP_Never.
  if (getLangOpts().CUDATargetOverloads)
    return IdentifyCUDAPreference(Caller,Callee) == CFP_Never;

  // The CUDADisableTargetCallChecks short-circuits this check: we assume all
  // cross-target calls are valid.
  if (getLangOpts().CUDADisableTargetCallChecks)
    return false;

  CUDAFunctionTarget CallerTarget = IdentifyCUDATarget(Caller),
                     CalleeTarget = IdentifyCUDATarget(Callee);

  // If one of the targets is invalid, the check always fails, no matter what
  // the other target is.
  if (CallerTarget == CFT_InvalidTarget || CalleeTarget == CFT_InvalidTarget)
    return true;

  // CUDA B.1.1 "The __device__ qualifier declares a function that is [...]
  // Callable from the device only."
  if (CallerTarget == CFT_Host && CalleeTarget == CFT_Device)
    return true;

  // CUDA B.1.2 "The __global__ qualifier declares a function that is [...]
  // Callable from the host only."
  // CUDA B.1.3 "The __host__ qualifier declares a function that is [...]
  // Callable from the host only."
  if ((CallerTarget == CFT_Device || CallerTarget == CFT_Global) &&
      (CalleeTarget == CFT_Host || CalleeTarget == CFT_Global))
    return true;

  // CUDA B.1.3 "The __device__ and __host__ qualifiers can be used together
  // however, in which case the function is compiled for both the host and the
  // device. The __CUDA_ARCH__ macro [...] can be used to differentiate code
  // paths between host and device."
  if (CallerTarget == CFT_HostDevice && CalleeTarget != CFT_HostDevice) {
    // If the caller is implicit then the check always passes.
    if (Caller->isImplicit()) return false;

    bool InDeviceMode = getLangOpts().CUDAIsDevice;
    if (!InDeviceMode && CalleeTarget != CFT_Host)
        return true;
    if (InDeviceMode && CalleeTarget != CFT_Device) {
      // Allow host device functions to call host functions if explicitly
      // requested.
      if (CalleeTarget == CFT_Host &&
          getLangOpts().CUDAAllowHostCallsFromHostDevice) {
        Diag(Caller->getLocation(),
             diag::warn_host_calls_from_host_device)
            << Callee->getNameAsString() << Caller->getNameAsString();
        return false;
      }

      return true;
    }
  }

  return false;
}

template <typename T, typename FetchDeclFn>
static void EraseUnwantedCUDAMatchesImpl(Sema &S, const FunctionDecl *Caller,
                                         llvm::SmallVectorImpl<T> &Matches,
                                         FetchDeclFn FetchDecl) {
  assert(S.getLangOpts().CUDATargetOverloads &&
         "Should not be called w/o enabled target overloads.");
  if (Matches.size() <= 1)
    return;

  // Find the best call preference among the functions in Matches.
  Sema::CUDAFunctionPreference P, BestCFP = Sema::CFP_Never;
  for (auto const &Match : Matches) {
    P = S.IdentifyCUDAPreference(Caller, FetchDecl(Match));
    if (P > BestCFP)
      BestCFP = P;
  }

  // Erase all functions with lower priority.
  for (unsigned I = 0, N = Matches.size(); I != N;)
    if (S.IdentifyCUDAPreference(Caller, FetchDecl(Matches[I])) < BestCFP) {
      Matches[I] = Matches[--N];
      Matches.resize(N);
    } else {
      ++I;
    }
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
