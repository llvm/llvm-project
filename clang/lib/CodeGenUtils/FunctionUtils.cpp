//===--- FunctionUtils.cpp - Shared function emission queries -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGenUtils/FunctionUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

namespace clang::CodeGenUtils {

bool shouldEmitLifetimeMarkers(const CodeGenOptions &CGOpts,
                               const LangOptions &LangOpts) {
  if (CGOpts.DisableLifetimeMarkers)
    return false;

  // Sanitizers may use markers.
  if (CGOpts.SanitizeAddressUseAfterScope ||
      LangOpts.Sanitize.has(SanitizerKind::HWAddress) ||
      LangOpts.Sanitize.has(SanitizerKind::Memory) ||
      LangOpts.Sanitize.has(SanitizerKind::MemtagStack))
    return true;

  // For now, only in optimized builds.
  return CGOpts.OptimizationLevel != 0;
}

// Emits an error if we don't have a valid set of target features for the
// called function.
void checkTargetFeatures(ASTContext &Ctx, DiagnosticsEngine &Diags,
                         const LangOptions &LangOpts, const CallExpr *E,
                         const FunctionDecl *Caller,
                         const FunctionDecl *TargetDecl,
                         InheritedTargetFeatures *Inherited) {
  // SemaChecking cannot handle these x86 builtins because they have different
  // parameter ranges depending on the caller's TargetAttribute.
  if (Ctx.getTargetInfo().getTriple().isX86()) {
    unsigned BuiltinID = TargetDecl->getBuiltinID();
    if (BuiltinID == X86::BI__builtin_ia32_cmpps ||
        BuiltinID == X86::BI__builtin_ia32_cmpss ||
        BuiltinID == X86::BI__builtin_ia32_cmppd ||
        BuiltinID == X86::BI__builtin_ia32_cmpsd) {
      llvm::StringMap<bool> TargetFeatureMap;
      Ctx.getFunctionFeatureMap(TargetFeatureMap, Caller);
      llvm::APSInt Result = *(E->getArg(2)->getIntegerConstantExpr(Ctx));
      if (Result.getSExtValue() > 7 && !TargetFeatureMap.lookup("avx"))
        Diags.Report(E->getBeginLoc(), diag::err_builtin_needs_feature)
            << TargetDecl->getDeclName() << "avx";
    }
  }
  checkTargetFeatures(Ctx, Diags, LangOpts, E->getBeginLoc(), Caller,
                      TargetDecl, Inherited);
}

// Returns the target features that the code of TargetDecl relies on and that
// Caller lacks.
static SmallVector<std::string, 1>
getMissingFeatures(ASTContext &Ctx, const FunctionDecl *Caller,
                   const FunctionDecl *TargetDecl) {
  SmallVector<std::string, 1> Missing;
  llvm::StringMap<bool> CallerFeatureMap;
  llvm::StringMap<bool> CalleeFeatureMap;
  if (!TargetDecl->isMultiVersion() && TargetDecl->hasAttr<TargetAttr>()) {
    Ctx.getFunctionFeatureMap(CallerFeatureMap, Caller);
    Ctx.getFunctionFeatureMap(CalleeFeatureMap, TargetDecl);

    // Get the required features for the callee.
    const TargetAttr *TD = TargetDecl->getAttr<TargetAttr>();
    ParsedTargetAttr ParsedAttr = Ctx.filterFunctionTargetAttrs(TD);

    SmallVector<StringRef, 1> ReqFeatures;
    for (const auto &F : ParsedAttr.Features) {
      if (F[0] == '+' && CalleeFeatureMap.lookup(F.substr(1)))
        ReqFeatures.push_back(StringRef(F).substr(1));
    }
    for (const auto &F : CalleeFeatureMap) {
      if (F.getValue())
        ReqFeatures.push_back(F.getKey());
    }
    auto It = llvm::find_if(ReqFeatures, [&](StringRef Feature) {
      return !CallerFeatureMap.lookup(Feature);
    });
    if (It != ReqFeatures.end())
      Missing.push_back(It->str());
  } else if (!Caller->isMultiVersion() && Caller->hasAttr<TargetAttr>()) {
    Ctx.getFunctionFeatureMap(CallerFeatureMap, Caller);
    Ctx.getFunctionFeatureMap(CalleeFeatureMap, TargetDecl);

    for (const auto &F : CalleeFeatureMap) {
      if (F.getValue() && !CallerFeatureMap.lookup(F.getKey()))
        Missing.push_back(F.getKey().str());
    }
  }
  return Missing;
}

static void reportMissingFeatures(DiagnosticsEngine &Diags, unsigned DiagID,
                                  SourceLocation Loc,
                                  const FunctionDecl *Caller,
                                  const FunctionDecl *TargetDecl,
                                  ArrayRef<std::string> Missing) {
  for (const std::string &Feature : Missing)
    Diags.Report(Loc, DiagID)
        << Caller->getDeclName() << TargetDecl->getDeclName() << Feature;
}

bool InheritedTargetFeatures::canInherit(ASTContext &Ctx,
                                         const FunctionDecl *FD) {
  // A function that is emitted even when unused has callers that are not
  // known here, which could not provide the features that it would inherit.
  return FD->hasAttr<AlwaysInlineAttr>() && !FD->isMultiVersion() &&
         !Ctx.DeclMustBeEmitted(FD);
}

bool InheritedTargetFeatures::contains(const FunctionDecl *FD) const {
  return Inherited.contains(FD->getCanonicalDecl());
}

void InheritedTargetFeatures::addTo(ASTContext &Ctx, const FunctionDecl *FD,
                                    llvm::StringMap<bool> &FeatureMap) const {
  auto Calls = Inherited.find(FD->getCanonicalDecl());
  if (Calls == Inherited.end())
    return;

  for (const InlinedCall &Call : Calls->second) {
    llvm::StringMap<bool> CalleeFeatureMap;
    Ctx.getFunctionFeatureMap(CalleeFeatureMap, Call.Callee);
    for (const auto &F : CalleeFeatureMap) {
      if (F.getValue())
        FeatureMap[F.getKey()] = true;
    }
  }
}

void InheritedTargetFeatures::addCall(ASTContext &Ctx, DiagnosticsEngine &Diags,
                                      SourceLocation Loc,
                                      const FunctionDecl *Caller,
                                      const FunctionDecl *Callee) {
  require(Ctx, Diags, Caller, Loc, {Callee, {Caller, Loc}});

  // This iterates over a copy, because require() can add to Inherited.
  for (const InlinedCall &Call : Inherited.lookup(Callee->getCanonicalDecl()))
    require(Ctx, Diags, Caller, Loc, Call);

  CallsTo[Callee->getCanonicalDecl()].push_back({Caller, Loc});
}

// Makes Host, which gets the code of the callee of Call through its call at
// Loc, provide the target features that this code relies on.
void InheritedTargetFeatures::require(ASTContext &Ctx, DiagnosticsEngine &Diags,
                                      const FunctionDecl *Host,
                                      SourceLocation Loc, InlinedCall Call) {
  SmallVector<std::string, 1> Missing =
      getMissingFeatures(Ctx, Host, Call.Callee);
  if (Missing.empty())
    return;

  if (!canInherit(Ctx, Host)) {
    reportMissingFeatures(Diags, diag::err_function_needs_feature, Loc, Host,
                          Call.Callee, Missing);
    if (Call.Site.Caller != Host)
      Diags.Report(Call.Site.Loc, diag::note_function_needs_feature_call)
          << Call.Site.Caller->getDeclName() << Call.Callee->getDeclName();
    return;
  }

  SmallVectorImpl<InlinedCall> &Calls = Inherited[Host->getCanonicalDecl()];
  // This also ends the recursion below for functions that call each other.
  if (llvm::any_of(Calls, [&](const InlinedCall &Other) {
        return declaresSameEntity(Other.Callee, Call.Callee);
      }))
    return;
  Calls.push_back(Call);

  auto Callers = CallsTo.find(Host->getCanonicalDecl());
  if (Callers == CallsTo.end())
    return;
  for (const CallSite &Site : Callers->second)
    require(Ctx, Diags, Site.Caller, Site.Loc, Call);
}

// Emits an error if we don't have a valid set of target features for the
// called function.
void checkTargetFeatures(ASTContext &Ctx, DiagnosticsEngine &Diags,
                         const LangOptions &LangOpts, SourceLocation Loc,
                         const FunctionDecl *Caller,
                         const FunctionDecl *TargetDecl,
                         InheritedTargetFeatures *Inherited) {
  if (!TargetDecl || !Caller)
    return;

  // When compiling in HipStdPar mode we have to be conservative in rejecting
  // target specific features in the FE, and defer the possible error to the
  // AcceleratorCodeSelection pass, wherein iff an unsupported target builtin is
  // referenced by an accelerator executable function, we emit an error.
  if (LangOpts.HIPStdPar && LangOpts.CUDAIsDevice)
    return;

  if (unsigned BuiltinID = TargetDecl->getBuiltinID()) {
    StringRef FeatureList(Ctx.BuiltinInfo.getRequiredFeatures(BuiltinID));
    if (FeatureList.empty())
      return;

    llvm::StringMap<bool> CallerFeatureMap;
    Ctx.getFunctionFeatureMap(CallerFeatureMap, Caller);
    if (!Builtin::evaluateRequiredTargetFeatures(FeatureList, CallerFeatureMap))
      Diags.Report(Loc, diag::err_builtin_needs_feature)
          << TargetDecl->getDeclName() << FeatureList;
  } else if (TargetDecl->hasAttr<AlwaysInlineAttr>() && Inherited) {
    Inherited->addCall(Ctx, Diags, Loc, Caller, TargetDecl);
  } else if (TargetDecl->hasAttr<AlwaysInlineAttr>()) {
    reportMissingFeatures(Diags, diag::err_function_needs_feature, Loc, Caller,
                          TargetDecl,
                          getMissingFeatures(Ctx, Caller, TargetDecl));
  } else if (Caller->hasAttr<FlattenAttr>()) {
    reportMissingFeatures(Diags, diag::err_flatten_function_needs_feature, Loc,
                          Caller, TargetDecl,
                          getMissingFeatures(Ctx, Caller, TargetDecl));
  }
}

} // namespace clang::CodeGenUtils
