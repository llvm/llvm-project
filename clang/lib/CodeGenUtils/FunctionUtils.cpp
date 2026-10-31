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
                         const FunctionDecl *TargetDecl) {
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
                      TargetDecl);
}

// Emits an error if we don't have a valid set of target features for the
// called function.
void checkTargetFeatures(ASTContext &Ctx, DiagnosticsEngine &Diags,
                         const LangOptions &LangOpts, SourceLocation Loc,
                         const FunctionDecl *Caller,
                         const FunctionDecl *TargetDecl) {
  if (!TargetDecl || !Caller)
    return;

  bool IsAlwaysInline = TargetDecl->hasAttr<AlwaysInlineAttr>();
  bool IsFlatten = Caller->hasAttr<FlattenAttr>();

  unsigned BuiltinID = TargetDecl->getBuiltinID();
  std::string MissingFeature;
  llvm::StringMap<bool> CallerFeatureMap;
  Ctx.getFunctionFeatureMap(CallerFeatureMap, Caller);
  // When compiling in HipStdPar mode we have to be conservative in rejecting
  // target specific features in the FE, and defer the possible error to the
  // AcceleratorCodeSelection pass, wherein iff an unsupported target builtin is
  // referenced by an accelerator executable function, we emit an error.
  bool IsHipStdPar = LangOpts.HIPStdPar && LangOpts.CUDAIsDevice;
  if (BuiltinID) {
    StringRef FeatureList(Ctx.BuiltinInfo.getRequiredFeatures(BuiltinID));
    if (!Builtin::evaluateRequiredTargetFeatures(FeatureList,
                                                 CallerFeatureMap) &&
        !IsHipStdPar)
      Diags.Report(Loc, diag::err_builtin_needs_feature)
          << TargetDecl->getDeclName() << FeatureList;
  } else if (!TargetDecl->isMultiVersion() &&
             TargetDecl->hasAttr<TargetAttr>()) {
    // Get the required features for the callee.
    const TargetAttr *TD = TargetDecl->getAttr<TargetAttr>();
    ParsedTargetAttr ParsedAttr = Ctx.filterFunctionTargetAttrs(TD);

    SmallVector<StringRef, 1> ReqFeatures;
    llvm::StringMap<bool> CalleeFeatureMap;
    Ctx.getFunctionFeatureMap(CalleeFeatureMap, TargetDecl);

    for (const auto &F : ParsedAttr.Features) {
      if (F[0] == '+' && CalleeFeatureMap.lookup(F.substr(1)))
        ReqFeatures.push_back(StringRef(F).substr(1));
    }
    for (const auto &F : CalleeFeatureMap) {
      if (F.getValue())
        ReqFeatures.push_back(F.getKey());
    }
    if (!llvm::all_of(ReqFeatures,
                      [&](StringRef Feature) {
                        if (!CallerFeatureMap.lookup(Feature)) {
                          MissingFeature = Feature.str();
                          return false;
                        }
                        return true;
                      }) &&
        !IsHipStdPar) {
      if (IsAlwaysInline)
        Diags.Report(Loc, diag::err_function_needs_feature)
            << Caller->getDeclName() << TargetDecl->getDeclName()
            << MissingFeature;
      else if (IsFlatten)
        Diags.Report(Loc, diag::err_flatten_function_needs_feature)
            << Caller->getDeclName() << TargetDecl->getDeclName()
            << MissingFeature;
    }
  } else if (!Caller->isMultiVersion() && Caller->hasAttr<TargetAttr>()) {
    llvm::StringMap<bool> CalleeFeatureMap;
    Ctx.getFunctionFeatureMap(CalleeFeatureMap, TargetDecl);

    for (const auto &F : CalleeFeatureMap) {
      if (F.getValue() &&
          (!CallerFeatureMap.lookup(F.getKey()) ||
           !CallerFeatureMap.find(F.getKey())->getValue()) &&
          !IsHipStdPar) {
        if (IsAlwaysInline)
          Diags.Report(Loc, diag::err_function_needs_feature)
              << Caller->getDeclName() << TargetDecl->getDeclName()
              << F.getKey();
        else if (IsFlatten)
          Diags.Report(Loc, diag::err_flatten_function_needs_feature)
              << Caller->getDeclName() << TargetDecl->getDeclName()
              << F.getKey();
      }
    }
  }
}

} // namespace clang::CodeGenUtils
