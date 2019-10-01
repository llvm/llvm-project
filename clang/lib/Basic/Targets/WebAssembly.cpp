//===--- WebAssembly.cpp - Implement WebAssembly target feature support ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements WebAssembly TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "Targets.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;
using namespace clang::targets;

const Builtin::Info WebAssemblyTargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS)                                               \
  {#ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, nullptr},
#define TARGET_BUILTIN(ID, TYPE, ATTRS, FEATURE)                               \
  {#ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, FEATURE},
#define LIBBUILTIN(ID, TYPE, ATTRS, HEADER)                                    \
  {#ID, TYPE, ATTRS, HEADER, ALL_LANGUAGES, nullptr},
#include "clang/Basic/BuiltinsWebAssembly.def"
};

static constexpr llvm::StringLiteral ValidCPUNames[] = {
    {"mvp"}, {"bleeding-edge"}, {"generic"}};

bool WebAssemblyTargetInfo::hasFeature(StringRef Feature) const {
  return llvm::StringSwitch<bool>(Feature)
      .Case("simd128", SIMDLevel >= SIMD128)
      .Case("unimplemented-simd128", SIMDLevel >= UnimplementedSIMD128)
      .Case("nontrapping-fptoint", HasNontrappingFPToInt)
      .Case("sign-ext", HasSignExt)
      .Case("exception-handling", HasExceptionHandling)
      .Case("bulk-memory", HasBulkMemory)
      .Case("atomics", HasAtomics)
      .Case("mutable-globals", HasMutableGlobals)
      .Case("multivalue", HasMultivalue)
      .Case("tail-call", HasTailCall)
      .Default(false);
}

bool WebAssemblyTargetInfo::isValidCPUName(StringRef Name) const {
  return llvm::find(ValidCPUNames, Name) != std::end(ValidCPUNames);
}

void WebAssemblyTargetInfo::fillValidCPUList(
    SmallVectorImpl<StringRef> &Values) const {
  Values.append(std::begin(ValidCPUNames), std::end(ValidCPUNames));
}

void WebAssemblyTargetInfo::getTargetDefines(const LangOptions &Opts,
                                             MacroBuilder &Builder) const {
  defineCPUMacros(Builder, "wasm", /*Tuning=*/false);
  if (SIMDLevel >= SIMD128)
    Builder.defineMacro("__wasm_simd128__");
  if (SIMDLevel >= UnimplementedSIMD128)
    Builder.defineMacro("__wasm_unimplemented_simd128__");
  if (HasNontrappingFPToInt)
    Builder.defineMacro("__wasm_nontrapping_fptoint__");
  if (HasSignExt)
    Builder.defineMacro("__wasm_sign_ext__");
  if (HasExceptionHandling)
    Builder.defineMacro("__wasm_exception_handling__");
  if (HasBulkMemory)
    Builder.defineMacro("__wasm_bulk_memory__");
  if (HasAtomics)
    Builder.defineMacro("__wasm_atomics__");
  if (HasMutableGlobals)
    Builder.defineMacro("__wasm_mutable_globals__");
  if (HasMultivalue)
    Builder.defineMacro("__wasm_multivalue__");
  if (HasTailCall)
    Builder.defineMacro("__wasm_tail_call__");
}

void WebAssemblyTargetInfo::setSIMDLevel(llvm::StringMap<bool> &Features,
                                         SIMDEnum Level) {
  switch (Level) {
  case UnimplementedSIMD128:
    Features["unimplemented-simd128"] = true;
    LLVM_FALLTHROUGH;
  case SIMD128:
    Features["simd128"] = true;
    LLVM_FALLTHROUGH;
  case NoSIMD:
    break;
  }
}

bool WebAssemblyTargetInfo::initFeatureMap(
    llvm::StringMap<bool> &Features, DiagnosticsEngine &Diags, StringRef CPU,
    const std::vector<std::string> &FeaturesVec) const {
  if (CPU == "bleeding-edge") {
    Features["nontrapping-fptoint"] = true;
    Features["sign-ext"] = true;
    Features["atomics"] = true;
    Features["mutable-globals"] = true;
    setSIMDLevel(Features, SIMD128);
  }
  // Other targets do not consider user-configured features here, but while we
  // are actively developing new features it is useful to let user-configured
  // features control availability of builtins
  setSIMDLevel(Features, SIMDLevel);
  if (HasNontrappingFPToInt)
    Features["nontrapping-fptoint"] = true;
  if (HasSignExt)
    Features["sign-ext"] = true;
  if (HasExceptionHandling)
    Features["exception-handling"] = true;
  if (HasBulkMemory)
    Features["bulk-memory"] = true;
  if (HasAtomics)
    Features["atomics"] = true;
  if (HasMutableGlobals)
    Features["mutable-globals"] = true;
  if (HasMultivalue)
    Features["multivalue"] = true;
  if (HasTailCall)
    Features["tail-call"] = true;

  return TargetInfo::initFeatureMap(Features, Diags, CPU, FeaturesVec);
}

bool WebAssemblyTargetInfo::handleTargetFeatures(
    std::vector<std::string> &Features, DiagnosticsEngine &Diags) {
  for (const auto &Feature : Features) {
    if (Feature == "+simd128") {
      SIMDLevel = std::max(SIMDLevel, SIMD128);
      continue;
    }
    if (Feature == "-simd128") {
      SIMDLevel = std::min(SIMDLevel, SIMDEnum(SIMD128 - 1));
      continue;
    }
    if (Feature == "+unimplemented-simd128") {
      SIMDLevel = std::max(SIMDLevel, SIMDEnum(UnimplementedSIMD128));
      continue;
    }
    if (Feature == "-unimplemented-simd128") {
      SIMDLevel = std::min(SIMDLevel, SIMDEnum(UnimplementedSIMD128 - 1));
      continue;
    }
    if (Feature == "+nontrapping-fptoint") {
      HasNontrappingFPToInt = true;
      continue;
    }
    if (Feature == "-nontrapping-fptoint") {
      HasNontrappingFPToInt = false;
      continue;
    }
    if (Feature == "+sign-ext") {
      HasSignExt = true;
      continue;
    }
    if (Feature == "-sign-ext") {
      HasSignExt = false;
      continue;
    }
    if (Feature == "+exception-handling") {
      HasExceptionHandling = true;
      continue;
    }
    if (Feature == "-exception-handling") {
      HasExceptionHandling = false;
      continue;
    }
    if (Feature == "+bulk-memory") {
      HasBulkMemory = true;
      continue;
    }
    if (Feature == "-bulk-memory") {
      HasBulkMemory = false;
      continue;
    }
    if (Feature == "+atomics") {
      HasAtomics = true;
      continue;
    }
    if (Feature == "-atomics") {
      HasAtomics = false;
      continue;
    }
    if (Feature == "+mutable-globals") {
      HasMutableGlobals = true;
      continue;
    }
    if (Feature == "-mutable-globals") {
      HasMutableGlobals = false;
      continue;
    }
    if (Feature == "+multivalue") {
      HasMultivalue = true;
      continue;
    }
    if (Feature == "-multivalue") {
      HasMultivalue = false;
      continue;
    }
    if (Feature == "+tail-call") {
      HasTailCall = true;
      continue;
    }
    if (Feature == "-tail-call") {
      HasTailCall = false;
      continue;
    }

    Diags.Report(diag::err_opt_not_valid_with_opt)
        << Feature << "-target-feature";
    return false;
  }
  return true;
}

ArrayRef<Builtin::Info> WebAssemblyTargetInfo::getTargetBuiltins() const {
  return llvm::makeArrayRef(BuiltinInfo, clang::WebAssembly::LastTSBuiltin -
                                             Builtin::FirstTSBuiltin);
}

void WebAssembly32TargetInfo::getTargetDefines(const LangOptions &Opts,
                                               MacroBuilder &Builder) const {
  WebAssemblyTargetInfo::getTargetDefines(Opts, Builder);
  defineCPUMacros(Builder, "wasm32", /*Tuning=*/false);
}

void WebAssembly64TargetInfo::getTargetDefines(const LangOptions &Opts,
                                               MacroBuilder &Builder) const {
  WebAssemblyTargetInfo::getTargetDefines(Opts, Builder);
  defineCPUMacros(Builder, "wasm64", /*Tuning=*/false);
}
