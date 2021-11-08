//===--- SPIR.cpp - Implement SPIR and SPIR-V target feature support ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements SPIR and SPIR-V TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "SPIR.h"
#include "Targets.h"

using namespace clang;
using namespace clang::targets;

void SPIRTargetInfo::getTargetDefines(const LangOptions &Opts,
                                      MacroBuilder &Builder) const {
  DefineStd(Builder, "SPIR", Opts);
}

void SPIR32TargetInfo::getTargetDefines(const LangOptions &Opts,
                                        MacroBuilder &Builder) const {
  SPIRTargetInfo::getTargetDefines(Opts, Builder);
  DefineStd(Builder, "SPIR32", Opts);
}

void SPIR64TargetInfo::getTargetDefines(const LangOptions &Opts,
                                        MacroBuilder &Builder) const {
  SPIRTargetInfo::getTargetDefines(Opts, Builder);
  DefineStd(Builder, "SPIR64", Opts);
}

void SPIRVTargetInfo::getTargetDefines(const LangOptions &Opts,
                                       MacroBuilder &Builder) const {
  DefineStd(Builder, "SPIRV", Opts);
}

void SPIRV32TargetInfo::getTargetDefines(const LangOptions &Opts,
                                         MacroBuilder &Builder) const {
  SPIRVTargetInfo::getTargetDefines(Opts, Builder);
  DefineStd(Builder, "SPIRV32", Opts);
}

void SPIRV64TargetInfo::getTargetDefines(const LangOptions &Opts,
                                         MacroBuilder &Builder) const {
  SPIRVTargetInfo::getTargetDefines(Opts, Builder);
  DefineStd(Builder, "SPIRV64", Opts);
}
