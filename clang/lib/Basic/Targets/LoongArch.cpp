//===--- LoongArch.cpp - Implement LoongArch target feature support -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements LoongArch TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "LoongArch.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/MacroBuilder.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/Support/TargetParser.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::targets;

ArrayRef<const char *> LoongArchTargetInfo::getGCCRegNames() const {
  // TODO: To be implemented in future.
  return {};
}

ArrayRef<TargetInfo::GCCRegAlias>
LoongArchTargetInfo::getGCCRegAliases() const {
  // TODO: To be implemented in future.
  return {};
}

bool LoongArchTargetInfo::validateAsmConstraint(
    const char *&Name, TargetInfo::ConstraintInfo &Info) const {
  // TODO: To be implemented in future.
  return false;
}

void LoongArchTargetInfo::getTargetDefines(const LangOptions &Opts,
                                           MacroBuilder &Builder) const {
  Builder.defineMacro("__loongarch__");
  // TODO: Define more macros.
}

ArrayRef<Builtin::Info> LoongArchTargetInfo::getTargetBuiltins() const {
  // TODO: To be implemented in future.
  return {};
}
