//===------- TargetDefines.h - Target define helpers ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a series of helper functions for defining target-specific
// macros.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_BASIC_TARGETDEFINES_H
#define LLVM_CLANG_LIB_BASIC_TARGETDEFINES_H

#include "clang/Basic/LangOptions.h"
#include "clang/Basic/MacroBuilder.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace targets {
/// Define a macro name and standard variants.  For example if MacroName is
/// "unix", then this will define "__unix", "__unix__", and "unix" when in GNU
/// mode.
LLVM_LIBRARY_VISIBILITY
void DefineStd(clang::MacroBuilder &Builder, llvm::StringRef MacroName,
               const clang::LangOptions &Opts);

LLVM_LIBRARY_VISIBILITY
void defineCPUMacros(clang::MacroBuilder &Builder, llvm::StringRef CPUName,
                     bool Tuning = true);

LLVM_LIBRARY_VISIBILITY
void addCygMingDefines(const clang::LangOptions &Opts,
                       clang::MacroBuilder &Builder);
} // namespace targets
} // namespace clang
#endif // LLVM_CLANG_LIB_BASIC_TARGETDEFINES_H
