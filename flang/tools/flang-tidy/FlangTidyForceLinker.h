//===--- FlangTidyForceLinker.h - flang-tidy --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_TOOLS_FLANG_TIDY_FLANGTIDYFORCELINKER_H
#define LLVM_FLANG_TOOLS_FLANG_TIDY_FLANGTIDYFORCELINKER_H

#include "llvm/Support/Compiler.h"

namespace Fortran::tidy {

// This anchor is used to force the linker to link the BugproneModule.
extern volatile int BugproneModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED BugproneModuleAnchorDestination =
    BugproneModuleAnchorSource;

// This anchor is used to force the linker to link the ModernizeModule.
extern volatile int ModernizeModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED ModernizeModuleAnchorDestination =
    ModernizeModuleAnchorSource;

// This anchor is used to force the linker to link the OpenMPModule.
extern volatile int OpenMPModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED OpenMPModuleAnchorDestination =
    OpenMPModuleAnchorSource;

// This anchor is used to force the linker to link the PerformanceModule.
extern volatile int PerformanceModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED PerformanceModuleAnchorDestination =
    PerformanceModuleAnchorSource;

// This anchor is used to force the linker to link the ReadabilityModule.
extern volatile int ReadabilityModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED ReadabilityModuleAnchorDestination =
    ReadabilityModuleAnchorSource;

} // namespace Fortran::tidy

#endif // LLVM_FLANG_TOOLS_FLANG_TIDY_FLANGTIDYFORCELINKER_H
