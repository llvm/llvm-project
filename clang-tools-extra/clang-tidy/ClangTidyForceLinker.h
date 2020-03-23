//===- ClangTidyForceLinker.h - clang-tidy --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYFORCELINKER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYFORCELINKER_H

#include "clang/Config/config.h"
#include "llvm/Support/Compiler.h"

namespace clang {
namespace tidy {

// This anchor is used to force the linker to link the AbseilModule.
extern volatile int AbseilModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED AbseilModuleAnchorDestination =
    AbseilModuleAnchorSource;

// This anchor is used to force the linker to link the AndroidModule.
extern volatile int AndroidModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED AndroidModuleAnchorDestination =
    AndroidModuleAnchorSource;

// This anchor is used to force the linker to link the BoostModule.
extern volatile int BoostModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED BoostModuleAnchorDestination =
    BoostModuleAnchorSource;

// This anchor is used to force the linker to link the BugproneModule.
extern volatile int BugproneModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED BugproneModuleAnchorDestination =
    BugproneModuleAnchorSource;

// This anchor is used to force the linker to link the CERTModule.
extern volatile int CERTModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED CERTModuleAnchorDestination =
    CERTModuleAnchorSource;

// This anchor is used to force the linker to link the CppCoreGuidelinesModule.
extern volatile int CppCoreGuidelinesModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED CppCoreGuidelinesModuleAnchorDestination =
    CppCoreGuidelinesModuleAnchorSource;

// This anchor is used to force the linker to link the DarwinModule.
extern volatile int DarwinModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED DarwinModuleAnchorDestination =
    DarwinModuleAnchorSource;

// This anchor is used to force the linker to link the FuchsiaModule.
extern volatile int FuchsiaModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED FuchsiaModuleAnchorDestination =
    FuchsiaModuleAnchorSource;

// This anchor is used to force the linker to link the GoogleModule.
extern volatile int GoogleModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED GoogleModuleAnchorDestination =
    GoogleModuleAnchorSource;

// This anchor is used to force the linker to link the HICPPModule.
extern volatile int HICPPModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED HICPPModuleAnchorDestination =
    HICPPModuleAnchorSource;

// This anchor is used to force the linker to link the LinuxKernelModule.
extern volatile int LinuxKernelModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED LinuxKernelModuleAnchorDestination =
    LinuxKernelModuleAnchorSource;

// This anchor is used to force the linker to link the LLVMModule.
extern volatile int LLVMModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED LLVMModuleAnchorDestination =
    LLVMModuleAnchorSource;

// This anchor is used to force the linker to link the LLVMLibcModule.
extern volatile int LLVMLibcModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED LLVMLibcModuleAnchorDestination =
    LLVMLibcModuleAnchorSource;

// This anchor is used to force the linker to link the MiscModule.
extern volatile int MiscModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED MiscModuleAnchorDestination =
    MiscModuleAnchorSource;

// This anchor is used to force the linker to link the ModernizeModule.
extern volatile int ModernizeModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED ModernizeModuleAnchorDestination =
    ModernizeModuleAnchorSource;

#if CLANG_ENABLE_STATIC_ANALYZER &&                                            \
    !defined(CLANG_TIDY_DISABLE_STATIC_ANALYZER_CHECKS)
// This anchor is used to force the linker to link the MPIModule.
extern volatile int MPIModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED MPIModuleAnchorDestination =
    MPIModuleAnchorSource;
#endif

// This anchor is used to force the linker to link the ObjCModule.
extern volatile int ObjCModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED ObjCModuleAnchorDestination =
    ObjCModuleAnchorSource;

// This anchor is used to force the linker to link the OpenMPModule.
extern volatile int OpenMPModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED OpenMPModuleAnchorDestination =
    OpenMPModuleAnchorSource;

// This anchor is used to force the linker to link the PerformanceModule.
extern volatile int PerformanceModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED PerformanceModuleAnchorDestination =
    PerformanceModuleAnchorSource;

// This anchor is used to force the linker to link the PortabilityModule.
extern volatile int PortabilityModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED PortabilityModuleAnchorDestination =
    PortabilityModuleAnchorSource;

// This anchor is used to force the linker to link the ReadabilityModule.
extern volatile int ReadabilityModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED ReadabilityModuleAnchorDestination =
    ReadabilityModuleAnchorSource;

// This anchor is used to force the linker to link the ZirconModule.
extern volatile int ZirconModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED ZirconModuleAnchorDestination =
    ZirconModuleAnchorSource;

} // namespace tidy
} // namespace clang

#endif
