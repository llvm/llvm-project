//===- ClangTidyForceLinker.h - clang-tidy --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYFORCELINKER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYFORCELINKER_H

#include "clang-tidy-config.h"
#include "llvm/Support/Compiler.h"

namespace clang::tidy {

// This anchor is used to force the linker to link the AbseilModule.
extern volatile int AbseilModuleAnchorSource;
[[maybe_unused]] static int AbseilModuleAnchorDestination =
    AbseilModuleAnchorSource;

// This anchor is used to force the linker to link the AlteraModule.
extern volatile int AlteraModuleAnchorSource;
[[maybe_unused]] static int AlteraModuleAnchorDestination =
    AlteraModuleAnchorSource;

// This anchor is used to force the linker to link the AndroidModule.
extern volatile int AndroidModuleAnchorSource;
[[maybe_unused]] static int AndroidModuleAnchorDestination =
    AndroidModuleAnchorSource;

// This anchor is used to force the linker to link the BoostModule.
extern volatile int BoostModuleAnchorSource;
[[maybe_unused]] static int BoostModuleAnchorDestination =
    BoostModuleAnchorSource;

// This anchor is used to force the linker to link the BugproneModule.
extern volatile int BugproneModuleAnchorSource;
[[maybe_unused]] static int BugproneModuleAnchorDestination =
    BugproneModuleAnchorSource;

// This anchor is used to force the linker to link the CERTModule.
extern volatile int CERTModuleAnchorSource;
[[maybe_unused]] static int CERTModuleAnchorDestination =
    CERTModuleAnchorSource;

// This anchor is used to force the linker to link the ConcurrencyModule.
extern volatile int ConcurrencyModuleAnchorSource;
[[maybe_unused]] static int ConcurrencyModuleAnchorDestination =
    ConcurrencyModuleAnchorSource;

// This anchor is used to force the linker to link the CppCoreGuidelinesModule.
extern volatile int CppCoreGuidelinesModuleAnchorSource;
[[maybe_unused]] static int CppCoreGuidelinesModuleAnchorDestination =
    CppCoreGuidelinesModuleAnchorSource;

#if CLANG_TIDY_ENABLE_QUERY_BASED_CUSTOM_CHECKS
// This anchor is used to force the linker to link the CustomModule.
extern volatile int CustomModuleAnchorSource;
[[maybe_unused]] static int CustomModuleAnchorDestination =
    CustomModuleAnchorSource;
#endif

// This anchor is used to force the linker to link the DarwinModule.
extern volatile int DarwinModuleAnchorSource;
[[maybe_unused]] static int DarwinModuleAnchorDestination =
    DarwinModuleAnchorSource;

// This anchor is used to force the linker to link the FuchsiaModule.
extern volatile int FuchsiaModuleAnchorSource;
[[maybe_unused]] static int FuchsiaModuleAnchorDestination =
    FuchsiaModuleAnchorSource;

// This anchor is used to force the linker to link the GoogleModule.
extern volatile int GoogleModuleAnchorSource;
[[maybe_unused]] static int GoogleModuleAnchorDestination =
    GoogleModuleAnchorSource;

// This anchor is used to force the linker to link the HICPPModule.
extern volatile int HICPPModuleAnchorSource;
[[maybe_unused]] static int HICPPModuleAnchorDestination =
    HICPPModuleAnchorSource;

// This anchor is used to force the linker to link the LinuxKernelModule.
extern volatile int LinuxKernelModuleAnchorSource;
[[maybe_unused]] static int LinuxKernelModuleAnchorDestination =
    LinuxKernelModuleAnchorSource;

// This anchor is used to force the linker to link the LLVMModule.
extern volatile int LLVMModuleAnchorSource;
[[maybe_unused]] static int LLVMModuleAnchorDestination =
    LLVMModuleAnchorSource;

// This anchor is used to force the linker to link the LLVMLibcModule.
extern volatile int LLVMLibcModuleAnchorSource;
[[maybe_unused]] static int LLVMLibcModuleAnchorDestination =
    LLVMLibcModuleAnchorSource;

// This anchor is used to force the linker to link the MiscModule.
extern volatile int MiscModuleAnchorSource;
[[maybe_unused]] static int MiscModuleAnchorDestination =
    MiscModuleAnchorSource;

// This anchor is used to force the linker to link the ModernizeModule.
extern volatile int ModernizeModuleAnchorSource;
[[maybe_unused]] static int ModernizeModuleAnchorDestination =
    ModernizeModuleAnchorSource;

#if CLANG_TIDY_ENABLE_STATIC_ANALYZER &&                                       \
    !defined(CLANG_TIDY_DISABLE_STATIC_ANALYZER_CHECKS)
// This anchor is used to force the linker to link the MPIModule.
extern volatile int MPIModuleAnchorSource;
[[maybe_unused]] static int MPIModuleAnchorDestination = MPIModuleAnchorSource;
#endif

// This anchor is used to force the linker to link the ObjCModule.
extern volatile int ObjCModuleAnchorSource;
[[maybe_unused]] static int ObjCModuleAnchorDestination =
    ObjCModuleAnchorSource;

// This anchor is used to force the linker to link the OpenMPModule.
extern volatile int OpenMPModuleAnchorSource;
[[maybe_unused]] static int OpenMPModuleAnchorDestination =
    OpenMPModuleAnchorSource;

// This anchor is used to force the linker to link the PerformanceModule.
extern volatile int PerformanceModuleAnchorSource;
[[maybe_unused]] static int PerformanceModuleAnchorDestination =
    PerformanceModuleAnchorSource;

// This anchor is used to force the linker to link the PortabilityModule.
extern volatile int PortabilityModuleAnchorSource;
[[maybe_unused]] static int PortabilityModuleAnchorDestination =
    PortabilityModuleAnchorSource;

// This anchor is used to force the linker to link the ReadabilityModule.
extern volatile int ReadabilityModuleAnchorSource;
[[maybe_unused]] static int ReadabilityModuleAnchorDestination =
    ReadabilityModuleAnchorSource;

// This anchor is used to force the linker to link the ZirconModule.
extern volatile int ZirconModuleAnchorSource;
[[maybe_unused]] static int ZirconModuleAnchorDestination =
    ZirconModuleAnchorSource;

} // namespace clang::tidy

#endif
