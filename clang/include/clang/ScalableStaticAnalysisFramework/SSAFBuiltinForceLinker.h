//===- SSAFBuiltinForceLinker.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file pulls in all built-in SSAF extractor and format registrations
/// by referencing their anchor symbols, preventing the static linker from
/// discarding the containing object files.
///
/// Include this header (with IWYU pragma: keep) in any translation unit that
/// must guarantee these registrations are active — typically the entry point
/// of a binary that uses clangScalableStaticAnalysisFrameworkCore.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_SSAFBUILTINFORCELINKER_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_SSAFBUILTINFORCELINKER_H

// TODO: Move these to the `clang::ssaf` namespace.

// This anchor is used to force the linker to link the JSONFormat registration.
extern volatile int SSAFJSONFormatAnchorSource;
[[maybe_unused]] static int SSAFJSONFormatAnchorDestination =
    SSAFJSONFormatAnchorSource;

// This anchor is used to force the linker to link the AnalysisRegistry.
extern volatile int SSAFAnalysisRegistryAnchorSource;
[[maybe_unused]] static int SSAFAnalysisRegistryAnchorDestination =
    SSAFAnalysisRegistryAnchorSource;

extern volatile int UnsafeBufferUsageSSAFJSONFormatAnchorSource;
[[maybe_unused]] static int UnsafeBufferUsageSSAFJSONFormatAnchorDestination =
    UnsafeBufferUsageSSAFJSONFormatAnchorSource;

// This anchor is used to force the linker to link the CallGraphExtractor.
extern volatile int CallGraphExtractorAnchorSource;
[[maybe_unused]] static int CallGraphExtractorAnchorDestination =
    CallGraphExtractorAnchorSource;

// This anchor is used to force the linker to link the CallGraph JSON format.
extern volatile int CallGraphJSONFormatAnchorSource;
[[maybe_unused]] static int CallGraphJSONFormatAnchorDestination =
    CallGraphJSONFormatAnchorSource;

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_SSAFBUILTINFORCELINKER_H
