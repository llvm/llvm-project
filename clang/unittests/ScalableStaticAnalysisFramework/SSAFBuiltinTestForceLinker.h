//===- SSAFBuiltinTestForceLinker.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file pulls in all test-only SSAF mock extractor and format
/// registrations by referencing their anchor symbols.
///
/// Include this header (with IWYU pragma: keep) in a translation unit that
/// is compiled into the SSAF unittest binary.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_SCALABLESTATICANALYSISFRAMEWORK_SSAFBUILTINTESTFORCELINKER_H
#define LLVM_CLANG_UNITTESTS_SCALABLESTATICANALYSISFRAMEWORK_SSAFBUILTINTESTFORCELINKER_H

// Force the linker to link NoOpExtractor registration.
extern volatile int SSAFNoOpExtractorAnchorSource;
[[maybe_unused]] static int SSAFNoOpExtractorAnchorDestination =
    SSAFNoOpExtractorAnchorSource;

// Force the linker to link MockSummaryExtractor1 registration.
extern volatile int SSAFMockSummaryExtractor1AnchorSource;
[[maybe_unused]] static int SSAFMockSummaryExtractor1AnchorDestination =
    SSAFMockSummaryExtractor1AnchorSource;

// Force the linker to link MockSummaryExtractor2 registration.
extern volatile int SSAFMockSummaryExtractor2AnchorSource;
[[maybe_unused]] static int SSAFMockSummaryExtractor2AnchorDestination =
    SSAFMockSummaryExtractor2AnchorSource;

// Force the linker to link FailingSerializationFormat registration.
extern volatile int SSAFFailingSerializationFormatAnchorSource;
[[maybe_unused]] static int SSAFFailingSerializationFormatAnchorDestination =
    SSAFFailingSerializationFormatAnchorSource;

// Force the linker to link MockSerializationFormat registration.
extern volatile int SSAFMockSerializationFormatAnchorSource;
[[maybe_unused]] static int SSAFMockSerializationFormatAnchorDestination =
    SSAFMockSerializationFormatAnchorSource;

// Force the linker to link FancyAnalysisData format info registration.
extern volatile int SSAFFancyAnalysisDataAnchorSource;
[[maybe_unused]] static int SSAFFancyAnalysisDataAnchorDestination =
    SSAFFancyAnalysisDataAnchorSource;

#endif // LLVM_CLANG_UNITTESTS_SCALABLESTATICANALYSISFRAMEWORK_SSAFBUILTINTESTFORCELINKER_H
