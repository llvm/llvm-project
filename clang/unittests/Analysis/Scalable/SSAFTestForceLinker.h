//===- SSAFTestForceLinker.h ------------------------------------*- C++ -*-===//
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

#ifndef LLVM_CLANG_UNITTESTS_ANALYSIS_SCALABLE_SSAFTESTFORCELINKER_H
#define LLVM_CLANG_UNITTESTS_ANALYSIS_SCALABLE_SSAFTESTFORCELINKER_H

#include "SSAFBuiltinTestForceLinker.h" // IWYU pragma: keep

#endif // LLVM_CLANG_UNITTESTS_ANALYSIS_SCALABLE_SSAFTESTFORCELINKER_H
