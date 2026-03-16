//===- SSAFForceLinker.h ----------------------------------------*- C++ -*-===//
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

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_SSAFFORCELINKER_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_SSAFFORCELINKER_H

#include "SSAFBuiltinForceLinker.h" // IWYU pragma: keep

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_SSAFFORCELINKER_H
