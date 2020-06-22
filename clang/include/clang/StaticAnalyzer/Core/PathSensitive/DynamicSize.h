//===- DynamicSize.h - Dynamic size related APIs ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines APIs that track and query dynamic size information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_DYNAMICSIZE_H
#define LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_DYNAMICSIZE_H

#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState_Fwd.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SValBuilder.h"

namespace clang {
namespace ento {

/// Get the stored dynamic size for the region \p MR.
DefinedOrUnknownSVal getDynamicSize(ProgramStateRef State, const MemRegion *MR,
                                    SValBuilder &SVB);

/// Get the stored element count of the region \p MR.
DefinedOrUnknownSVal getDynamicElementCount(ProgramStateRef State,
                                            const MemRegion *MR,
                                            SValBuilder &SVB,
                                            QualType ElementTy);

/// Get the dynamic size for a symbolic value that represents a buffer. If
/// there is an offsetting to the underlying buffer we consider that too.
/// Returns with an SVal that represents the size, this is Unknown if the
/// engine cannot deduce the size.
/// E.g.
///   char buf[3];
///   (buf); // size is 3
///   (buf + 1); // size is 2
///   (buf + 3); // size is 0
///   (buf + 4); // size is -1
///
///   char *bufptr;
///   (bufptr) // size is unknown
SVal getDynamicSizeWithOffset(ProgramStateRef State, const SVal &BufV);

} // namespace ento
} // namespace clang

#endif // LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_DYNAMICSIZE_H
