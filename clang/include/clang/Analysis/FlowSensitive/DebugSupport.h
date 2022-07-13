//===-- DebugSupport.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines functions which generate more readable forms of data
//  structures used in the dataflow analyses, for debugging purposes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_DEBUGSUPPORT_H_
#define LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_DEBUGSUPPORT_H_

#include <string>

#include "clang/Analysis/FlowSensitive/Value.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {
namespace dataflow {
/// Returns a string representation for the boolean value `B`.
///
/// Atomic booleans appearing in the boolean value `B` are assigned to labels
/// either specified in `AtomNames` or created by default rules as B0, B1, ...
///
/// Requirements:
///
///   Names assigned to atoms should not be repeated in `AtomNames`.
std::string debugString(
    const BoolValue &B,
    llvm::DenseMap<const AtomicBoolValue *, std::string> AtomNames = {{}});

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_DEBUGSUPPORT_H_
