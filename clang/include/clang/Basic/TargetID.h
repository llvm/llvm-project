//===--- TargetID.h - Utilities for target ID -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_TARGETID_H
#define LLVM_CLANG_BASIC_TARGETID_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/TargetParser/Triple.h"
#include <optional>
#include <string>
#include <utility>

namespace clang {

/// Get processor name from target ID.
/// Returns canonical processor name or empty if the processor name is invalid.
llvm::StringRef getProcessorFromTargetID(const llvm::Triple &T,
                                         llvm::StringRef OffloadArch);

/// A device triple paired with a target ID (processor and feature modifiers)
/// for that triple, e.g. {amdgcn-amd-amdhsa, "gfx906:xnack+"}.
using TargetIDEntry = std::pair<const llvm::Triple &, llvm::StringRef>;

/// Get the conflicted pair of target IDs for a compilation or a bundled code
/// object. Two entries conflict when they resolve to the same processor but
/// disagree on whether a feature (xnack/sramecc) is explicitly specified. If
/// there is no conflict, returns std::nullopt.
std::optional<std::pair<llvm::StringRef, llvm::StringRef>>
getConflictTargetIDCombination(llvm::ArrayRef<TargetIDEntry> Entries);

/// Sanitize a target ID string for use in a file name.
/// Replaces invalid characters (like ':') with safe characters (like '@').
/// Currently only replaces ':' with '@' on Windows.
std::string sanitizeTargetIDInFileName(llvm::StringRef TargetID);
} // namespace clang

#endif
