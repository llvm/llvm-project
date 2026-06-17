//===--- TargetID.h - Utilities for target ID -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_TARGETID_H
#define LLVM_CLANG_BASIC_TARGETID_H

#include <optional>
#include <set>
#include <string>

namespace llvm {
class Triple;
}

namespace clang {

/// Get processor name from target ID.
/// Returns canonical processor name or empty if the processor name is invalid.
llvm::StringRef getProcessorFromTargetID(const llvm::Triple &T,
                                         llvm::StringRef OffloadArch);

/// Get the conflicted pair of target IDs for a compilation or a bundled code
/// object, assuming \p TargetIDs are canonicalized. If there is no conflicts,
/// returns std::nullopt.
std::optional<std::pair<llvm::StringRef, llvm::StringRef>>
getConflictTargetIDCombination(const llvm::Triple &T,
                               const std::set<llvm::StringRef> &TargetIDs);

/// Sanitize a target ID string for use in a file name.
/// Replaces invalid characters (like ':') with safe characters (like '@').
/// Currently only replaces ':' with '@' on Windows.
std::string sanitizeTargetIDInFileName(llvm::StringRef TargetID);
} // namespace clang

#endif
