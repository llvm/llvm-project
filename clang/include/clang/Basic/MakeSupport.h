//===- MakeSupport.h - Make Utilities ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_MAKESUPPORT_H
#define LLVM_CLANG_BASIC_MAKESUPPORT_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace clang {

/// Quote target names for inclusion in GNU Make dependency files.
/// Only the characters '$', '#', ' ', '\t' are quoted.
void quoteMakeTarget(StringRef Target, SmallVectorImpl<char> &Res);

/// DependencyOutputFormat - Format for the compiler dependency file.
enum class DependencyOutputFormat { Make, NMake };

/// Write Make-style dependency output to the output stream, in the form:
///   target1 target2 ...: prereq1 prereq2 ...
///
/// \param Targets The targets, already quoted for Make via quoteMakeTarget().
/// \param Files The prerequisites; each is escaped for \p Format when written.
/// \param Format Escape prerequisites for GNU Make or NMake.
/// \param PhonyTargets If true, also emit an empty "prereq:" line for each
///   prerequisite (except \p InputFileIndex), so later deleting a prerequisite
///   doesn't break the build.
/// \param InputFileIndex Index in \p Files of the main input, skipped for
///   phony target emission.
void printMakeDependencyFile(
    llvm::raw_ostream &OS, llvm::ArrayRef<std::string> Targets,
    llvm::ArrayRef<std::string> Files,
    DependencyOutputFormat Format = DependencyOutputFormat::Make,
    bool PhonyTargets = false, unsigned InputFileIndex = 0);

} // namespace clang

#endif // LLVM_CLANG_BASIC_MAKESUPPORT_H
