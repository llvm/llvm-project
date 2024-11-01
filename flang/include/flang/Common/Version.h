//===- Version.h - Flang Version Number ---------------------*- Fortran -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines version macros and version-related utility functions
/// for Flang.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_COMMON_VERSION_H
#define LLVM_FLANG_COMMON_VERSION_H

#include "flang/Version.inc"
#include "llvm/ADT/StringRef.h"

namespace Fortran::common {
/// Retrieves the repository path (e.g., Git path) that
/// identifies the particular Flang branch, tag, or trunk from which this
/// Flang was built.
std::string getFlangRepositoryPath();

/// Retrieves the repository path from which LLVM was built.
///
/// This supports LLVM residing in a separate repository from flang.
std::string getLLVMRepositoryPath();

/// Retrieves the repository revision number (or identifier) from which
/// this Flang was built.
std::string getFlangRevision();

/// Retrieves the repository revision number (or identifier) from which
/// LLVM was built.
///
/// If Flang and LLVM are in the same repository, this returns the same
/// string as getFlangRevision.
std::string getLLVMRevision();

/// Retrieves the full repository version that is an amalgamation of
/// the information in getFlangRepositoryPath() and getFlangRevision().
std::string getFlangFullRepositoryVersion();

/// Retrieves a string representing the complete flang version,
/// which includes the flang version number, the repository version,
/// and the vendor tag.
std::string getFlangFullVersion();

/// Like getFlangFullVersion(), but with a custom tool name.
std::string getFlangToolFullVersion(llvm::StringRef ToolName);
} // namespace Fortran::common

#endif // LLVM_FLANG_COMMON_VERSION_H
