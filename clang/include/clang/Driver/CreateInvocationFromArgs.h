//===--- CreateInvocationFromArgs.h - CompilerInvocation from Args --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility for creating a CompilerInvocation from command-line arguments, for
// tools to use in preparation to parse a file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_CREATEINVOCATIONFROMARGS_H
#define LLVM_CLANG_DRIVER_CREATEINVOCATIONFROMARGS_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LLVM.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <memory>
#include <string>
#include <vector>

namespace clang {

class CompilerInvocation;
class DiagnosticsEngine;

/// Optional inputs to createInvocation.
struct CreateInvocationOptions {
  /// Receives diagnostics encountered while parsing command-line flags.
  /// If not provided, these are printed to stderr.
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags = nullptr;
  /// Used e.g. to probe for system headers locations.
  /// If not provided, the real filesystem is used.
  /// FIXME: the driver does perform some non-virtualized IO.
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS = nullptr;
  /// Whether to attempt to produce a non-null (possibly incorrect) invocation
  /// if any errors were encountered.
  /// By default, always return null on errors.
  bool RecoverOnError = false;
  /// Allow the driver to probe the filesystem for PCH files.
  /// This is used to replace -include with -include-pch in the cc1 args.
  /// FIXME: ProbePrecompiled=true is a poor, historical default.
  /// It misbehaves if the PCH file is from GCC, has the wrong version, etc.
  bool ProbePrecompiled = false;
  /// If set, the target is populated with the cc1 args produced by the driver.
  /// This may be populated even if createInvocation returns nullptr.
  std::vector<std::string> *CC1Args = nullptr;
};

/// Interpret clang arguments in preparation to parse a file.
///
/// This simulates a number of steps Clang takes when its driver is invoked:
/// - choosing actions (e.g compile + link) to run
/// - probing the system for settings like standard library locations
/// - spawning a cc1 subprocess to compile code, with more explicit arguments
/// - in the cc1 process, assembling those arguments into a CompilerInvocation
///   which is used to configure the parser
///
/// This simulation is lossy, e.g. in some situations one driver run would
/// result in multiple parses. (Multi-arch, CUDA, ...).
/// This function tries to select a reasonable invocation that tools should use.
///
/// Args[0] should be the driver name, such as "clang" or "/usr/bin/g++".
/// Absolute path is preferred - this affects searching for system headers.
///
/// May return nullptr if an invocation could not be determined.
/// See CreateInvocationOptions::RecoverOnError to try harder!
std::unique_ptr<CompilerInvocation>
createInvocation(ArrayRef<const char *> Args,
                 CreateInvocationOptions Opts = {});

} // namespace clang

#endif // LLVM_CLANG_DRIVER_CREATEINVOCATIONFROMARGS_H
