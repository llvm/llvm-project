//===--- CommonUtils.h - Common utilities for the toolchains ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_COMMONUTILS_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_COMMONUTILS_H

#include "clang/Driver/ToolChain.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Option/ArgList.h"

namespace clang {

// Add backslashes to escape spaces and other backslashes.
// This is used for the space-separated argument list specified with
// the -dwarf-debug-flags option.
void EscapeSpacesAndBackslashes(const char *Arg,
                                llvm::SmallVectorImpl<char> &Res);

/// Join the args in the given ArgList, escape spaces and backslashes and
/// return the joined string. This is used when saving the command line as a
/// result of using either the -frecord-command-line or -grecord-command-line
/// options. The lifetime of the returned c-string will match that of the Args
/// argument.
const char *RenderEscapedCommandLine(const clang::driver::ToolChain &TC,
                                     const llvm::opt::ArgList &Args);

/// Check if the command line should be recorded in the object file. This is
/// done if either -frecord-command-line or -grecord-command-line options have
/// been passed. This also does some error checking since -frecord-command-line
/// is currently only supported on ELF platforms. The last two boolean
/// arguments are out parameters and will be set depending on the command
/// line options that were passed.
bool ShouldRecordCommandLine(const clang::driver::ToolChain &TC,
                             const llvm::opt::ArgList &Args,
                             bool &FRecordCommandLine,
                             bool &GRecordCommandLine);
} // namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_COMMONUTILS_H
