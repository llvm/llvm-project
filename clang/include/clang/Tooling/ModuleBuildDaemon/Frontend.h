//===----------------------------- Frontend.h -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_FRONTEND_H
#define LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_FRONTEND_H

#include "clang/Frontend/CompilerInstance.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_socket_stream.h"

namespace clang::tooling::cc1modbuildd {

llvm::Error attemptHandshake(llvm::raw_socket_stream &Client);

llvm::Error spawnModuleBuildDaemon(const clang::CompilerInvocation &Clang,
                                   const char *Argv0,
                                   clang::DiagnosticsEngine &Diag,
                                   std::string BasePath);

llvm::Expected<std::unique_ptr<llvm::raw_socket_stream>>
getModuleBuildDaemon(const clang::CompilerInvocation &Clang, const char *Argv0,
                     clang::DiagnosticsEngine &Diag, llvm::StringRef BasePath);

// Sends request to module build daemon
llvm::Error registerTranslationUnit(llvm::ArrayRef<const char *> CC1Cmd,
                                    llvm::StringRef Argv0, llvm::StringRef CWD,
                                    llvm::raw_socket_stream &Client);

// Processes response from module build daemon
llvm::Expected<std::vector<std::string>>
getUpdatedCC1(llvm::raw_socket_stream &Client);

// Work in progress. Eventually function will modify CC1 command line to include
// path to modules already built by the daemon
llvm::Expected<std::vector<std::string>> updateCC1WithModuleBuildDaemon(
    const clang::CompilerInvocation &Clang, llvm::ArrayRef<const char *> CC1Cmd,
    const char *Argv0, llvm::StringRef CWD, clang::DiagnosticsEngine &Diag);

} // namespace clang::tooling::cc1modbuildd

#endif // LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_FRONTEND_H
