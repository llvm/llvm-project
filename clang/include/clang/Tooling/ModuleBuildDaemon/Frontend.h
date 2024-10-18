//===----------------------------- Frontend.h -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_FRONTEND_H
#define LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_FRONTEND_H

#include "clang/Frontend/CompilerInvocation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_socket_stream.h"

namespace clang::tooling::cc1modbuildd {

llvm::Error attemptHandshake(llvm::raw_socket_stream &Client,
                             clang::DiagnosticsEngine &Diag);

llvm::Error spawnModuleBuildDaemon(const clang::CompilerInvocation &Clang,
                                   const char *Argv0,
                                   clang::DiagnosticsEngine &Diag,
                                   std::string BasePath);

llvm::Expected<std::unique_ptr<llvm::raw_socket_stream>>
getModuleBuildDaemon(const clang::CompilerInvocation &Clang, const char *Argv0,
                     clang::DiagnosticsEngine &Diag, llvm::StringRef BasePath);

void spawnModuleBuildDaemonAndHandshake(const clang::CompilerInvocation &Clang,
                                        const char *Argv0,
                                        clang::DiagnosticsEngine &Diag);

} // namespace clang::tooling::cc1modbuildd

#endif // LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_CLIENT_H
