//===----------------------------- Protocol.h -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_CLIENT_H
#define LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_CLIENT_H

#include "clang/Frontend/CompilerInstance.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"

#define MAX_BUFFER 4096
#define SOCKET_FILE_NAME "mbd.sock"
#define STDOUT_FILE_NAME "mbd.out"
#define STDERR_FILE_NAME "mbd.err"

using namespace clang;
using namespace llvm;

namespace cc1modbuildd {

// Returns where to store log files and socket address. Of the format
// /tmp/clang-<BLAKE3HashOfClagnFullVersion>/
std::string getBasePath();

llvm::Error attemptHandshake(int SocketFD);

llvm::Error spawnModuleBuildDaemon(StringRef BasePath, const char *Argv0);

Expected<int> getModuleBuildDaemon(const char *Argv0, StringRef BasePath);

// Sends request to module build daemon
llvm::Error registerTranslationUnit(ArrayRef<const char *> CC1Cmd,
                                    StringRef Argv0, StringRef CWD,
                                    int ServerFD);

// Processes response from module build daemon
Expected<std::vector<std::string>> getUpdatedCC1(int ServerFD);

// Work in progress. Eventually function will modify CC1 command line to include
// path to modules already built by the daemon
Expected<std::vector<std::string>>
updateCC1WithModuleBuildDaemon(const CompilerInvocation &Clang,
                               ArrayRef<const char *> CC1Cmd, const char *Argv0,
                               StringRef CWD);

} // namespace cc1modbuildd

#endif // LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_PROTOCAL_H
