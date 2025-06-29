//===------------------------------ Utils.h -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functions required by both the frontend and the module build daemon
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_UTILS_H
#define LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_UTILS_H

#include "llvm/Support/Error.h"

#include <chrono>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
// winsock2.h must be included before afunix.h
// clang-format off
#include <winsock2.h>
#include <afunix.h>
// clang-format on
#else
#include <sys/un.h>
#endif

namespace clang::tooling::cc1modbuildd {

constexpr std::string_view SocketFileName = "mbd.sock";
constexpr std::string_view StdoutFileName = "mbd.out";
constexpr std::string_view StderrFileName = "mbd.err";
constexpr std::string_view ModuleBuildDaemonFlag = "-cc1modbuildd";

// A llvm::raw_socket_stream uses sockaddr_un
constexpr size_t SocketAddrMaxLength = sizeof(sockaddr_un::sun_path);

constexpr size_t BasePathMaxLength =
    SocketAddrMaxLength - SocketFileName.length();

// Get a temprary location where the daemon can store log files and a socket
// address. Of the format /tmp/clang-<BLAKE3HashOfClangFullVersion>/
std::string getBasePath();

// Check if the user provided BasePath is short enough
bool validBasePathLength(llvm::StringRef);

} // namespace clang::tooling::cc1modbuildd

#endif
