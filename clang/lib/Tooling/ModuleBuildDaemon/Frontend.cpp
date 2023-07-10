//===---------------------------- Frontend.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/ModuleBuildDaemon/Frontend.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Tooling/ModuleBuildDaemon/SocketSupport.h"
#include "clang/Tooling/ModuleBuildDaemon/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Program.h"

#include <cerrno>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <thread>

using namespace llvm;

namespace clang::tooling::cc1modbuildd {

llvm::Error attemptHandshake(raw_socket_stream &Client,
                             DiagnosticsEngine &Diag) {

  // Send HandshakeMsg to module build daemon
  HandshakeMsg Request{ActionType::HANDSHAKE, StatusType::REQUEST};
  if (llvm::Error Err = writeMsgStructToSocket(Client, Request))
    return std::move(Err);

  // Read response from module build daemon
  Expected<HandshakeMsg> MaybeResponse =
      readMsgStructFromSocket<HandshakeMsg>(Client);
  if (!MaybeResponse) {
    return std::move(MaybeResponse.takeError());
  }
  HandshakeMsg Response = std::move(*MaybeResponse);

  assert(Response.MsgAction == ActionType::HANDSHAKE &&
         "The response ActionType should only ever be HANDSHAKE");

  if (Response.MsgStatus == StatusType::SUCCESS) {
    return llvm::Error::success();
  }

  return llvm::make_error<StringError>(
      "Received handshake response 'FAILURE' from module build daemon",
      std::make_error_code(std::errc::operation_not_permitted));
}

llvm::Error spawnModuleBuildDaemon(const CompilerInvocation &Clang,
                                   const char *Argv0, DiagnosticsEngine &Diag,
                                   std::string BasePath) {

  std::vector<StringRef> Args = {Argv0, MODULE_BUILD_DAEMON_FLAG};
  if (!Clang.getFrontendOpts().ModuleBuildDaemonPath.empty())
    Args.push_back(BasePath.c_str());

  std::string ErrorBuffer;
  llvm::sys::ExecuteNoWait(Argv0, Args, std::nullopt, {}, 0, &ErrorBuffer,
                           nullptr, nullptr, /*DetachProcess*/ true);

  // llvm::sys::ExecuteNoWait can fail for a variety of reasons which can't be
  // generalized to one error code
  if (!ErrorBuffer.empty())
    return llvm::make_error<StringError>(ErrorBuffer, inconvertibleErrorCode());

  Diag.Report(diag::remark_mbd_spawn);
  return llvm::Error::success();
}

Expected<std::unique_ptr<raw_socket_stream>>
getModuleBuildDaemon(const CompilerInvocation &Clang, const char *Argv0,
                     DiagnosticsEngine &Diag, StringRef BasePath) {

  SmallString<128> SocketPath = BasePath;
  llvm::sys::path::append(SocketPath, SOCKET_FILE_NAME);

  if (llvm::sys::fs::exists(SocketPath)) {
    Expected<std::unique_ptr<raw_socket_stream>> MaybeClient =
        connectToSocket(SocketPath);
    if (MaybeClient) {
      return std::move(*MaybeClient);
    }
    consumeError(MaybeClient.takeError());
  }

  if (llvm::Error Err =
          spawnModuleBuildDaemon(Clang, Argv0, Diag, BasePath.str()))
    return std::move(Err);

  constexpr unsigned int MICROSEC_IN_SEC = 1000000;
  constexpr unsigned int MAX_WAIT_TIME = 30 * MICROSEC_IN_SEC;
  unsigned int CumulativeTime = 0;
  unsigned int WaitTime = 10;

  while (CumulativeTime <= MAX_WAIT_TIME) {
    // Wait a bit then check to see if the module build daemon has initialized
    std::this_thread::sleep_for(std::chrono::microseconds(WaitTime));

    if (llvm::sys::fs::exists(SocketPath)) {
      Expected<std::unique_ptr<raw_socket_stream>> MaybeClient =
          connectToSocket(SocketPath);
      if (MaybeClient) {
        Diag.Report(diag::remark_mbd_connection) << SocketPath;
        return std::move(*MaybeClient);
      }
      consumeError(MaybeClient.takeError());
    }

    CumulativeTime += WaitTime;
    WaitTime = WaitTime * 2;
  }

  // After waiting around 30 seconds give up and return an error
  return llvm::make_error<StringError>(
      "Max wait time exceeded: ",
      std::make_error_code(std::errc::no_such_process));
}

void spawnModuleBuildDaemonAndHandshake(const CompilerInvocation &Clang,
                                        const char *Argv0,
                                        DiagnosticsEngine &Diag) {

  // The module build daemon stores all output files and its socket address
  // under BasePath. Either set BasePath to a user provided option or create an
  // appropriate BasePath based on the BLAKE3 hash of the full clang version
  std::string BasePath;
  if (Clang.getFrontendOpts().ModuleBuildDaemonPath.empty())
    BasePath = getBasePath();
  else {
    // Get user provided BasePath and confirm it is short enough
    BasePath = Clang.getFrontendOpts().ModuleBuildDaemonPath;
    if (!validBasePathLength(BasePath)) {
      Diag.Report(diag::err_basepath_length) << BasePath << BASEPATH_MAX_LENGTH;
      return;
    }
  }

  // If module build daemon does not exist spawn module build daemon
  Expected<std::unique_ptr<raw_socket_stream>> MaybeClient =
      getModuleBuildDaemon(Clang, Argv0, Diag, BasePath);
  if (!MaybeClient) {
    Diag.Report(diag::err_mbd_connect) << MaybeClient.takeError();
    return;
  }
  raw_socket_stream &Client = **MaybeClient;
  if (llvm::Error HandshakeErr = attemptHandshake(Client, Diag)) {
    Diag.Report(diag::err_mbd_handshake) << std::move(HandshakeErr);
    return;
  }

  Diag.Report(diag::remark_mbd_handshake);
  return;
}

} // namespace clang::tooling::cc1modbuildd
