//===---------------------------- Frontend.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/ModuleBuildDaemon/Frontend.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Tooling/ModuleBuildDaemon/SocketSupport.h"
#include "clang/Tooling/ModuleBuildDaemon/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/ExponentialBackoff.h"
#include "llvm/Support/Program.h"

#include <cerrno>
#include <filesystem>
#include <fstream>
#include <signal.h>
#include <spawn.h>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

using namespace clang;
using namespace llvm;

namespace clang::tooling::cc1modbuildd {

llvm::Error attemptHandshake(llvm::raw_socket_stream &Client) {
  // Send HandshakeMsg to module build daemon
  HandshakeMsg Request{ActionType::HANDSHAKE, StatusType::REQUEST};
  if (llvm::Error Err = writeMsgStructToSocket(Client, Request))
    return Err;

  // Read response from module build daemon
  Expected<HandshakeMsg> MaybeResponse =
      readMsgStructFromSocket<HandshakeMsg>(Client);
  if (!MaybeResponse) {
    return MaybeResponse.takeError();
  }
  HandshakeMsg Response = std::move(*MaybeResponse);

  assert(Response.MsgAction == ActionType::HANDSHAKE &&
         "The response ActionType should only ever be HANDSHAKE");

  if (Response.MsgStatus == StatusType::SUCCESS) {
    return llvm::Error::success();
  }

  return llvm::make_error<llvm::StringError>(
      "Received handshake response 'FAILURE' from module build daemon",
      std::make_error_code(std::errc::operation_not_permitted));
}

llvm::Error spawnModuleBuildDaemon(const CompilerInvocation &Clang,
                                   const char *Argv0, DiagnosticsEngine &Diag,
                                   std::string BasePath) {
  std::vector<StringRef> Args = {Argv0, ModuleBuildDaemonFlag};
  if (!Clang.getFrontendOpts().ModuleBuildDaemonPath.empty())
    Args.push_back(BasePath.c_str());

  std::string ErrorBuffer;
  llvm::sys::ExecuteNoWait(Argv0, Args, std::nullopt, {}, 0, &ErrorBuffer,
                           nullptr, nullptr, /*DetachProcess*/ true);

  // llvm::sys::ExecuteNoWait can fail for a variety of reasons which can't be
  // generalized to one error code
  if (!ErrorBuffer.empty())
    return llvm::make_error<llvm::StringError>(ErrorBuffer,
                                               llvm::inconvertibleErrorCode());

  Diag.Report(diag::remark_mbd_spawn);
  return llvm::Error::success();
}

Expected<std::unique_ptr<llvm::raw_socket_stream>>
getModuleBuildDaemon(const CompilerInvocation &Clang, const char *Argv0,
                     DiagnosticsEngine &Diag, StringRef BasePath) {
  SmallString<128> SocketPath = BasePath;
  llvm::sys::path::append(SocketPath, SocketFileName);

  if (llvm::sys::fs::exists(SocketPath)) {
    Expected<std::unique_ptr<llvm::raw_socket_stream>> MaybeClient =
        llvm::raw_socket_stream::createConnectedUnix(SocketPath);
    if (MaybeClient)
      return std::move(*MaybeClient);
    consumeError(MaybeClient.takeError());
  }

  if (llvm::Error Err =
          spawnModuleBuildDaemon(Clang, Argv0, Diag, BasePath.str()))
    return std::move(Err);

  std::chrono::seconds MaxWaitTime(30);
  llvm::ExponentialBackoff Backoff(MaxWaitTime);
  do {
    if (llvm::sys::fs::exists(SocketPath)) {
      Expected<std::unique_ptr<llvm::raw_socket_stream>> MaybeClient =
          llvm::raw_socket_stream::createConnectedUnix(SocketPath);
      if (MaybeClient) {
        Diag.Report(diag::remark_mbd_connection) << SocketPath;
        return std::move(*MaybeClient);
      }
      consumeError(MaybeClient.takeError());
    }
  } while (Backoff.waitForNextAttempt());

  // After waiting around 30 seconds give up and return an error
  return llvm::make_error<llvm::StringError>(
      "Max wait time exceeded",
      std::make_error_code(std::errc::no_such_process));
}

llvm::Error registerTranslationUnit(ArrayRef<const char *> CC1Command,
                                    StringRef Argv0, StringRef CWD,
                                    llvm::raw_socket_stream &Client) {

  std::vector<std::string> StrCC1Command;
  for (const char *Arg : CC1Command)
    StrCC1Command.emplace_back(Arg);

  cc1modbuildd::RegisterMsg Request{ActionType::REGISTER, StatusType::REQUEST,
                                    CWD.str(), Argv0.str(), StrCC1Command};

  llvm::Error WriteErr = writeMsgStructToSocket(Client, Request);
  if (WriteErr)
    return std::move(WriteErr);

  return llvm::Error::success();
}

Expected<std::vector<std::string>>
getUpdatedCC1(llvm::raw_socket_stream &Server) {

  // Blocks cc1 invocation until module build daemon is done processing
  // translation unit. Currently receives a SUCCESS message and returns
  // llvm::Error::success() but will eventually recive updated cc1 command line
  Expected<RegisterMsg> MaybeServerResponse =
      readMsgStructFromSocket<RegisterMsg>(Server);
  if (!MaybeServerResponse)
    return std::move(MaybeServerResponse.takeError());
  RegisterMsg ServerResponse = std::move(*MaybeServerResponse);

  // Confirm response is REGISTER and MsgStatus is SUCCESS
  assert(ServerResponse.MsgAction == ActionType::REGISTER &&
         "At this point response ActionType should only ever be REGISTER");

  if (ServerResponse.MsgStatus == StatusType::SUCCESS)
    return ServerResponse.CC1CommandLine.value();

  return llvm::make_error<StringError>(
      "Daemon failed to processes registered translation unit",
      inconvertibleErrorCode());
}

Expected<std::vector<std::string>>
updateCC1WithModuleBuildDaemon(const CompilerInvocation &Clang,
                               ArrayRef<const char *> CC1Cmd, const char *Argv0,
                               StringRef CWD, DiagnosticsEngine &Diag) {
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
      Diag.Report(diag::err_path_length) << BasePath << BasePathMaxLength;
      return make_error<StringError>(inconvertibleErrorCode(),
                                     "BasePath is too long");
    }
  }

  // If module build daemon does not exist spawn module build daemon
  Expected<std::unique_ptr<llvm::raw_socket_stream>> MaybeClient =
      getModuleBuildDaemon(Clang, Argv0, Diag, BasePath);
  if (!MaybeClient) {
    Diag.Report(diag::err_mbd_connect) << MaybeClient.takeError();
    return make_error<StringError>(inconvertibleErrorCode(),
                                   "Could not connect to ModuleBuildDaemon");
  }
  llvm::raw_socket_stream &Client = **MaybeClient;

  if (llvm::Error HandshakeErr = attemptHandshake(Client)) {
    Diag.Report(diag::err_mbd_handshake) << std::move(HandshakeErr);
    return makeStringError(std::move(HandshakeErr),
                           "Failed to hadshake with daemon");
  }

  Diag.Report(diag::remark_mbd_handshake)
      << Clang.getFrontendOpts().Inputs[0].getFile();

  // Send translation unit information to module build daemon for processing
  if (llvm::Error RegisterErr =
          registerTranslationUnit(CC1Cmd, Argv0, CWD, Client))
    return makeStringError(std::move(RegisterErr),
                           "Register translation unti failed");

  // Wait for response from module build daemon. Response will hopefully be an
  // updated cc1 command line with additional -fmodule-file=<file> flags and
  // implicit module flags removed
  Expected<std::vector<std::string>> MaybeUpdatedCC1 = getUpdatedCC1(Client);
  if (!MaybeUpdatedCC1)
    return makeStringError(MaybeUpdatedCC1.takeError(),
                           "Failed to get updated CC1");
  return std::move(*MaybeUpdatedCC1);
}

} // namespace clang::tooling::cc1modbuildd
