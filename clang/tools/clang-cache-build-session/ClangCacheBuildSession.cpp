//===-- ClangCacheBuildSession.cpp - clang-cache-build-session tool -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Accepts a command that triggers a build and, while the command is running,
// all the clang invocations that are part of that build will be sharing the
// same dependency scanning daemon.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Process.h"

using namespace llvm;

int main(int Argc, const char **Argv) {
  InitLLVM X(Argc, Argv);

  SmallVector<const char *, 256> CmdArgs(Argv + 1, Argv + Argc);

  if (CmdArgs.empty()) {
    errs() << "Usage: clang-cache-build-session command ...\n";
    return 1;
  }

  // Set 'CLANG_CACHE_BUILD_SESSION_ID' to a unique identifier so that clang
  // invocations under the given command share the same depscan daemon while the
  // command is running.
  // Uses the process id to ensure parallel invocations of
  // `clang-cache-build-session` will not share the same identifier, and
  // 'elapsed nanoseconds since epoch' to ensure the same for consecutive
  // invocations.
  SmallString<32> SessionId;
  raw_svector_ostream(SessionId)
      << sys::Process::getProcessId() << '-'
      << std::chrono::system_clock::now().time_since_epoch().count();
  ::setenv("CLANG_CACHE_BUILD_SESSION_ID", SessionId.c_str(), 1);

  ErrorOr<std::string> ExecPathOrErr = sys::findProgramByName(CmdArgs.front());
  if (!ExecPathOrErr) {
    errs() << "error: cannot find executable " << CmdArgs.front() << '\n';
    return 1;
  }
  std::string ExecPath = std::move(*ExecPathOrErr);
  CmdArgs[0] = ExecPath.c_str();

  SmallVector<StringRef, 16> RefArgs;
  RefArgs.reserve(CmdArgs.size());
  for (const char *Arg : CmdArgs) {
    RefArgs.push_back(Arg);
  }

  std::string ErrMsg;
  int Result = sys::ExecuteAndWait(RefArgs.front(), RefArgs, /*Env*/ None,
                                   /*Redirects*/ {}, /*SecondsToWait*/ 0,
                                   /*MemoryLimit*/ 0, &ErrMsg);
  if (!ErrMsg.empty()) {
    errs() << "error: failed executing command: " << ErrMsg << '\n';
  }
  return Result;
}
