//===- FuzzerUtilLinux.cpp - Misc utils for Linux. ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Misc utils for Linux.
//===----------------------------------------------------------------------===//
#include "FuzzerDefs.h"
#if LIBFUZZER_LINUX || LIBFUZZER_NETBSD || LIBFUZZER_FREEBSD ||                \
    LIBFUZZER_OPENBSD
#include "FuzzerCommand.h"

#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>


namespace fuzzer {

int ExecuteCommand(const Command &Cmd) {
  std::string CmdLine = Cmd.toString();
  int exit_code = system(CmdLine.c_str());
  if (WIFEXITED(exit_code))
    return WEXITSTATUS(exit_code);
  return exit_code;
}

void DiscardOutput(int Fd) {
  FILE* Temp = fopen("/dev/null", "w");
  if (!Temp)
    return;
  dup2(fileno(Temp), Fd);
  fclose(Temp);
}

} // namespace fuzzer

#endif
