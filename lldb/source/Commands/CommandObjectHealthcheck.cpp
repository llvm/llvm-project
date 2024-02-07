//===-- CommandObjectHealthcheck.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandObjectHealthcheck.h"

#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Utility/Log.h"
#include "lldb/lldb-private.h"

#include "Plugins/Language/Swift/LogChannelSwift.h"

using namespace lldb;
using namespace lldb_private;

CommandObjectHealthcheck::CommandObjectHealthcheck(
    CommandInterpreter &interpreter)
    : CommandObjectParsed(
          interpreter, "swift-healthcheck",
          "Provides logging related to the Swift expression evaluator, "
          "including Swift compiler diagnostics. This makes it easier to "
          "identify project misconfigurations that result in module import "
          "failures in the debugger. The command is meant to be run after a "
          "expression evaluator failure has occurred.") {}

void CommandObjectHealthcheck::DoExecute(Args &args,
                                         CommandReturnObject &result) {
  std::error_code err;
  llvm::SmallString<128> temp_path;
  int temp_fd = -1;
  if (FileSpec temp_file_spec = HostInfo::GetProcessTempDir()) {
    temp_file_spec.AppendPathComponent("lldb-healthcheck-%%%%%%.log");
    err = llvm::sys::fs::createUniqueFile(temp_file_spec.GetPath(), temp_fd,
                                          temp_path);
  } else {
    err = llvm::sys::fs::createTemporaryFile("lldb-healthcheck", "log", temp_fd,
                                             temp_path);
  }

  if (temp_fd == -1) {
    result.AppendErrorWithFormat("could not write to temp file %s",
                                 err.message().c_str());
    return;
  }

  llvm::raw_fd_ostream temp_stream(temp_fd, true, true);
  llvm::StringRef data = GetSwiftHealthLogData();
  temp_stream << data;

  result.AppendMessageWithFormat("Health check written to %s\n",
                                 temp_path.c_str());
#if defined(__APPLE__)
  // When in an interactive graphical session and not, for example,
  // running LLDB running over ssh, open the log file straight away in
  // the user's configured editor or the default Console.app otherwise.
  if (llvm::StringRef(getprogname()).starts_with("lldb") &&
      Host::IsInteractiveGraphicSession()) {
    if (llvm::Error err =
            Host::OpenFileInExternalEditor("", FileSpec(temp_path), 0))
      return;
  }
#endif

}
