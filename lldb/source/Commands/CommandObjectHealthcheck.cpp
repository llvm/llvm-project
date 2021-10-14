//===-- CommandObjectHealthcheck.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandObjectHealthcheck.h"

#include "lldb/Host/HostInfo.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Utility/Log.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/lldb-private.h"

using namespace lldb;
using namespace lldb_private;

CommandObjectHealthcheck::CommandObjectHealthcheck(
    CommandInterpreter &interpreter)
    : CommandObjectParsed(interpreter, "healthcheck",
                          "Show the LLDB debugger health check diagnostics.",
                          "healthcheck") {}

bool CommandObjectHealthcheck::DoExecute(Args &args,
                                         CommandReturnObject &result) {
#ifdef LLDB_ENABLE_SWIFT
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
    return false;
  }

  llvm::raw_fd_ostream temp_stream(temp_fd, true, true);
  llvm::StringRef data = GetSwiftHealthLogData();
  temp_stream << data;

  result.AppendMessageWithFormat("Health check written to %s\n",
                                 temp_path.c_str());
#endif
  return true;
}