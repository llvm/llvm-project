//===-- CommandObjectStats.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectStats.h"
#include "lldb/Host/Host.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"

using namespace lldb;
using namespace lldb_private;

CommandObjectStats::CommandObjectStats(CommandInterpreter &interpreter)
    : CommandObjectParsed(
          interpreter, "stats", "Print statistics about a debugging session",
          nullptr) {
}

bool CommandObjectStats::DoExecute(Args &command, CommandReturnObject &result) {
  return true;
}

CommandObjectStats::~CommandObjectStats() {}
