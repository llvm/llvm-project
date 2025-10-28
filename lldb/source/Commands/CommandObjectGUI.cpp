//===-- CommandObjectGUI.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandObjectGUI.h"

#include "lldb/Core/IOHandlerCursesGUI.h"
#include "lldb/Host/Config.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"

using namespace lldb;
using namespace lldb_private;

// CommandObjectGUI

CommandObjectGUI::CommandObjectGUI(CommandInterpreter &interpreter)
    : CommandObjectParsed(interpreter, "gui",
                          "Switch into the curses based GUI mode.", "gui") {}

CommandObjectGUI::~CommandObjectGUI() = default;

void CommandObjectGUI::DoExecute(Args &args, CommandReturnObject &result) {
#if LLDB_ENABLE_CURSES
  Debugger &debugger = GetDebugger();

  FileSP input_sp = debugger.GetInputFileSP();
  FileSP output_sp = debugger.GetOutputFileSP();
  if (input_sp->GetStream() && output_sp->GetStream() &&
      input_sp->GetIsRealTerminal() && input_sp->GetIsInteractive()) {
    IOHandlerSP io_handler_sp(new IOHandlerCursesGUI(debugger));
    if (io_handler_sp)
      debugger.RunIOHandlerAsync(io_handler_sp);
    result.SetStatus(eReturnStatusSuccessFinishResult);
  } else {
    result.AppendError("the gui command requires an interactive terminal.");
  }
#else
  result.AppendError("lldb was not built with gui support");
#endif
}
