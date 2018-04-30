//===-- CommandObjectQuit.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectQuit.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/Process.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectQuit
//-------------------------------------------------------------------------

CommandObjectQuit::CommandObjectQuit(CommandInterpreter &interpreter)
    : CommandObjectParsed(interpreter, "quit", "Quit the LLDB debugger.",
                          "quit") {}

CommandObjectQuit::~CommandObjectQuit() {}

// returns true if there is at least one alive process is_a_detach will be true
// if all alive processes will be detached when you quit and false if at least
// one process will be killed instead
bool CommandObjectQuit::ShouldAskForConfirmation(bool &is_a_detach) {
  if (m_interpreter.GetPromptOnQuit() == false)
    return false;
  bool should_prompt = false;
  is_a_detach = true;
  for (uint32_t debugger_idx = 0; debugger_idx < Debugger::GetNumDebuggers();
       debugger_idx++) {
    DebuggerSP debugger_sp(Debugger::GetDebuggerAtIndex(debugger_idx));
    if (!debugger_sp)
      continue;
    const TargetList &target_list(debugger_sp->GetTargetList());
    for (uint32_t target_idx = 0;
         target_idx < static_cast<uint32_t>(target_list.GetNumTargets());
         target_idx++) {
      TargetSP target_sp(target_list.GetTargetAtIndex(target_idx));
      if (!target_sp)
        continue;
      ProcessSP process_sp(target_sp->GetProcessSP());
      if (process_sp && process_sp->IsValid() && process_sp->IsAlive() &&
          process_sp->WarnBeforeDetach()) {
        should_prompt = true;
        if (process_sp->GetShouldDetach() == false) {
          // if we need to kill at least one process, just say so and return
          is_a_detach = false;
          return should_prompt;
        }
      }
    }
  }
  return should_prompt;
}

bool CommandObjectQuit::DoExecute(Args &command, CommandReturnObject &result) {
  bool is_a_detach = true;
  if (ShouldAskForConfirmation(is_a_detach)) {
    StreamString message;
    message.Printf("Quitting LLDB will %s one or more processes. Do you really "
                   "want to proceed",
                   (is_a_detach ? "detach from" : "kill"));
    if (!m_interpreter.Confirm(message.GetString(), true)) {
      result.SetStatus(eReturnStatusFailed);
      return false;
    }
  }
  const uint32_t event_type =
      CommandInterpreter::eBroadcastBitQuitCommandReceived;
  m_interpreter.BroadcastEvent(event_type);
  result.SetStatus(eReturnStatusQuit);
  return true;
}
