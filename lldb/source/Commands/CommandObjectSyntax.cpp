//===-- CommandObjectSyntax.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "CommandObjectSyntax.h"
#include "CommandObjectHelp.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/Options.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectSyntax
//-------------------------------------------------------------------------

CommandObjectSyntax::CommandObjectSyntax(CommandInterpreter &interpreter)
    : CommandObjectParsed(
          interpreter, "syntax",
          "Shows the correct syntax for a given debugger command.",
          "syntax <command>") {
  CommandArgumentEntry arg;
  CommandArgumentData command_arg;

  // Define the first (and only) variant of this arg.
  command_arg.arg_type = eArgTypeCommandName;
  command_arg.arg_repetition = eArgRepeatPlain;

  // There is only one variant this argument could be; put it into the argument
  // entry.
  arg.push_back(command_arg);

  // Push the data for the first argument into the m_arguments vector.
  m_arguments.push_back(arg);
}

CommandObjectSyntax::~CommandObjectSyntax() = default;

bool CommandObjectSyntax::DoExecute(Args &command,
                                    CommandReturnObject &result) {
  CommandObject::CommandMap::iterator pos;
  CommandObject *cmd_obj;
  const size_t argc = command.GetArgumentCount();

  if (argc == 0) {
    result.AppendError("Must call 'syntax' with a valid command.");
    result.SetStatus(eReturnStatusFailed);
    return false;
  }

  cmd_obj = m_interpreter.GetCommandObject(command[0].ref);
  bool all_okay = llvm::all_of(
      command.entries().drop_front(), [&cmd_obj](const Args::ArgEntry &e) {
        if (!cmd_obj || !cmd_obj->IsMultiwordObject())
          return false;

        if (!(cmd_obj = cmd_obj->GetSubcommandObject(e.ref)))
          return false;

        return true;
      });

  if (!all_okay) {
    std::string cmd_string;
    command.GetCommandString(cmd_string);

    StreamString error_msg_stream;
    const bool generate_apropos = true;
    const bool generate_type_lookup = false;
    CommandObjectHelp::GenerateAdditionalHelpAvenuesMessage(
        &error_msg_stream, cmd_string, "", "", generate_apropos,
        generate_type_lookup);
    result.AppendErrorWithFormat("%s", error_msg_stream.GetData());
    result.SetStatus(eReturnStatusFailed);
    return false;
  }

  Stream &output_strm = result.GetOutputStream();
  if (cmd_obj->GetOptions() != nullptr) {
    output_strm.Printf("\nSyntax: %s\n", cmd_obj->GetSyntax().str().c_str());
    output_strm.Printf(
        "(Try 'help %s' for more information on command options syntax.)\n",
        cmd_obj->GetCommandName().str().c_str());
    result.SetStatus(eReturnStatusSuccessFinishNoResult);
  } else {
    output_strm.Printf("\nSyntax: %s\n", cmd_obj->GetSyntax().str().c_str());
    result.SetStatus(eReturnStatusSuccessFinishNoResult);
  }

  return result.Succeeded();
}
