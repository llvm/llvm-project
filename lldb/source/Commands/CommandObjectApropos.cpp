//===-- CommandObjectApropos.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandObjectApropos.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/Property.h"
#include "lldb/Utility/Args.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb;
using namespace lldb_private;

// CommandObjectApropos

CommandObjectApropos::CommandObjectApropos(CommandInterpreter &interpreter)
    : CommandObjectParsed(
          interpreter, "apropos",
          "List debugger commands related to a word or subject.", nullptr) {
  AddSimpleArgumentList(eArgTypeSearchWord);
}

CommandObjectApropos::~CommandObjectApropos() = default;

void CommandObjectApropos::DoExecute(Args &args, CommandReturnObject &result) {
  const size_t argc = args.GetArgumentCount();

  if (argc == 1) {
    auto search_word = args[0].ref();
    if (!search_word.empty()) {
      ReturnStatus return_status = eReturnStatusSuccessFinishNoResult;

      // Find all commands matching the search word.
      StringList commands_found;
      StringList commands_help;
      m_interpreter.FindCommandsForApropos(
          search_word, commands_found, commands_help, true, true, true, true);

      if (commands_found.GetSize() == 0) {
        result.AppendMessageWithFormat("No commands found pertaining to '%s'. "
                                       "Try 'help' to see a complete list of "
                                       "debugger commands.\n",
                                       args[0].c_str());
      } else {
        result.AppendMessageWithFormat(
            "The following commands may relate to '%s':\n", args[0].c_str());
        const size_t commands_max_len = commands_found.GetMaxStringLength();
        for (size_t i = 0; i < commands_found.GetSize(); ++i)
          m_interpreter.OutputFormattedHelpText(
              result.GetOutputStream(), commands_found.GetStringAtIndex(i),
              "--", commands_help.GetStringAtIndex(i), commands_max_len);
        return_status = eReturnStatusSuccessFinishResult;
      }

      // Find all the properties matching the search word.
      size_t properties_max_len = 0;
      std::vector<const Property *> properties;
      const size_t num_properties =
          GetDebugger().Apropos(search_word, properties);
      for (const Property *prop : properties) {
        StreamString qualified_name;
        prop->DumpQualifiedName(qualified_name);
        properties_max_len =
            std::max(properties_max_len, qualified_name.GetString().size());
      }

      if (num_properties == 0) {
        result.AppendMessageWithFormat(
            "No settings found pertaining to '%s'. "
            "Try 'settings show' to see a complete list of "
            "debugger settings.\n",
            args[0].c_str());

      } else {
        result.AppendMessageWithFormatv(
            "\nThe following settings variables may relate to '{0}': \n\n",
            search_word);

        const bool dump_qualified_name = true;
        for (size_t i = 0; i < num_properties; ++i)
          properties[i]->DumpDescription(
              m_interpreter, result.GetOutputStream(), properties_max_len,
              dump_qualified_name);
        return_status = eReturnStatusSuccessFinishResult;
      }

      result.SetStatus(return_status);
    } else {
      result.AppendError("'' is not a valid search word.\n");
    }
  } else {
    result.AppendError("'apropos' must be called with exactly one argument.\n");
  }
}
