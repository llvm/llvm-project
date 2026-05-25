//===-- CommandObjectApropos.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandObjectApropos.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/Property.h"
#include "lldb/Utility/Args.h"
#include "lldb/Utility/StreamString.h"
#include "llvm/Support/Regex.h"

using namespace lldb;
using namespace lldb_private;

// CommandObjectApropos

CommandObjectApropos::CommandObjectApropos(CommandInterpreter &interpreter)
    : CommandObjectParsed(
          interpreter, "apropos",
          "List debugger commands and settings related to a word or subject.",
          nullptr) {
  AddSimpleArgumentList(eArgTypeSearchWord);
}

CommandObjectApropos::~CommandObjectApropos() = default;

void CommandObjectApropos::DoExecute(Args &args, CommandReturnObject &result) {
  const size_t argc = args.GetArgumentCount();

  if (argc == 1) {
    auto search_word = args[0].ref();
    if (!search_word.empty()) {
      ReturnStatus return_status = eReturnStatusSuccessFinishNoResult;

      std::string escaped_search_word;
      std::optional<Stream::HighlightSettings> highlight;
      Debugger &dbg = GetDebugger();
      if (dbg.GetUseColor()) {
        escaped_search_word = llvm::Regex::escape(search_word);
        highlight.emplace(escaped_search_word, dbg.GetRegexMatchAnsiPrefix(),
                          dbg.GetRegexMatchAnsiSuffix(), true);
      }

      // Find all commands matching the search word.
      StringList commands_found;
      StringList commands_help;
      m_interpreter.FindCommandsForApropos(
          search_word, commands_found, commands_help, true, true, true, true);

      if (commands_found.GetSize() == 0) {
        result.AppendMessageWithFormatv(
            "No commands found pertaining to '{0}'. "
            "Try 'help' to see a complete list of "
            "debugger commands.",
            args[0].c_str());
      } else {
        result.AppendMessageWithFormatv(
            "The following commands may relate to '{0}':", args[0].c_str());
        const size_t commands_max_len = commands_found.GetMaxStringLength();
        for (size_t i = 0; i < commands_found.GetSize(); ++i)
          m_interpreter.OutputFormattedHelpText(
              result.GetOutputStream(), commands_found.GetStringAtIndex(i),
              "--", commands_help.GetStringAtIndex(i), commands_max_len,
              highlight);
        return_status = eReturnStatusSuccessFinishResult;
      }

      // Find all the properties matching the search word.
      size_t properties_max_len = 0;
      std::vector<const Property *> properties;
      std::vector<const Property *> property_paths;
      GetDebugger().Apropos(search_word, properties, property_paths);
      for (const Property *prop : properties) {
        StreamString qualified_name;
        prop->DumpQualifiedName(qualified_name);
        properties_max_len =
            std::max(properties_max_len, qualified_name.GetString().size());
      }

      if (properties.empty() && property_paths.empty()) {
        result.AppendMessageWithFormatv(
            "No settings found pertaining to '{0}'. "
            "Try 'settings show' to see a complete list of "
            "debugger settings.",
            args[0].c_str());

      } else {
        return_status = eReturnStatusSuccessFinishResult;

        if (!property_paths.empty()) {
          result.AppendMessageWithFormatv(
              "\nThe following settings paths may relate to '{0}': \n\n",
              search_word);

          auto &out_strm = result.GetOutputStream();
          out_strm.IndentMore();
          for (auto path : property_paths) {
            StreamString qual_name_strm;
            if (path->DumpQualifiedName(qual_name_strm, highlight)) {
              result.GetOutputStream().Indent();
              result.GetOutputStream() << qual_name_strm.GetString() << '\n';
            }
          }
          out_strm.IndentLess();

          result.AppendMessageWithFormatv("\n(use 'settings list <path>' to "
                                          "show settings with a given path)");
        }

        if (!properties.empty()) {
          result.AppendMessageWithFormatv(
              "\nThe following settings variables may relate to '{0}': \n\n",
              search_word);

          const bool dump_qualified_name = true;
          for (auto property : properties)
            property->DumpDescription(m_interpreter, result.GetOutputStream(),
                                      properties_max_len, dump_qualified_name,
                                      highlight);
        }
      }
      result.SetStatus(return_status);
    } else {
      result.AppendError("'' is not a valid search word.\n");
    }
  } else {
    result.AppendError("'apropos' must be called with exactly one argument.\n");
  }
}
