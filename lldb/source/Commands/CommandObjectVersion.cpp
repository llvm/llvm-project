//===-- CommandObjectVersion.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandObjectVersion.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Version/Version.h"
#include "llvm/ADT/StringExtras.h"

using namespace lldb;
using namespace lldb_private;

#define LLDB_OPTIONS_version
#include "CommandOptions.inc"

llvm::ArrayRef<OptionDefinition>
CommandObjectVersion::CommandOptions::GetDefinitions() {
  return llvm::ArrayRef(g_version_options);
}

CommandObjectVersion::CommandObjectVersion(CommandInterpreter &interpreter)
    : CommandObjectParsed(interpreter, "version",
                          "Show the LLDB debugger version.", "version") {}

CommandObjectVersion::~CommandObjectVersion() = default;

// Dump the array values on a single line.
static void dump(const StructuredData::Array &array, Stream &s) {
  std::vector<std::string> values;
  array.ForEach([&](StructuredData::Object *object) -> bool {
    values.emplace_back(object->GetStringValue().str());
    return true;
  });

  s << '[' << llvm::join(values, ", ") << ']';
}

// The default dump output is too verbose.
static void dump(const StructuredData::Dictionary &config, Stream &s) {
  config.ForEach(
      [&](llvm::StringRef key, StructuredData::Object *object) -> bool {
        assert(object);

        StructuredData::Dictionary *value_dict = object->GetAsDictionary();
        assert(value_dict);

        StructuredData::ObjectSP value_sp = value_dict->GetValueForKey("value");
        assert(value_sp);

        s << "  " << key << ": ";
        if (StructuredData::Boolean *boolean = value_sp->GetAsBoolean())
          s << (boolean->GetValue() ? "yes" : "no");
        else if (StructuredData::Array *array = value_sp->GetAsArray())
          dump(*array, s);
        s << '\n';

        return true;
      });
}

void CommandObjectVersion::DoExecute(Args &args, CommandReturnObject &result) {
  result.AppendMessageWithFormat("%s\n", lldb_private::GetVersion());

  if (m_options.verbose)
    dump(*Debugger::GetBuildConfiguration(), result.GetOutputStream());

  result.SetStatus(eReturnStatusSuccessFinishResult);
}
