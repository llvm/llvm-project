//===-- CommandObjectReproducer.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandObjectReproducer.h"

#include "lldb/Utility/Reproducer.h"

#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/OptionArgParser.h"
#include "lldb/Interpreter/OptionGroupBoolean.h"

using namespace lldb;
using namespace lldb_private;

class CommandObjectReproducerGenerate : public CommandObjectParsed {
public:
  CommandObjectReproducerGenerate(CommandInterpreter &interpreter)
      : CommandObjectParsed(
            interpreter, "reproducer generate",
            "Generate reproducer on disk. When the debugger is in capture "
            "mode, this command will output the reproducer to a directory on "
            "disk. In replay mode this command in a no-op.",
            nullptr) {}

  ~CommandObjectReproducerGenerate() override = default;

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override {
    if (!command.empty()) {
      result.AppendErrorWithFormat("'%s' takes no arguments",
                                   m_cmd_name.c_str());
      return false;
    }

    auto &r = repro::Reproducer::Instance();
    if (auto generator = r.GetGenerator()) {
      generator->Keep();
    } else if (r.GetLoader()) {
      // Make this operation a NOP in replay mode.
      result.SetStatus(eReturnStatusSuccessFinishNoResult);
      return result.Succeeded();
    } else {
      result.AppendErrorWithFormat("Unable to get the reproducer generator");
      result.SetStatus(eReturnStatusFailed);
      return false;
    }

    result.GetOutputStream()
        << "Reproducer written to '" << r.GetReproducerPath() << "'\n";
    result.GetOutputStream()
        << "Please have a look at the directory to assess if you're willing to "
           "share the contained information.\n";

    result.SetStatus(eReturnStatusSuccessFinishResult);
    return result.Succeeded();
  }
};

class CommandObjectReproducerStatus : public CommandObjectParsed {
public:
  CommandObjectReproducerStatus(CommandInterpreter &interpreter)
      : CommandObjectParsed(
            interpreter, "reproducer status",
            "Show the current reproducer status. In capture mode the debugger "
            "is collecting all the information it needs to create a "
            "reproducer.  In replay mode the reproducer is replaying a "
            "reproducer. When the reproducers are off, no data is collected "
            "and no reproducer can be generated.",
            nullptr) {}

  ~CommandObjectReproducerStatus() override = default;

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override {
    if (!command.empty()) {
      result.AppendErrorWithFormat("'%s' takes no arguments",
                                   m_cmd_name.c_str());
      return false;
    }

    auto &r = repro::Reproducer::Instance();
    if (r.GetGenerator()) {
      result.GetOutputStream() << "Reproducer is in capture mode.\n";
    } else if (r.GetLoader()) {
      result.GetOutputStream() << "Reproducer is in replay mode.\n";
    } else {
      result.GetOutputStream() << "Reproducer is off.\n";
    }

    result.SetStatus(eReturnStatusSuccessFinishResult);
    return result.Succeeded();
  }
};

CommandObjectReproducer::CommandObjectReproducer(
    CommandInterpreter &interpreter)
    : CommandObjectMultiword(
          interpreter, "reproducer",
          "Commands to inspect and manipulate the reproducer functionality.",
          "log <subcommand> [<command-options>]") {
  LoadSubCommand(
      "generate",
      CommandObjectSP(new CommandObjectReproducerGenerate(interpreter)));
  LoadSubCommand("status", CommandObjectSP(
                               new CommandObjectReproducerStatus(interpreter)));
}

CommandObjectReproducer::~CommandObjectReproducer() = default;
