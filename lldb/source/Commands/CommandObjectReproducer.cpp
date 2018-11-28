//===-- CommandObjectReproducer.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

static void AppendErrorToResult(llvm::Error e, CommandReturnObject &result) {
  std::string error_str = llvm::toString(std::move(e));
  result.AppendErrorWithFormat("%s", error_str.c_str());
}

class CommandObjectReproducerCaptureEnable : public CommandObjectParsed {
public:
  CommandObjectReproducerCaptureEnable(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "reproducer capture enable",
                            "Enable gathering information for reproducer",
                            nullptr) {}

  ~CommandObjectReproducerCaptureEnable() override = default;

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override {
    if (!command.empty()) {
      result.AppendErrorWithFormat("'%s' takes no arguments",
                                   m_cmd_name.c_str());
      return false;
    }

    if (auto e = m_interpreter.GetDebugger().SetReproducerCapture(true)) {
      AppendErrorToResult(std::move(e), result);
      return false;
    }

    result.SetStatus(eReturnStatusSuccessFinishNoResult);
    return result.Succeeded();
  }
};

class CommandObjectReproducerCaptureDisable : public CommandObjectParsed {
public:
  CommandObjectReproducerCaptureDisable(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "reproducer capture enable",
                            "Disable gathering information for reproducer",
                            nullptr) {}

  ~CommandObjectReproducerCaptureDisable() override = default;

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override {
    if (!command.empty()) {
      result.AppendErrorWithFormat("'%s' takes no arguments",
                                   m_cmd_name.c_str());
      return false;
    }

    if (auto e = m_interpreter.GetDebugger().SetReproducerCapture(false)) {
      AppendErrorToResult(std::move(e), result);
      return false;
    }

    result.SetStatus(eReturnStatusSuccessFinishNoResult);
    return result.Succeeded();
  }
};

class CommandObjectReproducerGenerate : public CommandObjectParsed {
public:
  CommandObjectReproducerGenerate(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "reproducer generate",
                            "Generate reproducer on disk.", nullptr) {}

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
    } else {
      result.AppendErrorWithFormat("Unable to get the reproducer generator");
      return false;
    }

    result.GetOutputStream()
        << "Reproducer written to '" << r.GetReproducerPath() << "'\n";

    result.SetStatus(eReturnStatusSuccessFinishResult);
    return result.Succeeded();
  }
};

class CommandObjectReproducerReplay : public CommandObjectParsed {
public:
  CommandObjectReproducerReplay(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "reproducer replay",
                            "Replay a reproducer.", nullptr) {
    CommandArgumentEntry arg1;
    CommandArgumentData path_arg;

    // Define the first (and only) variant of this arg.
    path_arg.arg_type = eArgTypePath;
    path_arg.arg_repetition = eArgRepeatPlain;

    // There is only one variant this argument could be; put it into the
    // argument entry.
    arg1.push_back(path_arg);

    // Push the data for the first argument into the m_arguments vector.
    m_arguments.push_back(arg1);
  }

  ~CommandObjectReproducerReplay() override = default;

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override {
    if (command.empty()) {
      result.AppendErrorWithFormat(
          "'%s' takes a single argument: the reproducer path",
          m_cmd_name.c_str());
      return false;
    }

    auto &r = repro::Reproducer::Instance();

    const char *repro_path = command.GetArgumentAtIndex(0);
    if (auto e = r.SetReplay(FileSpec(repro_path))) {
      std::string error_str = llvm::toString(std::move(e));
      result.AppendErrorWithFormat("%s", error_str.c_str());
      return false;
    }

    result.SetStatus(eReturnStatusSuccessFinishNoResult);
    return result.Succeeded();
  }
};

class CommandObjectReproducerCapture : public CommandObjectMultiword {
private:
public:
  CommandObjectReproducerCapture(CommandInterpreter &interpreter)
      : CommandObjectMultiword(
            interpreter, "reproducer capture",
            "Manage gathering of information needed to generate a reproducer.",
            NULL) {
    LoadSubCommand(
        "enable",
        CommandObjectSP(new CommandObjectReproducerCaptureEnable(interpreter)));
    LoadSubCommand("disable",
                   CommandObjectSP(
                       new CommandObjectReproducerCaptureDisable(interpreter)));
  }

  ~CommandObjectReproducerCapture() {}
};

CommandObjectReproducer::CommandObjectReproducer(
    CommandInterpreter &interpreter)
    : CommandObjectMultiword(interpreter, "reproducer",
                             "Commands controlling LLDB reproducers.",
                             "log <subcommand> [<command-options>]") {
  LoadSubCommand("capture", CommandObjectSP(new CommandObjectReproducerCapture(
                                interpreter)));
  LoadSubCommand(
      "generate",
      CommandObjectSP(new CommandObjectReproducerGenerate(interpreter)));
  LoadSubCommand("replay", CommandObjectSP(
                               new CommandObjectReproducerReplay(interpreter)));
}

CommandObjectReproducer::~CommandObjectReproducer() = default;
