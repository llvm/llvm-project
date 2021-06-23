//===-- CommandObjectTrace.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandObjectTrace.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/OptionParser.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/OptionArgParser.h"
#include "lldb/Interpreter/OptionGroupFormat.h"
#include "lldb/Interpreter/OptionValueBoolean.h"
#include "lldb/Interpreter/OptionValueLanguage.h"
#include "lldb/Interpreter/OptionValueString.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Trace.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

// CommandObjectTraceLoad
#define LLDB_OPTIONS_trace_load
#include "CommandOptions.inc"

#pragma mark CommandObjectTraceLoad

class CommandObjectTraceLoad : public CommandObjectParsed {
public:
  class CommandOptions : public Options {
  public:
    CommandOptions() : Options() { OptionParsingStarting(nullptr); }

    ~CommandOptions() override = default;

    Status SetOptionValue(uint32_t option_idx, StringRef option_arg,
                          ExecutionContext *execution_context) override {
      Status error;
      const int short_option = m_getopt_table[option_idx].val;

      switch (short_option) {
      case 'v': {
        m_verbose = true;
        break;
      }
      default:
        llvm_unreachable("Unimplemented option");
      }
      return error;
    }

    void OptionParsingStarting(ExecutionContext *execution_context) override {
      m_verbose = false;
    }

    ArrayRef<OptionDefinition> GetDefinitions() override {
      return makeArrayRef(g_trace_load_options);
    }

    bool m_verbose; // Enable verbose logging for debugging purposes.
  };

  CommandObjectTraceLoad(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "trace load",
                            "Load a processor trace session from a JSON file.",
                            "trace load"),
        m_options() {}

  ~CommandObjectTraceLoad() override = default;

  Options *GetOptions() override { return &m_options; }

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override {
    if (command.size() != 1) {
      result.AppendError(
          "a single path to a JSON file containing a trace session"
          "is required");
      return false;
    }

    auto end_with_failure = [&result](llvm::Error err) -> bool {
      result.AppendErrorWithFormat("%s\n",
                                   llvm::toString(std::move(err)).c_str());
      return false;
    };

    FileSpec json_file(command[0].ref());

    auto buffer_or_error = llvm::MemoryBuffer::getFile(json_file.GetPath());
    if (!buffer_or_error) {
      return end_with_failure(llvm::createStringError(
          std::errc::invalid_argument, "could not open input file: %s - %s.",
          json_file.GetPath().c_str(),
          buffer_or_error.getError().message().c_str()));
    }

    llvm::Expected<json::Value> session_file =
        json::parse(buffer_or_error.get()->getBuffer().str());
    if (!session_file)
      return end_with_failure(session_file.takeError());

    if (Expected<lldb::TraceSP> traceOrErr =
            Trace::FindPluginForPostMortemProcess(
                GetDebugger(), *session_file,
                json_file.GetDirectory().AsCString())) {
      lldb::TraceSP trace_sp = traceOrErr.get();
      if (m_options.m_verbose && trace_sp)
        result.AppendMessageWithFormat("loading trace with plugin %s\n",
                                       trace_sp->GetPluginName().AsCString());
    } else
      return end_with_failure(traceOrErr.takeError());

    result.SetStatus(eReturnStatusSuccessFinishResult);
    return true;
  }

  CommandOptions m_options;
};

// CommandObjectTraceDump
#define LLDB_OPTIONS_trace_dump
#include "CommandOptions.inc"

#pragma mark CommandObjectTraceDump

class CommandObjectTraceDump : public CommandObjectParsed {
public:
  class CommandOptions : public Options {
  public:
    CommandOptions() : Options() { OptionParsingStarting(nullptr); }

    ~CommandOptions() override = default;

    Status SetOptionValue(uint32_t option_idx, llvm::StringRef option_arg,
                          ExecutionContext *execution_context) override {
      Status error;
      const int short_option = m_getopt_table[option_idx].val;

      switch (short_option) {
      case 'v': {
        m_verbose = true;
        break;
      }
      default:
        llvm_unreachable("Unimplemented option");
      }
      return error;
    }

    void OptionParsingStarting(ExecutionContext *execution_context) override {
      m_verbose = false;
    }

    llvm::ArrayRef<OptionDefinition> GetDefinitions() override {
      return llvm::makeArrayRef(g_trace_dump_options);
    }

    bool m_verbose; // Enable verbose logging for debugging purposes.
  };

  CommandObjectTraceDump(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "trace dump",
                            "Dump the loaded processor trace data.",
                            "trace dump"),
        m_options() {}

  ~CommandObjectTraceDump() override = default;

  Options *GetOptions() override { return &m_options; }

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override {
    Status error;
    // TODO: fill in the dumping code here!
    if (error.Success()) {
      result.SetStatus(eReturnStatusSuccessFinishResult);
    } else {
      result.AppendErrorWithFormat("%s\n", error.AsCString());
    }
    return result.Succeeded();
  }

  CommandOptions m_options;
};

// CommandObjectTraceSchema
#define LLDB_OPTIONS_trace_schema
#include "CommandOptions.inc"

#pragma mark CommandObjectTraceSchema

class CommandObjectTraceSchema : public CommandObjectParsed {
public:
  class CommandOptions : public Options {
  public:
    CommandOptions() : Options() { OptionParsingStarting(nullptr); }

    ~CommandOptions() override = default;

    Status SetOptionValue(uint32_t option_idx, llvm::StringRef option_arg,
                          ExecutionContext *execution_context) override {
      Status error;
      const int short_option = m_getopt_table[option_idx].val;

      switch (short_option) {
      case 'v': {
        m_verbose = true;
        break;
      }
      default:
        llvm_unreachable("Unimplemented option");
      }
      return error;
    }

    void OptionParsingStarting(ExecutionContext *execution_context) override {
      m_verbose = false;
    }

    llvm::ArrayRef<OptionDefinition> GetDefinitions() override {
      return llvm::makeArrayRef(g_trace_schema_options);
    }

    bool m_verbose; // Enable verbose logging for debugging purposes.
  };

  CommandObjectTraceSchema(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "trace schema",
                            "Show the schema of the given trace plugin.",
                            "trace schema <plug-in>. Use the plug-in name "
                            "\"all\" to see all schemas.\n"),
        m_options() {}

  ~CommandObjectTraceSchema() override = default;

  Options *GetOptions() override { return &m_options; }

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override {
    Status error;
    if (command.empty()) {
      result.AppendError(
          "trace schema cannot be invoked without a plug-in as argument");
      return false;
    }

    StringRef plugin_name(command[0].c_str());
    if (plugin_name == "all") {
      size_t index = 0;
      while (true) {
        StringRef schema = PluginManager::GetTraceSchema(index++);
        if (schema.empty())
          break;

        result.AppendMessage(schema);
      }
    } else {
      if (Expected<StringRef> schemaOrErr =
              Trace::FindPluginSchema(plugin_name))
        result.AppendMessage(*schemaOrErr);
      else
        error = schemaOrErr.takeError();
    }

    if (error.Success()) {
      result.SetStatus(eReturnStatusSuccessFinishResult);
    } else {
      result.AppendErrorWithFormat("%s\n", error.AsCString());
    }
    return result.Succeeded();
  }

  CommandOptions m_options;
};

// CommandObjectTrace

CommandObjectTrace::CommandObjectTrace(CommandInterpreter &interpreter)
    : CommandObjectMultiword(interpreter, "trace",
                             "Commands for loading and using processor "
                             "trace information.",
                             "trace [<sub-command-options>]") {
  LoadSubCommand("load",
                 CommandObjectSP(new CommandObjectTraceLoad(interpreter)));
  LoadSubCommand("dump",
                 CommandObjectSP(new CommandObjectTraceDump(interpreter)));
  LoadSubCommand("schema",
                 CommandObjectSP(new CommandObjectTraceSchema(interpreter)));
}

CommandObjectTrace::~CommandObjectTrace() = default;

Expected<CommandObjectSP> CommandObjectTraceProxy::DoGetProxyCommandObject() {
  ProcessSP process_sp = m_interpreter.GetExecutionContext().GetProcessSP();

  if (!process_sp)
    return createStringError(inconvertibleErrorCode(),
                             "Process not available.");
  if (m_live_debug_session_only && !process_sp->IsLiveDebugSession())
    return createStringError(inconvertibleErrorCode(),
                             "Process must be alive.");

  if (Expected<TraceSP> trace_sp = process_sp->GetTarget().GetTraceOrCreate())
    return GetDelegateCommand(**trace_sp);
  else
    return createStringError(inconvertibleErrorCode(),
                             "Tracing is not supported. %s",
                             toString(trace_sp.takeError()).c_str());
}

CommandObject *CommandObjectTraceProxy::GetProxyCommandObject() {
  if (Expected<CommandObjectSP> delegate = DoGetProxyCommandObject()) {
    m_delegate_sp = *delegate;
    m_delegate_error.clear();
    return m_delegate_sp.get();
  } else {
    m_delegate_sp.reset();
    m_delegate_error = toString(delegate.takeError());
    return nullptr;
  }
}
