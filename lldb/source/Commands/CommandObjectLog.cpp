//===-- CommandObjectLog.cpp ------------------------------------*- C++ -*-===//
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
#include "CommandObjectLog.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/Timer.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/StringConvert.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Symbol/LineTable.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/RegularExpression.h"
#include "lldb/Utility/Stream.h"

using namespace lldb;
using namespace lldb_private;

static OptionDefinition g_log_options[] = {
    // clang-format off
  { LLDB_OPT_SET_1, false, "file",       'f', OptionParser::eRequiredArgument, nullptr, nullptr, 0, eArgTypeFilename, "Set the destination file to log to." },
  { LLDB_OPT_SET_1, false, "threadsafe", 't', OptionParser::eNoArgument,       nullptr, nullptr, 0, eArgTypeNone,     "Enable thread safe logging to avoid interweaved log lines." },
  { LLDB_OPT_SET_1, false, "verbose",    'v', OptionParser::eNoArgument,       nullptr, nullptr, 0, eArgTypeNone,     "Enable verbose logging." },
  { LLDB_OPT_SET_1, false, "sequence",   's', OptionParser::eNoArgument,       nullptr, nullptr, 0, eArgTypeNone,     "Prepend all log lines with an increasing integer sequence id." },
  { LLDB_OPT_SET_1, false, "timestamp",  'T', OptionParser::eNoArgument,       nullptr, nullptr, 0, eArgTypeNone,     "Prepend all log lines with a timestamp." },
  { LLDB_OPT_SET_1, false, "pid-tid",    'p', OptionParser::eNoArgument,       nullptr, nullptr, 0, eArgTypeNone,     "Prepend all log lines with the process and thread ID that generates the log line." },
  { LLDB_OPT_SET_1, false, "thread-name",'n', OptionParser::eNoArgument,       nullptr, nullptr, 0, eArgTypeNone,     "Prepend all log lines with the thread name for the thread that generates the log line." },
  { LLDB_OPT_SET_1, false, "stack",      'S', OptionParser::eNoArgument,       nullptr, nullptr, 0, eArgTypeNone,     "Append a stack backtrace to each log line." },
  { LLDB_OPT_SET_1, false, "append",     'a', OptionParser::eNoArgument,       nullptr, nullptr, 0, eArgTypeNone,     "Append to the log file instead of overwriting." },
  { LLDB_OPT_SET_1, false, "file-function",'F',OptionParser::eNoArgument,      nullptr, nullptr, 0, eArgTypeNone,     "Prepend the names of files and function that generate the logs." },
    // clang-format on
};

class CommandObjectLogEnable : public CommandObjectParsed {
public:
  //------------------------------------------------------------------
  // Constructors and Destructors
  //------------------------------------------------------------------
  CommandObjectLogEnable(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "log enable",
                            "Enable logging for a single log channel.",
                            nullptr),
        m_options() {
    CommandArgumentEntry arg1;
    CommandArgumentEntry arg2;
    CommandArgumentData channel_arg;
    CommandArgumentData category_arg;

    // Define the first (and only) variant of this arg.
    channel_arg.arg_type = eArgTypeLogChannel;
    channel_arg.arg_repetition = eArgRepeatPlain;

    // There is only one variant this argument could be; put it into the
    // argument entry.
    arg1.push_back(channel_arg);

    category_arg.arg_type = eArgTypeLogCategory;
    category_arg.arg_repetition = eArgRepeatPlus;

    arg2.push_back(category_arg);

    // Push the data for the first argument into the m_arguments vector.
    m_arguments.push_back(arg1);
    m_arguments.push_back(arg2);
  }

  ~CommandObjectLogEnable() override = default;

  Options *GetOptions() override { return &m_options; }

  class CommandOptions : public Options {
  public:
    CommandOptions() : Options(), log_file(), log_options(0) {}

    ~CommandOptions() override = default;

    Error SetOptionValue(uint32_t option_idx, llvm::StringRef option_arg,
                         ExecutionContext *execution_context) override {
      Error error;
      const int short_option = m_getopt_table[option_idx].val;

      switch (short_option) {
      case 'f':
        log_file.SetFile(option_arg, true);
        break;
      case 't':
        log_options |= LLDB_LOG_OPTION_THREADSAFE;
        break;
      case 'v':
        log_options |= LLDB_LOG_OPTION_VERBOSE;
        break;
      case 's':
        log_options |= LLDB_LOG_OPTION_PREPEND_SEQUENCE;
        break;
      case 'T':
        log_options |= LLDB_LOG_OPTION_PREPEND_TIMESTAMP;
        break;
      case 'p':
        log_options |= LLDB_LOG_OPTION_PREPEND_PROC_AND_THREAD;
        break;
      case 'n':
        log_options |= LLDB_LOG_OPTION_PREPEND_THREAD_NAME;
        break;
      case 'S':
        log_options |= LLDB_LOG_OPTION_BACKTRACE;
        break;
      case 'a':
        log_options |= LLDB_LOG_OPTION_APPEND;
        break;
      case 'F':
        log_options |= LLDB_LOG_OPTION_PREPEND_FILE_FUNCTION;
        break;
      default:
        error.SetErrorStringWithFormat("unrecognized option '%c'",
                                       short_option);
        break;
      }

      return error;
    }

    void OptionParsingStarting(ExecutionContext *execution_context) override {
      log_file.Clear();
      log_options = 0;
    }

    llvm::ArrayRef<OptionDefinition> GetDefinitions() override {
      return llvm::makeArrayRef(g_log_options);
    }

    // Instance variables to hold the values for command options.

    FileSpec log_file;
    uint32_t log_options;
  };

protected:
  bool DoExecute(Args &args, CommandReturnObject &result) override {
    if (args.GetArgumentCount() < 2) {
      result.AppendErrorWithFormat(
          "%s takes a log channel and one or more log types.\n",
          m_cmd_name.c_str());
      return false;
    }

    // Store into a std::string since we're about to shift the channel off.
    const std::string channel = args[0].ref;
    args.Shift(); // Shift off the channel
    char log_file[PATH_MAX];
    if (m_options.log_file)
      m_options.log_file.GetPath(log_file, sizeof(log_file));
    else
      log_file[0] = '\0';
    bool success = m_interpreter.GetDebugger().EnableLog(
        channel.c_str(), args.GetConstArgumentVector(), log_file,
        m_options.log_options, result.GetErrorStream());
    if (success)
      result.SetStatus(eReturnStatusSuccessFinishNoResult);
    else
      result.SetStatus(eReturnStatusFailed);
    return result.Succeeded();
  }

  CommandOptions m_options;
};

class CommandObjectLogDisable : public CommandObjectParsed {
public:
  //------------------------------------------------------------------
  // Constructors and Destructors
  //------------------------------------------------------------------
  CommandObjectLogDisable(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "log disable",
                            "Disable one or more log channel categories.",
                            nullptr) {
    CommandArgumentEntry arg1;
    CommandArgumentEntry arg2;
    CommandArgumentData channel_arg;
    CommandArgumentData category_arg;

    // Define the first (and only) variant of this arg.
    channel_arg.arg_type = eArgTypeLogChannel;
    channel_arg.arg_repetition = eArgRepeatPlain;

    // There is only one variant this argument could be; put it into the
    // argument entry.
    arg1.push_back(channel_arg);

    category_arg.arg_type = eArgTypeLogCategory;
    category_arg.arg_repetition = eArgRepeatPlus;

    arg2.push_back(category_arg);

    // Push the data for the first argument into the m_arguments vector.
    m_arguments.push_back(arg1);
    m_arguments.push_back(arg2);
  }

  ~CommandObjectLogDisable() override = default;

protected:
  bool DoExecute(Args &args, CommandReturnObject &result) override {
    if (args.empty()) {
      result.AppendErrorWithFormat(
          "%s takes a log channel and one or more log types.\n",
          m_cmd_name.c_str());
      return false;
    }

    Log::Callbacks log_callbacks;

    const std::string channel = args[0].ref;
    args.Shift(); // Shift off the channel
    if (Log::GetLogChannelCallbacks(ConstString(channel), log_callbacks)) {
      log_callbacks.disable(args.GetConstArgumentVector(),
                            &result.GetErrorStream());
      result.SetStatus(eReturnStatusSuccessFinishNoResult);
    } else if (channel == "all") {
      Log::DisableAllLogChannels(&result.GetErrorStream());
    } else {
      LogChannelSP log_channel_sp(LogChannel::FindPlugin(channel.data()));
      if (log_channel_sp) {
        log_channel_sp->Disable(args.GetConstArgumentVector(),
                                &result.GetErrorStream());
        result.SetStatus(eReturnStatusSuccessFinishNoResult);
      } else
        result.AppendErrorWithFormat("Invalid log channel '%s'.\n",
                                     channel.data());
    }
    return result.Succeeded();
  }
};

class CommandObjectLogList : public CommandObjectParsed {
public:
  //------------------------------------------------------------------
  // Constructors and Destructors
  //------------------------------------------------------------------
  CommandObjectLogList(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "log list",
                            "List the log categories for one or more log "
                            "channels.  If none specified, lists them all.",
                            nullptr) {
    CommandArgumentEntry arg;
    CommandArgumentData channel_arg;

    // Define the first (and only) variant of this arg.
    channel_arg.arg_type = eArgTypeLogChannel;
    channel_arg.arg_repetition = eArgRepeatStar;

    // There is only one variant this argument could be; put it into the
    // argument entry.
    arg.push_back(channel_arg);

    // Push the data for the first argument into the m_arguments vector.
    m_arguments.push_back(arg);
  }

  ~CommandObjectLogList() override = default;

protected:
  bool DoExecute(Args &args, CommandReturnObject &result) override {
    if (args.empty()) {
      Log::ListAllLogChannels(&result.GetOutputStream());
      result.SetStatus(eReturnStatusSuccessFinishResult);
    } else {
      for (auto &entry : args.entries()) {
        Log::Callbacks log_callbacks;

        if (Log::GetLogChannelCallbacks(ConstString(entry.ref),
                                        log_callbacks)) {
          log_callbacks.list_categories(&result.GetOutputStream());
          result.SetStatus(eReturnStatusSuccessFinishResult);
        } else if (entry.ref == "all") {
          Log::ListAllLogChannels(&result.GetOutputStream());
          result.SetStatus(eReturnStatusSuccessFinishResult);
        } else {
          LogChannelSP log_channel_sp(LogChannel::FindPlugin(entry.c_str()));
          if (log_channel_sp) {
            log_channel_sp->ListCategories(&result.GetOutputStream());
            result.SetStatus(eReturnStatusSuccessFinishNoResult);
          } else
            result.AppendErrorWithFormat("Invalid log channel '%s'.\n",
                                         entry.c_str());
        }
      }
    }
    return result.Succeeded();
  }
};

class CommandObjectLogTimer : public CommandObjectParsed {
public:
  //------------------------------------------------------------------
  // Constructors and Destructors
  //------------------------------------------------------------------
  CommandObjectLogTimer(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "log timers",
                            "Enable, disable, dump, and reset LLDB internal "
                            "performance timers.",
                            "log timers < enable <depth> | disable | dump | "
                            "increment <bool> | reset >") {}

  ~CommandObjectLogTimer() override = default;

protected:
  bool DoExecute(Args &args, CommandReturnObject &result) override {
    result.SetStatus(eReturnStatusFailed);

    if (args.GetArgumentCount() == 1) {
      auto sub_command = args[0].ref;

      if (sub_command.equals_lower("enable")) {
        Timer::SetDisplayDepth(UINT32_MAX);
        result.SetStatus(eReturnStatusSuccessFinishNoResult);
      } else if (sub_command.equals_lower("disable")) {
        Timer::DumpCategoryTimes(&result.GetOutputStream());
        Timer::SetDisplayDepth(0);
        result.SetStatus(eReturnStatusSuccessFinishResult);
      } else if (sub_command.equals_lower("dump")) {
        Timer::DumpCategoryTimes(&result.GetOutputStream());
        result.SetStatus(eReturnStatusSuccessFinishResult);
      } else if (sub_command.equals_lower("reset")) {
        Timer::ResetCategoryTimes();
        result.SetStatus(eReturnStatusSuccessFinishResult);
      }
    } else if (args.GetArgumentCount() == 2) {
      auto sub_command = args[0].ref;
      auto param = args[1].ref;

      if (sub_command.equals_lower("enable")) {
        uint32_t depth;
        if (param.consumeInteger(0, depth)) {
          result.AppendError(
              "Could not convert enable depth to an unsigned integer.");
        } else {
          Timer::SetDisplayDepth(depth);
          result.SetStatus(eReturnStatusSuccessFinishNoResult);
        }
      } else if (sub_command.equals_lower("increment")) {
        bool success;
        bool increment = Args::StringToBoolean(param, false, &success);
        if (success) {
          Timer::SetQuiet(!increment);
          result.SetStatus(eReturnStatusSuccessFinishNoResult);
        } else
          result.AppendError("Could not convert increment value to boolean.");
      }
    }

    if (!result.Succeeded()) {
      result.AppendError("Missing subcommand");
      result.AppendErrorWithFormat("Usage: %s\n", m_cmd_syntax.c_str());
    }
    return result.Succeeded();
  }
};

CommandObjectLog::CommandObjectLog(CommandInterpreter &interpreter)
    : CommandObjectMultiword(interpreter, "log",
                             "Commands controlling LLDB internal logging.",
                             "log <subcommand> [<command-options>]") {
  LoadSubCommand("enable",
                 CommandObjectSP(new CommandObjectLogEnable(interpreter)));
  LoadSubCommand("disable",
                 CommandObjectSP(new CommandObjectLogDisable(interpreter)));
  LoadSubCommand("list",
                 CommandObjectSP(new CommandObjectLogList(interpreter)));
  LoadSubCommand("timers",
                 CommandObjectSP(new CommandObjectLogTimer(interpreter)));
}

CommandObjectLog::~CommandObjectLog() = default;
