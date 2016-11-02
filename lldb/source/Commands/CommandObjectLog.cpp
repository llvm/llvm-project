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
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/Stream.h"
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

using namespace lldb;
using namespace lldb_private;

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

  //    int
  //    HandleArgumentCompletion (Args &input,
  //                              int &cursor_index,
  //                              int &cursor_char_position,
  //                              OptionElementVector &opt_element_vector,
  //                              int match_start_point,
  //                              int max_return_elements,
  //                              bool &word_complete,
  //                              StringList &matches)
  //    {
  //        std::string completion_str (input.GetArgumentAtIndex(cursor_index));
  //        completion_str.erase (cursor_char_position);
  //
  //        if (cursor_index == 1)
  //        {
  //            //
  //            Log::AutoCompleteChannelName (completion_str.c_str(), matches);
  //        }
  //        return matches.GetSize();
  //    }
  //

  class CommandOptions : public Options {
  public:
    CommandOptions() : Options(), log_file(), log_options(0) {}

    ~CommandOptions() override = default;

    Error SetOptionValue(uint32_t option_idx, const char *option_arg,
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
      case 'g':
        log_options |= LLDB_LOG_OPTION_DEBUG;
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

    const OptionDefinition *GetDefinitions() override { return g_option_table; }

    // Options table: Required for subclasses of Options.

    static OptionDefinition g_option_table[];

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
    } else {
      std::string channel(args.GetArgumentAtIndex(0));
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
    }
    return result.Succeeded();
  }

  CommandOptions m_options;
};

OptionDefinition CommandObjectLogEnable::CommandOptions::g_option_table[] = {
    // clang-format off
  {LLDB_OPT_SET_1, false, "file",       'f', OptionParser::eRequiredArgument, nullptr, nullptr, 0, eArgTypeFilename, "Set the destination file to log to."},
  {LLDB_OPT_SET_1, false, "threadsafe", 't', OptionParser::eNoArgument,       nullptr, nullptr, 0, eArgTypeNone,     "Enable thread safe logging to avoid interweaved log lines."},
  {LLDB_OPT_SET_1, false, "verbose",    'v', OptionParser::eNoArgument,       nullptr, nullptr, 0, eArgTypeNone,     "Enable verbose logging."},
  {LLDB_OPT_SET_1, false, "debug",      'g', OptionParser::eNoArgument,       nullptr, nullptr, 0, eArgTypeNone,     "Enable debug logging."},
  {LLDB_OPT_SET_1, false, "sequence",   's', OptionParser::eNoArgument,       nullptr, nullptr, 0, eArgTypeNone,     "Prepend all log lines with an increasing integer sequence id."},
  {LLDB_OPT_SET_1, false, "timestamp",  'T', OptionParser::eNoArgument,       nullptr, nullptr, 0, eArgTypeNone,     "Prepend all log lines with a timestamp."},
  {LLDB_OPT_SET_1, false, "pid-tid",    'p', OptionParser::eNoArgument,       nullptr, nullptr, 0, eArgTypeNone,     "Prepend all log lines with the process and thread ID that generates the log line."},
  {LLDB_OPT_SET_1, false, "thread-name",'n', OptionParser::eNoArgument,       nullptr, nullptr, 0, eArgTypeNone,     "Prepend all log lines with the thread name for the thread that generates the log line."},
  {LLDB_OPT_SET_1, false, "stack",      'S', OptionParser::eNoArgument,       nullptr, nullptr, 0, eArgTypeNone,     "Append a stack backtrace to each log line."},
  {LLDB_OPT_SET_1, false, "append",     'a', OptionParser::eNoArgument,       nullptr, nullptr, 0, eArgTypeNone,     "Append to the log file instead of overwriting."},
  {0, false, nullptr, 0, 0, nullptr, nullptr, 0, eArgTypeNone, nullptr}
    // clang-format on
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
    const size_t argc = args.GetArgumentCount();
    if (argc == 0) {
      result.AppendErrorWithFormat(
          "%s takes a log channel and one or more log types.\n",
          m_cmd_name.c_str());
    } else {
      Log::Callbacks log_callbacks;

      std::string channel(args.GetArgumentAtIndex(0));
      args.Shift(); // Shift off the channel
      if (Log::GetLogChannelCallbacks(ConstString(channel.c_str()),
                                      log_callbacks)) {
        log_callbacks.disable(args.GetConstArgumentVector(),
                              &result.GetErrorStream());
        result.SetStatus(eReturnStatusSuccessFinishNoResult);
      } else if (channel == "all") {
        Log::DisableAllLogChannels(&result.GetErrorStream());
      } else {
        LogChannelSP log_channel_sp(LogChannel::FindPlugin(channel.c_str()));
        if (log_channel_sp) {
          log_channel_sp->Disable(args.GetConstArgumentVector(),
                                  &result.GetErrorStream());
          result.SetStatus(eReturnStatusSuccessFinishNoResult);
        } else
          result.AppendErrorWithFormat("Invalid log channel '%s'.\n",
                                       args.GetArgumentAtIndex(0));
      }
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
    const size_t argc = args.GetArgumentCount();
    if (argc == 0) {
      Log::ListAllLogChannels(&result.GetOutputStream());
      result.SetStatus(eReturnStatusSuccessFinishResult);
    } else {
      for (size_t i = 0; i < argc; ++i) {
        Log::Callbacks log_callbacks;

        std::string channel(args.GetArgumentAtIndex(i));
        if (Log::GetLogChannelCallbacks(ConstString(channel.c_str()),
                                        log_callbacks)) {
          log_callbacks.list_categories(&result.GetOutputStream());
          result.SetStatus(eReturnStatusSuccessFinishResult);
        } else if (channel == "all") {
          Log::ListAllLogChannels(&result.GetOutputStream());
          result.SetStatus(eReturnStatusSuccessFinishResult);
        } else {
          LogChannelSP log_channel_sp(LogChannel::FindPlugin(channel.c_str()));
          if (log_channel_sp) {
            log_channel_sp->ListCategories(&result.GetOutputStream());
            result.SetStatus(eReturnStatusSuccessFinishNoResult);
          } else
            result.AppendErrorWithFormat("Invalid log channel '%s'.\n",
                                         args.GetArgumentAtIndex(0));
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
    const size_t argc = args.GetArgumentCount();
    result.SetStatus(eReturnStatusFailed);

    if (argc == 1) {
      const char *sub_command = args.GetArgumentAtIndex(0);

      if (strcasecmp(sub_command, "enable") == 0) {
        Timer::SetDisplayDepth(UINT32_MAX);
        result.SetStatus(eReturnStatusSuccessFinishNoResult);
      } else if (strcasecmp(sub_command, "disable") == 0) {
        Timer::DumpCategoryTimes(&result.GetOutputStream());
        Timer::SetDisplayDepth(0);
        result.SetStatus(eReturnStatusSuccessFinishResult);
      } else if (strcasecmp(sub_command, "dump") == 0) {
        Timer::DumpCategoryTimes(&result.GetOutputStream());
        result.SetStatus(eReturnStatusSuccessFinishResult);
      } else if (strcasecmp(sub_command, "reset") == 0) {
        Timer::ResetCategoryTimes();
        result.SetStatus(eReturnStatusSuccessFinishResult);
      }
    } else if (argc == 2) {
      const char *sub_command = args.GetArgumentAtIndex(0);

      if (strcasecmp(sub_command, "enable") == 0) {
        bool success;
        uint32_t depth =
            StringConvert::ToUInt32(args.GetArgumentAtIndex(1), 0, 0, &success);
        if (success) {
          Timer::SetDisplayDepth(depth);
          result.SetStatus(eReturnStatusSuccessFinishNoResult);
        } else
          result.AppendError(
              "Could not convert enable depth to an unsigned integer.");
      }
      if (strcasecmp(sub_command, "increment") == 0) {
        bool success;
        bool increment =
            Args::StringToBoolean(args.GetArgumentAtIndex(1), false, &success);
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
