//===-- CommandObjectStats.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandObjectStats.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Host/OptionParser.h"
#include "lldb/Interpreter/CommandOptionArgumentTable.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/OptionArgParser.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

class CommandObjectStatsEnable : public CommandObjectParsed {
public:
  CommandObjectStatsEnable(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "enable",
                            "Enable statistics collection", nullptr,
                            eCommandProcessMustBePaused) {}

  ~CommandObjectStatsEnable() override = default;

protected:
  void DoExecute(Args &command, CommandReturnObject &result) override {
    if (DebuggerStats::GetCollectingStats()) {
      result.AppendError("statistics already enabled");
      return;
    }

    DebuggerStats::SetCollectingStats(true);
    result.SetStatus(eReturnStatusSuccessFinishResult);
  }
};

class CommandObjectStatsDisable : public CommandObjectParsed {
public:
  CommandObjectStatsDisable(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "disable",
                            "Disable statistics collection", nullptr,
                            eCommandProcessMustBePaused) {}

  ~CommandObjectStatsDisable() override = default;

protected:
  void DoExecute(Args &command, CommandReturnObject &result) override {
    if (!DebuggerStats::GetCollectingStats()) {
      result.AppendError("need to enable statistics before disabling them");
      return;
    }

    DebuggerStats::SetCollectingStats(false);
    result.SetStatus(eReturnStatusSuccessFinishResult);
  }
};

#define LLDB_OPTIONS_statistics_dump
#include "CommandOptions.inc"

class CommandObjectStatsDump : public CommandObjectParsed {
  class CommandOptions : public Options {
  public:
    CommandOptions() { OptionParsingStarting(nullptr); }

    Status SetOptionValue(uint32_t option_idx, llvm::StringRef option_arg,
                          ExecutionContext *execution_context) override {
      Status error;
      const int short_option = m_getopt_table[option_idx].val;

      switch (short_option) {
      case 'a':
        m_all_targets = true;
        break;
      case 's':
        m_stats_options.summary_only = true;
        // In the "summary" mode, the following sections should be omitted by
        // default unless their corresponding options are explicitly given.
        // If such options were processed before 's', `m_seen_options` should
        // contain them, and we will skip setting them to `false` here. If such
        // options are not yet processed, we will set them to `false` here, and
        // they will be overwritten when the options are processed.
        if (m_seen_options.find((int)'r') == m_seen_options.end())
          m_stats_options.include_targets = false;
        if (m_seen_options.find((int)'m') == m_seen_options.end())
          m_stats_options.include_modules = false;
        if (m_seen_options.find((int)'t') == m_seen_options.end())
          m_stats_options.include_transcript = false;
        break;
      case 'f':
        m_stats_options.load_all_debug_info = true;
        break;
      case 'r':
        m_stats_options.include_targets = OptionArgParser::ToBoolean(
            "--targets", option_arg, true /* doesn't matter */, error);
        break;
      case 'm':
        m_stats_options.include_modules = OptionArgParser::ToBoolean(
            "--modules", option_arg, true /* doesn't matter */, error);
        break;
      case 't':
        m_stats_options.include_transcript = OptionArgParser::ToBoolean(
            "--transcript", option_arg, true /* doesn't matter */, error);
        break;
      default:
        llvm_unreachable("Unimplemented option");
      }
      return error;
    }

    void OptionParsingStarting(ExecutionContext *execution_context) override {
      m_all_targets = false;
      m_stats_options = StatisticsOptions();
    }

    llvm::ArrayRef<OptionDefinition> GetDefinitions() override {
      return llvm::ArrayRef(g_statistics_dump_options);
    }

    const StatisticsOptions &GetStatisticsOptions() { return m_stats_options; }

    bool m_all_targets = false;
    StatisticsOptions m_stats_options = StatisticsOptions();
  };

public:
  CommandObjectStatsDump(CommandInterpreter &interpreter)
      : CommandObjectParsed(
            interpreter, "statistics dump", "Dump metrics in JSON format",
            "statistics dump [<options>]", eCommandRequiresTarget) {}

  ~CommandObjectStatsDump() override = default;

  Options *GetOptions() override { return &m_options; }

protected:
  void DoExecute(Args &command, CommandReturnObject &result) override {
    Target *target = nullptr;
    if (!m_options.m_all_targets)
      target = m_exe_ctx.GetTargetPtr();

    result.AppendMessageWithFormatv(
        "{0:2}", DebuggerStats::ReportStatistics(
                     GetDebugger(), target, m_options.GetStatisticsOptions()));
    result.SetStatus(eReturnStatusSuccessFinishResult);
  }

  CommandOptions m_options;
};

CommandObjectStats::CommandObjectStats(CommandInterpreter &interpreter)
    : CommandObjectMultiword(interpreter, "statistics",
                             "Print statistics about a debugging session",
                             "statistics <subcommand> [<subcommand-options>]") {
  LoadSubCommand("enable",
                 CommandObjectSP(new CommandObjectStatsEnable(interpreter)));
  LoadSubCommand("disable",
                 CommandObjectSP(new CommandObjectStatsDisable(interpreter)));
  LoadSubCommand("dump",
                 CommandObjectSP(new CommandObjectStatsDump(interpreter)));
}

CommandObjectStats::~CommandObjectStats() = default;
