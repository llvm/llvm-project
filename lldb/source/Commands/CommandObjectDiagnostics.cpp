//===-- CommandObjectDiagnostics.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandObjectDiagnostics.h"
#include "lldb/Core/BugReporter.h"
#include "lldb/Core/Diagnostics.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/OptionParser.h"
#include "lldb/Interpreter/CommandOptionArgumentTable.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/OptionArgParser.h"
#include "lldb/Interpreter/OptionValueEnumeration.h"
#include "lldb/Interpreter/OptionValueUInt64.h"
#include "lldb/Interpreter/Options.h"

#include "llvm/Support/JSON.h"

using namespace lldb;
using namespace lldb_private;

#define LLDB_OPTIONS_diagnostics_dump
#include "CommandOptions.inc"

#define LLDB_OPTIONS_diagnostics_report
#include "CommandOptions.inc"

class CommandObjectDiagnosticsDump : public CommandObjectParsed {
public:
  // Constructors and Destructors
  CommandObjectDiagnosticsDump(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "diagnostics dump",
                            "Dump diagnostics to disk", nullptr) {}

  ~CommandObjectDiagnosticsDump() override = default;

  class CommandOptions : public Options {
  public:
    CommandOptions() = default;

    ~CommandOptions() override = default;

    Status SetOptionValue(uint32_t option_idx, llvm::StringRef option_arg,
                          ExecutionContext *execution_context) override {
      Status error;
      const int short_option = m_getopt_table[option_idx].val;

      switch (short_option) {
      case 'd':
        directory.SetDirectory(option_arg);
        break;
      default:
        llvm_unreachable("Unimplemented option");
      }
      return error;
    }

    void OptionParsingStarting(ExecutionContext *execution_context) override {
      directory.Clear();
    }

    llvm::ArrayRef<OptionDefinition> GetDefinitions() override {
      return llvm::ArrayRef(g_diagnostics_dump_options);
    }

    FileSpec directory;
  };

  Options *GetOptions() override { return &m_options; }

protected:
  llvm::Expected<FileSpec> GetDirectory() {
    if (m_options.directory) {
      auto ec =
          llvm::sys::fs::create_directories(m_options.directory.GetPath());
      if (ec)
        return llvm::errorCodeToError(ec);
      return m_options.directory;
    }
    return Diagnostics::CreateUniqueDirectory();
  }

  void DoExecute(Args &args, CommandReturnObject &result) override {
    llvm::Expected<FileSpec> directory = GetDirectory();

    if (!directory) {
      result.AppendError(llvm::toString(directory.takeError()));
      return;
    }

    // Collect the diagnostics bundle into the directory.
    llvm::Expected<Diagnostics::Report> report =
        Diagnostics::Instance().Collect(GetDebugger(), m_exe_ctx, *directory);
    if (!report) {
      result.AppendErrorWithFormat("failed to write diagnostics to %s",
                                   directory->GetPath().c_str());
      result.AppendError(llvm::toString(report.takeError()));
      return;
    }

    // Print the report as JSON so the user can review what a bug report would
    // carry. The bundle directory and its files are listed under "attachments".
    result.GetOutputStream().Format("{0:2}\n", toJSON(*report));

    result.SetStatus(eReturnStatusSuccessFinishResult);
  }

  CommandOptions m_options;
};

class CommandObjectDiagnosticsReport : public CommandObjectParsed {
public:
  CommandObjectDiagnosticsReport(CommandInterpreter &interpreter)
      : CommandObjectParsed(
            interpreter, "diagnostics report",
            "Assemble a diagnostics bundle and file it as a bug report.",
            nullptr) {}

  ~CommandObjectDiagnosticsReport() override = default;

  class CommandOptions : public Options {
  public:
    CommandOptions() = default;

    ~CommandOptions() override = default;

    Status SetOptionValue(uint32_t option_idx, llvm::StringRef option_arg,
                          ExecutionContext *execution_context) override {
      Status error;
      const int short_option = m_getopt_table[option_idx].val;

      switch (short_option) {
      case 'd':
        directory.SetDirectory(option_arg);
        break;
      case 'n':
        no_open = true;
        break;
      case 'P':
        plugin = option_arg.str();
        break;
      default:
        llvm_unreachable("Unimplemented option");
      }
      return error;
    }

    void OptionParsingStarting(ExecutionContext *execution_context) override {
      directory.Clear();
      no_open = false;
      plugin.clear();
    }

    llvm::ArrayRef<OptionDefinition> GetDefinitions() override {
      return llvm::ArrayRef(g_diagnostics_report_options);
    }

    FileSpec directory;
    bool no_open = false;
    std::string plugin;
  };

  Options *GetOptions() override { return &m_options; }

protected:
  llvm::Expected<FileSpec> GetDirectory() {
    if (m_options.directory) {
      auto ec =
          llvm::sys::fs::create_directories(m_options.directory.GetPath());
      if (ec)
        return llvm::errorCodeToError(ec);
      return m_options.directory;
    }
    return Diagnostics::CreateUniqueDirectory();
  }

  void DoExecute(Args &args, CommandReturnObject &result) override {
    llvm::Expected<FileSpec> directory = GetDirectory();
    if (!directory) {
      result.AppendError(llvm::toString(directory.takeError()));
      return;
    }

    llvm::Expected<Diagnostics::Report> report =
        Diagnostics::Instance().Collect(GetDebugger(), m_exe_ctx, *directory);
    if (!report) {
      result.AppendError(llvm::toString(report.takeError()));
      return;
    }

    Stream &out = result.GetOutputStream();
    out << "Bug report written to " << directory->GetPath() << "\n";
    if (!report->attachments.files.empty()) {
      out << "Attach the following files to the issue:\n";
      for (const std::string &file : report->attachments.files)
        out << "  [ ] " << file << "\n";
    }
    result.AppendWarning("the report may contain file paths, command history "
                         "and program data. Review it before attaching it to a "
                         "public issue");

    if (m_options.no_open) {
      result.SetStatus(eReturnStatusSuccessFinishResult);
      return;
    }

    // No-tracker handling lives in the fallback reporter's File(), not here.
    std::unique_ptr<BugReporter> reporter =
        PluginManager::CreateBugReporterInstance(m_options.plugin);
    if (!reporter) {
      if (!m_options.plugin.empty())
        result.AppendErrorWithFormat("no bug reporter named '%s'",
                                     m_options.plugin.c_str());
      else
        result.AppendError("no bug reporter is available");
      return;
    }
    if (llvm::Error error = reporter->File(*report)) {
      result.AppendError(llvm::toString(std::move(error)));
      return;
    }

    out << "Opened a pre-filled " << reporter->GetPluginName() << " report.\n";
    result.SetStatus(eReturnStatusSuccessFinishResult);
  }

  CommandOptions m_options;
};

CommandObjectDiagnostics::CommandObjectDiagnostics(
    CommandInterpreter &interpreter)
    : CommandObjectMultiword(interpreter, "diagnostics",
                             "Commands controlling LLDB diagnostics.",
                             "diagnostics <subcommand> [<command-options>]") {
  LoadSubCommand(
      "dump", CommandObjectSP(new CommandObjectDiagnosticsDump(interpreter)));
  LoadSubCommand("report", CommandObjectSP(new CommandObjectDiagnosticsReport(
                               interpreter)));
}

CommandObjectDiagnostics::~CommandObjectDiagnostics() = default;
