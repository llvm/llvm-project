//===- CLOptionsSetup.cpp - Helpers to setup debug CL options ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Debug/CLOptionsSetup.h"

#include "aiir/Debug/Counter.h"
#include "aiir/Debug/DebuggerExecutionContextHook.h"
#include "aiir/Debug/ExecutionContext.h"
#include "aiir/Debug/Observers/ActionLogging.h"
#include "aiir/Debug/Observers/ActionProfiler.h"
#include "aiir/IR/AIIRContext.h"
#include "aiir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace aiir;
using namespace aiir::tracing;
using namespace llvm;

namespace {
struct DebugConfigCLOptions : public DebugConfig {
  DebugConfigCLOptions() {
    static cl::opt<std::string, /*ExternalStorage=*/true> logActionsTo{
        "log-actions-to",
        cl::desc("Log action execution to a file, or stderr if "
                 " '-' is passed"),
        cl::location(logActionsToFlag)};

    static cl::opt<std::string, /*ExternalStorage=*/true> profileActionsTo{
        "profile-actions-to",
        cl::desc("Profile action execution to a file, or stderr if "
                 " '-' is passed"),
        cl::location(profileActionsToFlag)};

    static cl::list<std::string> logActionLocationFilter(
        "log-aiir-actions-filter",
        cl::desc(
            "Comma separated list of locations to filter actions from logging"),
        cl::CommaSeparated,
        cl::cb<void, std::string>([&](const std::string &location) {
          static bool registerOnce = [&] {
            addLogActionLocFilter(&locBreakpointManager);
            return true;
          }();
          (void)registerOnce;
          static std::vector<std::string> locations;
          locations.push_back(location);
          StringRef locStr = locations.back();

          // Parse the individual location filters and set the breakpoints.
          auto diag = [](Twine msg) { llvm::errs() << msg << "\n"; };
          auto locBreakpoint =
              tracing::FileLineColLocBreakpoint::parseFromString(locStr, diag);
          if (failed(locBreakpoint)) {
            llvm::errs() << "Invalid location filter: " << locStr << "\n";
            exit(1);
          }
          auto [file, line, col] = *locBreakpoint;
          locBreakpointManager.addBreakpoint(file, line, col);
        }));

    static cl::opt<bool, /*ExternalStorage=*/true> enableDebuggerHook(
        "aiir-enable-debugger-hook",
        cl::desc("Enable Debugger hook for debugging AIIR Actions"),
        cl::location(enableDebuggerActionHookFlag), cl::init(false));
  }
  tracing::FileLineColLocBreakpointManager locBreakpointManager;
};

} // namespace

static ManagedStatic<DebugConfigCLOptions> clOptionsConfig;
void DebugConfig::registerCLOptions() { *clOptionsConfig; }

DebugConfig DebugConfig::createFromCLOptions() { return *clOptionsConfig; }

class InstallDebugHandler::Impl {
public:
  Impl(AIIRContext &context, const DebugConfig &config) {
    if (config.getLogActionsTo().empty() &&
        config.getProfileActionsTo().empty() &&
        !config.isDebuggerActionHookEnabled()) {
      if (tracing::DebugCounter::isActivated())
        context.registerActionHandler(tracing::DebugCounter());
      return;
    }
    errs() << "ExecutionContext registered on the context";
    if (tracing::DebugCounter::isActivated())
      emitError(UnknownLoc::get(&context),
                "Debug counters are incompatible with --log-actions-to and "
                "--aiir-enable-debugger-hook options and are disabled");
    if (!config.getLogActionsTo().empty()) {
      std::string errorMessage;
      logActionsFile = openOutputFile(config.getLogActionsTo(), &errorMessage);
      if (!logActionsFile) {
        emitError(UnknownLoc::get(&context),
                  "Opening file for --log-actions-to failed: ")
            << errorMessage << "\n";
        return;
      }
      logActionsFile->keep();
      raw_fd_ostream &logActionsStream = logActionsFile->os();
      actionLogger = std::make_unique<tracing::ActionLogger>(logActionsStream);
      for (const auto *locationBreakpoint : config.getLogActionsLocFilters())
        actionLogger->addBreakpointManager(locationBreakpoint);
      executionContext.registerObserver(actionLogger.get());
    }

    if (!config.getProfileActionsTo().empty()) {
      std::string errorMessage;
      profileActionsFile =
          openOutputFile(config.getProfileActionsTo(), &errorMessage);
      if (!profileActionsFile) {
        emitError(UnknownLoc::get(&context),
                  "Opening file for --profile-actions-to failed: ")
            << errorMessage << "\n";
        return;
      }
      profileActionsFile->keep();
      raw_fd_ostream &profileActionsStream = profileActionsFile->os();
      actionProfiler =
          std::make_unique<tracing::ActionProfiler>(profileActionsStream);
      executionContext.registerObserver(actionProfiler.get());
    }

    if (config.isDebuggerActionHookEnabled()) {
      errs() << " (with Debugger hook)";
      setupDebuggerExecutionContextHook(executionContext);
    }
    errs() << "\n";
    context.registerActionHandler(executionContext);
  }

private:
  std::unique_ptr<ToolOutputFile> logActionsFile;
  tracing::ExecutionContext executionContext;
  std::unique_ptr<tracing::ActionLogger> actionLogger;
  std::vector<std::unique_ptr<tracing::FileLineColLocBreakpoint>>
      locationBreakpoints;
  std::unique_ptr<ToolOutputFile> profileActionsFile;
  std::unique_ptr<tracing::ActionProfiler> actionProfiler;
};

InstallDebugHandler::InstallDebugHandler(AIIRContext &context,
                                         const DebugConfig &config)
    : impl(std::make_unique<Impl>(context, config)) {}

InstallDebugHandler::~InstallDebugHandler() = default;
