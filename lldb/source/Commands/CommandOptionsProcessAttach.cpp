//===-- CommandOptionsProcessAttach.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandOptionsProcessAttach.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/IOHandlerPicker.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/OptionParser.h"
#include "lldb/Interpreter/CommandCompletions.h"
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/CommandOptionArgumentTable.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/OptionArgParser.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Target.h"

#include "llvm/ADT/ArrayRef.h"

using namespace llvm;
using namespace lldb;
using namespace lldb_private;

#define LLDB_OPTIONS_process_attach
#include "CommandOptions.inc"

Status CommandOptionsProcessAttach::SetOptionValue(
    uint32_t option_idx, llvm::StringRef option_arg,
    ExecutionContext *execution_context) {
  Status error;
  const int short_option = g_process_attach_options[option_idx].short_option;
  switch (short_option) {
  case 'c':
    attach_info.SetContinueOnceAttached(true);
    break;

  case 'p': {
    lldb::pid_t pid;
    if (option_arg.getAsInteger(0, pid)) {
      return Status::FromErrorStringWithFormatv("invalid process ID '{0}'",
                                                option_arg);
    } else {
      attach_info.SetProcessID(pid);
    }
  } break;

  case 'P':
    attach_info.SetProcessPluginName(option_arg);
    break;

  case 'n':
    attach_info.GetExecutableFile().SetFile(option_arg,
                                            FileSpec::Style::native);
    break;

  case 'w':
    attach_info.SetWaitForLaunch(true);
    break;

  case 'i':
    attach_info.SetIgnoreExisting(false);
    break;

  default:
    llvm_unreachable("Unimplemented option");
  }
  return error;
}

llvm::ArrayRef<OptionDefinition> CommandOptionsProcessAttach::GetDefinitions() {
  return llvm::ArrayRef(g_process_attach_options);
}

llvm::Expected<lldb::pid_t>
lldb_private::PickProcessToAttach(Debugger &debugger, Platform &platform,
                                  CommandReturnObject &result) {
  ProcessInstanceInfoMatch match_info;
  ProcessInstanceInfoList proc_infos;
  platform.FindProcesses(match_info, proc_infos);
  if (proc_infos.empty())
    return llvm::createStringError(
        "no processes found on the current platform");

  llvm::SmallVector<PickerColumn> columns = {"PID", "NAME", "ARCH"};
  llvm::SmallVector<PickerItem, 0> items;
  items.reserve(proc_infos.size());
  for (const ProcessInstanceInfo &proc : proc_infos) {
    const ArchSpec &arch = proc.GetArchitecture();
    PickerItem item;
    item.values = {std::to_string(proc.GetProcessID()), proc.GetName().str(),
                   arch.IsValid() ? arch.GetTriple().getArchName().str() : ""};
    items.push_back(std::move(item));
  }

  auto picker_sp = std::make_shared<IOHandlerPicker>(
      debugger, "Select a process to attach to:", std::move(columns),
      std::move(items), PickerMode::SingleSelect);
  debugger.RunIOHandlerSync(picker_sp);

  const PickerResult &picked = picker_sp->GetResult();
  if (picked.was_canceled || picked.selected_indices.empty())
    return llvm::createStringError("no process selected");

  const ProcessInstanceInfo &proc = proc_infos[picked.selected_indices.front()];
  lldb::pid_t pid = proc.GetProcessID();
  if (llvm::StringRef name = proc.GetName(); !name.empty())
    result.AppendMessageWithFormatv("Attaching to \"{0}\" ({1})...", name, pid);
  else
    result.AppendMessageWithFormatv("Attaching to process {0}...", pid);
  return pid;
}
