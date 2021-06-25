//===-- CommandObjectMemoryTag.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandObjectMemoryTag.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/OptionArgParser.h"
#include "lldb/Target/Process.h"

using namespace lldb;
using namespace lldb_private;

#define LLDB_OPTIONS_memory_tag_read
#include "CommandOptions.inc"

class CommandObjectMemoryTagRead : public CommandObjectParsed {
public:
  CommandObjectMemoryTagRead(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "tag",
                            "Read memory tags for the given range of memory.",
                            nullptr,
                            eCommandRequiresTarget | eCommandRequiresProcess |
                                eCommandProcessMustBePaused) {
    // Address
    m_arguments.push_back(
        CommandArgumentEntry{CommandArgumentData(eArgTypeAddressOrExpression)});
    // Optional end address
    m_arguments.push_back(CommandArgumentEntry{
        CommandArgumentData(eArgTypeAddressOrExpression, eArgRepeatOptional)});
  }

  ~CommandObjectMemoryTagRead() override = default;

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override {
    if ((command.GetArgumentCount() < 1) || (command.GetArgumentCount() > 2)) {
      result.AppendError(
          "wrong number of arguments; expected at least <address-expression>, "
          "at most <address-expression> <end-address-expression>");
      return false;
    }

    Status error;
    addr_t start_addr = OptionArgParser::ToAddress(
        &m_exe_ctx, command[0].ref(), LLDB_INVALID_ADDRESS, &error);
    if (start_addr == LLDB_INVALID_ADDRESS) {
      result.AppendErrorWithFormatv("Invalid address expression, {0}",
                                    error.AsCString());
      return false;
    }

    // Default 1 byte beyond start, rounds up to at most 1 granule later
    addr_t end_addr = start_addr + 1;

    if (command.GetArgumentCount() > 1) {
      end_addr = OptionArgParser::ToAddress(&m_exe_ctx, command[1].ref(),
                                            LLDB_INVALID_ADDRESS, &error);
      if (end_addr == LLDB_INVALID_ADDRESS) {
        result.AppendErrorWithFormatv("Invalid end address expression, {0}",
                                      error.AsCString());
        return false;
      }
    }

    Process *process = m_exe_ctx.GetProcessPtr();
    llvm::Expected<const MemoryTagManager *> tag_manager_or_err =
        process->GetMemoryTagManager(start_addr, end_addr);

    if (!tag_manager_or_err) {
      result.SetError(Status(tag_manager_or_err.takeError()));
      return false;
    }

    const MemoryTagManager *tag_manager = *tag_manager_or_err;
    ptrdiff_t len = tag_manager->AddressDiff(end_addr, start_addr);
    llvm::Expected<std::vector<lldb::addr_t>> tags =
        process->ReadMemoryTags(tag_manager, start_addr, len);

    if (!tags) {
      result.SetError(Status(tags.takeError()));
      return false;
    }

    result.AppendMessageWithFormatv("Logical tag: {0:x}",
                                    tag_manager->GetLogicalTag(start_addr));
    result.AppendMessage("Allocation tags:");

    MemoryTagManager::TagRange initial_range(start_addr, len);
    addr_t addr = tag_manager->ExpandToGranule(initial_range).GetRangeBase();
    for (auto tag : *tags) {
      addr_t next_addr = addr + tag_manager->GetGranuleSize();
      // Showing tagged adresses here until we have non address bit handling
      result.AppendMessageWithFormatv("[{0:x}, {1:x}): {2:x}", addr, next_addr,
                                      tag);
      addr = next_addr;
    }

    result.SetStatus(eReturnStatusSuccessFinishResult);
    return true;
  }
};

CommandObjectMemoryTag::CommandObjectMemoryTag(CommandInterpreter &interpreter)
    : CommandObjectMultiword(
          interpreter, "tag", "Commands for manipulating memory tags",
          "memory tag <sub-command> [<sub-command-options>]") {
  CommandObjectSP read_command_object(
      new CommandObjectMemoryTagRead(interpreter));
  read_command_object->SetCommandName("memory tag read");
  LoadSubCommand("read", read_command_object);
}

CommandObjectMemoryTag::~CommandObjectMemoryTag() = default;
