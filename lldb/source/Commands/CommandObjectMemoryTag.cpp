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
        process->GetMemoryTagManager();

    if (!tag_manager_or_err) {
      result.SetError(Status(tag_manager_or_err.takeError()));
      return false;
    }

    const MemoryTagManager *tag_manager = *tag_manager_or_err;

    MemoryRegionInfos memory_regions;
    // If this fails the list of regions is cleared, so we don't need to read
    // the return status here.
    process->GetMemoryRegions(memory_regions);
    llvm::Expected<MemoryTagManager::TagRange> tagged_range =
        tag_manager->MakeTaggedRange(start_addr, end_addr, memory_regions);

    if (!tagged_range) {
      result.SetError(Status(tagged_range.takeError()));
      return false;
    }

    llvm::Expected<std::vector<lldb::addr_t>> tags = process->ReadMemoryTags(
        tagged_range->GetRangeBase(), tagged_range->GetByteSize());

    if (!tags) {
      result.SetError(Status(tags.takeError()));
      return false;
    }

    result.AppendMessageWithFormatv("Logical tag: {0:x}",
                                    tag_manager->GetLogicalTag(start_addr));
    result.AppendMessage("Allocation tags:");

    addr_t addr = tagged_range->GetRangeBase();
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

#define LLDB_OPTIONS_memory_tag_write
#include "CommandOptions.inc"

class CommandObjectMemoryTagWrite : public CommandObjectParsed {
public:
  CommandObjectMemoryTagWrite(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "tag",
                            "Write memory tags starting from the granule that "
                            "contains the given address.",
                            nullptr,
                            eCommandRequiresTarget | eCommandRequiresProcess |
                                eCommandProcessMustBePaused) {
    // Address
    m_arguments.push_back(
        CommandArgumentEntry{CommandArgumentData(eArgTypeAddressOrExpression)});
    // One or more tag values
    m_arguments.push_back(CommandArgumentEntry{
        CommandArgumentData(eArgTypeValue, eArgRepeatPlus)});
  }

  ~CommandObjectMemoryTagWrite() override = default;

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override {
    if (command.GetArgumentCount() < 2) {
      result.AppendError("wrong number of arguments; expected "
                         "<address-expression> <tag> [<tag> [...]]");
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

    command.Shift(); // shift off start address

    std::vector<lldb::addr_t> tags;
    for (auto &entry : command) {
      lldb::addr_t tag_value;
      // getAsInteger returns true on failure
      if (entry.ref().getAsInteger(0, tag_value)) {
        result.AppendErrorWithFormat(
            "'%s' is not a valid unsigned decimal string value.\n",
            entry.c_str());
        return false;
      }
      tags.push_back(tag_value);
    }

    Process *process = m_exe_ctx.GetProcessPtr();
    llvm::Expected<const MemoryTagManager *> tag_manager_or_err =
        process->GetMemoryTagManager();

    if (!tag_manager_or_err) {
      result.SetError(Status(tag_manager_or_err.takeError()));
      return false;
    }

    const MemoryTagManager *tag_manager = *tag_manager_or_err;

    MemoryRegionInfos memory_regions;
    // If this fails the list of regions is cleared, so we don't need to read
    // the return status here.
    process->GetMemoryRegions(memory_regions);

    // We have to assume start_addr is not granule aligned.
    // So if we simply made a range:
    // (start_addr, start_addr + (N * granule_size))
    // We would end up with a range that isn't N granules but N+1
    // granules. To avoid this we'll align the start first using the method that
    // doesn't check memory attributes. (if the final range is untagged we'll
    // handle that error later)
    lldb::addr_t aligned_start_addr =
        tag_manager->ExpandToGranule(MemoryTagManager::TagRange(start_addr, 1))
            .GetRangeBase();

    // Now we've aligned the start address so if we ask for another range
    // using the number of tags N, we'll get back a range that is also N
    // granules in size.
    llvm::Expected<MemoryTagManager::TagRange> tagged_range =
        tag_manager->MakeTaggedRange(
            aligned_start_addr,
            aligned_start_addr + (tags.size() * tag_manager->GetGranuleSize()),
            memory_regions);

    if (!tagged_range) {
      result.SetError(Status(tagged_range.takeError()));
      return false;
    }

    Status status = process->WriteMemoryTags(tagged_range->GetRangeBase(),
                                             tagged_range->GetByteSize(), tags);

    if (status.Fail()) {
      result.SetError(status);
      return false;
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

  CommandObjectSP write_command_object(
      new CommandObjectMemoryTagWrite(interpreter));
  write_command_object->SetCommandName("memory tag write");
  LoadSubCommand("write", write_command_object);
}

CommandObjectMemoryTag::~CommandObjectMemoryTag() = default;
