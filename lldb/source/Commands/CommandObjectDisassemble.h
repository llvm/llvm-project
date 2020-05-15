//===-- CommandObjectDisassemble.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_COMMANDS_COMMANDOBJECTDISASSEMBLE_H
#define LLDB_SOURCE_COMMANDS_COMMANDOBJECTDISASSEMBLE_H

#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Utility/ArchSpec.h"

namespace lldb_private {

// CommandObjectDisassemble

class CommandObjectDisassemble : public CommandObjectParsed {
public:
  class CommandOptions : public Options {
  public:
    CommandOptions();

    ~CommandOptions() override;

    Status SetOptionValue(uint32_t option_idx, llvm::StringRef option_arg,
                          ExecutionContext *execution_context) override;

    void OptionParsingStarting(ExecutionContext *execution_context) override;

    llvm::ArrayRef<OptionDefinition> GetDefinitions() override;

    const char *GetPluginName() {
      return (plugin_name.empty() ? nullptr : plugin_name.c_str());
    }

    const char *GetFlavorString() {
      if (flavor_string.empty() || flavor_string == "default")
        return nullptr;
      return flavor_string.c_str();
    }

    Status OptionParsingFinished(ExecutionContext *execution_context) override;

    bool show_mixed; // Show mixed source/assembly
    bool show_bytes;
    uint32_t num_lines_context;
    uint32_t num_instructions;
    bool raw;
    std::string func_name;
    bool current_function;
    lldb::addr_t start_addr;
    lldb::addr_t end_addr;
    bool at_pc;
    bool frame_line;
    std::string plugin_name;
    std::string flavor_string;
    ArchSpec arch;
    bool some_location_specified; // If no location was specified, we'll select
                                  // "at_pc".  This should be set
    // in SetOptionValue if anything the selects a location is set.
    lldb::addr_t symbol_containing_addr;
    bool force = false;
  };

  CommandObjectDisassemble(CommandInterpreter &interpreter);

  ~CommandObjectDisassemble() override;

  Options *GetOptions() override { return &m_options; }

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override;

  llvm::Expected<std::vector<AddressRange>>
  GetRangesForSelectedMode(CommandReturnObject &result);

  llvm::Expected<std::vector<AddressRange>> GetContainingAddressRanges();
  llvm::Expected<std::vector<AddressRange>> GetCurrentFunctionRanges();
  llvm::Expected<std::vector<AddressRange>> GetCurrentLineRanges();
  llvm::Expected<std::vector<AddressRange>>
  GetNameRanges(CommandReturnObject &result);
  llvm::Expected<std::vector<AddressRange>> GetPCRanges();
  llvm::Expected<std::vector<AddressRange>> GetStartEndAddressRanges();

  llvm::Error CheckRangeSize(const AddressRange &range, llvm::StringRef what);

  CommandOptions m_options;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_COMMANDS_COMMANDOBJECTDISASSEMBLE_H
