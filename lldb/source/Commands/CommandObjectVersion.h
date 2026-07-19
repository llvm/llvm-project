//===-- CommandObjectVersion.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_COMMANDS_COMMANDOBJECTVERSION_H
#define LLDB_SOURCE_COMMANDS_COMMANDOBJECTVERSION_H

#include "lldb/Host/OptionParser.h"
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/Options.h"

namespace lldb_private {

class CommandObjectVersion : public CommandObjectParsed {
public:
  CommandObjectVersion(CommandInterpreter &interpreter);

  ~CommandObjectVersion() override;

  class CommandOptions : public Options {
  public:
    CommandOptions() = default;

    ~CommandOptions() override = default;

    Status SetOptionValue(uint32_t option_idx, llvm::StringRef option_arg,
                          ExecutionContext *execution_context) override {
      Status error;
      const int short_option = m_getopt_table[option_idx].val;

      switch (short_option) {
      case 'v':
        verbose = true;
        break;
      default:
        llvm_unreachable("Unimplemented option");
      }

      return error;
    }

    void OptionParsingStarting(ExecutionContext *execution_context) override {
      verbose = false;
    }

    llvm::ArrayRef<OptionDefinition> GetDefinitions() override;

    bool verbose;
  };

  Options *GetOptions() override { return &m_options; }

protected:
  void DoExecute(Args &args, CommandReturnObject &result) override;

private:
  CommandOptions m_options;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_COMMANDS_COMMANDOBJECTVERSION_H
