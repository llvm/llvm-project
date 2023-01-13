//===-- CommandObjectPlatform.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_COMMANDS_COMMANDOBJECTPLATFORM_H
#define LLDB_SOURCE_COMMANDS_COMMANDOBJECTPLATFORM_H

#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Interpreter/OptionGroupPlatform.h"

namespace lldb_private {

// CommandObjectPlatform

class CommandObjectPlatform : public CommandObjectMultiword {
public:
  CommandObjectPlatform(CommandInterpreter &interpreter);

  ~CommandObjectPlatform() override;

private:
  CommandObjectPlatform(const CommandObjectPlatform &) = delete;
  const CommandObjectPlatform &
  operator=(const CommandObjectPlatform &) = delete;
};

class CommandObjectPlatformSelect : public CommandObjectParsed {
public:
  CommandObjectPlatformSelect(CommandInterpreter &interpreter);

  ~CommandObjectPlatformSelect() override = default;

  void HandleCompletion(CompletionRequest &request) override;

  Options *GetOptions() override;

protected:
  bool DoExecute(Args &args, CommandReturnObject &result) override;

  OptionGroupOptions m_option_group;
  OptionGroupPlatform m_platform_options;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_COMMANDS_COMMANDOBJECTPLATFORM_H
