//===-- CommandObjectLanguage.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandObjectLanguage.h"



#include "lldb/Target/LanguageRuntime.h"

using namespace lldb;
using namespace lldb_private;

CommandObjectLanguage::CommandObjectLanguage(CommandInterpreter &interpreter)
    : CommandObjectMultiword(
          interpreter, "language", "Commands specific to a source language.",
          "language <language-name> <subcommand> [<subcommand-options>]") {
  // Let the LanguageRuntime populates this command with subcommands
  LanguageRuntime::InitializeCommands(this);
  SetHelpLong(
      R"(
Language specific subcommands may be used directly (without the `language
<language-name>` prefix), when stopped on a frame written in that language. For
example, from a C++ frame, users may run `demangle` directly, instead of
`language cplusplus demangle`.

Language specific subcommands are only available when the command name cannot be
misinterpreted. Take the `demangle` command for example, if a Python command
named `demangle-tree` were loaded, then the invocation `demangle` would run
`demangle-tree`, not `language cplusplus demangle`.
      )");
}

CommandObjectLanguage::~CommandObjectLanguage() = default;
