//===-- ScriptedInterfaceUsages.h ---------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_SCRIPTEDINTERFACEUSAGES_H
#define LLDB_INTERPRETER_SCRIPTEDINTERFACEUSAGES_H

#include "lldb/lldb-types.h"

#include "lldb/Utility/Stream.h"
#include "llvm/ADT/StringRef.h"

namespace lldb_private {
class ScriptedInterfaceUsages {
public:
  ScriptedInterfaceUsages() = default;
  ScriptedInterfaceUsages(const std::vector<llvm::StringRef> ci_usages,
                          const std::vector<llvm::StringRef> sbapi_usages)
      : m_command_interpreter_usages(ci_usages), m_sbapi_usages(sbapi_usages) {}

  const std::vector<llvm::StringRef> &GetCommandInterpreterUsages() const {
    return m_command_interpreter_usages;
  }

  const std::vector<llvm::StringRef> &GetSBAPIUsages() const {
    return m_sbapi_usages;
  }

  enum class UsageKind { CommandInterpreter, API };

  void Dump(Stream &s, UsageKind kind) const;

private:
  std::vector<llvm::StringRef> m_command_interpreter_usages;
  std::vector<llvm::StringRef> m_sbapi_usages;
};
} // namespace lldb_private

#endif // LLDB_INTERPRETER_SCRIPTEDINTERFACEUSAGES_H
