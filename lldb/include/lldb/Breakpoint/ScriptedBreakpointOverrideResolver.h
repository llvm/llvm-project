//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_BREAKPOINT_SCRIPTEDBREAKPOINTOVERRIDERESOLVER_H
#define LLDB_BREAKPOINT_SCRIPTEDBREAKPOINTOVERRIDERESOLVER_H

#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/Target.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class ScriptedBreakpointResolverOverride
    : public Target::BreakpointResolverOverride {
public:
  ScriptedBreakpointResolverOverride(Target &target,
                                     const std::string &description,
                                     const std::string &class_name,
                                     StructuredDataImpl &args_data)
      : Target::BreakpointResolverOverride(target, description),
        m_args_data(args_data), m_class_name(class_name) {}

  Target::BreakpointResolverOverrideUP
  CopyIntoNewTarget(Target &target) override {
    return Target::BreakpointResolverOverrideUP(
        new ScriptedBreakpointResolverOverride(target, m_desc, m_class_name,
                                               m_args_data));
  }

  lldb::BreakpointResolverSP
  CheckForOverride(Target &target,
                   lldb::BreakpointResolverSP initial_sp) override;

  llvm::Error Validate() override;

private:
  StructuredDataImpl m_args_data;
  std::string m_class_name;
};
} // namespace lldb_private
#endif // LLDB_BREAKPOINT_SCRIPTEDBREAKPOINTOVERRIDERESOLVER_H
