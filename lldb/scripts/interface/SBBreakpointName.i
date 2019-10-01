//===-- SWIG interface for SBBreakpointName.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {
%feature("docstring",
"Represents a breakpoint name registered in a given SBTarget.

Breakpoint names provide a way to act on groups of breakpoints.  When you add a
name to a group of breakpoints, you can then use the name in all the command
line lldb commands for that name.  You can also configure the SBBreakpointName
options and those options will be propagated to any SBBreakpoints currently
using that name.  Adding a name to a breakpoint will also apply any of the
set options to that breakpoint.

You can also set permissions on a breakpoint name to disable listing, deleting
and disabling breakpoints.  That will disallow the given operation for breakpoints
except when the breakpoint is mentioned by ID.  So for instance deleting all the
breakpoints won't delete breakpoints so marked."
) SBBreakpointName;
class LLDB_API SBBreakpointName {
public:
  SBBreakpointName();

  SBBreakpointName(SBTarget &target, const char *name);

  SBBreakpointName(SBBreakpoint &bkpt, const char *name);

  SBBreakpointName(const lldb::SBBreakpointName &rhs);

  ~SBBreakpointName();

  const lldb::SBBreakpointName &operator=(const lldb::SBBreakpointName &rhs);

  // Tests to see if the opaque breakpoint object in this object matches the
  // opaque breakpoint object in "rhs".
  bool operator==(const lldb::SBBreakpointName &rhs);

  bool operator!=(const lldb::SBBreakpointName &rhs);

  explicit operator bool() const;

  bool IsValid() const;

  const char *GetName() const;

  void SetEnabled(bool enable);

  bool IsEnabled();

  void SetOneShot(bool one_shot);

  bool IsOneShot() const;

  void SetIgnoreCount(uint32_t count);

  uint32_t GetIgnoreCount() const;

  void SetCondition(const char *condition);

  const char *GetCondition();

  void SetAutoContinue(bool auto_continue);

  bool GetAutoContinue();

  void SetThreadID(lldb::tid_t sb_thread_id);

  lldb::tid_t GetThreadID();

  void SetThreadIndex(uint32_t index);

  uint32_t GetThreadIndex() const;

  void SetThreadName(const char *thread_name);

  const char *GetThreadName() const;

  void SetQueueName(const char *queue_name);

  const char *GetQueueName() const;

  void SetScriptCallbackFunction(const char *callback_function_name);

  void SetCommandLineCommands(SBStringList &commands);

  bool GetCommandLineCommands(SBStringList &commands);

  SBError SetScriptCallbackBody(const char *script_body_text);

  const char *GetHelpString() const;
  void SetHelpString(const char *help_string);

  bool GetAllowList() const;
  void SetAllowList(bool value);

  bool GetAllowDelete();
  void SetAllowDelete(bool value);

  bool GetAllowDisable();
  void SetAllowDisable(bool value);

  bool GetDescription(lldb::SBStream &description);

};

} // namespace lldb

