//===-- SWIG Interface for SBTraceCursor.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents a trace cursor."
) SBTrace;
class LLDB_API SBTraceCursor {
public:
  SBTraceCursor();

  SBTraceCursor(lldb::TraceCursorSP trace_cursor_sp);

  void SetForwards(bool forwards);

  bool IsForwards() const;

  void Next();

  bool HasValue();

  bool GoToId(lldb::user_id_t id);

  bool HasId(lldb::user_id_t id) const;

  lldb::user_id_t GetId() const;

  bool Seek(int64_t offset, lldb::TraceCursorSeekType origin);

  lldb::TraceItemKind GetItemKind() const;

  bool IsError() const;

  const char *GetError() const;

  bool IsEvent() const;

  lldb::TraceEvent GetEventType() const;

  const char *GetEventTypeAsString() const;

  bool IsInstruction() const;

  lldb::addr_t GetLoadAddress() const;

  lldb::cpu_id_t GetCPU() const;

  bool IsValid() const;

  explicit operator bool() const;
};
} // namespace lldb
