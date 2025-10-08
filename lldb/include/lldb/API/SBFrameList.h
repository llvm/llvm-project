//===-- SBFrameList.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_API_SBFRAMELIST_H
#define LLDB_API_SBFRAMELIST_H

#include "lldb/API/SBDefines.h"

namespace lldb_private {
class ScriptInterpreter;
namespace python {
class SWIGBridge;
}
namespace lua {
class SWIGBridge;
}
} // namespace lldb_private

namespace lldb {

class LLDB_API SBFrameList {
public:
  SBFrameList();

  SBFrameList(const lldb::SBFrameList &rhs);

  ~SBFrameList();

  const lldb::SBFrameList &operator=(const lldb::SBFrameList &rhs);

  explicit operator bool() const;

  bool IsValid() const;

  uint32_t GetSize() const;

  lldb::SBFrame GetFrameAtIndex(uint32_t idx) const;

  void Clear();

  void Append(const lldb::SBFrame &frame);

  void Append(const lldb::SBFrameList &frame_list);

  bool GetDescription(lldb::SBStream &description) const;

protected:
  friend class SBThread;

  friend class lldb_private::python::SWIGBridge;
  friend class lldb_private::lua::SWIGBridge;
  friend class lldb_private::ScriptInterpreter;

private:
  SBFrameList(const lldb::StackFrameListSP &frame_list_sp);

  void SetOpaque(const lldb::StackFrameListSP &frame_list_sp);

  lldb::StackFrameListSP m_opaque_sp;
};

} // namespace lldb

#endif // LLDB_API_SBFRAMELIST_H
