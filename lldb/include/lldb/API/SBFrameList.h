//===----------------------------------------------------------------------===//
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

/// Represents a list of SBFrame objects.
///
/// SBFrameList provides a way to iterate over stack frames lazily,
/// materializing frames on-demand as they are accessed. This is more
/// efficient than eagerly creating all frames upfront.
class LLDB_API SBFrameList {
public:
  SBFrameList();

  SBFrameList(const lldb::SBFrameList &rhs);

  ~SBFrameList();

  const lldb::SBFrameList &operator=(const lldb::SBFrameList &rhs);

  explicit operator bool() const;

  bool IsValid() const;

  /// Returns the number of frames in the list.
  uint32_t GetSize() const;

  /// Returns the frame at the given index.
  ///
  /// \param[in] idx
  ///     The index of the frame to retrieve (0-based).
  ///
  /// \return
  ///     An SBFrame object for the frame at the specified index.
  ///     Returns an invalid SBFrame if idx is out of range.
  lldb::SBFrame GetFrameAtIndex(uint32_t idx) const;

  /// Get the thread associated with this frame list.
  ///
  /// \return
  ///     An SBThread object representing the thread.
  lldb::SBThread GetThread() const;

  /// Clear all frames from this list.
  void Clear();

  /// Get a description of this frame list.
  ///
  /// \param[in] description
  ///     The stream to write the description to.
  ///
  /// \return
  ///     True if the description was successfully written.
  bool GetDescription(lldb::SBStream &description) const;

protected:
  friend class SBThread;

  friend class lldb_private::python::SWIGBridge;
  friend class lldb_private::lua::SWIGBridge;
  friend class lldb_private::ScriptInterpreter;

private:
  SBFrameList(const lldb::StackFrameListSP &frame_list_sp);

  void SetFrameList(const lldb::StackFrameListSP &frame_list_sp);

  // This needs to be a shared_ptr since an SBFrameList can be passed to
  // scripting affordances like ScriptedFrameProviders but also out of
  // convenience because Thread::GetStackFrameList returns a StackFrameListSP.
  lldb::StackFrameListSP m_opaque_sp;
};

} // namespace lldb

#endif // LLDB_API_SBFRAMELIST_H
