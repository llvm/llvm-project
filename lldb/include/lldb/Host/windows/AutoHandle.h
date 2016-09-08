//===-- AutoHandle.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_lldb_Host_windows_AutoHandle_h_
#define LLDB_lldb_Host_windows_AutoHandle_h_

namespace lldb_private {

class AutoHandle {
public:
  AutoHandle(HANDLE handle, HANDLE invalid_value = INVALID_HANDLE_VALUE)
      : m_handle(handle), m_invalid_value(invalid_value) {}

  ~AutoHandle() {
    if (m_handle != m_invalid_value)
      ::CloseHandle(m_handle);
  }

  bool IsValid() const { return m_handle != m_invalid_value; }

  HANDLE get() const { return m_handle; }

private:
  HANDLE m_handle;
  HANDLE m_invalid_value;
};
}

#endif
