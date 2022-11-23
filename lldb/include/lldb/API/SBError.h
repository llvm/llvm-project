//===-- SBError.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_API_SBERROR_H
#define LLDB_API_SBERROR_H

#include "lldb/API/SBDefines.h"

namespace lldb_private {
class ScriptInterpreter;
} // namespace lldb_private

namespace lldb {

class LLDB_API SBError {
public:
  SBError();

  SBError(const lldb::SBError &rhs);

  SBError(const lldb_private::Status &error);

  ~SBError();

  const SBError &operator=(const lldb::SBError &rhs);

  /// Get the error string as a NULL terminated UTF8 c-string.
  ///
  /// This SBError object owns the returned string and this object must be kept
  /// around long enough to use the returned string.
  const char *GetCString() const;

  void Clear();

  bool Fail() const;

  bool Success() const;

  uint32_t GetError() const;

  lldb::ErrorType GetType() const;

  void SetError(uint32_t err, lldb::ErrorType type);

  void SetErrorToErrno();

  void SetErrorToGenericError();

  void SetErrorString(const char *err_str);

  int SetErrorStringWithFormat(const char *format, ...)
      __attribute__((format(printf, 2, 3)));

  explicit operator bool() const;

  bool IsValid() const;

  bool GetDescription(lldb::SBStream &description);

protected:
  friend class SBBreakpoint;
  friend class SBBreakpointLocation;
  friend class SBBreakpointName;
  friend class SBCommandReturnObject;
  friend class SBCommunication;
  friend class SBData;
  friend class SBDebugger;
  friend class SBFile;
  friend class SBHostOS;
  friend class SBPlatform;
  friend class SBProcess;
  friend class SBReproducer;
  friend class SBStructuredData;
  friend class SBTarget;
  friend class SBThread;
  friend class SBTrace;
  friend class SBValue;
  friend class SBValueList;
  friend class SBWatchpoint;

  friend class lldb_private::ScriptInterpreter;

  lldb_private::Status *get();

  lldb_private::Status *operator->();

  const lldb_private::Status &operator*() const;

  lldb_private::Status &ref();

  void SetError(const lldb_private::Status &lldb_error);

private:
  std::unique_ptr<lldb_private::Status> m_opaque_up;

  void CreateIfNeeded();
};

} // namespace lldb

#endif // LLDB_API_SBERROR_H
