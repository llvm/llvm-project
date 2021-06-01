//===-- SBCommandReturnObject.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_API_SBCOMMANDRETURNOBJECT_H
#define LLDB_API_SBCOMMANDRETURNOBJECT_H

#include <cstdio>

#include <memory>

#include "lldb/API/SBDefines.h"

namespace lldb_private {
class SBCommandReturnObjectImpl;
}

namespace lldb {

class LLDB_API SBCommandReturnObject {
public:
  SBCommandReturnObject();

  SBCommandReturnObject(lldb_private::CommandReturnObject &ref);

  // rvalue ctor+assignment are incompatible with Reproducers.

  SBCommandReturnObject(const lldb::SBCommandReturnObject &rhs);

  ~SBCommandReturnObject();

  lldb::SBCommandReturnObject &
  operator=(const lldb::SBCommandReturnObject &rhs);

  explicit operator bool() const;

  bool IsValid() const;

  const char *GetOutput();

  const char *GetError();

  size_t PutOutput(FILE *fh); // DEPRECATED

  size_t PutOutput(SBFile file);

  size_t PutOutput(FileSP file);

  size_t GetOutputSize();

  size_t GetErrorSize();

  size_t PutError(FILE *fh); // DEPRECATED

  size_t PutError(SBFile file);

  size_t PutError(FileSP file);

  void Clear();

  lldb::ReturnStatus GetStatus();

  void SetStatus(lldb::ReturnStatus status);

  bool Succeeded();

  bool HasResult();

  void AppendMessage(const char *message);

  void AppendWarning(const char *message);

  bool GetDescription(lldb::SBStream &description);

  void SetImmediateOutputFile(FILE *fh); // DEPRECATED

  void SetImmediateErrorFile(FILE *fh); // DEPRECATED

  void SetImmediateOutputFile(FILE *fh, bool transfer_ownership); // DEPRECATED

  void SetImmediateErrorFile(FILE *fh, bool transfer_ownership); // DEPRECATED

  void SetImmediateOutputFile(SBFile file);

  void SetImmediateErrorFile(SBFile file);

  void SetImmediateOutputFile(FileSP file);

  void SetImmediateErrorFile(FileSP file);

  void PutCString(const char *string, int len = -1);

  size_t Printf(const char *format, ...) __attribute__((format(printf, 2, 3)));

  const char *GetOutput(bool only_if_no_immediate);

  const char *GetError(bool only_if_no_immediate);

  void SetError(lldb::SBError &error,
                const char *fallback_error_cstr = nullptr);

  void SetError(const char *error_cstr);

protected:
  friend class SBCommandInterpreter;
  friend class SBOptions;

  lldb_private::CommandReturnObject *operator->() const;

  lldb_private::CommandReturnObject *get() const;

  lldb_private::CommandReturnObject &operator*() const;

private:
  lldb_private::CommandReturnObject &ref() const;

  std::unique_ptr<lldb_private::SBCommandReturnObjectImpl> m_opaque_up;
};

} // namespace lldb

#endif // LLDB_API_SBCOMMANDRETURNOBJECT_H
