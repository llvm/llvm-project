//===-- SBFile.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBFile_h_
#define LLDB_SBFile_h_

#include "lldb/API/SBDefines.h"

namespace lldb {

class LLDB_API SBFile {
  friend class SBInstruction;
  friend class SBInstructionList;
  friend class SBDebugger;
  friend class SBCommandReturnObject;
  friend class SBProcess;

public:
  SBFile();
  SBFile(FileSP file_sp);
  SBFile(FILE *file, bool transfer_ownership);
  SBFile(int fd, const char *mode, bool transfer_ownership);
  ~SBFile();

  SBError Read(uint8_t *buf, size_t num_bytes, size_t *bytes_read);
  SBError Write(const uint8_t *buf, size_t num_bytes, size_t *bytes_written);
  SBError Flush();
  bool IsValid() const;
  SBError Close();

  operator bool() const;
  bool operator!() const;

  FileSP GetFile() const;

private:
  FileSP m_opaque_sp;
};

} // namespace lldb

#endif // LLDB_SBFile_h_
