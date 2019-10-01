//===-- StreamFile.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_StreamFile_h_
#define liblldb_StreamFile_h_

#include "lldb/Host/File.h"
#include "lldb/Utility/Stream.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"

#include <stdint.h>
#include <stdio.h>

namespace lldb_private {

class StreamFile : public Stream {
public:
  // Constructors and Destructors
  StreamFile();

  StreamFile(uint32_t flags, uint32_t addr_size, lldb::ByteOrder byte_order);

  StreamFile(int fd, bool transfer_ownership);

  StreamFile(const char *path);

  StreamFile(const char *path, uint32_t options,
             uint32_t permissions = lldb::eFilePermissionsFileDefault);

  StreamFile(FILE *fh, bool transfer_ownership);

  ~StreamFile() override;

  File &GetFile() { return m_file; }

  const File &GetFile() const { return m_file; }

  void Flush() override;


protected:
  // Classes that inherit from StreamFile can see and modify these
  File m_file;
  size_t WriteImpl(const void *s, size_t length) override;

private:
  DISALLOW_COPY_AND_ASSIGN(StreamFile);
};

} // namespace lldb_private

#endif // liblldb_StreamFile_h_
