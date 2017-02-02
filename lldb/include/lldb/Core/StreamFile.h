//===-- StreamFile.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_StreamFile_h_
#define liblldb_StreamFile_h_

// C Includes
// C++ Includes

#include <string>

// Other libraries and framework includes
// Project includes

#include "lldb/Host/File.h"
#include "lldb/Utility/Stream.h"

namespace lldb_private {

class StreamFile : public Stream {
public:
  //------------------------------------------------------------------
  // Constructors and Destructors
  //------------------------------------------------------------------
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

  size_t Write(const void *s, size_t length) override;

protected:
  //------------------------------------------------------------------
  // Classes that inherit from StreamFile can see and modify these
  //------------------------------------------------------------------
  File m_file;

private:
  DISALLOW_COPY_AND_ASSIGN(StreamFile);
};

} // namespace lldb_private

#endif // liblldb_StreamFile_h_
