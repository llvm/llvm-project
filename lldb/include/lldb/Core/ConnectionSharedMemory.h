//===-- ConnectionSharedMemory.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ConnectionSharedMemory_h_
#define liblldb_ConnectionSharedMemory_h_

// C Includes
// C++ Includes
#include <string>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/Connection.h"
#include "lldb/Core/DataBufferMemoryMap.h"

namespace lldb_private {

class ConnectionSharedMemory : public Connection {
public:
  ConnectionSharedMemory();

  ~ConnectionSharedMemory() override;

  bool IsConnected() const override;

  virtual lldb::ConnectionStatus BytesAvailable(uint32_t timeout_usec,
                                                Error *error_ptr);

  lldb::ConnectionStatus Connect(const char *s, Error *error_ptr) override;

  lldb::ConnectionStatus Disconnect(Error *error_ptr) override;

  size_t Read(void *dst, size_t dst_len, uint32_t timeout_usec,
              lldb::ConnectionStatus &status, Error *error_ptr) override;

  size_t Write(const void *src, size_t src_len, lldb::ConnectionStatus &status,
               Error *error_ptr) override;

  std::string GetURI() override;

  lldb::ConnectionStatus Open(bool create, const char *name, size_t size,
                              Error *error_ptr);

protected:
  std::string m_name;
  int m_fd; // One buffer that contains all we need
  DataBufferMemoryMap m_mmap;

private:
  DISALLOW_COPY_AND_ASSIGN(ConnectionSharedMemory);
};

} // namespace lldb_private

#endif // liblldb_ConnectionSharedMemory_h_
