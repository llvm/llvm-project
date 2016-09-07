//===-- ConnectionMachPort.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#if defined(__APPLE__)

#ifndef liblldb_ConnectionMachPort_h_
#define liblldb_ConnectionMachPort_h_

// C Includes
#include <mach/kern_return.h>
#include <mach/port.h>

// C++ Includes
#include <string>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/Connection.h"

class ConnectionMachPort : public lldb_private::Connection {
public:
  ConnectionMachPort();

  virtual ~ConnectionMachPort();

  virtual bool IsConnected() const;

  virtual lldb::ConnectionStatus BytesAvailable(uint32_t timeout_usec,
                                                lldb_private::Error *error_ptr);

  virtual lldb::ConnectionStatus Connect(const char *s,
                                         lldb_private::Error *error_ptr);

  virtual lldb::ConnectionStatus Disconnect(lldb_private::Error *error_ptr);

  virtual size_t Read(void *dst, size_t dst_len, uint32_t timeout_usec,
                      lldb::ConnectionStatus &status,
                      lldb_private::Error *error_ptr);

  virtual size_t Write(const void *src, size_t src_len,
                       lldb::ConnectionStatus &status,
                       lldb_private::Error *error_ptr);

  virtual std::string GetURI();

  lldb::ConnectionStatus BootstrapCheckIn(const char *port_name,
                                          lldb_private::Error *error_ptr);

  lldb::ConnectionStatus BootstrapLookup(const char *port_name,
                                         lldb_private::Error *error_ptr);

  struct PayloadType {
    uint32_t command;
    uint32_t data_length;
    uint8_t data[32];
  };

  kern_return_t Send(const PayloadType &payload);

  kern_return_t Receive(PayloadType &payload);

protected:
  mach_port_t m_task;
  mach_port_t m_port;

private:
  std::string m_uri;

  DISALLOW_COPY_AND_ASSIGN(ConnectionMachPort);
};

#endif // liblldb_ConnectionMachPort_h_

#endif // #if defined(__APPLE__)
