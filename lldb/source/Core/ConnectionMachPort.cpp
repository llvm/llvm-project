//===-- ConnectionMachPort.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#if defined(__APPLE__)

#include "lldb/Core/ConnectionMachPort.h"

// C Includes
#include <mach/mach.h>
#include <servers/bootstrap.h>

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Communication.h"
#include "lldb/Core/Log.h"

using namespace lldb;
using namespace lldb_private;

struct MessageType {
  mach_msg_header_t head;
  ConnectionMachPort::PayloadType payload;
};

ConnectionMachPort::ConnectionMachPort()
    : Connection(), m_task(mach_task_self()), m_port(MACH_PORT_TYPE_NONE) {}

ConnectionMachPort::~ConnectionMachPort() { Disconnect(NULL); }

bool ConnectionMachPort::IsConnected() const {
  return m_port != MACH_PORT_TYPE_NONE;
}

ConnectionStatus ConnectionMachPort::Connect(const char *s, Error *error_ptr) {
  if (IsConnected()) {
    if (error_ptr)
      error_ptr->SetErrorString("already connected");
    return eConnectionStatusError;
  }

  if (s == NULL || s[0] == '\0') {
    if (error_ptr)
      error_ptr->SetErrorString("empty connect URL");
    return eConnectionStatusError;
  }

  ConnectionStatus status = eConnectionStatusError;

  if (0 == strncmp(s, "bootstrap-checkin://", strlen("bootstrap-checkin://"))) {
    s += strlen("bootstrap-checkin://");

    if (*s) {
      status = BootstrapCheckIn(s, error_ptr);
    } else {
      if (error_ptr)
        error_ptr->SetErrorString("bootstrap port name is empty");
    }
  } else if (0 ==
             strncmp(s, "bootstrap-lookup://", strlen("bootstrap-lookup://"))) {
    s += strlen("bootstrap-lookup://");
    if (*s) {
      status = BootstrapLookup(s, error_ptr);
    } else {
      if (error_ptr)
        error_ptr->SetErrorString("bootstrap port name is empty");
    }
  } else {
    if (error_ptr)
      error_ptr->SetErrorStringWithFormat("unsupported connection URL: '%s'",
                                          s);
  }

  if (status == eConnectionStatusSuccess) {
    if (error_ptr)
      error_ptr->Clear();
    m_uri.assign(s);
  } else {
    Disconnect(NULL);
  }

  return status;
}

ConnectionStatus ConnectionMachPort::BootstrapCheckIn(const char *port,
                                                      Error *error_ptr) {
  mach_port_t bootstrap_port = MACH_PORT_TYPE_NONE;

  /* Getting bootstrap server port */
  kern_return_t kret =
      task_get_bootstrap_port(mach_task_self(), &bootstrap_port);
  if (kret == KERN_SUCCESS) {
    name_t port_name;
    int len = snprintf(port_name, sizeof(port_name), "%s", port);
    if (static_cast<size_t>(len) < sizeof(port_name)) {
      kret = ::bootstrap_check_in(bootstrap_port, port_name, &m_port);
    } else {
      Disconnect(NULL);
      if (error_ptr)
        error_ptr->SetErrorString("bootstrap is too long");
      return eConnectionStatusError;
    }
  }

  if (kret != KERN_SUCCESS) {
    Disconnect(NULL);
    if (error_ptr)
      error_ptr->SetError(kret, eErrorTypeMachKernel);
    return eConnectionStatusError;
  }
  return eConnectionStatusSuccess;
}

lldb::ConnectionStatus ConnectionMachPort::BootstrapLookup(const char *port,
                                                           Error *error_ptr) {
  name_t port_name;

  if (port && port[0]) {
    if (static_cast<size_t>(::snprintf(port_name, sizeof(port_name), "%s",
                                       port)) >= sizeof(port_name)) {
      if (error_ptr)
        error_ptr->SetErrorString("port netname is too long");
      return eConnectionStatusError;
    }
  } else {
    if (error_ptr)
      error_ptr->SetErrorString("empty port netname");
    return eConnectionStatusError;
  }

  mach_port_t bootstrap_port = MACH_PORT_TYPE_NONE;

  /* Getting bootstrap server port */
  kern_return_t kret =
      task_get_bootstrap_port(mach_task_self(), &bootstrap_port);
  if (kret == KERN_SUCCESS) {
    kret = ::bootstrap_look_up(bootstrap_port, port_name, &m_port);
  }

  if (kret != KERN_SUCCESS) {
    if (error_ptr)
      error_ptr->SetError(kret, eErrorTypeMachKernel);
    return eConnectionStatusError;
  }

  return eConnectionStatusSuccess;
}

ConnectionStatus ConnectionMachPort::Disconnect(Error *error_ptr) {
  kern_return_t kret;

  // TODO: verify if we need to netname_check_out for
  // either or both
  if (m_port != MACH_PORT_TYPE_NONE) {
    kret = ::mach_port_deallocate(m_task, m_port);
    if (error_ptr)
      error_ptr->SetError(kret, eErrorTypeMachKernel);
    m_port = MACH_PORT_TYPE_NONE;
  }
  m_uri.clear();

  return eConnectionStatusSuccess;
}

size_t ConnectionMachPort::Read(void *dst, size_t dst_len,
                                uint32_t timeout_usec, ConnectionStatus &status,
                                Error *error_ptr) {
  PayloadType payload;

  kern_return_t kret = Receive(payload);
  if (kret == KERN_SUCCESS) {
    memcpy(dst, payload.data, payload.data_length);
    status = eConnectionStatusSuccess;
    return payload.data_length;
  }

  if (error_ptr)
    error_ptr->SetError(kret, eErrorTypeMachKernel);
  status = eConnectionStatusError;
  return 0;
}

size_t ConnectionMachPort::Write(const void *src, size_t src_len,
                                 ConnectionStatus &status, Error *error_ptr) {
  PayloadType payload;
  payload.command = 0;
  payload.data_length = src_len;
  const size_t max_payload_size = sizeof(payload.data);
  if (src_len > max_payload_size)
    payload.data_length = max_payload_size;
  memcpy(payload.data, src, payload.data_length);

  if (Send(payload) == KERN_SUCCESS) {
    status = eConnectionStatusSuccess;
    return payload.data_length;
  }
  status = eConnectionStatusError;
  return 0;
}

std::string ConnectionMachPort::GetURI() { return m_uri; }

ConnectionStatus ConnectionMachPort::BytesAvailable(uint32_t timeout_usec,
                                                    Error *error_ptr) {
  return eConnectionStatusLostConnection;
}

kern_return_t ConnectionMachPort::Send(const PayloadType &payload) {
  struct MessageType message;

  /* (i) Form the message : */

  /* (i.a) Fill the header fields : */
  message.head.msgh_bits = MACH_MSGH_BITS_REMOTE(MACH_MSG_TYPE_MAKE_SEND) |
                           MACH_MSGH_BITS_OTHER(MACH_MSGH_BITS_COMPLEX);
  message.head.msgh_size = sizeof(MessageType);
  message.head.msgh_local_port = MACH_PORT_NULL;
  message.head.msgh_remote_port = m_port;

  /* (i.b) Explain the message type ( an integer ) */
  //	message.type.msgt_name = MACH_MSG_TYPE_INTEGER_32;
  //	message.type.msgt_size = 32;
  //	message.type.msgt_number = 1;
  //	message.type.msgt_inline = TRUE;
  //	message.type.msgt_longform = FALSE;
  //	message.type.msgt_deallocate = FALSE;
  /* message.type.msgt_unused = 0; */ /* not needed, I think */

  /* (i.c) Fill the message with the given integer : */
  message.payload = payload;

  /* (ii) Send the message : */
  kern_return_t kret =
      ::mach_msg(&message.head, MACH_SEND_MSG, message.head.msgh_size, 0,
                 MACH_PORT_NULL, MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);

  return kret;
}

kern_return_t ConnectionMachPort::Receive(PayloadType &payload) {
  MessageType message;
  message.head.msgh_size = sizeof(MessageType);

  kern_return_t kret =
      ::mach_msg(&message.head, MACH_RCV_MSG, 0, sizeof(MessageType), m_port,
                 MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);

  if (kret == KERN_SUCCESS)
    payload = message.payload;

  return kret;
}

#endif // #if defined(__APPLE__)
