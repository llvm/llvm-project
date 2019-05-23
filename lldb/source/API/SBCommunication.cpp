//===-- SBCommunication.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBCommunication.h"
#include "SBReproducerPrivate.h"
#include "lldb/API/SBBroadcaster.h"
#include "lldb/Core/Communication.h"
#include "lldb/Host/ConnectionFileDescriptor.h"
#include "lldb/Host/Host.h"

using namespace lldb;
using namespace lldb_private;

SBCommunication::SBCommunication() : m_opaque(nullptr), m_opaque_owned(false) {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBCommunication);
}

SBCommunication::SBCommunication(const char *broadcaster_name)
    : m_opaque(new Communication(broadcaster_name)), m_opaque_owned(true) {
  LLDB_RECORD_CONSTRUCTOR(SBCommunication, (const char *), broadcaster_name);
}

SBCommunication::~SBCommunication() {
  if (m_opaque && m_opaque_owned)
    delete m_opaque;
  m_opaque = nullptr;
  m_opaque_owned = false;
}

bool SBCommunication::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBCommunication, IsValid);
  return this->operator bool();
}
SBCommunication::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBCommunication, operator bool);

  return m_opaque != nullptr;
}

bool SBCommunication::GetCloseOnEOF() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBCommunication, GetCloseOnEOF);

  if (m_opaque)
    return m_opaque->GetCloseOnEOF();
  return false;
}

void SBCommunication::SetCloseOnEOF(bool b) {
  LLDB_RECORD_METHOD(void, SBCommunication, SetCloseOnEOF, (bool), b);

  if (m_opaque)
    m_opaque->SetCloseOnEOF(b);
}

ConnectionStatus SBCommunication::Connect(const char *url) {
  LLDB_RECORD_METHOD(lldb::ConnectionStatus, SBCommunication, Connect,
                     (const char *), url);

  if (m_opaque) {
    if (!m_opaque->HasConnection())
      m_opaque->SetConnection(Host::CreateDefaultConnection(url).release());
    return m_opaque->Connect(url, nullptr);
  }
  return eConnectionStatusNoConnection;
}

ConnectionStatus SBCommunication::AdoptFileDesriptor(int fd, bool owns_fd) {
  LLDB_RECORD_METHOD(lldb::ConnectionStatus, SBCommunication,
                     AdoptFileDesriptor, (int, bool), fd, owns_fd);

  ConnectionStatus status = eConnectionStatusNoConnection;
  if (m_opaque) {
    if (m_opaque->HasConnection()) {
      if (m_opaque->IsConnected())
        m_opaque->Disconnect();
    }
    m_opaque->SetConnection(new ConnectionFileDescriptor(fd, owns_fd));
    if (m_opaque->IsConnected())
      status = eConnectionStatusSuccess;
    else
      status = eConnectionStatusLostConnection;
  }
  return status;
}

ConnectionStatus SBCommunication::Disconnect() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::ConnectionStatus, SBCommunication,
                             Disconnect);

  ConnectionStatus status = eConnectionStatusNoConnection;
  if (m_opaque)
    status = m_opaque->Disconnect();
  return status;
}

bool SBCommunication::IsConnected() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBCommunication, IsConnected);

  return m_opaque ? m_opaque->IsConnected() : false;
}

size_t SBCommunication::Read(void *dst, size_t dst_len, uint32_t timeout_usec,
                             ConnectionStatus &status) {
  LLDB_RECORD_DUMMY(size_t, SBCommunication, Read,
                    (void *, size_t, uint32_t, lldb::ConnectionStatus &), dst,
                    dst_len, timeout_usec, status);

  size_t bytes_read = 0;
  Timeout<std::micro> timeout = timeout_usec == UINT32_MAX
                                    ? Timeout<std::micro>(llvm::None)
                                    : std::chrono::microseconds(timeout_usec);
  if (m_opaque)
    bytes_read = m_opaque->Read(dst, dst_len, timeout, status, nullptr);
  else
    status = eConnectionStatusNoConnection;

  return bytes_read;
}

size_t SBCommunication::Write(const void *src, size_t src_len,
                              ConnectionStatus &status) {
  LLDB_RECORD_DUMMY(size_t, SBCommunication, Write,
                    (const void *, size_t, lldb::ConnectionStatus &), src,
                    src_len, status);

  size_t bytes_written = 0;
  if (m_opaque)
    bytes_written = m_opaque->Write(src, src_len, status, nullptr);
  else
    status = eConnectionStatusNoConnection;

  return bytes_written;
}

bool SBCommunication::ReadThreadStart() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBCommunication, ReadThreadStart);

  return m_opaque ? m_opaque->StartReadThread() : false;
}

bool SBCommunication::ReadThreadStop() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBCommunication, ReadThreadStop);

  return m_opaque ? m_opaque->StopReadThread() : false;
}

bool SBCommunication::ReadThreadIsRunning() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBCommunication, ReadThreadIsRunning);

  return m_opaque ? m_opaque->ReadThreadIsRunning() : false;
}

bool SBCommunication::SetReadThreadBytesReceivedCallback(
    ReadThreadBytesReceived callback, void *callback_baton) {
  LLDB_RECORD_DUMMY(bool, SBCommunication, SetReadThreadBytesReceivedCallback,
                    (lldb::SBCommunication::ReadThreadBytesReceived, void *),
                    callback, callback_baton);

  bool result = false;
  if (m_opaque) {
    m_opaque->SetReadThreadBytesReceivedCallback(callback, callback_baton);
    result = true;
  }
  return result;
}

SBBroadcaster SBCommunication::GetBroadcaster() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBBroadcaster, SBCommunication,
                             GetBroadcaster);

  SBBroadcaster broadcaster(m_opaque, false);
  return LLDB_RECORD_RESULT(broadcaster);
}

const char *SBCommunication::GetBroadcasterClass() {
  LLDB_RECORD_STATIC_METHOD_NO_ARGS(const char *, SBCommunication,
                                    GetBroadcasterClass);

  return Communication::GetStaticBroadcasterClass().AsCString();
}

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBCommunication>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBCommunication, ());
  LLDB_REGISTER_CONSTRUCTOR(SBCommunication, (const char *));
  LLDB_REGISTER_METHOD_CONST(bool, SBCommunication, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBCommunication, operator bool, ());
  LLDB_REGISTER_METHOD(bool, SBCommunication, GetCloseOnEOF, ());
  LLDB_REGISTER_METHOD(void, SBCommunication, SetCloseOnEOF, (bool));
  LLDB_REGISTER_METHOD(lldb::ConnectionStatus, SBCommunication, Connect,
                       (const char *));
  LLDB_REGISTER_METHOD(lldb::ConnectionStatus, SBCommunication,
                       AdoptFileDesriptor, (int, bool));
  LLDB_REGISTER_METHOD(lldb::ConnectionStatus, SBCommunication, Disconnect,
                       ());
  LLDB_REGISTER_METHOD_CONST(bool, SBCommunication, IsConnected, ());
  LLDB_REGISTER_METHOD(bool, SBCommunication, ReadThreadStart, ());
  LLDB_REGISTER_METHOD(bool, SBCommunication, ReadThreadStop, ());
  LLDB_REGISTER_METHOD(bool, SBCommunication, ReadThreadIsRunning, ());
  LLDB_REGISTER_METHOD(lldb::SBBroadcaster, SBCommunication, GetBroadcaster,
                       ());
  LLDB_REGISTER_STATIC_METHOD(const char *, SBCommunication,
                              GetBroadcasterClass, ());
}

}
}
