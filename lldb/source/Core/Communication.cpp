//===-- Communication.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Communication.h"

#include "lldb/Utility/Connection.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"

#include "llvm/Support/Compiler.h"

#include <algorithm>
#include <cstring>
#include <memory>

#include <cerrno>
#include <cinttypes>
#include <cstdio>

using namespace lldb;
using namespace lldb_private;

Communication::Communication()
    : m_connection_sp(), m_connection_mutex(), m_close_on_eof(true) {}

Communication::~Communication() { Disconnect(nullptr); }

ConnectionStatus Communication::Connect(const char *url, Status *error_ptr) {
  std::unique_lock guard(m_connection_mutex);

  LLDB_LOG(GetLog(LLDBLog::Communication),
           "{0} Communication::Connect (url = {1})", this, url);

  DisconnectUnlocked();

  if (m_connection_sp)
    return m_connection_sp->Connect(url, error_ptr);
  if (error_ptr)
    error_ptr->SetErrorString("Invalid connection.");
  return eConnectionStatusNoConnection;
}

ConnectionStatus Communication::Disconnect(Status *error_ptr) {
  std::unique_lock guard(m_connection_mutex);
  return DisconnectUnlocked(error_ptr);
}

ConnectionStatus Communication::DisconnectUnlocked(Status *error_ptr) {
  LLDB_LOG(GetLog(LLDBLog::Communication), "{0} Communication::Disconnect ()",
           this);

  if (m_connection_sp) {
    ConnectionStatus status = m_connection_sp->Disconnect(error_ptr);
    return status;
  }
  return eConnectionStatusNoConnection;
}

bool Communication::IsConnected() const {
  std::shared_lock guard(m_connection_mutex);
  return (m_connection_sp ? m_connection_sp->IsConnected() : false);
}

bool Communication::HasConnection() const {
  std::shared_lock guard(m_connection_mutex);
  return m_connection_sp.get() != nullptr;
}

size_t Communication::Read(void *dst, size_t dst_len,
                           const Timeout<std::micro> &timeout,
                           ConnectionStatus &status, Status *error_ptr) {
  std::shared_lock guard(m_connection_mutex);
  return ReadUnlocked(dst, dst_len, timeout, status, error_ptr);
}

size_t Communication::Write(const void *src, size_t src_len,
                            ConnectionStatus &status, Status *error_ptr) {
  // We need to lock the write mutex so no concurrent writes happen, but also
  // lock the connection mutex so it's not reset mid write. We need both mutexes
  // because reads and writes from the connection can happen concurrently.
  std::shared_lock guard(m_connection_mutex);
  std::lock_guard<std::mutex> guard_write(m_write_mutex);
  return WriteUnlocked(src, src_len, status, error_ptr);
}

size_t Communication::WriteUnlocked(const void *src, size_t src_len,
                                    ConnectionStatus &status,
                                    Status *error_ptr) {
  if (!m_connection_sp) {
    if (error_ptr)
      error_ptr->SetErrorString("Invalid connection.");
    status = eConnectionStatusNoConnection;
    return 0;
  }

  LLDB_LOG(GetLog(LLDBLog::Communication),
           "{0} Communication::Write (src = {1}, src_len = {2}"
           ") connection = {3}",
           this, src, (uint64_t)src_len, m_connection_sp.get());

  return m_connection_sp->Write(src, src_len, status, error_ptr);
}

size_t Communication::WriteAll(const void *src, size_t src_len,
                               ConnectionStatus &status, Status *error_ptr) {
  std::shared_lock guard(m_connection_mutex);
  std::lock_guard<std::mutex> guard_write(m_write_mutex);
  size_t total_written = 0;
  do
    total_written +=
        WriteUnlocked(static_cast<const char *>(src) + total_written,
                      src_len - total_written, status, error_ptr);
  while (status == eConnectionStatusSuccess && total_written < src_len);
  return total_written;
}

size_t Communication::ReadUnlocked(void *dst, size_t dst_len,
                                   const Timeout<std::micro> &timeout,
                                   ConnectionStatus &status,
                                   Status *error_ptr) {
  Log *log = GetLog(LLDBLog::Communication);
  LLDB_LOG(
      log,
      "this = {0}, dst = {1}, dst_len = {2}, timeout = {3}, connection = {4}",
      this, dst, dst_len, timeout, m_connection_sp.get());
  if (m_connection_sp)
    return m_connection_sp->Read(dst, dst_len, timeout, status, error_ptr);

  if (error_ptr)
    error_ptr->SetErrorString("Invalid connection.");
  status = eConnectionStatusNoConnection;
  return 0;
}

void Communication::SetConnection(std::unique_ptr<Connection> connection) {
  std::unique_lock guard(m_connection_mutex);
  DisconnectUnlocked(nullptr);
  m_connection_sp = std::move(connection);
}

std::string
Communication::ConnectionStatusAsString(lldb::ConnectionStatus status) {
  switch (status) {
  case eConnectionStatusSuccess:
    return "success";
  case eConnectionStatusError:
    return "error";
  case eConnectionStatusTimedOut:
    return "timed out";
  case eConnectionStatusNoConnection:
    return "no connection";
  case eConnectionStatusLostConnection:
    return "lost connection";
  case eConnectionStatusEndOfFile:
    return "end of file";
  case eConnectionStatusInterrupted:
    return "interrupted";
  }

  return "@" + std::to_string(status);
}
