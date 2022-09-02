//===-- Communication.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Communication.h"

#include "lldb/Host/HostThread.h"
#include "lldb/Host/ThreadLauncher.h"
#include "lldb/Utility/Connection.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Event.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Listener.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"

#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Compiler.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>

#include <cerrno>
#include <cinttypes>
#include <cstdio>

using namespace lldb;
using namespace lldb_private;

ConstString &Communication::GetStaticBroadcasterClass() {
  static ConstString class_name("lldb.communication");
  return class_name;
}

Communication::Communication(const char *name)
    : Broadcaster(nullptr, name), m_connection_sp(),
      m_read_thread_enabled(false), m_read_thread_did_exit(false), m_bytes(),
      m_bytes_mutex(), m_write_mutex(), m_synchronize_mutex(),
      m_callback(nullptr), m_callback_baton(nullptr), m_close_on_eof(true)

{

  LLDB_LOG(GetLog(LLDBLog::Object | LLDBLog::Communication),
           "{0} Communication::Communication (name = {1})", this, name);

  SetEventName(eBroadcastBitDisconnected, "disconnected");
  SetEventName(eBroadcastBitReadThreadGotBytes, "got bytes");
  SetEventName(eBroadcastBitReadThreadDidExit, "read thread did exit");
  SetEventName(eBroadcastBitReadThreadShouldExit, "read thread should exit");
  SetEventName(eBroadcastBitPacketAvailable, "packet available");
  SetEventName(eBroadcastBitNoMorePendingInput, "no more pending input");

  CheckInWithManager();
}

Communication::~Communication() {
  LLDB_LOG(GetLog(LLDBLog::Object | LLDBLog::Communication),
           "{0} Communication::~Communication (name = {1})", this,
           GetBroadcasterName().AsCString());
  Clear();
}

void Communication::Clear() {
  SetReadThreadBytesReceivedCallback(nullptr, nullptr);
  StopReadThread(nullptr);
  Disconnect(nullptr);
}

ConnectionStatus Communication::Connect(const char *url, Status *error_ptr) {
  Clear();

  LLDB_LOG(GetLog(LLDBLog::Communication),
           "{0} Communication::Connect (url = {1})", this, url);

  lldb::ConnectionSP connection_sp(m_connection_sp);
  if (connection_sp)
    return connection_sp->Connect(url, error_ptr);
  if (error_ptr)
    error_ptr->SetErrorString("Invalid connection.");
  return eConnectionStatusNoConnection;
}

ConnectionStatus Communication::Disconnect(Status *error_ptr) {
  LLDB_LOG(GetLog(LLDBLog::Communication), "{0} Communication::Disconnect ()",
           this);

  assert((!m_read_thread_enabled || m_read_thread_did_exit) &&
         "Disconnecting while the read thread is running is racy!");
  lldb::ConnectionSP connection_sp(m_connection_sp);
  if (connection_sp) {
    ConnectionStatus status = connection_sp->Disconnect(error_ptr);
    // We currently don't protect connection_sp with any mutex for multi-
    // threaded environments. So lets not nuke our connection class without
    // putting some multi-threaded protections in. We also probably don't want
    // to pay for the overhead it might cause if every time we access the
    // connection we have to take a lock.
    //
    // This unique pointer will cleanup after itself when this object goes
    // away, so there is no need to currently have it destroy itself
    // immediately upon disconnect.
    // connection_sp.reset();
    return status;
  }
  return eConnectionStatusNoConnection;
}

bool Communication::IsConnected() const {
  lldb::ConnectionSP connection_sp(m_connection_sp);
  return (connection_sp ? connection_sp->IsConnected() : false);
}

bool Communication::HasConnection() const {
  return m_connection_sp.get() != nullptr;
}

size_t Communication::Read(void *dst, size_t dst_len,
                           const Timeout<std::micro> &timeout,
                           ConnectionStatus &status, Status *error_ptr) {
  Log *log = GetLog(LLDBLog::Communication);
  LLDB_LOG(
      log,
      "this = {0}, dst = {1}, dst_len = {2}, timeout = {3}, connection = {4}",
      this, dst, dst_len, timeout, m_connection_sp.get());

  if (m_read_thread_enabled) {
    // We have a dedicated read thread that is getting data for us
    size_t cached_bytes = GetCachedBytes(dst, dst_len);
    if (cached_bytes > 0) {
      status = eConnectionStatusSuccess;
      return cached_bytes;
    }
    if (timeout && timeout->count() == 0) {
      if (error_ptr)
        error_ptr->SetErrorString("Timed out.");
      status = eConnectionStatusTimedOut;
      return 0;
    }

    if (!m_connection_sp) {
      if (error_ptr)
        error_ptr->SetErrorString("Invalid connection.");
      status = eConnectionStatusNoConnection;
      return 0;
    }

    ListenerSP listener_sp(Listener::MakeListener("Communication::Read"));
    listener_sp->StartListeningForEvents(
        this, eBroadcastBitReadThreadGotBytes | eBroadcastBitReadThreadDidExit);
    EventSP event_sp;
    while (listener_sp->GetEvent(event_sp, timeout)) {
      const uint32_t event_type = event_sp->GetType();
      if (event_type & eBroadcastBitReadThreadGotBytes) {
        return GetCachedBytes(dst, dst_len);
      }

      if (event_type & eBroadcastBitReadThreadDidExit) {
        // If the thread exited of its own accord, it either means it
        // hit an end-of-file condition or an error.
        status = m_pass_status;
        if (error_ptr)
          *error_ptr = std::move(m_pass_error);

        if (GetCloseOnEOF())
          Disconnect(nullptr);
        return 0;
      }
    }

    if (error_ptr)
      error_ptr->SetErrorString("Timed out.");
    status = eConnectionStatusTimedOut;
    return 0;
  }

  // We aren't using a read thread, just read the data synchronously in this
  // thread.
  return ReadFromConnection(dst, dst_len, timeout, status, error_ptr);
}

size_t Communication::Write(const void *src, size_t src_len,
                            ConnectionStatus &status, Status *error_ptr) {
  lldb::ConnectionSP connection_sp(m_connection_sp);

  std::lock_guard<std::mutex> guard(m_write_mutex);
  LLDB_LOG(GetLog(LLDBLog::Communication),
           "{0} Communication::Write (src = {1}, src_len = {2}"
           ") connection = {3}",
           this, src, (uint64_t)src_len, connection_sp.get());

  if (connection_sp)
    return connection_sp->Write(src, src_len, status, error_ptr);

  if (error_ptr)
    error_ptr->SetErrorString("Invalid connection.");
  status = eConnectionStatusNoConnection;
  return 0;
}

size_t Communication::WriteAll(const void *src, size_t src_len,
                               ConnectionStatus &status, Status *error_ptr) {
  size_t total_written = 0;
  do
    total_written += Write(static_cast<const char *>(src) + total_written,
                           src_len - total_written, status, error_ptr);
  while (status == eConnectionStatusSuccess && total_written < src_len);
  return total_written;
}

bool Communication::StartReadThread(Status *error_ptr) {
  if (error_ptr)
    error_ptr->Clear();

  if (m_read_thread.IsJoinable())
    return true;

  LLDB_LOG(GetLog(LLDBLog::Communication),
           "{0} Communication::StartReadThread ()", this);

  const std::string thread_name =
      llvm::formatv("<lldb.comm.{0}>", GetBroadcasterName());

  m_read_thread_enabled = true;
  m_read_thread_did_exit = false;
  auto maybe_thread = ThreadLauncher::LaunchThread(
      thread_name, [this] { return ReadThread(); });
  if (maybe_thread) {
    m_read_thread = *maybe_thread;
  } else {
    if (error_ptr)
      *error_ptr = Status(maybe_thread.takeError());
    else {
      LLDB_LOG(GetLog(LLDBLog::Host), "failed to launch host thread: {}",
               llvm::toString(maybe_thread.takeError()));
    }
  }

  if (!m_read_thread.IsJoinable())
    m_read_thread_enabled = false;

  return m_read_thread_enabled;
}

bool Communication::StopReadThread(Status *error_ptr) {
  if (!m_read_thread.IsJoinable())
    return true;

  LLDB_LOG(GetLog(LLDBLog::Communication),
           "{0} Communication::StopReadThread ()", this);

  m_read_thread_enabled = false;

  BroadcastEvent(eBroadcastBitReadThreadShouldExit, nullptr);

  // error = m_read_thread.Cancel();

  Status error = m_read_thread.Join(nullptr);
  return error.Success();
}

bool Communication::JoinReadThread(Status *error_ptr) {
  if (!m_read_thread.IsJoinable())
    return true;

  Status error = m_read_thread.Join(nullptr);
  return error.Success();
}

size_t Communication::GetCachedBytes(void *dst, size_t dst_len) {
  std::lock_guard<std::recursive_mutex> guard(m_bytes_mutex);
  if (!m_bytes.empty()) {
    // If DST is nullptr and we have a thread, then return the number of bytes
    // that are available so the caller can call again
    if (dst == nullptr)
      return m_bytes.size();

    const size_t len = std::min<size_t>(dst_len, m_bytes.size());

    ::memcpy(dst, m_bytes.c_str(), len);
    m_bytes.erase(m_bytes.begin(), m_bytes.begin() + len);

    return len;
  }
  return 0;
}

void Communication::AppendBytesToCache(const uint8_t *bytes, size_t len,
                                       bool broadcast,
                                       ConnectionStatus status) {
  LLDB_LOG(GetLog(LLDBLog::Communication),
           "{0} Communication::AppendBytesToCache (src = {1}, src_len = {2}, "
           "broadcast = {3})",
           this, bytes, (uint64_t)len, broadcast);
  if ((bytes == nullptr || len == 0) &&
      (status != lldb::eConnectionStatusEndOfFile))
    return;
  if (m_callback) {
    // If the user registered a callback, then call it and do not broadcast
    m_callback(m_callback_baton, bytes, len);
  } else if (bytes != nullptr && len > 0) {
    std::lock_guard<std::recursive_mutex> guard(m_bytes_mutex);
    m_bytes.append((const char *)bytes, len);
    if (broadcast)
      BroadcastEventIfUnique(eBroadcastBitReadThreadGotBytes);
  }
}

size_t Communication::ReadFromConnection(void *dst, size_t dst_len,
                                         const Timeout<std::micro> &timeout,
                                         ConnectionStatus &status,
                                         Status *error_ptr) {
  lldb::ConnectionSP connection_sp(m_connection_sp);
  if (connection_sp)
    return connection_sp->Read(dst, dst_len, timeout, status, error_ptr);

  if (error_ptr)
    error_ptr->SetErrorString("Invalid connection.");
  status = eConnectionStatusNoConnection;
  return 0;
}

bool Communication::ReadThreadIsRunning() { return m_read_thread_enabled; }

lldb::thread_result_t Communication::ReadThread() {
  Log *log = GetLog(LLDBLog::Communication);

  LLDB_LOG(log, "Communication({0}) thread starting...", this);

  uint8_t buf[1024];

  Status error;
  ConnectionStatus status = eConnectionStatusSuccess;
  bool done = false;
  bool disconnect = false;
  while (!done && m_read_thread_enabled) {
    size_t bytes_read = ReadFromConnection(
        buf, sizeof(buf), std::chrono::seconds(5), status, &error);
    if (bytes_read > 0 || status == eConnectionStatusEndOfFile)
      AppendBytesToCache(buf, bytes_read, true, status);

    switch (status) {
    case eConnectionStatusSuccess:
      break;

    case eConnectionStatusEndOfFile:
      done = true;
      disconnect = GetCloseOnEOF();
      break;
    case eConnectionStatusError: // Check GetError() for details
      if (error.GetType() == eErrorTypePOSIX && error.GetError() == EIO) {
        // EIO on a pipe is usually caused by remote shutdown
        disconnect = GetCloseOnEOF();
        done = true;
      }
      if (error.Fail())
        LLDB_LOG(log, "error: {0}, status = {1}", error,
                 Communication::ConnectionStatusAsString(status));
      break;
    case eConnectionStatusInterrupted: // Synchronization signal from
                                       // SynchronizeWithReadThread()
      // The connection returns eConnectionStatusInterrupted only when there is
      // no input pending to be read, so we can signal that.
      BroadcastEvent(eBroadcastBitNoMorePendingInput);
      break;
    case eConnectionStatusNoConnection:   // No connection
    case eConnectionStatusLostConnection: // Lost connection while connected to
                                          // a valid connection
      done = true;
      [[fallthrough]];
    case eConnectionStatusTimedOut: // Request timed out
      if (error.Fail())
        LLDB_LOG(log, "error: {0}, status = {1}", error,
                 Communication::ConnectionStatusAsString(status));
      break;
    }
  }
  m_pass_status = status;
  m_pass_error = std::move(error);
  LLDB_LOG(log, "Communication({0}) thread exiting...", this);

  // Handle threads wishing to synchronize with us.
  {
    // Prevent new ones from showing up.
    m_read_thread_did_exit = true;

    // Unblock any existing thread waiting for the synchronization signal.
    BroadcastEvent(eBroadcastBitNoMorePendingInput);

    // Wait for the thread to finish...
    std::lock_guard<std::mutex> guard(m_synchronize_mutex);
    // ... and disconnect.
    if (disconnect)
      Disconnect();
  }

  // Let clients know that this thread is exiting
  BroadcastEvent(eBroadcastBitReadThreadDidExit);
  return {};
}

void Communication::SetReadThreadBytesReceivedCallback(
    ReadThreadBytesReceived callback, void *callback_baton) {
  m_callback = callback;
  m_callback_baton = callback_baton;
}

void Communication::SynchronizeWithReadThread() {
  // Only one thread can do the synchronization dance at a time.
  std::lock_guard<std::mutex> guard(m_synchronize_mutex);

  // First start listening for the synchronization event.
  ListenerSP listener_sp(
      Listener::MakeListener("Communication::SyncronizeWithReadThread"));
  listener_sp->StartListeningForEvents(this, eBroadcastBitNoMorePendingInput);

  // If the thread is not running, there is no point in synchronizing.
  if (!m_read_thread_enabled || m_read_thread_did_exit)
    return;

  // Notify the read thread.
  m_connection_sp->InterruptRead();

  // Wait for the synchronization event.
  EventSP event_sp;
  listener_sp->GetEvent(event_sp, llvm::None);
}

void Communication::SetConnection(std::unique_ptr<Connection> connection) {
  Disconnect(nullptr);
  StopReadThread(nullptr);
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
