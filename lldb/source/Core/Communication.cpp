//===-- Communication.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
#include <cstring>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/Communication.h"
#include "lldb/Core/Connection.h"
#include "lldb/Core/Event.h"
#include "lldb/Core/Listener.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Timer.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Host/ThreadLauncher.h"

using namespace lldb;
using namespace lldb_private;

ConstString &Communication::GetStaticBroadcasterClass() {
  static ConstString class_name("lldb.communication");
  return class_name;
}

Communication::Communication(const char *name)
    : Broadcaster(nullptr, name), m_connection_sp(),
      m_read_thread_enabled(false), m_read_thread_did_exit(false), m_bytes(),
      m_bytes_mutex(Mutex::eMutexTypeRecursive),
      m_write_mutex(Mutex::eMutexTypeNormal),
      m_synchronize_mutex(Mutex::eMutexTypeNormal), m_callback(nullptr),
      m_callback_baton(nullptr), m_close_on_eof(true)

{
  lldb_private::LogIfAnyCategoriesSet(
      LIBLLDB_LOG_OBJECT | LIBLLDB_LOG_COMMUNICATION,
      "%p Communication::Communication (name = %s)", this, name);

  SetEventName(eBroadcastBitDisconnected, "disconnected");
  SetEventName(eBroadcastBitReadThreadGotBytes, "got bytes");
  SetEventName(eBroadcastBitReadThreadDidExit, "read thread did exit");
  SetEventName(eBroadcastBitReadThreadShouldExit, "read thread should exit");
  SetEventName(eBroadcastBitPacketAvailable, "packet available");
  SetEventName(eBroadcastBitNoMorePendingInput, "no more pending input");

  CheckInWithManager();
}

Communication::~Communication() {
  lldb_private::LogIfAnyCategoriesSet(
      LIBLLDB_LOG_OBJECT | LIBLLDB_LOG_COMMUNICATION,
      "%p Communication::~Communication (name = %s)", this,
      GetBroadcasterName().AsCString());
  Clear();
}

void Communication::Clear() {
  SetReadThreadBytesReceivedCallback(nullptr, nullptr);
  Disconnect(nullptr);
  StopReadThread(nullptr);
}

ConnectionStatus Communication::Connect(const char *url, Error *error_ptr) {
  Clear();

  lldb_private::LogIfAnyCategoriesSet(LIBLLDB_LOG_COMMUNICATION,
                                      "%p Communication::Connect (url = %s)",
                                      this, url);

  lldb::ConnectionSP connection_sp(m_connection_sp);
  if (connection_sp)
    return connection_sp->Connect(url, error_ptr);
  if (error_ptr)
    error_ptr->SetErrorString("Invalid connection.");
  return eConnectionStatusNoConnection;
}

ConnectionStatus Communication::Disconnect(Error *error_ptr) {
  lldb_private::LogIfAnyCategoriesSet(LIBLLDB_LOG_COMMUNICATION,
                                      "%p Communication::Disconnect ()", this);

  lldb::ConnectionSP connection_sp(m_connection_sp);
  if (connection_sp) {
    ConnectionStatus status = connection_sp->Disconnect(error_ptr);
    // We currently don't protect connection_sp with any mutex for
    // multi-threaded environments. So lets not nuke our connection class
    // without putting some multi-threaded protections in. We also probably
    // don't want to pay for the overhead it might cause if every time we
    // access the connection we have to take a lock.
    //
    // This unique pointer will cleanup after itself when this object goes away,
    // so there is no need to currently have it destroy itself immediately
    // upon disconnnect.
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

size_t Communication::Read(void *dst, size_t dst_len, uint32_t timeout_usec,
                           ConnectionStatus &status, Error *error_ptr) {
  lldb_private::LogIfAnyCategoriesSet(
      LIBLLDB_LOG_COMMUNICATION,
      "%p Communication::Read (dst = %p, dst_len = %" PRIu64
      ", timeout = %u usec) connection = %p",
      this, dst, (uint64_t)dst_len, timeout_usec, m_connection_sp.get());

  if (m_read_thread_enabled) {
    // We have a dedicated read thread that is getting data for us
    size_t cached_bytes = GetCachedBytes(dst, dst_len);
    if (cached_bytes > 0 || timeout_usec == 0) {
      status = eConnectionStatusSuccess;
      return cached_bytes;
    }

    if (!m_connection_sp) {
      if (error_ptr)
        error_ptr->SetErrorString("Invalid connection.");
      status = eConnectionStatusNoConnection;
      return 0;
    }
    // Set the timeout appropriately
    TimeValue timeout_time;
    if (timeout_usec != UINT32_MAX) {
      timeout_time = TimeValue::Now();
      timeout_time.OffsetWithMicroSeconds(timeout_usec);
    }

    ListenerSP listener_sp(Listener::MakeListener("Communication::Read"));
    listener_sp->StartListeningForEvents(
        this, eBroadcastBitReadThreadGotBytes | eBroadcastBitReadThreadDidExit);
    EventSP event_sp;
    while (listener_sp->WaitForEvent(
        timeout_time.IsValid() ? &timeout_time : nullptr, event_sp)) {
      const uint32_t event_type = event_sp->GetType();
      if (event_type & eBroadcastBitReadThreadGotBytes) {
        return GetCachedBytes(dst, dst_len);
      }

      if (event_type & eBroadcastBitReadThreadDidExit) {
        if (GetCloseOnEOF())
          Disconnect(nullptr);
        break;
      }
    }
    return 0;
  }

  // We aren't using a read thread, just read the data synchronously in this
  // thread.
  lldb::ConnectionSP connection_sp(m_connection_sp);
  if (connection_sp) {
    return connection_sp->Read(dst, dst_len, timeout_usec, status, error_ptr);
  }

  if (error_ptr)
    error_ptr->SetErrorString("Invalid connection.");
  status = eConnectionStatusNoConnection;
  return 0;
}

size_t Communication::Write(const void *src, size_t src_len,
                            ConnectionStatus &status, Error *error_ptr) {
  lldb::ConnectionSP connection_sp(m_connection_sp);

  Mutex::Locker locker(m_write_mutex);
  lldb_private::LogIfAnyCategoriesSet(
      LIBLLDB_LOG_COMMUNICATION,
      "%p Communication::Write (src = %p, src_len = %" PRIu64
      ") connection = %p",
      this, src, (uint64_t)src_len, connection_sp.get());

  if (connection_sp)
    return connection_sp->Write(src, src_len, status, error_ptr);

  if (error_ptr)
    error_ptr->SetErrorString("Invalid connection.");
  status = eConnectionStatusNoConnection;
  return 0;
}

bool Communication::StartReadThread(Error *error_ptr) {
  if (error_ptr)
    error_ptr->Clear();

  if (m_read_thread.IsJoinable())
    return true;

  lldb_private::LogIfAnyCategoriesSet(
      LIBLLDB_LOG_COMMUNICATION, "%p Communication::StartReadThread ()", this);

  char thread_name[1024];
  snprintf(thread_name, sizeof(thread_name), "<lldb.comm.%s>",
           GetBroadcasterName().AsCString());

  m_read_thread_enabled = true;
  m_read_thread_did_exit = false;
  m_read_thread = ThreadLauncher::LaunchThread(
      thread_name, Communication::ReadThread, this, error_ptr);
  if (!m_read_thread.IsJoinable())
    m_read_thread_enabled = false;
  return m_read_thread_enabled;
}

bool Communication::StopReadThread(Error *error_ptr) {
  if (!m_read_thread.IsJoinable())
    return true;

  lldb_private::LogIfAnyCategoriesSet(
      LIBLLDB_LOG_COMMUNICATION, "%p Communication::StopReadThread ()", this);

  m_read_thread_enabled = false;

  BroadcastEvent(eBroadcastBitReadThreadShouldExit, nullptr);

  // error = m_read_thread.Cancel();

  Error error = m_read_thread.Join(nullptr);
  return error.Success();
}

bool Communication::JoinReadThread(Error *error_ptr) {
  if (!m_read_thread.IsJoinable())
    return true;

  Error error = m_read_thread.Join(nullptr);
  return error.Success();
}

size_t Communication::GetCachedBytes(void *dst, size_t dst_len) {
  Mutex::Locker locker(m_bytes_mutex);
  if (!m_bytes.empty()) {
    // If DST is nullptr and we have a thread, then return the number
    // of bytes that are available so the caller can call again
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
  lldb_private::LogIfAnyCategoriesSet(
      LIBLLDB_LOG_COMMUNICATION,
      "%p Communication::AppendBytesToCache (src = %p, src_len = %" PRIu64
      ", broadcast = %i)",
      this, bytes, (uint64_t)len, broadcast);
  if ((bytes == nullptr || len == 0) &&
      (status != lldb::eConnectionStatusEndOfFile))
    return;
  if (m_callback) {
    // If the user registered a callback, then call it and do not broadcast
    m_callback(m_callback_baton, bytes, len);
  } else if (bytes != nullptr && len > 0) {
    Mutex::Locker locker(m_bytes_mutex);
    m_bytes.append((const char *)bytes, len);
    if (broadcast)
      BroadcastEventIfUnique(eBroadcastBitReadThreadGotBytes);
  }
}

size_t Communication::ReadFromConnection(void *dst, size_t dst_len,
                                         uint32_t timeout_usec,
                                         ConnectionStatus &status,
                                         Error *error_ptr) {
  lldb::ConnectionSP connection_sp(m_connection_sp);
  return (
      connection_sp
          ? connection_sp->Read(dst, dst_len, timeout_usec, status, error_ptr)
          : 0);
}

bool Communication::ReadThreadIsRunning() { return m_read_thread_enabled; }

lldb::thread_result_t Communication::ReadThread(lldb::thread_arg_t p) {
  Communication *comm = (Communication *)p;

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_COMMUNICATION));

  if (log)
    log->Printf("%p Communication::ReadThread () thread starting...", p);

  uint8_t buf[1024];

  Error error;
  ConnectionStatus status = eConnectionStatusSuccess;
  bool done = false;
  while (!done && comm->m_read_thread_enabled) {
    size_t bytes_read = comm->ReadFromConnection(
        buf, sizeof(buf), 5 * TimeValue::MicroSecPerSec, status, &error);
    if (bytes_read > 0)
      comm->AppendBytesToCache(buf, bytes_read, true, status);
    else if ((bytes_read == 0) && status == eConnectionStatusEndOfFile) {
      if (comm->GetCloseOnEOF())
        comm->Disconnect();
      comm->AppendBytesToCache(buf, bytes_read, true, status);
    }

    switch (status) {
    case eConnectionStatusSuccess:
      break;

    case eConnectionStatusEndOfFile:
      done = true;
      break;
    case eConnectionStatusError: // Check GetError() for details
      if (error.GetType() == eErrorTypePOSIX && error.GetError() == EIO) {
        // EIO on a pipe is usually caused by remote shutdown
        comm->Disconnect();
        done = true;
      }
      if (log)
        error.LogIfError(
            log, "%p Communication::ReadFromConnection () => status = %s", p,
            Communication::ConnectionStatusAsCString(status));
      break;
    case eConnectionStatusInterrupted: // Synchronization signal from
                                       // SynchronizeWithReadThread()
      // The connection returns eConnectionStatusInterrupted only when there is
      // no
      // input pending to be read, so we can signal that.
      comm->BroadcastEvent(eBroadcastBitNoMorePendingInput);
      break;
    case eConnectionStatusNoConnection:   // No connection
    case eConnectionStatusLostConnection: // Lost connection while connected to
                                          // a valid connection
      done = true;
      LLVM_FALLTHROUGH;
    case eConnectionStatusTimedOut: // Request timed out
      if (log)
        error.LogIfError(
            log, "%p Communication::ReadFromConnection () => status = %s", p,
            Communication::ConnectionStatusAsCString(status));
      break;
    }
  }
  log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_COMMUNICATION);
  if (log)
    log->Printf("%p Communication::ReadThread () thread exiting...", p);

  comm->m_read_thread_did_exit = true;
  // Let clients know that this thread is exiting
  comm->BroadcastEvent(eBroadcastBitNoMorePendingInput);
  comm->BroadcastEvent(eBroadcastBitReadThreadDidExit);
  return NULL;
}

void Communication::SetReadThreadBytesReceivedCallback(
    ReadThreadBytesReceived callback, void *callback_baton) {
  m_callback = callback;
  m_callback_baton = callback_baton;
}

void Communication::SynchronizeWithReadThread() {
  // Only one thread can do the synchronization dance at a time.
  Mutex::Locker locker(m_synchronize_mutex);

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
  listener_sp->WaitForEvent(nullptr, event_sp);
}

void Communication::SetConnection(Connection *connection) {
  Disconnect(nullptr);
  StopReadThread(nullptr);
  m_connection_sp.reset(connection);
}

const char *
Communication::ConnectionStatusAsCString(lldb::ConnectionStatus status) {
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

  static char unknown_state_string[64];
  snprintf(unknown_state_string, sizeof(unknown_state_string),
           "ConnectionStatus = %i", status);
  return unknown_state_string;
}
