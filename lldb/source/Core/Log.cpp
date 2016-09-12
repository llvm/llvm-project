//===-- Log.cpp -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

// Other libraries and framework includes
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

// Project includes
#include "lldb/Core/Log.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/ThisThread.h"
#include "lldb/Host/TimeValue.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Utility/NameMatches.h"

using namespace lldb;
using namespace lldb_private;

Log::Log() : m_stream_sp(), m_options(0), m_mask_bits(0) {}

Log::Log(const StreamSP &stream_sp)
    : m_stream_sp(stream_sp), m_options(0), m_mask_bits(0) {}

Log::~Log() = default;

Flags &Log::GetOptions() { return m_options; }

const Flags &Log::GetOptions() const { return m_options; }

Flags &Log::GetMask() { return m_mask_bits; }

const Flags &Log::GetMask() const { return m_mask_bits; }

void Log::PutCString(const char *cstr) { Printf("%s", cstr); }

//----------------------------------------------------------------------
// Simple variable argument logging with flags.
//----------------------------------------------------------------------
void Log::Printf(const char *format, ...) {
  va_list args;
  va_start(args, format);
  VAPrintf(format, args);
  va_end(args);
}

//----------------------------------------------------------------------
// All logging eventually boils down to this function call. If we have
// a callback registered, then we call the logging callback. If we have
// a valid file handle, we also log to the file.
//----------------------------------------------------------------------
void Log::VAPrintf(const char *format, va_list args) {
  // Make a copy of our stream shared pointer in case someone disables our
  // log while we are logging and releases the stream
  StreamSP stream_sp(m_stream_sp);
  if (stream_sp) {
    static uint32_t g_sequence_id = 0;
    StreamString header;

    // Add a sequence ID if requested
    if (m_options.Test(LLDB_LOG_OPTION_PREPEND_SEQUENCE))
      header.Printf("%u ", ++g_sequence_id);

    // Timestamp if requested
    if (m_options.Test(LLDB_LOG_OPTION_PREPEND_TIMESTAMP)) {
      TimeValue now = TimeValue::Now();
      header.Printf("%9d.%9.9d ", now.seconds(), now.nanoseconds());
    }

    // Add the process and thread if requested
    if (m_options.Test(LLDB_LOG_OPTION_PREPEND_PROC_AND_THREAD))
      header.Printf("[%4.4x/%4.4" PRIx64 "]: ", getpid(),
                    Host::GetCurrentThreadID());

    // Add the thread name if requested
    if (m_options.Test(LLDB_LOG_OPTION_PREPEND_THREAD_NAME)) {
      llvm::SmallString<32> thread_name;
      ThisThread::GetName(thread_name);
      if (!thread_name.empty())
        header.Printf("%s ", thread_name.c_str());
    }

    header.PrintfVarArg(format, args);
    header.PutCString("\n");

    if (m_options.Test(LLDB_LOG_OPTION_BACKTRACE)) {
      std::string back_trace;
      llvm::raw_string_ostream stream(back_trace);
      llvm::sys::PrintStackTrace(stream);
      stream.flush();
      header.PutCString(back_trace.c_str());
    }

    if (m_options.Test(LLDB_LOG_OPTION_THREADSAFE)) {
      static std::recursive_mutex g_LogThreadedMutex;
      std::lock_guard<std::recursive_mutex> guard(g_LogThreadedMutex);
      stream_sp->PutCString(header.GetString().c_str());
      stream_sp->Flush();
    } else {
      stream_sp->PutCString(header.GetString().c_str());
      stream_sp->Flush();
    }
  }
}

//----------------------------------------------------------------------
// Print debug strings if and only if the global debug option is set to
// a non-zero value.
//----------------------------------------------------------------------
void Log::Debug(const char *format, ...) {
  if (!GetOptions().Test(LLDB_LOG_OPTION_DEBUG))
    return;

  va_list args;
  va_start(args, format);
  VAPrintf(format, args);
  va_end(args);
}

//----------------------------------------------------------------------
// Print debug strings if and only if the global debug option is set to
// a non-zero value.
//----------------------------------------------------------------------
void Log::DebugVerbose(const char *format, ...) {
  if (!GetOptions().AllSet(LLDB_LOG_OPTION_DEBUG | LLDB_LOG_OPTION_VERBOSE))
    return;

  va_list args;
  va_start(args, format);
  VAPrintf(format, args);
  va_end(args);
}

//----------------------------------------------------------------------
// Log only if all of the bits are set
//----------------------------------------------------------------------
void Log::LogIf(uint32_t bits, const char *format, ...) {
  if (!m_options.AllSet(bits))
    return;

  va_list args;
  va_start(args, format);
  VAPrintf(format, args);
  va_end(args);
}

//----------------------------------------------------------------------
// Printing of errors that are not fatal.
//----------------------------------------------------------------------
void Log::Error(const char *format, ...) {
  va_list args;
  va_start(args, format);
  VAError(format, args);
  va_end(args);
}

void Log::VAError(const char *format, va_list args) {
  char *arg_msg = nullptr;
  ::vasprintf(&arg_msg, format, args);

  if (arg_msg == nullptr)
    return;

  Printf("error: %s", arg_msg);
  free(arg_msg);
}

//----------------------------------------------------------------------
// Printing of errors that ARE fatal. Exit with ERR exit code
// immediately.
//----------------------------------------------------------------------
void Log::FatalError(int err, const char *format, ...) {
  char *arg_msg = nullptr;
  va_list args;
  va_start(args, format);
  ::vasprintf(&arg_msg, format, args);
  va_end(args);

  if (arg_msg != nullptr) {
    Printf("error: %s", arg_msg);
    ::free(arg_msg);
  }
  ::exit(err);
}

//----------------------------------------------------------------------
// Printing of warnings that are not fatal only if verbose mode is
// enabled.
//----------------------------------------------------------------------
void Log::Verbose(const char *format, ...) {
  if (!m_options.Test(LLDB_LOG_OPTION_VERBOSE))
    return;

  va_list args;
  va_start(args, format);
  VAPrintf(format, args);
  va_end(args);
}

//----------------------------------------------------------------------
// Printing of warnings that are not fatal only if verbose mode is
// enabled.
//----------------------------------------------------------------------
void Log::WarningVerbose(const char *format, ...) {
  if (!m_options.Test(LLDB_LOG_OPTION_VERBOSE))
    return;

  char *arg_msg = nullptr;
  va_list args;
  va_start(args, format);
  ::vasprintf(&arg_msg, format, args);
  va_end(args);

  if (arg_msg == nullptr)
    return;

  Printf("warning: %s", arg_msg);
  free(arg_msg);
}

//----------------------------------------------------------------------
// Printing of warnings that are not fatal.
//----------------------------------------------------------------------
void Log::Warning(const char *format, ...) {
  char *arg_msg = nullptr;
  va_list args;
  va_start(args, format);
  ::vasprintf(&arg_msg, format, args);
  va_end(args);

  if (arg_msg == nullptr)
    return;

  Printf("warning: %s", arg_msg);
  free(arg_msg);
}

typedef std::map<ConstString, Log::Callbacks> CallbackMap;
typedef CallbackMap::iterator CallbackMapIter;

typedef std::map<ConstString, LogChannelSP> LogChannelMap;
typedef LogChannelMap::iterator LogChannelMapIter;

// Surround our callback map with a singleton function so we don't have any
// global initializers.
static CallbackMap &GetCallbackMap() {
  static CallbackMap g_callback_map;
  return g_callback_map;
}

static LogChannelMap &GetChannelMap() {
  static LogChannelMap g_channel_map;
  return g_channel_map;
}

void Log::RegisterLogChannel(const ConstString &channel,
                             const Log::Callbacks &log_callbacks) {
  GetCallbackMap().insert(std::make_pair(channel, log_callbacks));
}

bool Log::UnregisterLogChannel(const ConstString &channel) {
  return GetCallbackMap().erase(channel) != 0;
}

bool Log::GetLogChannelCallbacks(const ConstString &channel,
                                 Log::Callbacks &log_callbacks) {
  CallbackMap &callback_map = GetCallbackMap();
  CallbackMapIter pos = callback_map.find(channel);
  if (pos != callback_map.end()) {
    log_callbacks = pos->second;
    return true;
  }
  ::memset(&log_callbacks, 0, sizeof(log_callbacks));
  return false;
}

bool Log::EnableLogChannel(lldb::StreamSP &log_stream_sp, uint32_t log_options,
                           const char *channel, const char **categories,
                           Stream &error_stream) {
  Log::Callbacks log_callbacks;
  if (Log::GetLogChannelCallbacks(ConstString(channel), log_callbacks)) {
    log_callbacks.enable(log_stream_sp, log_options, categories, &error_stream);
    return true;
  }

  LogChannelSP log_channel_sp(LogChannel::FindPlugin(channel));
  if (log_channel_sp) {
    if (log_channel_sp->Enable(log_stream_sp, log_options, &error_stream,
                               categories)) {
      return true;
    } else {
      error_stream.Printf("Invalid log channel '%s'.\n", channel);
      return false;
    }
  } else {
    error_stream.Printf("Invalid log channel '%s'.\n", channel);
    return false;
  }
}

void Log::EnableAllLogChannels(StreamSP &log_stream_sp, uint32_t log_options,
                               const char **categories, Stream *feedback_strm) {
  CallbackMap &callback_map = GetCallbackMap();
  CallbackMapIter pos, end = callback_map.end();

  for (pos = callback_map.begin(); pos != end; ++pos)
    pos->second.enable(log_stream_sp, log_options, categories, feedback_strm);

  LogChannelMap &channel_map = GetChannelMap();
  LogChannelMapIter channel_pos, channel_end = channel_map.end();
  for (channel_pos = channel_map.begin(); channel_pos != channel_end;
       ++channel_pos) {
    channel_pos->second->Enable(log_stream_sp, log_options, feedback_strm,
                                categories);
  }
}

void Log::AutoCompleteChannelName(const char *channel_name,
                                  StringList &matches) {
  LogChannelMap &map = GetChannelMap();
  LogChannelMapIter pos, end = map.end();
  for (pos = map.begin(); pos != end; ++pos) {
    const char *pos_channel_name = pos->first.GetCString();
    if (channel_name && channel_name[0]) {
      if (NameMatches(channel_name, eNameMatchStartsWith, pos_channel_name)) {
        matches.AppendString(pos_channel_name);
      }
    } else
      matches.AppendString(pos_channel_name);
  }
}

void Log::DisableAllLogChannels(Stream *feedback_strm) {
  CallbackMap &callback_map = GetCallbackMap();
  CallbackMapIter pos, end = callback_map.end();
  const char *categories[] = {"all", nullptr};

  for (pos = callback_map.begin(); pos != end; ++pos)
    pos->second.disable(categories, feedback_strm);

  LogChannelMap &channel_map = GetChannelMap();
  LogChannelMapIter channel_pos, channel_end = channel_map.end();
  for (channel_pos = channel_map.begin(); channel_pos != channel_end;
       ++channel_pos)
    channel_pos->second->Disable(categories, feedback_strm);
}

void Log::Initialize() {
  Log::Callbacks log_callbacks = {DisableLog, EnableLog, ListLogCategories};
  Log::RegisterLogChannel(ConstString("lldb"), log_callbacks);
}

void Log::Terminate() { DisableAllLogChannels(nullptr); }

void Log::ListAllLogChannels(Stream *strm) {
  CallbackMap &callback_map = GetCallbackMap();
  LogChannelMap &channel_map = GetChannelMap();

  if (callback_map.empty() && channel_map.empty()) {
    strm->PutCString("No logging channels are currently registered.\n");
    return;
  }

  CallbackMapIter pos, end = callback_map.end();
  for (pos = callback_map.begin(); pos != end; ++pos)
    pos->second.list_categories(strm);

  uint32_t idx = 0;
  const char *name;
  for (idx = 0;
       (name = PluginManager::GetLogChannelCreateNameAtIndex(idx)) != nullptr;
       ++idx) {
    LogChannelSP log_channel_sp(LogChannel::FindPlugin(name));
    if (log_channel_sp)
      log_channel_sp->ListCategories(strm);
  }
}

bool Log::GetVerbose() const {
  // FIXME: This has to be centralized between the stream and the log...
  if (m_options.Test(LLDB_LOG_OPTION_VERBOSE))
    return true;

  // Make a copy of our stream shared pointer in case someone disables our
  // log while we are logging and releases the stream
  StreamSP stream_sp(m_stream_sp);
  if (stream_sp)
    return stream_sp->GetVerbose();
  return false;
}

//------------------------------------------------------------------
// Returns true if the debug flag bit is set in this stream.
//------------------------------------------------------------------
bool Log::GetDebug() const {
  // Make a copy of our stream shared pointer in case someone disables our
  // log while we are logging and releases the stream
  StreamSP stream_sp(m_stream_sp);
  if (stream_sp)
    return stream_sp->GetDebug();
  return false;
}

LogChannelSP LogChannel::FindPlugin(const char *plugin_name) {
  LogChannelSP log_channel_sp;
  LogChannelMap &channel_map = GetChannelMap();
  ConstString log_channel_name(plugin_name);
  LogChannelMapIter pos = channel_map.find(log_channel_name);
  if (pos == channel_map.end()) {
    ConstString const_plugin_name(plugin_name);
    LogChannelCreateInstance create_callback =
        PluginManager::GetLogChannelCreateCallbackForPluginName(
            const_plugin_name);
    if (create_callback) {
      log_channel_sp.reset(create_callback());
      if (log_channel_sp) {
        // Cache the one and only loaded instance of each log channel
        // plug-in after it has been loaded once.
        channel_map[log_channel_name] = log_channel_sp;
      }
    }
  } else {
    // We have already loaded an instance of this log channel class,
    // so just return the cached instance.
    log_channel_sp = pos->second;
  }
  return log_channel_sp;
}

LogChannel::LogChannel() : m_log_ap() {}

LogChannel::~LogChannel() = default;
