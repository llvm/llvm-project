//===-- Log.cpp -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Project includes
#include "lldb/Core/Log.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/ThisThread.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Utility/NameMatches.h"
#include "lldb/Utility/StreamString.h"

// Other libraries and framework includes
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

// C Includes
// C++ Includes
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

using namespace lldb;
using namespace lldb_private;

Log::Log() : m_stream_sp(), m_options(0), m_mask_bits(0) {}

Log::Log(const std::shared_ptr<llvm::raw_ostream> &stream_sp)
    : m_stream_sp(stream_sp), m_options(0), m_mask_bits(0) {}

Log::~Log() = default;

Flags &Log::GetOptions() { return m_options; }

const Flags &Log::GetOptions() const { return m_options; }

Flags &Log::GetMask() { return m_mask_bits; }

const Flags &Log::GetMask() const { return m_mask_bits; }

void Log::PutCString(const char *cstr) { Printf("%s", cstr); }
void Log::PutString(llvm::StringRef str) { PutCString(str.str().c_str()); }

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
  std::string message_string;
  llvm::raw_string_ostream message(message_string);
  WriteHeader(message, "", "");

  char *text;
  vasprintf(&text, format, args);
  message << text;
  free(text);

  message << "\n";

  WriteMessage(message.str());
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

bool Log::EnableLogChannel(
    const std::shared_ptr<llvm::raw_ostream> &log_stream_sp,
    uint32_t log_options, const char *channel, const char **categories,
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

void Log::EnableAllLogChannels(
    const std::shared_ptr<llvm::raw_ostream> &log_stream_sp,
    uint32_t log_options, const char **categories, Stream *feedback_strm) {
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

bool Log::GetVerbose() const { return m_options.Test(LLDB_LOG_OPTION_VERBOSE); }

void Log::WriteHeader(llvm::raw_ostream &OS, llvm::StringRef file,
                      llvm::StringRef function) {
  static uint32_t g_sequence_id = 0;
  // Add a sequence ID if requested
  if (m_options.Test(LLDB_LOG_OPTION_PREPEND_SEQUENCE))
    OS << ++g_sequence_id << " ";

  // Timestamp if requested
  if (m_options.Test(LLDB_LOG_OPTION_PREPEND_TIMESTAMP)) {
    auto now = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch());
    OS << llvm::formatv("{0:f9} ", now.count());
  }

  // Add the process and thread if requested
  if (m_options.Test(LLDB_LOG_OPTION_PREPEND_PROC_AND_THREAD))
    OS << llvm::formatv("[{0,0+4}/{1,0+4}] ", getpid(),
                        Host::GetCurrentThreadID());

  // Add the thread name if requested
  if (m_options.Test(LLDB_LOG_OPTION_PREPEND_THREAD_NAME)) {
    llvm::SmallString<32> thread_name;
    ThisThread::GetName(thread_name);
    if (!thread_name.empty())
      OS << thread_name;
  }

  if (m_options.Test(LLDB_LOG_OPTION_BACKTRACE))
    llvm::sys::PrintStackTrace(OS);

  if (m_options.Test(LLDB_LOG_OPTION_PREPEND_FILE_FUNCTION) &&
      (!file.empty() || !function.empty())) {
    file = llvm::sys::path::filename(file).take_front(40);
    function = function.take_front(40);
    OS << llvm::formatv("{0,-60:60} ", (file + ":" + function).str());
  }
}

void Log::WriteMessage(const std::string &message) {
  // Make a copy of our stream shared pointer in case someone disables our
  // log while we are logging and releases the stream
  auto stream_sp = m_stream_sp;
  if (!stream_sp)
    return;

  if (m_options.Test(LLDB_LOG_OPTION_THREADSAFE)) {
    static std::recursive_mutex g_LogThreadedMutex;
    std::lock_guard<std::recursive_mutex> guard(g_LogThreadedMutex);
    *stream_sp << message;
    stream_sp->flush();
  } else {
    *stream_sp << message;
    stream_sp->flush();
  }
}

void Log::Format(llvm::StringRef file, llvm::StringRef function,
                 const llvm::formatv_object_base &payload) {
  std::string message_string;
  llvm::raw_string_ostream message(message_string);
  WriteHeader(message, file, function);
  message << payload << "\n";
  WriteMessage(message.str());
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
