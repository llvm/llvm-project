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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/ManagedStatic.h"
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

namespace {
  struct ChannelAndLog {
    Log log;
    Log::Channel &channel;

    ChannelAndLog(Log::Channel &channel) : channel(channel) {}
  };
  typedef llvm::StringMap<ChannelAndLog> ChannelMap;
}

static llvm::ManagedStatic<ChannelMap> g_channel_map;

static void ListCategories(Stream &stream, const ChannelMap::value_type &entry) {
  stream.Format("Logging categories for '{0}':\n", entry.first());
  stream.Format("  all - all available logging categories\n");
  stream.Format("  default - default set of logging categories\n");
  for (const auto &category : entry.second.channel.categories)
    stream.Format("  {0} - {1}\n", category.name, category.description);
}

static uint32_t GetFlags(Stream &stream, const ChannelMap::value_type &entry,
                         llvm::ArrayRef<const char *> categories) {
  bool list_categories = false;
  uint32_t flags = 0;
  for (const char *category : categories) {
    if (llvm::StringRef("all").equals_lower(category)) {
      flags |= UINT32_MAX;
      continue;
    }
    if (llvm::StringRef("default").equals_lower(category)) {
      flags |= entry.second.channel.default_flags;
      continue;
    }
    auto cat = llvm::find_if(
        entry.second.channel.categories,
        [&](const Log::Category &c) { return c.name.equals_lower(category); });
    if (cat != entry.second.channel.categories.end()) {
      flags |= cat->flag;
      continue;
    }
    stream.Format("error: unrecognized log category '{0}'\n", category);
    list_categories = true;
  }
  if (list_categories)
    ListCategories(stream, entry);
  return flags;
}

void Log::Channel::Enable(Log &log,
                          const std::shared_ptr<llvm::raw_ostream> &stream_sp,
                          uint32_t options, uint32_t flags) {
  log.GetMask().Set(flags);
  if (log.GetMask().Get()) {
    log.GetOptions().Set(options);
    log.SetStream(stream_sp);
    log_ptr.store(&log, std::memory_order_release);
  }
}

void Log::Channel::Disable(uint32_t flags) {
  Log *log = log_ptr.load(std::memory_order_acquire);
  if (!log)
    return;
  log->GetMask().Clear(flags);
  if (!log->GetMask().Get()) {
    log->SetStream(nullptr);
    log_ptr.store(nullptr, std::memory_order_release);
  }
}

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

void Log::Register(llvm::StringRef name, Channel &channel) {
  auto iter = g_channel_map->try_emplace(name, channel);
  assert(iter.second == true);
  (void)iter;
}

void Log::Unregister(llvm::StringRef name) {
  auto iter = g_channel_map->find(name);
  assert(iter != g_channel_map->end());
  iter->second.channel.Disable(UINT32_MAX);
  g_channel_map->erase(iter);
}

bool Log::EnableLogChannel(
    const std::shared_ptr<llvm::raw_ostream> &log_stream_sp,
    uint32_t log_options, llvm::StringRef channel,
    llvm::ArrayRef<const char *> categories, Stream &error_stream) {
  auto iter = g_channel_map->find(channel);
  if (iter == g_channel_map->end()) {
    error_stream.Format("Invalid log channel '{0}'.\n", channel);
    return false;
  }
  uint32_t flags = categories.empty()
                       ? iter->second.channel.default_flags
                       : GetFlags(error_stream, *iter, categories);
  iter->second.channel.Enable(iter->second.log, log_stream_sp, log_options,
                              flags);
  return true;
}

bool Log::DisableLogChannel(llvm::StringRef channel,
                            llvm::ArrayRef<const char *> categories,
                            Stream &error_stream) {
  auto iter = g_channel_map->find(channel);
  if (iter == g_channel_map->end()) {
    error_stream.Format("Invalid log channel '{0}'.\n", channel);
    return false;
  }
  uint32_t flags = categories.empty()
                       ? UINT32_MAX
                       : GetFlags(error_stream, *iter, categories);
  iter->second.channel.Disable(flags);
  return true;
}

bool Log::ListChannelCategories(llvm::StringRef channel, Stream &stream) {
  auto ch = g_channel_map->find(channel);
  if (ch == g_channel_map->end()) {
    stream.Format("Invalid log channel '{0}'.\n", channel);
    return false;
  }
  ListCategories(stream, *ch);
  return true;
}

void Log::DisableAllLogChannels(Stream *feedback_strm) {
  for (auto &entry : *g_channel_map)
    entry.second.channel.Disable(UINT32_MAX);
}

void Log::ListAllLogChannels(Stream *strm) {
  if (g_channel_map->empty()) {
    strm->PutCString("No logging channels are currently registered.\n");
    return;
  }

  for (const auto &channel : *g_channel_map)
    ListCategories(*strm, channel);
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
  auto stream_sp = GetStream();
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
