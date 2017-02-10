//===-- Logging.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Logging.h"

// C Includes
// C++ Includes
#include <atomic>
#include <cstring>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Interpreter/Args.h"

using namespace lldb;
using namespace lldb_private;

// We want to avoid global constructors where code needs to be run so here we
// control access to our static g_log_sp by hiding it in a singleton function
// that will construct the static g_lob_sp the first time this function is
// called.

static std::atomic<bool> g_log_enabled{false};
static Log *g_log = nullptr;

static Log *GetLog() {
  if (!g_log_enabled)
    return nullptr;
  return g_log;
}

uint32_t lldb_private::GetLogMask() {
  Log *log(GetLog());
  if (log)
    return log->GetMask().Get();
  return 0;
}

Log *lldb_private::GetLogIfAllCategoriesSet(uint32_t mask) {
  Log *log(GetLog());
  if (log && mask) {
    uint32_t log_mask = log->GetMask().Get();
    if ((log_mask & mask) != mask)
      return nullptr;
  }
  return log;
}

void lldb_private::LogIfAllCategoriesSet(uint32_t mask, const char *format,
                                         ...) {
  Log *log(GetLogIfAllCategoriesSet(mask));
  if (log) {
    va_list args;
    va_start(args, format);
    log->VAPrintf(format, args);
    va_end(args);
  }
}

void lldb_private::LogIfAnyCategoriesSet(uint32_t mask, const char *format,
                                         ...) {
  Log *log(GetLogIfAnyCategoriesSet(mask));
  if (log != nullptr) {
    va_list args;
    va_start(args, format);
    log->VAPrintf(format, args);
    va_end(args);
  }
}

Log *lldb_private::GetLogIfAnyCategoriesSet(uint32_t mask) {
  Log *log(GetLog());
  if (log != nullptr && mask && (mask & log->GetMask().Get()))
    return log;
  return nullptr;
}

void lldb_private::DisableLog(const char **categories, Stream *feedback_strm) {
  Log *log(GetLog());

  if (log != nullptr) {
    uint32_t flag_bits = 0;
    if (categories && categories[0]) {
      flag_bits = log->GetMask().Get();
      for (size_t i = 0; categories[i] != nullptr; ++i) {
        const char *arg = categories[i];

        if (0 == ::strcasecmp(arg, "all"))
          flag_bits &= ~LIBLLDB_LOG_ALL;
        else if (0 == ::strcasecmp(arg, "api"))
          flag_bits &= ~LIBLLDB_LOG_API;
        else if (0 == ::strncasecmp(arg, "break", 5))
          flag_bits &= ~LIBLLDB_LOG_BREAKPOINTS;
        else if (0 == ::strcasecmp(arg, "commands"))
          flag_bits &= ~LIBLLDB_LOG_COMMANDS;
        else if (0 == ::strcasecmp(arg, "default"))
          flag_bits &= ~LIBLLDB_LOG_DEFAULT;
        else if (0 == ::strcasecmp(arg, "dyld"))
          flag_bits &= ~LIBLLDB_LOG_DYNAMIC_LOADER;
        else if (0 == ::strncasecmp(arg, "event", 5))
          flag_bits &= ~LIBLLDB_LOG_EVENTS;
        else if (0 == ::strncasecmp(arg, "expr", 4))
          flag_bits &= ~LIBLLDB_LOG_EXPRESSIONS;
        else if (0 == ::strncasecmp(arg, "object", 6))
          flag_bits &= ~LIBLLDB_LOG_OBJECT;
        else if (0 == ::strcasecmp(arg, "process"))
          flag_bits &= ~LIBLLDB_LOG_PROCESS;
        else if (0 == ::strcasecmp(arg, "platform"))
          flag_bits &= ~LIBLLDB_LOG_PLATFORM;
        else if (0 == ::strcasecmp(arg, "script"))
          flag_bits &= ~LIBLLDB_LOG_SCRIPT;
        else if (0 == ::strcasecmp(arg, "state"))
          flag_bits &= ~LIBLLDB_LOG_STATE;
        else if (0 == ::strcasecmp(arg, "step"))
          flag_bits &= ~LIBLLDB_LOG_STEP;
        else if (0 == ::strcasecmp(arg, "thread"))
          flag_bits &= ~LIBLLDB_LOG_THREAD;
        else if (0 == ::strcasecmp(arg, "target"))
          flag_bits &= ~LIBLLDB_LOG_TARGET;
        else if (0 == ::strncasecmp(arg, "watch", 5))
          flag_bits &= ~LIBLLDB_LOG_WATCHPOINTS;
        else if (0 == ::strncasecmp(arg, "temp", 4))
          flag_bits &= ~LIBLLDB_LOG_TEMPORARY;
        else if (0 == ::strncasecmp(arg, "comm", 4))
          flag_bits &= ~LIBLLDB_LOG_COMMUNICATION;
        else if (0 == ::strncasecmp(arg, "conn", 4))
          flag_bits &= ~LIBLLDB_LOG_CONNECTION;
        else if (0 == ::strncasecmp(arg, "host", 4))
          flag_bits &= ~LIBLLDB_LOG_HOST;
        else if (0 == ::strncasecmp(arg, "unwind", 6))
          flag_bits &= ~LIBLLDB_LOG_UNWIND;
        else if (0 == ::strncasecmp(arg, "types", 5))
          flag_bits &= ~LIBLLDB_LOG_TYPES;
        else if (0 == ::strncasecmp(arg, "symbol", 6))
          flag_bits &= ~LIBLLDB_LOG_SYMBOLS;
        else if (0 == ::strcasecmp(arg, "system-runtime"))
          flag_bits &= ~LIBLLDB_LOG_SYSTEM_RUNTIME;
        else if (0 == ::strncasecmp(arg, "module", 6))
          flag_bits &= ~LIBLLDB_LOG_MODULES;
        else if (0 == ::strncasecmp(arg, "mmap", 4))
          flag_bits &= ~LIBLLDB_LOG_MMAP;
        else if (0 == ::strcasecmp(arg, "os"))
          flag_bits &= ~LIBLLDB_LOG_OS;
        else if (0 == ::strcasecmp(arg, "jit"))
          flag_bits &= ~LIBLLDB_LOG_JIT_LOADER;
        else if (0 == ::strcasecmp(arg, "language"))
          flag_bits &= ~LIBLLDB_LOG_LANGUAGE;
        else if (0 == ::strncasecmp(arg, "formatters", 10))
          flag_bits &= ~LIBLLDB_LOG_DATAFORMATTERS;
        else if (0 == ::strncasecmp(arg, "demangle", 8))
          flag_bits &= ~LIBLLDB_LOG_DEMANGLE;
        else {
          feedback_strm->Printf("error:  unrecognized log category '%s'\n",
                                arg);
          ListLogCategories(feedback_strm);
          return;
        }
      }
    }
    log->GetMask().Reset(flag_bits);
    if (flag_bits == 0) {
      log->SetStream(nullptr);
      g_log_enabled = false;
    }
  }
}

Log *lldb_private::EnableLog(
    const std::shared_ptr<llvm::raw_ostream> &log_stream_sp,
    uint32_t log_options, const char **categories, Stream *feedback_strm) {
  // Try see if there already is a log - that way we can reuse its settings.
  // We could reuse the log in toto, but we don't know that the stream is the
  // same.
  uint32_t flag_bits;
  if (g_log != nullptr)
    flag_bits = g_log->GetMask().Get();
  else
    flag_bits = 0;

  // Now make a new log with this stream if one was provided
  if (log_stream_sp) {
    if (g_log != nullptr)
      g_log->SetStream(log_stream_sp);
    else
      g_log = new Log(log_stream_sp);
  }

  if (g_log != nullptr) {
    for (size_t i = 0; categories[i] != nullptr; ++i) {
      const char *arg = categories[i];

      if (0 == ::strcasecmp(arg, "all"))
        flag_bits |= LIBLLDB_LOG_ALL;
      else if (0 == ::strcasecmp(arg, "api"))
        flag_bits |= LIBLLDB_LOG_API;
      else if (0 == ::strncasecmp(arg, "break", 5))
        flag_bits |= LIBLLDB_LOG_BREAKPOINTS;
      else if (0 == ::strcasecmp(arg, "commands"))
        flag_bits |= LIBLLDB_LOG_COMMANDS;
      else if (0 == ::strncasecmp(arg, "commu", 5))
        flag_bits |= LIBLLDB_LOG_COMMUNICATION;
      else if (0 == ::strncasecmp(arg, "conn", 4))
        flag_bits |= LIBLLDB_LOG_CONNECTION;
      else if (0 == ::strcasecmp(arg, "default"))
        flag_bits |= LIBLLDB_LOG_DEFAULT;
      else if (0 == ::strcasecmp(arg, "dyld"))
        flag_bits |= LIBLLDB_LOG_DYNAMIC_LOADER;
      else if (0 == ::strncasecmp(arg, "event", 5))
        flag_bits |= LIBLLDB_LOG_EVENTS;
      else if (0 == ::strncasecmp(arg, "expr", 4))
        flag_bits |= LIBLLDB_LOG_EXPRESSIONS;
      else if (0 == ::strncasecmp(arg, "host", 4))
        flag_bits |= LIBLLDB_LOG_HOST;
      else if (0 == ::strncasecmp(arg, "mmap", 4))
        flag_bits |= LIBLLDB_LOG_MMAP;
      else if (0 == ::strncasecmp(arg, "module", 6))
        flag_bits |= LIBLLDB_LOG_MODULES;
      else if (0 == ::strncasecmp(arg, "object", 6))
        flag_bits |= LIBLLDB_LOG_OBJECT;
      else if (0 == ::strcasecmp(arg, "os"))
        flag_bits |= LIBLLDB_LOG_OS;
      else if (0 == ::strcasecmp(arg, "platform"))
        flag_bits |= LIBLLDB_LOG_PLATFORM;
      else if (0 == ::strcasecmp(arg, "process"))
        flag_bits |= LIBLLDB_LOG_PROCESS;
      else if (0 == ::strcasecmp(arg, "script"))
        flag_bits |= LIBLLDB_LOG_SCRIPT;
      else if (0 == ::strcasecmp(arg, "state"))
        flag_bits |= LIBLLDB_LOG_STATE;
      else if (0 == ::strcasecmp(arg, "step"))
        flag_bits |= LIBLLDB_LOG_STEP;
      else if (0 == ::strncasecmp(arg, "symbol", 6))
        flag_bits |= LIBLLDB_LOG_SYMBOLS;
      else if (0 == ::strcasecmp(arg, "system-runtime"))
        flag_bits |= LIBLLDB_LOG_SYSTEM_RUNTIME;
      else if (0 == ::strcasecmp(arg, "target"))
        flag_bits |= LIBLLDB_LOG_TARGET;
      else if (0 == ::strncasecmp(arg, "temp", 4))
        flag_bits |= LIBLLDB_LOG_TEMPORARY;
      else if (0 == ::strcasecmp(arg, "thread"))
        flag_bits |= LIBLLDB_LOG_THREAD;
      else if (0 == ::strncasecmp(arg, "types", 5))
        flag_bits |= LIBLLDB_LOG_TYPES;
      else if (0 == ::strncasecmp(arg, "unwind", 6))
        flag_bits |= LIBLLDB_LOG_UNWIND;
      else if (0 == ::strncasecmp(arg, "watch", 5))
        flag_bits |= LIBLLDB_LOG_WATCHPOINTS;
      else if (0 == ::strcasecmp(arg, "jit"))
        flag_bits |= LIBLLDB_LOG_JIT_LOADER;
      else if (0 == ::strcasecmp(arg, "language"))
        flag_bits |= LIBLLDB_LOG_LANGUAGE;
      else if (0 == ::strncasecmp(arg, "formatters", 10))
        flag_bits |= LIBLLDB_LOG_DATAFORMATTERS;
      else if (0 == ::strncasecmp(arg, "demangle", 8))
        flag_bits |= LIBLLDB_LOG_DEMANGLE;
      else {
        feedback_strm->Printf("error: unrecognized log category '%s'\n", arg);
        ListLogCategories(feedback_strm);
        return g_log;
      }
    }

    g_log->GetMask().Reset(flag_bits);
    g_log->GetOptions().Reset(log_options);
  }
  g_log_enabled = true;
  return g_log;
}

void lldb_private::ListLogCategories(Stream *strm) {
  strm->Printf(
      "Logging categories for 'lldb':\n"
      "  all - turn on all available logging categories\n"
      "  api - enable logging of API calls and return values\n"
      "  break - log breakpoints\n"
      "  commands - log command argument parsing\n"
      "  communication - log communication activities\n"
      "  connection - log connection details\n"
      "  default - enable the default set of logging categories for liblldb\n"
      "  demangle - log mangled names to catch demangler crashes\n"
      "  dyld - log shared library related activities\n"
      "  events - log broadcaster, listener and event queue activities\n"
      "  expr - log expressions\n"
      "  formatters - log data formatters related activities\n"
      "  host - log host activities\n"
      "  jit - log JIT events in the target\n"
      "  language - log language runtime events\n"
      "  mmap - log mmap related activities\n"
      "  module - log module activities such as when modules are created, "
      "destroyed, replaced, and more\n"
      "  object - log object construction/destruction for important objects\n"
      "  os - log OperatingSystem plugin related activities\n"
      "  platform - log platform events and activities\n"
      "  process - log process events and activities\n"
      "  script - log events about the script interpreter\n"
      "  state - log private and public process state changes\n"
      "  step - log step related activities\n"
      "  symbol - log symbol related issues and warnings\n"
      "  system-runtime - log system runtime events\n"
      "  target - log target events and activities\n"
      "  thread - log thread events and activities\n"
      "  types - log type system related activities\n"
      "  unwind - log stack unwind activities\n"
      "  verbose - enable verbose logging\n"
      "  watch - log watchpoint related activities\n");
}
