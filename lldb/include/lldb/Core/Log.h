//===-- Log.h ---------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Log_h_
#define liblldb_Log_h_

// C Includes
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Flags.h"
#include "lldb/Core/Logging.h"
#include "lldb/Core/PluginInterface.h"
#include "lldb/lldb-private.h"

//----------------------------------------------------------------------
// Logging Options
//----------------------------------------------------------------------
#define LLDB_LOG_OPTION_THREADSAFE (1u << 0)
#define LLDB_LOG_OPTION_VERBOSE (1u << 1)
#define LLDB_LOG_OPTION_DEBUG (1u << 2)
#define LLDB_LOG_OPTION_PREPEND_SEQUENCE (1u << 3)
#define LLDB_LOG_OPTION_PREPEND_TIMESTAMP (1u << 4)
#define LLDB_LOG_OPTION_PREPEND_PROC_AND_THREAD (1u << 5)
#define LLDB_LOG_OPTION_PREPEND_THREAD_NAME (1U << 6)
#define LLDB_LOG_OPTION_BACKTRACE (1U << 7)
#define LLDB_LOG_OPTION_APPEND (1U << 8)

//----------------------------------------------------------------------
// Logging Functions
//----------------------------------------------------------------------
namespace lldb_private {

class Log {
public:
  //------------------------------------------------------------------
  // Callback definitions for abstracted plug-in log access.
  //------------------------------------------------------------------
  typedef void (*DisableCallback)(const char **categories,
                                  Stream *feedback_strm);
  typedef Log *(*EnableCallback)(lldb::StreamSP &log_stream_sp,
                                 uint32_t log_options, const char **categories,
                                 Stream *feedback_strm);
  typedef void (*ListCategoriesCallback)(Stream *strm);

  struct Callbacks {
    DisableCallback disable;
    EnableCallback enable;
    ListCategoriesCallback list_categories;
  };

  //------------------------------------------------------------------
  // Static accessors for logging channels
  //------------------------------------------------------------------
  static void RegisterLogChannel(const ConstString &channel,
                                 const Log::Callbacks &log_callbacks);

  static bool UnregisterLogChannel(const ConstString &channel);

  static bool GetLogChannelCallbacks(const ConstString &channel,
                                     Log::Callbacks &log_callbacks);

  static bool EnableLogChannel(lldb::StreamSP &log_stream_sp,
                               uint32_t log_options, const char *channel,
                               const char **categories, Stream &error_stream);

  static void EnableAllLogChannels(lldb::StreamSP &log_stream_sp,
                                   uint32_t log_options,
                                   const char **categories,
                                   Stream *feedback_strm);

  static void DisableAllLogChannels(Stream *feedback_strm);

  static void ListAllLogChannels(Stream *strm);

  static void Initialize();

  static void Terminate();

  //------------------------------------------------------------------
  // Auto completion
  //------------------------------------------------------------------
  static void AutoCompleteChannelName(const char *channel_name,
                                      StringList &matches);

  //------------------------------------------------------------------
  // Member functions
  //------------------------------------------------------------------
  Log();

  Log(const lldb::StreamSP &stream_sp);

  virtual ~Log();

  virtual void PutCString(const char *cstr);

  // CLEANUP: Add llvm::raw_ostream &Stream() function.
  virtual void Printf(const char *format, ...)
      __attribute__((format(printf, 2, 3)));

  virtual void VAPrintf(const char *format, va_list args);

  virtual void LogIf(uint32_t mask, const char *fmt, ...)
      __attribute__((format(printf, 3, 4)));

  virtual void Debug(const char *fmt, ...)
      __attribute__((format(printf, 2, 3)));

  virtual void DebugVerbose(const char *fmt, ...)
      __attribute__((format(printf, 2, 3)));

  virtual void Error(const char *fmt, ...)
      __attribute__((format(printf, 2, 3)));

  virtual void VAError(const char *format, va_list args);

  virtual void FatalError(int err, const char *fmt, ...)
      __attribute__((format(printf, 3, 4)));

  virtual void Verbose(const char *fmt, ...)
      __attribute__((format(printf, 2, 3)));

  virtual void Warning(const char *fmt, ...)
      __attribute__((format(printf, 2, 3)));

  virtual void WarningVerbose(const char *fmt, ...)
      __attribute__((format(printf, 2, 3)));

  Flags &GetOptions();

  const Flags &GetOptions() const;

  Flags &GetMask();

  const Flags &GetMask() const;

  bool GetVerbose() const;

  bool GetDebug() const;

  void SetStream(const lldb::StreamSP &stream_sp) { m_stream_sp = stream_sp; }

protected:
  //------------------------------------------------------------------
  // Member variables
  //------------------------------------------------------------------
  lldb::StreamSP m_stream_sp;
  Flags m_options;
  Flags m_mask_bits;

private:
  DISALLOW_COPY_AND_ASSIGN(Log);
};

class LogChannel : public PluginInterface {
public:
  LogChannel();

  ~LogChannel() override;

  static lldb::LogChannelSP FindPlugin(const char *plugin_name);

  // categories is an array of chars that ends with a NULL element.
  virtual void Disable(const char **categories, Stream *feedback_strm) = 0;

  virtual bool
  Enable(lldb::StreamSP &log_stream_sp, uint32_t log_options,
         Stream *feedback_strm, // Feedback stream for argument errors etc
         const char **categories) = 0; // The categories to enable within this
                                       // logging stream, if empty, enable
                                       // default set

  virtual void ListCategories(Stream *strm) = 0;

protected:
  std::unique_ptr<Log> m_log_ap;

private:
  DISALLOW_COPY_AND_ASSIGN(LogChannel);
};

} // namespace lldb_private

#endif // liblldb_Log_h_
