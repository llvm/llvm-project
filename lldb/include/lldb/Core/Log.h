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

// Project includes
#include "lldb/Core/Logging.h"
#include "lldb/Core/PluginInterface.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Flags.h"
#include "lldb/lldb-private.h"

// Other libraries and framework includes
#include "llvm/Support/FormatVariadic.h"
// C++ Includes
#include <cstdarg>
#include <cstdint>
// C Includes

//----------------------------------------------------------------------
// Logging Options
//----------------------------------------------------------------------
#define LLDB_LOG_OPTION_THREADSAFE (1u << 0)
#define LLDB_LOG_OPTION_VERBOSE (1u << 1)
#define LLDB_LOG_OPTION_PREPEND_SEQUENCE (1u << 3)
#define LLDB_LOG_OPTION_PREPEND_TIMESTAMP (1u << 4)
#define LLDB_LOG_OPTION_PREPEND_PROC_AND_THREAD (1u << 5)
#define LLDB_LOG_OPTION_PREPEND_THREAD_NAME (1U << 6)
#define LLDB_LOG_OPTION_BACKTRACE (1U << 7)
#define LLDB_LOG_OPTION_APPEND (1U << 8)
#define LLDB_LOG_OPTION_PREPEND_FILE_FUNCTION (1U << 9)

//----------------------------------------------------------------------
// Logging Functions
//----------------------------------------------------------------------
namespace lldb_private {

class Log final {
public:
  //------------------------------------------------------------------
  // Callback definitions for abstracted plug-in log access.
  //------------------------------------------------------------------
  typedef void (*DisableCallback)(const char **categories,
                                  Stream *feedback_strm);
  typedef Log *(*EnableCallback)(
      const std::shared_ptr<llvm::raw_ostream> &log_stream_sp,
      uint32_t log_options, const char **categories, Stream *feedback_strm);
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

  static bool
  EnableLogChannel(const std::shared_ptr<llvm::raw_ostream> &log_stream_sp,
                   uint32_t log_options, const char *channel,
                   const char **categories, Stream &error_stream);

  static void
  EnableAllLogChannels(const std::shared_ptr<llvm::raw_ostream> &log_stream_sp,
                       uint32_t log_options, const char **categories,
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

  Log(const std::shared_ptr<llvm::raw_ostream> &stream_sp);

  ~Log();

  void PutCString(const char *cstr);
  void PutString(llvm::StringRef str);

  template <typename... Args>
  void Format(llvm::StringRef file, llvm::StringRef function,
              const char *format, Args &&... args) {
    Format(file, function, llvm::formatv(format, std::forward<Args>(args)...));
  }

  // CLEANUP: Add llvm::raw_ostream &Stream() function.
  void Printf(const char *format, ...) __attribute__((format(printf, 2, 3)));

  void VAPrintf(const char *format, va_list args);

  void LogIf(uint32_t mask, const char *fmt, ...)
      __attribute__((format(printf, 3, 4)));

  void Error(const char *fmt, ...) __attribute__((format(printf, 2, 3)));

  void VAError(const char *format, va_list args);

  void Verbose(const char *fmt, ...) __attribute__((format(printf, 2, 3)));

  void Warning(const char *fmt, ...) __attribute__((format(printf, 2, 3)));

  Flags &GetOptions();

  const Flags &GetOptions() const;

  Flags &GetMask();

  const Flags &GetMask() const;

  bool GetVerbose() const;

  void SetStream(const std::shared_ptr<llvm::raw_ostream> &stream_sp) {
    m_stream_sp = stream_sp;
  }

protected:
  //------------------------------------------------------------------
  // Member variables
  //------------------------------------------------------------------
  std::shared_ptr<llvm::raw_ostream> m_stream_sp;
  Flags m_options;
  Flags m_mask_bits;

private:
  DISALLOW_COPY_AND_ASSIGN(Log);

  void WriteHeader(llvm::raw_ostream &OS, llvm::StringRef file,
                   llvm::StringRef function);
  void WriteMessage(const std::string &message);

  void Format(llvm::StringRef file, llvm::StringRef function,
              const llvm::formatv_object_base &payload);
};

class LogChannel : public PluginInterface {
public:
  LogChannel();

  ~LogChannel() override;

  static lldb::LogChannelSP FindPlugin(const char *plugin_name);

  // categories is an array of chars that ends with a NULL element.
  virtual void Disable(const char **categories, Stream *feedback_strm) = 0;

  virtual bool
  Enable(const std::shared_ptr<llvm::raw_ostream> &log_stream_sp,
         uint32_t log_options,
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

#define LLDB_LOG(log, ...)                                                     \
  do {                                                                         \
    ::lldb_private::Log *log_private = (log);                                  \
    if (log_private)                                                           \
      log_private->Format(__FILE__, __FUNCTION__, __VA_ARGS__);                \
  } while (0)

#define LLDB_LOGV(log, ...)                                                    \
  do {                                                                         \
    ::lldb_private::Log *log_private = (log);                                  \
    if (log_private && log_private->GetVerbose())                              \
      log_private->Format(__FILE__, __FUNCTION__, __VA_ARGS__);                \
  } while (0)

#endif // liblldb_Log_h_
