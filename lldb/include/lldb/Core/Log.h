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
#include "lldb/Utility/Flags.h"
#include "lldb/lldb-private.h"

// Other libraries and framework includes
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/RWMutex.h"
// C++ Includes
#include <atomic>
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
  // Description of a log channel category.
  struct Category {
    llvm::StringLiteral name;
    llvm::StringLiteral description;
    uint32_t flag;
  };

  // This class describes a log channel. It also encapsulates the behavior
  // necessary to enable a log channel in an atomic manner.
  class Channel {
    std::atomic<Log *> log_ptr;

  public:
    const llvm::ArrayRef<Category> categories;
    const uint32_t default_flags;

    constexpr Channel(llvm::ArrayRef<Log::Category> categories,
                      uint32_t default_flags)
        : log_ptr(nullptr), categories(categories),
          default_flags(default_flags) {}

    // This function is safe to call at any time
    // FIXME: Not true yet, mask access is not atomic
    Log *GetLogIfAll(uint32_t mask) {
      Log *log = log_ptr.load(std::memory_order_acquire);
      if (log && log->GetMask().AllSet(mask))
        return log;
      return nullptr;
    }

    // This function is safe to call at any time
    // FIXME: Not true yet, mask access is not atomic
    Log *GetLogIfAny(uint32_t mask) {
      Log *log = log_ptr.load(std::memory_order_acquire);
      if (log && log->GetMask().AnySet(mask))
        return log;
      return nullptr;
    }

    // Calls to Enable and disable need to be serialized externally.
    void Enable(Log &log, const std::shared_ptr<llvm::raw_ostream> &stream_sp,
                uint32_t options, uint32_t flags);

    // Calls to Enable and disable need to be serialized externally.
    void Disable(uint32_t flags);
  };

  //------------------------------------------------------------------
  // Static accessors for logging channels
  //------------------------------------------------------------------
  static void Register(llvm::StringRef name, Channel &channel);
  static void Unregister(llvm::StringRef name);

  static bool
  EnableLogChannel(const std::shared_ptr<llvm::raw_ostream> &log_stream_sp,
                   uint32_t log_options, llvm::StringRef channel,
                   llvm::ArrayRef<const char *> categories,
                   Stream &error_stream);

  static bool DisableLogChannel(llvm::StringRef channel,
                                llvm::ArrayRef<const char *> categories,
                                Stream &error_stream);

  static bool ListChannelCategories(llvm::StringRef channel, Stream &stream);

  static void DisableAllLogChannels(Stream *feedback_strm);

  static void ListAllLogChannels(Stream *strm);

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
    llvm::sys::ScopedWriter lock(m_stream_mutex);
    m_stream_sp = stream_sp;
  }

  std::shared_ptr<llvm::raw_ostream> GetStream() {
    llvm::sys::ScopedReader lock(m_stream_mutex);
    return m_stream_sp;
  }

protected:
  //------------------------------------------------------------------
  // Member variables
  //------------------------------------------------------------------
  llvm::sys::RWMutex m_stream_mutex; // Protects m_stream_sp
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
