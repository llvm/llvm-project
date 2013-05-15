//===-- lldb-private-log.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_lldb_private_log_h_
#define liblldb_lldb_private_log_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"

//----------------------------------------------------------------------
// Log Bits specific to logging in lldb
//----------------------------------------------------------------------
#define LIBLLDB_LOG_VERBOSE             (1u << 0)
#define LIBLLDB_LOG_PROCESS             (1u << 1)
#define LIBLLDB_LOG_THREAD              (1u << 2)
#define LIBLLDB_LOG_DYNAMIC_LOADER      (1u << 3)
#define LIBLLDB_LOG_EVENTS              (1u << 4)
#define LIBLLDB_LOG_BREAKPOINTS         (1u << 5)
#define LIBLLDB_LOG_WATCHPOINTS         (1u << 6)
#define LIBLLDB_LOG_STEP                (1u << 7)
#define LIBLLDB_LOG_EXPRESSIONS         (1u << 8)
#define LIBLLDB_LOG_TEMPORARY           (1u << 9)
#define LIBLLDB_LOG_STATE               (1u << 10)
#define LIBLLDB_LOG_OBJECT              (1u << 11)
#define LIBLLDB_LOG_COMMUNICATION       (1u << 12)
#define LIBLLDB_LOG_CONNECTION          (1u << 13)
#define LIBLLDB_LOG_HOST                (1u << 14)
#define LIBLLDB_LOG_UNWIND              (1u << 15)
#define LIBLLDB_LOG_API                 (1u << 16)
#define LIBLLDB_LOG_SCRIPT              (1u << 17)
#define LIBLLDB_LOG_COMMANDS            (1U << 18)
#define LIBLLDB_LOG_TYPES               (1u << 19)
#define LIBLLDB_LOG_SYMBOLS             (1u << 20)
#define LIBLLDB_LOG_MODULES             (1u << 21)
#define LIBLLDB_LOG_TARGET              (1u << 22)
#define LIBLLDB_LOG_MMAP                (1u << 23)
#define LIBLLDB_LOG_ALL                 (UINT32_MAX)
#define LIBLLDB_LOG_DEFAULT             (LIBLLDB_LOG_PROCESS              |\
                                         LIBLLDB_LOG_THREAD               |\
                                         LIBLLDB_LOG_DYNAMIC_LOADER       |\
                                         LIBLLDB_LOG_BREAKPOINTS          |\
                                         LIBLLDB_LOG_WATCHPOINTS          |\
                                         LIBLLDB_LOG_STEP                 |\
                                         LIBLLDB_LOG_STATE                |\
                                         LIBLLDB_LOG_SYMBOLS              |\
                                         LIBLLDB_LOG_TARGET               |\
                                         LIBLLDB_LOG_COMMANDS)

namespace lldb_private {

void
LogIfAllCategoriesSet (uint32_t mask, const char *format, ...);

void
LogIfAnyCategoriesSet (uint32_t mask, const char *format, ...);

Log *
GetLogIfAllCategoriesSet (uint32_t mask);

Log *
GetLogIfAnyCategoriesSet (uint32_t mask);

uint32_t
GetLogMask ();

bool
IsLogVerbose ();

void
DisableLog (const char **categories, Stream *feedback_strm);

Log *
EnableLog (lldb::StreamSP &log_stream_sp, uint32_t log_options, const char **categories, Stream *feedback_strm);

void
ListLogCategories (Stream *strm);

} // namespace lldb_private

#endif  // liblldb_lldb_private_log_h_
