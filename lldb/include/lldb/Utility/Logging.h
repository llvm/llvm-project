//===-- Logging.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_LOGGING_H
#define LLDB_UTILITY_LOGGING_H

#include "lldb/Utility/Log.h"
#include "llvm/ADT/BitmaskEnum.h"
#include <cstdint>

namespace lldb_private {

enum class LLDBLog : Log::MaskType {
  API = Log::ChannelFlag<0>,
  AST = Log::ChannelFlag<1>,
  Breakpoints = Log::ChannelFlag<2>,
  Commands = Log::ChannelFlag<3>,
  Communication = Log::ChannelFlag<4>,
  Connection = Log::ChannelFlag<5>,
  DataFormatters = Log::ChannelFlag<6>,
  Demangle = Log::ChannelFlag<7>,
  DynamicLoader = Log::ChannelFlag<8>,
  Events = Log::ChannelFlag<9>,
  Expressions = Log::ChannelFlag<10>,
  Host = Log::ChannelFlag<11>,
  JITLoader = Log::ChannelFlag<12>,
  Language = Log::ChannelFlag<13>,
  MMap = Log::ChannelFlag<14>,
  Modules = Log::ChannelFlag<15>,
  Object = Log::ChannelFlag<16>,
  OS = Log::ChannelFlag<17>,
  Platform = Log::ChannelFlag<18>,
  Process = Log::ChannelFlag<19>,
  Script = Log::ChannelFlag<20>,
  State = Log::ChannelFlag<21>,
  Step = Log::ChannelFlag<22>,
  Symbols = Log::ChannelFlag<23>,
  SystemRuntime = Log::ChannelFlag<24>,
  Target = Log::ChannelFlag<25>,
  Temporary = Log::ChannelFlag<26>,
  Thread = Log::ChannelFlag<27>,
  Types = Log::ChannelFlag<28>,
  Unwind = Log::ChannelFlag<29>,
  Watchpoints = Log::ChannelFlag<30>,
  LLVM_MARK_AS_BITMASK_ENUM(Watchpoints),
};

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

// Log Bits specific to logging in lldb
#define LIBLLDB_LOG_PROCESS ::lldb_private::LLDBLog::Process
#define LIBLLDB_LOG_THREAD ::lldb_private::LLDBLog::Thread
#define LIBLLDB_LOG_DYNAMIC_LOADER ::lldb_private::LLDBLog::DynamicLoader
#define LIBLLDB_LOG_EVENTS ::lldb_private::LLDBLog::Events
#define LIBLLDB_LOG_BREAKPOINTS ::lldb_private::LLDBLog::Breakpoints
#define LIBLLDB_LOG_WATCHPOINTS ::lldb_private::LLDBLog::Watchpoints
#define LIBLLDB_LOG_STEP ::lldb_private::LLDBLog::Step
#define LIBLLDB_LOG_EXPRESSIONS ::lldb_private::LLDBLog::Expressions
#define LIBLLDB_LOG_TEMPORARY ::lldb_private::LLDBLog::Temporary
#define LIBLLDB_LOG_STATE ::lldb_private::LLDBLog::State
#define LIBLLDB_LOG_OBJECT ::lldb_private::LLDBLog::Object
#define LIBLLDB_LOG_COMMUNICATION ::lldb_private::LLDBLog::Communication
#define LIBLLDB_LOG_CONNECTION ::lldb_private::LLDBLog::Connection
#define LIBLLDB_LOG_HOST ::lldb_private::LLDBLog::Host
#define LIBLLDB_LOG_UNWIND ::lldb_private::LLDBLog::Unwind
#define LIBLLDB_LOG_API ::lldb_private::LLDBLog::API
#define LIBLLDB_LOG_SCRIPT ::lldb_private::LLDBLog::Script
#define LIBLLDB_LOG_COMMANDS ::lldb_private::LLDBLog::Commands
#define LIBLLDB_LOG_TYPES ::lldb_private::LLDBLog::Types
#define LIBLLDB_LOG_SYMBOLS ::lldb_private::LLDBLog::Symbols
#define LIBLLDB_LOG_MODULES ::lldb_private::LLDBLog::Modules
#define LIBLLDB_LOG_TARGET ::lldb_private::LLDBLog::Target
#define LIBLLDB_LOG_MMAP ::lldb_private::LLDBLog::MMap
#define LIBLLDB_LOG_OS ::lldb_private::LLDBLog::OS
#define LIBLLDB_LOG_PLATFORM ::lldb_private::LLDBLog::Platform
#define LIBLLDB_LOG_SYSTEM_RUNTIME ::lldb_private::LLDBLog::SystemRuntime
#define LIBLLDB_LOG_JIT_LOADER ::lldb_private::LLDBLog::JITLoader
#define LIBLLDB_LOG_LANGUAGE ::lldb_private::LLDBLog::Language
#define LIBLLDB_LOG_DATAFORMATTERS ::lldb_private::LLDBLog::DataFormatters
#define LIBLLDB_LOG_DEMANGLE ::lldb_private::LLDBLog::Demangle
#define LIBLLDB_LOG_AST ::lldb_private::LLDBLog::AST

Log *GetLogIfAllCategoriesSet(LLDBLog mask);

Log *GetLogIfAnyCategoriesSet(LLDBLog mask);

void InitializeLldbChannel();

template <> Log::Channel &LogChannelFor<LLDBLog>();
} // namespace lldb_private

#endif // LLDB_UTILITY_LOGGING_H
