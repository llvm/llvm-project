//===-- sanitizer_common_nolibc.cc ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains stubs for libc function to facilitate optional use of
// libc in no-libcdep sources.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#include "sanitizer_common.h"
#include "sanitizer_libc.h"

namespace __sanitizer {

#if SANITIZER_LINUX
bool ShouldLogAfterPrintf() { return false; }
void LogMessageOnPrintf(const char *str) {}
#endif
void WriteToSyslog(const char *buffer) {}
void Abort() { internal__exit(1); }

}  // namespace __sanitizer
