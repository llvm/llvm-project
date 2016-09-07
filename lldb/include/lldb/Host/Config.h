//===-- Config.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Config_h_
#define liblldb_Config_h_

#if defined(__APPLE__)

#include "lldb/Host/macosx/Config.h"

#elif defined(__ANDROID_NDK__)

#include "lldb/Host/android/Config.h"

#elif defined(__linux__) || defined(__GNU__)

#include "lldb/Host/linux/Config.h"

#elif defined(__FreeBSD__) || defined(__FreeBSD_kernel__) ||                   \
    defined(__OpenBSD__)

#include "lldb/Host/freebsd/Config.h"

#elif defined(__NetBSD__)

#include "lldb/Host/netbsd/Config.h"

#elif defined(__MINGW__) || defined(__MINGW32__)

#include "lldb/Host/mingw/Config.h"

#elif defined(_MSC_VER)

#include "lldb/Host/msvc/Config.h"

#else

#error undefined platform

#endif

#endif // #ifndef liblldb_Config_h_
