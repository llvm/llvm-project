//===-- lldb-private.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_lldb_private_h_
#define lldb_lldb_private_h_

#if defined(__cplusplus)

#ifdef _WIN32
#include "lldb/Host/windows/win32.h"
#endif

#ifdef __ANDROID_NDK__
#include "lldb/Host/android/Android.h"
#endif

#include "lldb/lldb-public.h"
#include "lldb/lldb-private-enumerations.h"
#include "lldb/lldb-private-interfaces.h"
#include "lldb/lldb-private-types.h"

namespace lldb_private {

const char *
GetVersion ();

} // namespace lldb_private


#endif  // defined(__cplusplus)


#endif  // lldb_lldb_private_h_
