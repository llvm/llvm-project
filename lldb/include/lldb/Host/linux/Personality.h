//===-- Personality.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// This file defines personality functions & structures

#ifndef liblldb_Host_linux_Personality_h_
#define liblldb_Host_linux_Personality_h_

#ifdef __ANDROID__
#include <android/api-level.h>
#endif

#if defined(__ANDROID_API__) && __ANDROID_API__ < 21
#include <linux/personality.h>
#else
#include <sys/personality.h>
#endif

#endif // liblldb_Host_linux_Personality_h_
