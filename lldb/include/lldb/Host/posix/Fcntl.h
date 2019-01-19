//===-- Fcntl.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file defines fcntl functions & structures

#ifndef liblldb_Host_posix_Fcntl_h_
#define liblldb_Host_posix_Fcntl_h_

#ifdef __ANDROID__
#include <android/api-level.h>
#endif

#include <fcntl.h>

#if defined(__ANDROID_API__) && __ANDROID_API__ < 21
#define F_DUPFD_CLOEXEC (F_LINUX_SPECIFIC_BASE + 6)
#endif

#endif // liblldb_Host_posix_Fcntl_h_
