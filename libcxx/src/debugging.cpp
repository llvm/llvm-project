//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__linux__)
#  include "debugging/linux.ipp"
#elif defined(__FREEBSD__) || defined(__APPLE__)
#  include "debugging/bsd_like.ipp"
#elif defined(_WIN32)
#  include "debugging/windows.ipp"
#elif defined(_AIX)
#  include "debugging/aix.ipp"
#else
#  include "debugging/default.ipp"
#endif
