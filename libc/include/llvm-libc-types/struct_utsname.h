//===-- Definition of struct utsname --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_STRUCT_UTSNAME_H__
#define __LLVM_LIBC_TYPES_STRUCT_UTSNAME_H__

#ifdef __linux__
#define __UTS_NAME_LENGTH 65
#else
// Arbitray default. Should be specialized for each platform.
#define __UTS_NAME_LENGTH 1024
#endif

struct utsname {
  char sysname[__UTS_NAME_LENGTH];
  char nodename[__UTS_NAME_LENGTH];
  char release[__UTS_NAME_LENGTH];
  char version[__UTS_NAME_LENGTH];
  char machine[__UTS_NAME_LENGTH];
#ifdef __linux__
  char domainname[__UTS_NAME_LENGTH];
#endif
};

#undef __UTS_NAME_LENGTH

#endif // __LLVM_LIBC_TYPES_STRUCT_UTSNAME_H__
