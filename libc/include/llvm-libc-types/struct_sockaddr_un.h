//===-- Definition of struct sockaddr_un ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_STRUCT_SOCKADDR_UN_H__
#define __LLVM_LIBC_TYPES_STRUCT_SOCKADDR_UN_H__

#include <llvm-libc-types/sa_family_t.h>

// This is the sockaddr specialization for AF_UNIX or AF_LOCAL sockets, as
// defined by posix.

struct sockaddr_un {
  sa_family_t sun_family; /* AF_UNIX */
  char sun_path[108];     /* Pathname */
};

#endif // __LLVM_LIBC_TYPES_STRUCT_SOCKADDR_UN_H__
