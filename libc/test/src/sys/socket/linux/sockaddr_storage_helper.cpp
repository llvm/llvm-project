//===-- Helpers for the struct sockaddr_storage test ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/sys_socket_macros.h"
#include "hdr/types/struct_sockaddr_storage.h"
#include "hdr/types/struct_sockaddr_un.h"
#include "include/llvm-libc-types/sa_family_t.h"

// POSIX requires (and many applications make use of this) the ability to cast
// one sockaddr pointer to another. This verifies that the compiler does not
// assume the two pointers do not point to the same object (alias). It is in a
// different compile unit to prevent the compiler from noticing (at least
// without LTO) that the two variables point to the same object. Noticing that
// wouldn't cause the test to fail, but it might cause it to not test the
// desired property.
sa_family_t test_sockaddr_aliasing(struct sockaddr_storage *ss,
                                   struct sockaddr_un *sun) {
  ss->ss_family = AF_UNSPEC;
  sun->sun_family = AF_UNIX;
  return ss->ss_family;
}
