//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of idtype_t type.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_IDTYPE_T_H
#define LLVM_LIBC_TYPES_IDTYPE_T_H

#ifdef P_ALL
#undef P_ALL
#endif

#ifdef P_PID
#undef P_PID
#endif

#ifdef P_PGID
#undef P_PGID
#endif

#ifdef P_PIDFD
#undef P_PIDFD
#endif

typedef enum {
  P_ALL,
  P_PID,
  P_PGID,
  P_PIDFD,
} idtype_t;

#endif // LLVM_LIBC_TYPES_IDTYPE_T_H
