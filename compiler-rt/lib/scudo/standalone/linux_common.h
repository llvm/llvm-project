//===-- linux_common.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_LINUX_COMMON_H_
#define SCUDO_LINUX_COMMON_H_

#include "platform.h"

#if SCUDO_LINUX

#include "internal_defs.h"

namespace scudo {

// Internal map fatal error. This must not call map(). SizeIfOOM shall
// hold the requested size on an out-of-memory error, 0 otherwise.
void NORETURN dieOnMapError(uptr SizeIfOOM = 0);

// Internal unmap fatal error. This must not call map().
void NORETURN dieOnUnmapError(uptr Addr, uptr Size);

// Internal protect fatal error. This must not call map().
void NORETURN dieOnProtectError(uptr Addr, uptr Size, int Prot);

} // namespace scudo

#endif // SCUDO_LINUX

#endif // SCUDO_LINUX_COMMON_H_
