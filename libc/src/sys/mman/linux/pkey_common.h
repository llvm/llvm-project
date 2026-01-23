//===---------- Linux implementation of the Linux pkey_mprotect function --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/macros/properties/architectures.h"

#if defined(LIBC_TARGET_ARCH_IS_X86_64)
#include "src/sys/mman/linux/x86_64/pkey_common.h"
#else
#include "src/sys/mman/linux/generic/pkey_common.h"
#endif
