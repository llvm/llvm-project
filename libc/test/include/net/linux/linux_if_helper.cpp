//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <linux/if.h>
#include <stddef.h>

extern const size_t LINUX_IFMAP_SIZE = sizeof(struct ifmap);
extern const size_t LINUX_IFMAP_ALIGN = alignof(struct ifmap);
extern const size_t LINUX_IFREQ_SIZE = sizeof(struct ifreq);
extern const size_t LINUX_IFREQ_ALIGN = alignof(struct ifreq);
