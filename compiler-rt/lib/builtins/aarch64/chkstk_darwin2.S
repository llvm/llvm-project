// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../assembly.h"

#ifdef __aarch64__

.subsections_via_symbols

.p2align 2
.globl ____chkstk_darwin
.private_extern ____chkstk_darwin
____chkstk_darwin:
// This is the fallback implementation if we weren't able to link libSystem.
        b    ___chkstk_darwin_probe

#endif // __aarch64__
