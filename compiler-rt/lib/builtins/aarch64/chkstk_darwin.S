// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../assembly.h"

#ifdef __aarch64__

.subsections_via_symbols

.p2align 2
.globl ___chkstk_darwin
.private_extern ___chkstk_darwin
___chkstk_darwin:
// Try to use the faster libSystem implementation (3 underscores) if available.
#if !__STDC_HOSTED__
        // In a freestanding environment, it's definitely not available,
        // don't bother even checking.
        b       ___chkstk_darwin_probe

#else // __STDC_HOSTED__
        // Otherwise, use a weak import to support backdeployment.
        .weak_reference ____chkstk_darwin

Lloh0:
        adrp    x16, ____chkstk_darwin@GOTPAGE
Lloh1:
#if __LP64__
        ldr     x16, [x16, ____chkstk_darwin@GOTPAGEOFF]
        cbnz     x16, Lavailable
#else
        ldr     w16, [x16, ____chkstk_darwin@GOTPAGEOFF]
        cbnz    w16, Lavailable
#endif
        .loh AdrpLdrGot Lloh0, Lloh1
        b ___chkstk_darwin_probe
Lavailable:
        br      x16
#endif // __STDC_HOSTED__

.p2align 2
.globl ___chkstk_darwin_probe
.private_extern ___chkstk_darwin_probe
___chkstk_darwin_probe:
        cmp     x9, #1, lsl #12
        mov     x10, sp
        b.lo    Lend

Lloop:
        sub     x10, x10, #1, lsl #12
        ldrb    wzr, [x10]
        sub     x9, x9, #1, lsl #12
        cmp     x9, #1, lsl #12
        b.hi    Lloop

Lend:
        sub     x10, x10, x9
        ldrb    wzr, [x10]
        ret

#endif // __aarch64__
