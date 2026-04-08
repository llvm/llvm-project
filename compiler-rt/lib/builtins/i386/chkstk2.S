// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../assembly.h"

#ifdef __i386__

// _chkstk (_alloca) routine - probe stack between %esp and (%esp-%eax) in 4k increments,
// then decrement %esp by %eax.  Preserves all registers except %esp and flags.
// This routine is windows specific
// http://msdn.microsoft.com/en-us/library/ms648426.aspx

// Clang on i386 mingw generates calls to "_alloca" (which gets decorated to
// "__alloca").
//
// GCC before 4.6 generated calls a symbol which after decoration is named
// "___chkstk", with three leading underscores. We provide that here as well.
//
// MSVC produces calls to the symbol "__chkstk", with two leading underscores.
// That one has the same signature as this one - but we don't provide that
// symbol here. (If we'd do that, we should do it in a separate object file
// to avoid potential symbol collisions - see
// commit 248aeac1ad2cf4f583490dd1312a5b448d2bb8cc for details.)
//
// GCC after 4.6 generates calls to "___chkstk_ms", which does not decrement
// %esp - that function is defined in chkstk.S.

.text
.balign 4
DEFINE_COMPILERRT_FUNCTION(_alloca) // _chkstk and _alloca are the same function
// This gets decorated into "___chkstk"; GCC < 4.6 references this symbol.
DEFINE_COMPILERRT_FUNCTION(__chkstk)
        push   %ecx
        cmp    $0x1000,%eax
        lea    8(%esp),%ecx     // esp before calling this routine -> ecx
        jb     1f
2:
        sub    $0x1000,%ecx
        test   %ecx,(%ecx)
        sub    $0x1000,%eax
        cmp    $0x1000,%eax
        ja     2b
1:
        sub    %eax,%ecx
        test   %ecx,(%ecx)

        lea    4(%esp),%eax     // load pointer to the return address into eax
        mov    %ecx,%esp        // install the new top of stack pointer into esp
        mov    -4(%eax),%ecx    // restore ecx
        push   (%eax)           // push return address onto the stack
        sub    %esp,%eax        // restore the original value in eax
        ret
END_COMPILERRT_FUNCTION(__chkstk)
END_COMPILERRT_FUNCTION(_alloca)

#endif // __i386__
