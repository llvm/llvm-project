/*
 * Copyright (c) 2020 Bitdefender
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef ND_CRT_H
#define ND_CRT_H

#include "/home/bernhard/data/entwicklung/2024/llvm-mingw/2024-10-18/llvm-mingw/llvm-project/compiler-rt/lib/interception/bddisasm/inc/bddisasm_types.h"

#ifndef UNREFERENCED_PARAMETER
#define UNREFERENCED_PARAMETER(P)       ((void)(P))
#endif

// By default, an integrator is expected to provide nd_vsnprintf_s and nd_strcat_s.
// bddisasm needs both for NdToText, while bdshemu needs nd_vsnprintf_s for emulation tracing.
// If BDDISASM_NO_FORMAT is defined at compile time these requirements are removed. Instruction formatting will no
// longer be available in bddisasm and emulation tracing will no longer be available in bdshemu.
#ifndef BDDISASM_NO_FORMAT
#include <stdarg.h>

extern int nd_vsnprintf_s(
    char *buffer,
    ND_SIZET sizeOfBuffer,
    ND_SIZET count,
    const char *format,
    va_list argptr
    );

char *
nd_strcat_s(
    char *dst,
    ND_SIZET dst_size,
    const char *src
    );
#endif // !BDDISASM_NO_FORMAT

// Declared here only. Expecting it to be defined in the integrator.
extern void *nd_memset(void *s, int c, ND_SIZET n);

#define nd_memzero(Dest, Size)         nd_memset((Dest), 0, (Size))


// Handy macros.
#define RET_EQ(x, y, z)     if ((x) == (y)) { return (z); }
#define RET_GE(x, y, z)     if ((x) >= (y)) { return (z); }
#define RET_GT(x, y, z)     if ((x) >  (y)) { return (z); }


#endif // ND_CRT_H
