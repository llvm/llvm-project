/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/**
 * \file
 * \brief ent3f.h macros for building RTE routine names and arg lists
 */

#if defined(_WIN64)
#include <io.h> // for _access, _chmod, _ulink
#include <direct.h> // for _chdir
#endif

#undef DCHAR
#undef DCLEN
#undef CADR
#undef CLEN

#define ENT3F(UC, LC) LC##_
#define ENT3FSU(UC, LC) LC##__
/* macros to declare character arguments */
#define DCHAR(ARG) char *ARG##_adr
#define DCLEN(ARG) , int ARG##_len

#if defined(_WIN64)
#define j0 _j0
#define j1 _j1
#define jn _jn
#define y0 _y0
#define y1 _y1
#define yn _yn
#define access _access
#define chdir _chdir
#define chmod _chmod
#define getpid _getpid
#define putenv _putenv
#define unlink _unlink
#endif

/* macros to access character arguments */
#define CADR(ARG) (ARG##_adr)
#define CLEN(ARG) (ARG##_len)

/* declarations in runtime must match declarations in MS msvcrt.dll
 * to achieve consistent DLL linkage.
 */
#define WIN_CDECL
#define WIN_MSVCRT_IMP extern
