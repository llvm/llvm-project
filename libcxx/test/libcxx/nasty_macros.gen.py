#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

# Test that headers are not tripped up by the surrounding code defining various
# alphabetic macros.

# RUN: %{python} %s %{libcxx}/utils

import sys
sys.path.append(sys.argv[1])
from libcxx.test.header_information import lit_header_restrictions, public_headers

for header in public_headers:
  print(f"""\
//--- {header}.compile.pass.cpp
{lit_header_restrictions.get(header, '')}

#define NASTY_MACRO This should not be expanded!!!

// libc++ does not use single-letter names as a matter of principle.
// But Windows' own <wchar.h>, <math.h>, and <exception> use many of these
// (at least C,E,F,I,M,N,P,S,X,Y,Z) as uglified function parameter names,
// so don't define these on Windows.
//
#ifndef _WIN32
#define _A NASTY_MACRO
#define _B NASTY_MACRO
#define _C NASTY_MACRO
#define _D NASTY_MACRO
#define _E NASTY_MACRO
#define _F NASTY_MACRO
#define _G NASTY_MACRO
#define _H NASTY_MACRO
#define _I NASTY_MACRO
#define _J NASTY_MACRO
#define _K NASTY_MACRO
#define _L NASTY_MACRO
#define _M NASTY_MACRO
#define _N NASTY_MACRO
#define _O NASTY_MACRO
#define _P NASTY_MACRO
#define _Q NASTY_MACRO
#define _R NASTY_MACRO
#define _S NASTY_MACRO
#define _T NASTY_MACRO
#define _U NASTY_MACRO
#define _V NASTY_MACRO
#define _W NASTY_MACRO
#define _X NASTY_MACRO
#define _Y NASTY_MACRO
#define _Z NASTY_MACRO
#endif

// FreeBSD's <sys/types.h> uses _M
//
#ifdef __FreeBSD__
# undef _M
#endif

// Test that libc++ doesn't use names that collide with FreeBSD system macros.
#ifndef __FreeBSD__
#  define __null_sentinel NASTY_MACRO
#  define __generic
#endif

// tchar.h defines these macros on Windows
#ifndef _WIN32
# define _UI   NASTY_MACRO
# define _PUC  NASTY_MACRO
# define _CPUC NASTY_MACRO
# define _PC   NASTY_MACRO
# define _CRPC NASTY_MACRO
# define _CPC  NASTY_MACRO
#endif

// yvals.h on MINGW defines this macro
#ifndef _WIN32
# define _C2 NASTY_MACRO
#endif

// Test that libc++ doesn't use names that collide with Win32 API macros.
// Obviously we can only define these on non-Windows platforms.
#ifndef _WIN32
# define __allocator NASTY_MACRO
# define __bound NASTY_MACRO
# define __deallocate NASTY_MACRO
# define __deref NASTY_MACRO
# define __format_string NASTY_MACRO
# define __full NASTY_MACRO
# define __in NASTY_MACRO
# define __inout NASTY_MACRO
# define __nz NASTY_MACRO
# define __out NASTY_MACRO
# define __part NASTY_MACRO
# define __post NASTY_MACRO
# define __pre NASTY_MACRO
#endif

#define __input NASTY_MACRO
#define __output NASTY_MACRO

#define __acquire NASTY_MACRO
#define __release NASTY_MACRO

// These names are not reserved, so the user can macro-define them.
// These are intended to find improperly _Uglified template parameters.
#define A NASTY_MACRO
#define Arg NASTY_MACRO
#define Args NASTY_MACRO
#define As NASTY_MACRO
#define B NASTY_MACRO
#define Bs NASTY_MACRO
#define C NASTY_MACRO
#define Cp NASTY_MACRO
#define Cs NASTY_MACRO
// Windows setjmp.h contains a struct member named 'D' on ARM/AArch64.
#ifndef _WIN32
# define D NASTY_MACRO
#endif
#define Dp NASTY_MACRO
#define Ds NASTY_MACRO
#define E NASTY_MACRO
#define Ep NASTY_MACRO
#define Es NASTY_MACRO
#define R NASTY_MACRO
#define Rp NASTY_MACRO
#define Rs NASTY_MACRO
#define T NASTY_MACRO
#define Tp NASTY_MACRO
#define Ts NASTY_MACRO
#define Type NASTY_MACRO
#define Types NASTY_MACRO
#define U NASTY_MACRO
#define Up NASTY_MACRO
#define Us NASTY_MACRO
#define V NASTY_MACRO
#define Vp NASTY_MACRO
#define Vs NASTY_MACRO
#define X NASTY_MACRO
#define Xp NASTY_MACRO
#define Xs NASTY_MACRO

// The classic Windows min/max macros
#define min NASTY_MACRO
#define max NASTY_MACRO

// Test to make sure curses has no conflicting macros with the standard library
#define move NASTY_MACRO
#define erase NASTY_MACRO
#define refresh NASTY_MACRO

#include <{header}>
""")
