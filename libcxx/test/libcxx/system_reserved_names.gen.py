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
from libcxx.header_information import lit_header_restrictions, public_headers

for header in public_headers:
  print(f"""\
//--- {header}.compile.pass.cpp
{lit_header_restrictions.get(header, '')}

#define SYSTEM_RESERVED_NAME This name should not be used in libc++

// libc++ does not use single-letter names as a matter of principle.
// But Windows' own <wchar.h>, <math.h>, and <exception> use many of these
// (at least C,E,F,I,M,N,P,S,X,Y,Z) as uglified function parameter names,
// so don't define these on Windows.
//
#ifndef _WIN32
#define _A SYSTEM_RESERVED_NAME
#define _B SYSTEM_RESERVED_NAME
#define _C SYSTEM_RESERVED_NAME
#define _D SYSTEM_RESERVED_NAME
#define _E SYSTEM_RESERVED_NAME
#define _F SYSTEM_RESERVED_NAME
#define _G SYSTEM_RESERVED_NAME
#define _H SYSTEM_RESERVED_NAME
#define _I SYSTEM_RESERVED_NAME
#define _J SYSTEM_RESERVED_NAME
#define _K SYSTEM_RESERVED_NAME
#define _L SYSTEM_RESERVED_NAME
#define _M SYSTEM_RESERVED_NAME
#define _N SYSTEM_RESERVED_NAME
#define _O SYSTEM_RESERVED_NAME
#define _P SYSTEM_RESERVED_NAME
#define _Q SYSTEM_RESERVED_NAME
#define _R SYSTEM_RESERVED_NAME
#define _S SYSTEM_RESERVED_NAME
#define _T SYSTEM_RESERVED_NAME
#define _U SYSTEM_RESERVED_NAME
#define _V SYSTEM_RESERVED_NAME
#define _W SYSTEM_RESERVED_NAME
#define _X SYSTEM_RESERVED_NAME
#define _Y SYSTEM_RESERVED_NAME
#define _Z SYSTEM_RESERVED_NAME
#endif

// FreeBSD's <sys/types.h> uses _M
//
#ifdef __FreeBSD__
# undef _M
#endif

// Test that libc++ doesn't use names that collide with FreeBSD system macros.
#ifndef __FreeBSD__
#  define __null_sentinel SYSTEM_RESERVED_NAME
#  define __generic SYSTEM_RESERVED_NAME
#endif

// tchar.h defines these macros on Windows
#ifndef _WIN32
# define _UI   SYSTEM_RESERVED_NAME
# define _PUC  SYSTEM_RESERVED_NAME
# define _CPUC SYSTEM_RESERVED_NAME
# define _PC   SYSTEM_RESERVED_NAME
# define _CRPC SYSTEM_RESERVED_NAME
# define _CPC  SYSTEM_RESERVED_NAME
#endif

// yvals.h on MINGW defines this macro
#ifndef _WIN32
# define _C2 SYSTEM_RESERVED_NAME
#endif

// Test that libc++ doesn't use names that collide with Win32 API macros.
// Obviously we can only define these on non-Windows platforms.
#ifndef _WIN32
# define __allocator SYSTEM_RESERVED_NAME
# define __bound SYSTEM_RESERVED_NAME
# define __deallocate SYSTEM_RESERVED_NAME
# define __deref SYSTEM_RESERVED_NAME
# define __format_string SYSTEM_RESERVED_NAME
# define __full SYSTEM_RESERVED_NAME
# define __in SYSTEM_RESERVED_NAME
# define __inout SYSTEM_RESERVED_NAME
# define __nz SYSTEM_RESERVED_NAME
# define __out SYSTEM_RESERVED_NAME
# define __part SYSTEM_RESERVED_NAME
# define __post SYSTEM_RESERVED_NAME
# define __pre SYSTEM_RESERVED_NAME
#endif

#define __input SYSTEM_RESERVED_NAME
#define __output SYSTEM_RESERVED_NAME

#define __acquire SYSTEM_RESERVED_NAME
#define __release SYSTEM_RESERVED_NAME

// These names are not reserved, so the user can macro-define them.
// These are intended to find improperly _Uglified template parameters.
#define A SYSTEM_RESERVED_NAME
#define Arg SYSTEM_RESERVED_NAME
#define Args SYSTEM_RESERVED_NAME
#define As SYSTEM_RESERVED_NAME
#define B SYSTEM_RESERVED_NAME
#define Bs SYSTEM_RESERVED_NAME
#define C SYSTEM_RESERVED_NAME
#define Cp SYSTEM_RESERVED_NAME
#define Cs SYSTEM_RESERVED_NAME
// Windows setjmp.h contains a struct member named 'D' on ARM/AArch64.
#ifndef _WIN32
# define D SYSTEM_RESERVED_NAME
#endif
#define Dp SYSTEM_RESERVED_NAME
#define Ds SYSTEM_RESERVED_NAME
#define E SYSTEM_RESERVED_NAME
#define Ep SYSTEM_RESERVED_NAME
#define Es SYSTEM_RESERVED_NAME
#define R SYSTEM_RESERVED_NAME
#define Rp SYSTEM_RESERVED_NAME
#define Rs SYSTEM_RESERVED_NAME
#define T SYSTEM_RESERVED_NAME
#define Tp SYSTEM_RESERVED_NAME
#define Ts SYSTEM_RESERVED_NAME
#define Type SYSTEM_RESERVED_NAME
#define Types SYSTEM_RESERVED_NAME
#define U SYSTEM_RESERVED_NAME
#define Up SYSTEM_RESERVED_NAME
#define Us SYSTEM_RESERVED_NAME
#define V SYSTEM_RESERVED_NAME
#define Vp SYSTEM_RESERVED_NAME
#define Vs SYSTEM_RESERVED_NAME
#define X SYSTEM_RESERVED_NAME
#define Xp SYSTEM_RESERVED_NAME
#define Xs SYSTEM_RESERVED_NAME

// The classic Windows min/max macros
#define min SYSTEM_RESERVED_NAME
#define max SYSTEM_RESERVED_NAME

// Test to make sure curses has no conflicting macros with the standard library
#define move SYSTEM_RESERVED_NAME
#define erase SYSTEM_RESERVED_NAME
#define refresh SYSTEM_RESERVED_NAME

#include <{header}>
""")
