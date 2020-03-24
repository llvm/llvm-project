// Verify that a continue statement in a cilk_for loop can only reach
// the reattach for the loop body, and that the CFG generated for such
// a loop is valid.
//
// RUN: %clang_cc1 %s -std=c99 -triple x86_64-unknown-linux-gnu -O1 -fcilkplus -ftapir=none -verify -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

# 1 "<built-in>"
# 1 "oski.c"
#if 0 /* expanded by -frewrite-includes */
#include <assert.h>
#endif /* expanded by -frewrite-includes */
# 1 "oski.c"
# 1 "/usr/include/assert.h" 1 3 4
/* Copyright (C) 1991-2014 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */

/*
 *	ISO C99 Standard: 7.2 Diagnostics	<assert.h>
 */

#ifdef	_ASSERT_H

# undef	_ASSERT_H
# undef	assert
# undef __ASSERT_VOID_CAST

# ifdef	__USE_GNU
#  undef assert_perror
# endif
# 31 "/usr/include/assert.h" 3 4

#endif /* assert.h	*/
# 33 "/usr/include/assert.h" 3 4

#define	_ASSERT_H	1
#if 0 /* expanded by -frewrite-includes */
#include <features.h>
#endif /* expanded by -frewrite-includes */
# 35 "/usr/include/assert.h" 3 4
# 1 "/usr/include/features.h" 1 3 4
/* Copyright (C) 1991-2014 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */

#ifndef	_FEATURES_H
#define	_FEATURES_H	1

/* These are defined by the user (or the compiler)
   to specify the desired environment:

   __STRICT_ANSI__	ISO Standard C.
   _ISOC99_SOURCE	Extensions to ISO C89 from ISO C99.
   _ISOC11_SOURCE	Extensions to ISO C99 from ISO C11.
   _POSIX_SOURCE	IEEE Std 1003.1.
   _POSIX_C_SOURCE	If ==1, like _POSIX_SOURCE; if >=2 add IEEE Std 1003.2;
			if >=199309L, add IEEE Std 1003.1b-1993;
			if >=199506L, add IEEE Std 1003.1c-1995;
			if >=200112L, all of IEEE 1003.1-2004
			if >=200809L, all of IEEE 1003.1-2008
   _XOPEN_SOURCE	Includes POSIX and XPG things.  Set to 500 if
			Single Unix conformance is wanted, to 600 for the
			sixth revision, to 700 for the seventh revision.
   _XOPEN_SOURCE_EXTENDED XPG things and X/Open Unix extensions.
   _LARGEFILE_SOURCE	Some more functions for correct standard I/O.
   _LARGEFILE64_SOURCE	Additional functionality from LFS for large files.
   _FILE_OFFSET_BITS=N	Select default filesystem interface.
   _BSD_SOURCE		ISO C, POSIX, and 4.3BSD things.
   _SVID_SOURCE		ISO C, POSIX, and SVID things.
   _ATFILE_SOURCE	Additional *at interfaces.
   _GNU_SOURCE		All of the above, plus GNU extensions.
   _DEFAULT_SOURCE	The default set of features (taking precedence over
			__STRICT_ANSI__).
   _REENTRANT		Select additionally reentrant object.
   _THREAD_SAFE		Same as _REENTRANT, often used by other systems.
   _FORTIFY_SOURCE	If set to numeric value > 0 additional security
			measures are defined, according to level.

   The `-ansi' switch to the GNU C compiler, and standards conformance
   options such as `-std=c99', define __STRICT_ANSI__.  If none of
   these are defined, or if _DEFAULT_SOURCE is defined, the default is
   to have _SVID_SOURCE, _BSD_SOURCE, and _POSIX_SOURCE set to one and
   _POSIX_C_SOURCE set to 200809L.  If more than one of these are
   defined, they accumulate.  For example __STRICT_ANSI__,
   _POSIX_SOURCE and _POSIX_C_SOURCE together give you ISO C, 1003.1,
   and 1003.2, but nothing else.

   These are defined by this file and are used by the
   header files to decide what to declare or define:

   __USE_ISOC11		Define ISO C11 things.
   __USE_ISOC99		Define ISO C99 things.
   __USE_ISOC95		Define ISO C90 AMD1 (C95) things.
   __USE_POSIX		Define IEEE Std 1003.1 things.
   __USE_POSIX2		Define IEEE Std 1003.2 things.
   __USE_POSIX199309	Define IEEE Std 1003.1, and .1b things.
   __USE_POSIX199506	Define IEEE Std 1003.1, .1b, .1c and .1i things.
   __USE_XOPEN		Define XPG things.
   __USE_XOPEN_EXTENDED	Define X/Open Unix things.
   __USE_UNIX98		Define Single Unix V2 things.
   __USE_XOPEN2K        Define XPG6 things.
   __USE_XOPEN2KXSI     Define XPG6 XSI things.
   __USE_XOPEN2K8       Define XPG7 things.
   __USE_XOPEN2K8XSI    Define XPG7 XSI things.
   __USE_LARGEFILE	Define correct standard I/O things.
   __USE_LARGEFILE64	Define LFS things with separate names.
   __USE_FILE_OFFSET64	Define 64bit interface as default.
   __USE_BSD		Define 4.3BSD things.
   __USE_SVID		Define SVID things.
   __USE_MISC		Define things common to BSD and System V Unix.
   __USE_ATFILE		Define *at interfaces and AT_* constants for them.
   __USE_GNU		Define GNU extensions.
   __USE_REENTRANT	Define reentrant/thread-safe *_r functions.
   __USE_FORTIFY_LEVEL	Additional security measures used, according to level.

   The macros `__GNU_LIBRARY__', `__GLIBC__', and `__GLIBC_MINOR__' are
   defined by this file unconditionally.  `__GNU_LIBRARY__' is provided
   only for compatibility.  All new code should use the other symbols
   to test for features.

   All macros listed above as possibly being defined by this file are
   explicitly undefined if they are not explicitly defined.
   Feature-test macros that are not defined by the user or compiler
   but are implied by the other feature-test macros defined (or by the
   lack of any definitions) are defined by the file.  */


/* Undefine everything, so we get a clean slate.  */
#undef	__USE_ISOC11
#undef	__USE_ISOC99
#undef	__USE_ISOC95
#undef	__USE_ISOCXX11
#undef	__USE_POSIX
#undef	__USE_POSIX2
#undef	__USE_POSIX199309
#undef	__USE_POSIX199506
#undef	__USE_XOPEN
#undef	__USE_XOPEN_EXTENDED
#undef	__USE_UNIX98
#undef	__USE_XOPEN2K
#undef	__USE_XOPEN2KXSI
#undef	__USE_XOPEN2K8
#undef	__USE_XOPEN2K8XSI
#undef	__USE_LARGEFILE
#undef	__USE_LARGEFILE64
#undef	__USE_FILE_OFFSET64
#undef	__USE_BSD
#undef	__USE_SVID
#undef	__USE_MISC
#undef	__USE_ATFILE
#undef	__USE_GNU
#undef	__USE_REENTRANT
#undef	__USE_FORTIFY_LEVEL
#undef	__KERNEL_STRICT_NAMES

/* Suppress kernel-name space pollution unless user expressedly asks
   for it.  */
#ifndef _LOOSE_KERNEL_NAMES
# define __KERNEL_STRICT_NAMES
#endif
# 133 "/usr/include/features.h" 3 4

/* Convenience macros to test the versions of glibc and gcc.
   Use them like this:
   #if __GNUC_PREREQ (2,8)
   ... code requiring gcc 2.8 or later ...
   #endif
   Note - they won't work for gcc1 or glibc1, since the _MINOR macros
   were not defined then.  */
#if defined __GNUC__ && defined __GNUC_MINOR__
# define __GNUC_PREREQ(maj, min) \
	((__GNUC__ << 16) + __GNUC_MINOR__ >= ((maj) << 16) + (min))
#else
# 145 "/usr/include/features.h" 3 4
# define __GNUC_PREREQ(maj, min) 0
#endif
# 147 "/usr/include/features.h" 3 4


/* If _GNU_SOURCE was defined by the user, turn on all the other features.  */
#ifdef _GNU_SOURCE
# undef  _ISOC95_SOURCE
# define _ISOC95_SOURCE	1
# undef  _ISOC99_SOURCE
# define _ISOC99_SOURCE	1
# undef  _ISOC11_SOURCE
# define _ISOC11_SOURCE	1
# undef  _POSIX_SOURCE
# define _POSIX_SOURCE	1
# undef  _POSIX_C_SOURCE
# define _POSIX_C_SOURCE	200809L
# undef  _XOPEN_SOURCE
# define _XOPEN_SOURCE	700
# undef  _XOPEN_SOURCE_EXTENDED
# define _XOPEN_SOURCE_EXTENDED	1
# undef	 _LARGEFILE64_SOURCE
# define _LARGEFILE64_SOURCE	1
# undef  _DEFAULT_SOURCE
# define _DEFAULT_SOURCE	1
# undef  _BSD_SOURCE
# define _BSD_SOURCE	1
# undef  _SVID_SOURCE
# define _SVID_SOURCE	1
# undef  _ATFILE_SOURCE
# define _ATFILE_SOURCE	1
#endif
# 176 "/usr/include/features.h" 3 4

/* If nothing (other than _GNU_SOURCE and _DEFAULT_SOURCE) is defined,
   define _DEFAULT_SOURCE, _BSD_SOURCE and _SVID_SOURCE.  */
#if (defined _DEFAULT_SOURCE					\
     || (!defined __STRICT_ANSI__				\
	 && !defined _ISOC99_SOURCE				\
	 && !defined _POSIX_SOURCE && !defined _POSIX_C_SOURCE	\
	 && !defined _XOPEN_SOURCE				\
	 && !defined _BSD_SOURCE && !defined _SVID_SOURCE))
# undef  _DEFAULT_SOURCE
# define _DEFAULT_SOURCE	1
# undef  _BSD_SOURCE
# define _BSD_SOURCE	1
# undef  _SVID_SOURCE
# define _SVID_SOURCE	1
#endif
# 192 "/usr/include/features.h" 3 4

/* This is to enable the ISO C11 extension.  */
#if (defined _ISOC11_SOURCE \
     || (defined __STDC_VERSION__ && __STDC_VERSION__ >= 201112L))
# define __USE_ISOC11	1
#endif
# 198 "/usr/include/features.h" 3 4

/* This is to enable the ISO C99 extension.  */
#if (defined _ISOC99_SOURCE || defined _ISOC11_SOURCE \
     || (defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L))
# define __USE_ISOC99	1
#endif
# 204 "/usr/include/features.h" 3 4

/* This is to enable the ISO C90 Amendment 1:1995 extension.  */
#if (defined _ISOC99_SOURCE || defined _ISOC11_SOURCE \
     || (defined __STDC_VERSION__ && __STDC_VERSION__ >= 199409L))
# define __USE_ISOC95	1
#endif
# 210 "/usr/include/features.h" 3 4

/* This is to enable compatibility for ISO C++11.

   So far g++ does not provide a macro.  Check the temporary macro for
   now, too.  */
#if ((defined __cplusplus && __cplusplus >= 201103L)			      \
     || defined __GXX_EXPERIMENTAL_CXX0X__)
# define __USE_ISOCXX11	1
#endif
# 219 "/usr/include/features.h" 3 4

/* If none of the ANSI/POSIX macros are defined, or if _DEFAULT_SOURCE
   is defined, use POSIX.1-2008 (or another version depending on
   _XOPEN_SOURCE).  */
#ifdef _DEFAULT_SOURCE
# if !defined _POSIX_SOURCE && !defined _POSIX_C_SOURCE
#  define __USE_POSIX_IMPLICITLY	1
# endif
# 227 "/usr/include/features.h" 3 4
# undef  _POSIX_SOURCE
# define _POSIX_SOURCE	1
# undef  _POSIX_C_SOURCE
# define _POSIX_C_SOURCE	200809L
#endif
# 232 "/usr/include/features.h" 3 4
#if ((!defined __STRICT_ANSI__ || (_XOPEN_SOURCE - 0) >= 500) && \
     !defined _POSIX_SOURCE && !defined _POSIX_C_SOURCE)
# define _POSIX_SOURCE	1
# if defined _XOPEN_SOURCE && (_XOPEN_SOURCE - 0) < 500
#  define _POSIX_C_SOURCE	2
# elif defined _XOPEN_SOURCE && (_XOPEN_SOURCE - 0) < 600
# 238 "/usr/include/features.h" 3 4
#  define _POSIX_C_SOURCE	199506L
# elif defined _XOPEN_SOURCE && (_XOPEN_SOURCE - 0) < 700
# 240 "/usr/include/features.h" 3 4
#  define _POSIX_C_SOURCE	200112L
# else
# 242 "/usr/include/features.h" 3 4
#  define _POSIX_C_SOURCE	200809L
# endif
# 244 "/usr/include/features.h" 3 4
# define __USE_POSIX_IMPLICITLY	1
#endif
# 246 "/usr/include/features.h" 3 4

#if defined _POSIX_SOURCE || _POSIX_C_SOURCE >= 1 || defined _XOPEN_SOURCE
# define __USE_POSIX	1
#endif
# 250 "/usr/include/features.h" 3 4

#if defined _POSIX_C_SOURCE && _POSIX_C_SOURCE >= 2 || defined _XOPEN_SOURCE
# define __USE_POSIX2	1
#endif
# 254 "/usr/include/features.h" 3 4

#if (_POSIX_C_SOURCE - 0) >= 199309L
# define __USE_POSIX199309	1
#endif
# 258 "/usr/include/features.h" 3 4

#if (_POSIX_C_SOURCE - 0) >= 199506L
# define __USE_POSIX199506	1
#endif
# 262 "/usr/include/features.h" 3 4

#if (_POSIX_C_SOURCE - 0) >= 200112L
# define __USE_XOPEN2K		1
# undef __USE_ISOC95
# define __USE_ISOC95		1
# undef __USE_ISOC99
# define __USE_ISOC99		1
#endif
# 270 "/usr/include/features.h" 3 4

#if (_POSIX_C_SOURCE - 0) >= 200809L
# define __USE_XOPEN2K8		1
# undef  _ATFILE_SOURCE
# define _ATFILE_SOURCE	1
#endif
# 276 "/usr/include/features.h" 3 4

#ifdef	_XOPEN_SOURCE
# define __USE_XOPEN	1
# if (_XOPEN_SOURCE - 0) >= 500
#  define __USE_XOPEN_EXTENDED	1
#  define __USE_UNIX98	1
#  undef _LARGEFILE_SOURCE
#  define _LARGEFILE_SOURCE	1
#  if (_XOPEN_SOURCE - 0) >= 600
#   if (_XOPEN_SOURCE - 0) >= 700
#    define __USE_XOPEN2K8	1
#    define __USE_XOPEN2K8XSI	1
#   endif
# 289 "/usr/include/features.h" 3 4
#   define __USE_XOPEN2K	1
#   define __USE_XOPEN2KXSI	1
#   undef __USE_ISOC95
#   define __USE_ISOC95		1
#   undef __USE_ISOC99
#   define __USE_ISOC99		1
#  endif
# 296 "/usr/include/features.h" 3 4
# else
# 297 "/usr/include/features.h" 3 4
#  ifdef _XOPEN_SOURCE_EXTENDED
#   define __USE_XOPEN_EXTENDED	1
#  endif
# 300 "/usr/include/features.h" 3 4
# endif
# 301 "/usr/include/features.h" 3 4
#endif
# 302 "/usr/include/features.h" 3 4

#ifdef _LARGEFILE_SOURCE
# define __USE_LARGEFILE	1
#endif
# 306 "/usr/include/features.h" 3 4

#ifdef _LARGEFILE64_SOURCE
# define __USE_LARGEFILE64	1
#endif
# 310 "/usr/include/features.h" 3 4

#if defined _FILE_OFFSET_BITS && _FILE_OFFSET_BITS == 64
# define __USE_FILE_OFFSET64	1
#endif
# 314 "/usr/include/features.h" 3 4

#if defined _BSD_SOURCE || defined _SVID_SOURCE
# define __USE_MISC	1
#endif
# 318 "/usr/include/features.h" 3 4

#ifdef	_BSD_SOURCE
# define __USE_BSD	1
#endif
# 322 "/usr/include/features.h" 3 4

#ifdef	_SVID_SOURCE
# define __USE_SVID	1
#endif
# 326 "/usr/include/features.h" 3 4

#ifdef	_ATFILE_SOURCE
# define __USE_ATFILE	1
#endif
# 330 "/usr/include/features.h" 3 4

#ifdef	_GNU_SOURCE
# define __USE_GNU	1
#endif
# 334 "/usr/include/features.h" 3 4

#if defined _REENTRANT || defined _THREAD_SAFE
# define __USE_REENTRANT	1
#endif
# 338 "/usr/include/features.h" 3 4

#if defined _FORTIFY_SOURCE && _FORTIFY_SOURCE > 0 \
    && __GNUC_PREREQ (4, 1) && defined __OPTIMIZE__ && __OPTIMIZE__ > 0
# if _FORTIFY_SOURCE > 1
#  define __USE_FORTIFY_LEVEL 2
# else
# 344 "/usr/include/features.h" 3 4
#  define __USE_FORTIFY_LEVEL 1
# endif
# 346 "/usr/include/features.h" 3 4
#else
# 347 "/usr/include/features.h" 3 4
# define __USE_FORTIFY_LEVEL 0
#endif
# 349 "/usr/include/features.h" 3 4

/* Get definitions of __STDC_* predefined macros, if the compiler has
   not preincluded this header automatically.  */
#if 0 /* expanded by -frewrite-includes */
#include <stdc-predef.h>
#endif /* expanded by -frewrite-includes */
# 352 "/usr/include/features.h" 3 4
# 1 "/usr/include/stdc-predef.h" 1 3 4
/* Copyright (C) 1991-2014 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */

#ifndef	_STDC_PREDEF_H
#define	_STDC_PREDEF_H	1

/* This header is separate from features.h so that the compiler can
   include it implicitly at the start of every compilation.  It must
   not itself include <features.h> or any other header that includes
   <features.h> because the implicit include comes before any feature
   test macros that may be defined in a source file before it first
   explicitly includes a system header.  GCC knows the name of this
   header in order to preinclude it.  */

/* glibc's intent is to support the IEC 559 math functionality, real
   and complex.  If the GCC (4.9 and later) predefined macros
   specifying compiler intent are available, use them to determine
   whether the overall intent is to support these features; otherwise,
   presume an older compiler has intent to support these features and
   define these macros by default.  */

#ifdef __GCC_IEC_559
# if __GCC_IEC_559 > 0
#  define __STDC_IEC_559__		1
# endif
# 40 "/usr/include/stdc-predef.h" 3 4
#else
# 41 "/usr/include/stdc-predef.h" 3 4
# define __STDC_IEC_559__		1
#endif
# 43 "/usr/include/stdc-predef.h" 3 4

#ifdef __GCC_IEC_559_COMPLEX
# if __GCC_IEC_559_COMPLEX > 0
#  define __STDC_IEC_559_COMPLEX__	1
# endif
# 48 "/usr/include/stdc-predef.h" 3 4
#else
# 49 "/usr/include/stdc-predef.h" 3 4
# define __STDC_IEC_559_COMPLEX__	1
#endif
# 51 "/usr/include/stdc-predef.h" 3 4

/* wchar_t uses ISO/IEC 10646 (2nd ed., published 2011-03-15) /
   Unicode 6.0.  */
#define __STDC_ISO_10646__		201103L

/* We do not support C11 <threads.h>.  */
#define __STDC_NO_THREADS__		1

#endif
# 60 "/usr/include/stdc-predef.h" 3 4
# 353 "/usr/include/features.h" 2 3 4

/* This macro indicates that the installed library is the GNU C Library.
   For historic reasons the value now is 6 and this will stay from now
   on.  The use of this variable is deprecated.  Use __GLIBC__ and
   __GLIBC_MINOR__ now (see below) when you want to test for a specific
   GNU C library version and use the values in <gnu/lib-names.h> to get
   the sonames of the shared libraries.  */
#undef  __GNU_LIBRARY__
#define __GNU_LIBRARY__ 6

/* Major and minor version number of the GNU C library package.  Use
   these macros to test for features in specific releases.  */
#define	__GLIBC__	2
#define	__GLIBC_MINOR__	19

#define __GLIBC_PREREQ(maj, min) \
	((__GLIBC__ << 16) + __GLIBC_MINOR__ >= ((maj) << 16) + (min))

/* This is here only because every header file already includes this one.  */
#ifndef __ASSEMBLER__
# ifndef _SYS_CDEFS_H
#if 0 /* expanded by -frewrite-includes */
#  include <sys/cdefs.h>
#endif /* expanded by -frewrite-includes */
# 374 "/usr/include/features.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 1 3 4
/* Copyright (C) 1992-2014 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */

#ifndef	_SYS_CDEFS_H
#define	_SYS_CDEFS_H	1

/* We are almost always included from features.h. */
#ifndef _FEATURES_H
#if 0 /* expanded by -frewrite-includes */
# include <features.h>
#endif /* expanded by -frewrite-includes */
# 23 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
# 24 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
#endif
# 25 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

/* The GNU libc does not support any K&R compilers or the traditional mode
   of ISO C compilers anymore.  Check for some of the combinations not
   anymore supported.  */
#if defined __GNUC__ && !defined __STDC__
# error "You need a ISO C conforming compiler to use the glibc headers"
#endif
# 32 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

/* Some user header file might have defined this before.  */
#undef	__P
#undef	__PMT

#ifdef __GNUC__

/* All functions, except those with callbacks or those that
   synchronize memory, are leaf functions.  */
# if __GNUC_PREREQ (4, 6) && !defined _LIBC
#  define __LEAF , __leaf__
#  define __LEAF_ATTR __attribute__ ((__leaf__))
# else
# 45 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
#  define __LEAF
#  define __LEAF_ATTR
# endif
# 48 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

/* GCC can always grok prototypes.  For C++ programs we add throw()
   to help it optimize the function calls.  But this works only with
   gcc 2.8.x and egcs.  For gcc 3.2 and up we even mark C functions
   as non-throwing using a function attribute since programs can use
   the -fexceptions options for C code as well.  */
# if !defined __cplusplus && __GNUC_PREREQ (3, 3)
#  define __THROW	__attribute__ ((__nothrow__ __LEAF))
#  define __THROWNL	__attribute__ ((__nothrow__))
#  define __NTH(fct)	__attribute__ ((__nothrow__ __LEAF)) fct
# else
# 59 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
#  if defined __cplusplus && __GNUC_PREREQ (2,8)
#   define __THROW	throw ()
#   define __THROWNL	throw ()
#   define __NTH(fct)	__LEAF_ATTR fct throw ()
#  else
# 64 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
#   define __THROW
#   define __THROWNL
#   define __NTH(fct)	fct
#  endif
# 68 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
# endif
# 69 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

#else	/* Not GCC.  */
# 71 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

# define __inline		/* No inline functions.  */

# define __THROW
# define __THROWNL
# define __NTH(fct)	fct

#endif	/* GCC.  */
# 79 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

/* These two macros are not used in glibc anymore.  They are kept here
   only because some other projects expect the macros to be defined.  */
#define __P(args)	args
#define __PMT(args)	args

/* For these things, GCC behaves the ANSI way normally,
   and the non-ANSI way under -traditional.  */

#define __CONCAT(x,y)	x ## y
#define __STRING(x)	#x

/* This is not a typedef so `const __ptr_t' does the right thing.  */
#define __ptr_t void *
#define __long_double_t  long double


/* C++ needs to know that types and declarations are C, not C++.  */
#ifdef	__cplusplus
# define __BEGIN_DECLS	extern "C" {
# define __END_DECLS	}
#else
# 101 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
# define __BEGIN_DECLS
# define __END_DECLS
#endif
# 104 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4


/* The standard library needs the functions from the ISO C90 standard
   in the std namespace.  At the same time we want to be safe for
   future changes and we include the ISO C99 code in the non-standard
   namespace __c99.  The C++ wrapper header take case of adding the
   definitions to the global namespace.  */
#if defined __cplusplus && defined _GLIBCPP_USE_NAMESPACES
# define __BEGIN_NAMESPACE_STD	namespace std {
# define __END_NAMESPACE_STD	}
# define __USING_NAMESPACE_STD(name) using std::name;
# define __BEGIN_NAMESPACE_C99	namespace __c99 {
# define __END_NAMESPACE_C99	}
# define __USING_NAMESPACE_C99(name) using __c99::name;
#else
# 119 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
/* For compatibility we do not add the declarations into any
   namespace.  They will end up in the global namespace which is what
   old code expects.  */
# define __BEGIN_NAMESPACE_STD
# define __END_NAMESPACE_STD
# define __USING_NAMESPACE_STD(name)
# define __BEGIN_NAMESPACE_C99
# define __END_NAMESPACE_C99
# define __USING_NAMESPACE_C99(name)
#endif
# 129 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4


/* Fortify support.  */
#define __bos(ptr) __builtin_object_size (ptr, __USE_FORTIFY_LEVEL > 1)
#define __bos0(ptr) __builtin_object_size (ptr, 0)
#define __fortify_function __extern_always_inline __attribute_artificial__

#if __GNUC_PREREQ (4,3)
# define __warndecl(name, msg) \
  extern void name (void) __attribute__((__warning__ (msg)))
# define __warnattr(msg) __attribute__((__warning__ (msg)))
# define __errordecl(name, msg) \
  extern void name (void) __attribute__((__error__ (msg)))
#else
# 143 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
# define __warndecl(name, msg) extern void name (void)
# define __warnattr(msg)
# define __errordecl(name, msg) extern void name (void)
#endif
# 147 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

/* Support for flexible arrays.  */
#if __GNUC_PREREQ (2,97)
/* GCC 2.97 supports C99 flexible array members.  */
# define __flexarr	[]
#else
# 153 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
# ifdef __GNUC__
#  define __flexarr	[0]
# else
# 156 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
#  if defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
#   define __flexarr	[]
#  else
# 159 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
/* Some other non-C99 compiler.  Approximate with [1].  */
#   define __flexarr	[1]
#  endif
# 162 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
# endif
# 163 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
#endif
# 164 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4


/* __asm__ ("xyz") is used throughout the headers to rename functions
   at the assembly language level.  This is wrapped by the __REDIRECT
   macro, in order to support compilers that can do this some other
   way.  When compilers don't support asm-names at all, we have to do
   preprocessor tricks instead (which don't have exactly the right
   semantics, but it's the best we can do).

   Example:
   int __REDIRECT(setpgrp, (__pid_t pid, __pid_t pgrp), setpgid); */

#if defined __GNUC__ && __GNUC__ >= 2

# define __REDIRECT(name, proto, alias) name proto __asm__ (__ASMNAME (#alias))
# ifdef __cplusplus
#  define __REDIRECT_NTH(name, proto, alias) \
     name proto __THROW __asm__ (__ASMNAME (#alias))
#  define __REDIRECT_NTHNL(name, proto, alias) \
     name proto __THROWNL __asm__ (__ASMNAME (#alias))
# else
# 185 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
#  define __REDIRECT_NTH(name, proto, alias) \
     name proto __asm__ (__ASMNAME (#alias)) __THROW
#  define __REDIRECT_NTHNL(name, proto, alias) \
     name proto __asm__ (__ASMNAME (#alias)) __THROWNL
# endif
# 190 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
# define __ASMNAME(cname)  __ASMNAME2 (__USER_LABEL_PREFIX__, cname)
# define __ASMNAME2(prefix, cname) __STRING (prefix) cname

/*
#elif __SOME_OTHER_COMPILER__

# define __REDIRECT(name, proto, alias) name proto; \
	_Pragma("let " #name " = " #alias)
*/
#endif
# 200 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

/* GCC has various useful declarations that can be made with the
   `__attribute__' syntax.  All of the ways we use this do fine if
   they are omitted for compilers that don't understand it. */
#if !defined __GNUC__ || __GNUC__ < 2
# define __attribute__(xyz)	/* Ignore */
#endif
# 207 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

/* At some point during the gcc 2.96 development the `malloc' attribute
   for functions was introduced.  We don't want to use it unconditionally
   (although this would be possible) since it generates warnings.  */
#if __GNUC_PREREQ (2,96)
# define __attribute_malloc__ __attribute__ ((__malloc__))
#else
# 214 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
# define __attribute_malloc__ /* Ignore */
#endif
# 216 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

/* Tell the compiler which arguments to an allocation function
   indicate the size of the allocation.  */
#if __GNUC_PREREQ (4, 3)
# define __attribute_alloc_size__(params) \
  __attribute__ ((__alloc_size__ params))
#else
# 223 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
# define __attribute_alloc_size__(params) /* Ignore.  */
#endif
# 225 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

/* At some point during the gcc 2.96 development the `pure' attribute
   for functions was introduced.  We don't want to use it unconditionally
   (although this would be possible) since it generates warnings.  */
#if __GNUC_PREREQ (2,96)
# define __attribute_pure__ __attribute__ ((__pure__))
#else
# 232 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
# define __attribute_pure__ /* Ignore */
#endif
# 234 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

/* This declaration tells the compiler that the value is constant.  */
#if __GNUC_PREREQ (2,5)
# define __attribute_const__ __attribute__ ((__const__))
#else
# 239 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
# define __attribute_const__ /* Ignore */
#endif
# 241 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

/* At some point during the gcc 3.1 development the `used' attribute
   for functions was introduced.  We don't want to use it unconditionally
   (although this would be possible) since it generates warnings.  */
#if __GNUC_PREREQ (3,1)
# define __attribute_used__ __attribute__ ((__used__))
# define __attribute_noinline__ __attribute__ ((__noinline__))
#else
# 249 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
# define __attribute_used__ __attribute__ ((__unused__))
# define __attribute_noinline__ /* Ignore */
#endif
# 252 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

/* gcc allows marking deprecated functions.  */
#if __GNUC_PREREQ (3,2)
# define __attribute_deprecated__ __attribute__ ((__deprecated__))
#else
# 257 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
# define __attribute_deprecated__ /* Ignore */
#endif
# 259 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

/* At some point during the gcc 2.8 development the `format_arg' attribute
   for functions was introduced.  We don't want to use it unconditionally
   (although this would be possible) since it generates warnings.
   If several `format_arg' attributes are given for the same function, in
   gcc-3.0 and older, all but the last one are ignored.  In newer gccs,
   all designated arguments are considered.  */
#if __GNUC_PREREQ (2,8)
# define __attribute_format_arg__(x) __attribute__ ((__format_arg__ (x)))
#else
# 269 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
# define __attribute_format_arg__(x) /* Ignore */
#endif
# 271 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

/* At some point during the gcc 2.97 development the `strfmon' format
   attribute for functions was introduced.  We don't want to use it
   unconditionally (although this would be possible) since it
   generates warnings.  */
#if __GNUC_PREREQ (2,97)
# define __attribute_format_strfmon__(a,b) \
  __attribute__ ((__format__ (__strfmon__, a, b)))
#else
# 280 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
# define __attribute_format_strfmon__(a,b) /* Ignore */
#endif
# 282 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

/* The nonull function attribute allows to mark pointer parameters which
   must not be NULL.  */
#if __GNUC_PREREQ (3,3)
# define __nonnull(params) __attribute__ ((__nonnull__ params))
#else
# 288 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
# define __nonnull(params)
#endif
# 290 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

/* If fortification mode, we warn about unused results of certain
   function calls which can lead to problems.  */
#if __GNUC_PREREQ (3,4)
# define __attribute_warn_unused_result__ \
   __attribute__ ((__warn_unused_result__))
# if __USE_FORTIFY_LEVEL > 0
#  define __wur __attribute_warn_unused_result__
# endif
# 299 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
#else
# 300 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
# define __attribute_warn_unused_result__ /* empty */
#endif
# 302 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
#ifndef __wur
# define __wur /* Ignore */
#endif
# 305 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

/* Forces a function to be always inlined.  */
#if __GNUC_PREREQ (3,2)
# define __always_inline __inline __attribute__ ((__always_inline__))
#else
# 310 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
# define __always_inline __inline
#endif
# 312 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

/* Associate error messages with the source location of the call site rather
   than with the source location inside the function.  */
#if __GNUC_PREREQ (4,3)
# define __attribute_artificial__ __attribute__ ((__artificial__))
#else
# 318 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
# define __attribute_artificial__ /* Ignore */
#endif
# 320 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

#ifdef __GNUC__
/* One of these will be defined if the __gnu_inline__ attribute is
   available.  In C++, __GNUC_GNU_INLINE__ will be defined even though
   __inline does not use the GNU inlining rules.  If neither macro is
   defined, this version of GCC only supports GNU inline semantics. */
# if defined __GNUC_STDC_INLINE__ || defined __GNUC_GNU_INLINE__
#  define __extern_inline extern __inline __attribute__ ((__gnu_inline__))
#  define __extern_always_inline \
  extern __always_inline __attribute__ ((__gnu_inline__))
# else
# 331 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
#  define __extern_inline extern __inline
#  define __extern_always_inline extern __always_inline
# endif
# 334 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
#else /* Not GCC.  */
# 335 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
# define __extern_inline  /* Ignore */
# define __extern_always_inline /* Ignore */
#endif
# 338 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

/* GCC 4.3 and above allow passing all anonymous arguments of an
   __extern_always_inline function to some other vararg function.  */
#if __GNUC_PREREQ (4,3)
# define __va_arg_pack() __builtin_va_arg_pack ()
# define __va_arg_pack_len() __builtin_va_arg_pack_len ()
#endif
# 345 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

/* It is possible to compile containing GCC extensions even if GCC is
   run in pedantic mode if the uses are carefully marked using the
   `__extension__' keyword.  But this is not generally available before
   version 2.8.  */
#if !__GNUC_PREREQ (2,8)
# define __extension__		/* Ignore */
#endif
# 353 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

/* __restrict is known in EGCS 1.2 and above. */
#if !__GNUC_PREREQ (2,92)
# define __restrict	/* Ignore */
#endif
# 358 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

/* ISO C99 also allows to declare arrays as non-overlapping.  The syntax is
     array_name[restrict]
   GCC 3.1 supports this.  */
#if __GNUC_PREREQ (3,1) && !defined __GNUG__
# define __restrict_arr	__restrict
#else
# 365 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
# ifdef __GNUC__
#  define __restrict_arr	/* Not supported in old GCC.  */
# else
# 368 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
#  if defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
#   define __restrict_arr	restrict
#  else
# 371 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
/* Some other non-C99 compiler.  */
#   define __restrict_arr	/* Not supported.  */
#  endif
# 374 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
# endif
# 375 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
#endif
# 376 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

#if __GNUC__ >= 3
# define __glibc_unlikely(cond)	__builtin_expect ((cond), 0)
# define __glibc_likely(cond)	__builtin_expect ((cond), 1)
#else
# 381 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
# define __glibc_unlikely(cond)	(cond)
# define __glibc_likely(cond)	(cond)
#endif
# 384 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

#if 0 /* expanded by -frewrite-includes */
#include <bits/wordsize.h>
#endif /* expanded by -frewrite-includes */
# 385 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/wordsize.h" 1 3 4
/* Determine the wordsize from the preprocessor defines.  */

#if defined __x86_64__ && !defined __ILP32__
# define __WORDSIZE	64
#else
# 6 "/usr/include/x86_64-linux-gnu/bits/wordsize.h" 3 4
# define __WORDSIZE	32
#endif
# 8 "/usr/include/x86_64-linux-gnu/bits/wordsize.h" 3 4

#ifdef __x86_64__
# define __WORDSIZE_TIME64_COMPAT32	1
/* Both x86-64 and x32 use the 64-bit system call interface.  */
# define __SYSCALL_WORDSIZE		64
#endif
# 14 "/usr/include/x86_64-linux-gnu/bits/wordsize.h" 3 4
# 386 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 2 3 4

#if defined __LONG_DOUBLE_MATH_OPTIONAL && defined __NO_LONG_DOUBLE_MATH
# define __LDBL_COMPAT 1
# ifdef __REDIRECT
#  define __LDBL_REDIR1(name, proto, alias) __REDIRECT (name, proto, alias)
#  define __LDBL_REDIR(name, proto) \
  __LDBL_REDIR1 (name, proto, __nldbl_##name)
#  define __LDBL_REDIR1_NTH(name, proto, alias) __REDIRECT_NTH (name, proto, alias)
#  define __LDBL_REDIR_NTH(name, proto) \
  __LDBL_REDIR1_NTH (name, proto, __nldbl_##name)
#  define __LDBL_REDIR1_DECL(name, alias) \
  extern __typeof (name) name __asm (__ASMNAME (#alias));
#  define __LDBL_REDIR_DECL(name) \
  extern __typeof (name) name __asm (__ASMNAME ("__nldbl_" #name));
#  define __REDIRECT_LDBL(name, proto, alias) \
  __LDBL_REDIR1 (name, proto, __nldbl_##alias)
#  define __REDIRECT_NTH_LDBL(name, proto, alias) \
  __LDBL_REDIR1_NTH (name, proto, __nldbl_##alias)
# endif
# 405 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
#endif
# 406 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
#if !defined __LDBL_COMPAT || !defined __REDIRECT
# define __LDBL_REDIR1(name, proto, alias) name proto
# define __LDBL_REDIR(name, proto) name proto
# define __LDBL_REDIR1_NTH(name, proto, alias) name proto __THROW
# define __LDBL_REDIR_NTH(name, proto) name proto __THROW
# define __LDBL_REDIR_DECL(name)
# ifdef __REDIRECT
#  define __REDIRECT_LDBL(name, proto, alias) __REDIRECT (name, proto, alias)
#  define __REDIRECT_NTH_LDBL(name, proto, alias) \
  __REDIRECT_NTH (name, proto, alias)
# endif
# 417 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
#endif
# 418 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4

#endif	 /* sys/cdefs.h */
# 420 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
# 375 "/usr/include/features.h" 2 3 4
# endif
# 376 "/usr/include/features.h" 3 4

/* If we don't have __REDIRECT, prototypes will be missing if
   __USE_FILE_OFFSET64 but not __USE_LARGEFILE[64]. */
# if defined __USE_FILE_OFFSET64 && !defined __REDIRECT
#  define __USE_LARGEFILE	1
#  define __USE_LARGEFILE64	1
# endif
# 383 "/usr/include/features.h" 3 4

#endif	/* !ASSEMBLER */
# 385 "/usr/include/features.h" 3 4

/* Decide whether we can define 'extern inline' functions in headers.  */
#if __GNUC_PREREQ (2, 7) && defined __OPTIMIZE__ \
    && !defined __OPTIMIZE_SIZE__ && !defined __NO_INLINE__ \
    && defined __extern_inline
# define __USE_EXTERN_INLINES	1
#endif
# 392 "/usr/include/features.h" 3 4


/* This is here only because every header file already includes this one.
   Get the definitions of all the appropriate `__stub_FUNCTION' symbols.
   <gnu/stubs.h> contains `#define __stub_FUNCTION' when FUNCTION is a stub
   that will always return failure (and set errno to ENOSYS).  */
#if 0 /* expanded by -frewrite-includes */
#include <gnu/stubs.h>
#endif /* expanded by -frewrite-includes */
# 398 "/usr/include/features.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/gnu/stubs.h" 1 3 4
/* This file is automatically generated.
   This file selects the right generated file of `__stub_FUNCTION' macros
   based on the architecture being compiled for.  */


#if !defined __x86_64__
#if 0 /* expanded by -frewrite-includes */
# include <gnu/stubs-32.h>
#endif /* expanded by -frewrite-includes */
# 7 "/usr/include/x86_64-linux-gnu/gnu/stubs.h" 3 4
# 8 "/usr/include/x86_64-linux-gnu/gnu/stubs.h" 3 4
#endif
# 9 "/usr/include/x86_64-linux-gnu/gnu/stubs.h" 3 4
#if defined __x86_64__ && defined __LP64__
#if 0 /* expanded by -frewrite-includes */
# include <gnu/stubs-64.h>
#endif /* expanded by -frewrite-includes */
# 10 "/usr/include/x86_64-linux-gnu/gnu/stubs.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/gnu/stubs-64.h" 1 3 4
/* This file is automatically generated.
   It defines a symbol `__stub_FUNCTION' for each function
   in the C library which is a stub, meaning it will fail
   every time called, usually setting errno to ENOSYS.  */

#ifdef _LIBC
# error Applications may not define the macro _LIBC
#endif
# 9 "/usr/include/x86_64-linux-gnu/gnu/stubs-64.h" 3 4

#define __stub_bdflush
#define __stub_chflags
#define __stub_fattach
#define __stub_fchflags
#define __stub_fdetach
#define __stub_getmsg
#define __stub_gtty
#define __stub_lchmod
#define __stub_putmsg
#define __stub_revoke
#define __stub_setlogin
#define __stub_sigreturn
#define __stub_sstk
#define __stub_stty
# 11 "/usr/include/x86_64-linux-gnu/gnu/stubs.h" 2 3 4
#endif
# 12 "/usr/include/x86_64-linux-gnu/gnu/stubs.h" 3 4
#if defined __x86_64__ && defined __ILP32__
#if 0 /* expanded by -frewrite-includes */
# include <gnu/stubs-x32.h>
#endif /* expanded by -frewrite-includes */
# 13 "/usr/include/x86_64-linux-gnu/gnu/stubs.h" 3 4
# 14 "/usr/include/x86_64-linux-gnu/gnu/stubs.h" 3 4
#endif
# 15 "/usr/include/x86_64-linux-gnu/gnu/stubs.h" 3 4
# 399 "/usr/include/features.h" 2 3 4


#endif	/* features.h  */
# 402 "/usr/include/features.h" 3 4
# 36 "/usr/include/assert.h" 2 3 4

#if defined __cplusplus && __GNUC_PREREQ (2,95)
# define __ASSERT_VOID_CAST static_cast<void>
#else
# 40 "/usr/include/assert.h" 3 4
# define __ASSERT_VOID_CAST (void)
#endif
# 42 "/usr/include/assert.h" 3 4

/* void assert (int expression);

   If NDEBUG is defined, do nothing.
   If not, and EXPRESSION is zero, print an error message and abort.  */

#ifdef	NDEBUG

# define assert(expr)		(__ASSERT_VOID_CAST (0))

/* void assert_perror (int errnum);

   If NDEBUG is defined, do nothing.  If not, and ERRNUM is not zero, print an
   error message with the error text for ERRNUM and abort.
   (This is a GNU extension.) */

# ifdef	__USE_GNU
#  define assert_perror(errnum)	(__ASSERT_VOID_CAST (0))
# endif
# 61 "/usr/include/assert.h" 3 4

#else /* Not NDEBUG.  */
# 63 "/usr/include/assert.h" 3 4

#ifndef _ASSERT_H_DECLS
#define _ASSERT_H_DECLS
__BEGIN_DECLS

/* This prints an "Assertion failed" message and aborts.  */
extern void __assert_fail (const char *__assertion, const char *__file,
			   unsigned int __line, const char *__function)
     __THROW __attribute__ ((__noreturn__));

/* Likewise, but prints the error text for ERRNUM.  */
extern void __assert_perror_fail (int __errnum, const char *__file,
				  unsigned int __line, const char *__function)
     __THROW __attribute__ ((__noreturn__));


/* The following is not at all used here but needed for standard
   compliance.  */
extern void __assert (const char *__assertion, const char *__file, int __line)
     __THROW __attribute__ ((__noreturn__));


__END_DECLS
#endif /* Not _ASSERT_H_DECLS */
# 87 "/usr/include/assert.h" 3 4

# define assert(expr)							\
  ((expr)								\
   ? __ASSERT_VOID_CAST (0)						\
   : __assert_fail (__STRING(expr), __FILE__, __LINE__, __ASSERT_FUNCTION))

# ifdef	__USE_GNU
#  define assert_perror(errnum)						\
  (!(errnum)								\
   ? __ASSERT_VOID_CAST (0)						\
   : __assert_perror_fail ((errnum), __FILE__, __LINE__, __ASSERT_FUNCTION))
# endif
# 99 "/usr/include/assert.h" 3 4

/* Version 2.4 and later of GCC define a magical variable `__PRETTY_FUNCTION__'
   which contains the name of the function currently being defined.
   This is broken in G++ before version 2.6.
   C9x has a similar variable called __func__, but prefer the GCC one since
   it demangles C++ function names.  */
# if defined __cplusplus ? __GNUC_PREREQ (2, 6) : __GNUC_PREREQ (2, 4)
#   define __ASSERT_FUNCTION	__PRETTY_FUNCTION__
# else
# 108 "/usr/include/assert.h" 3 4
#  if defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
#   define __ASSERT_FUNCTION	__func__
#  else
# 111 "/usr/include/assert.h" 3 4
#   define __ASSERT_FUNCTION	((const char *) 0)
#  endif
# 113 "/usr/include/assert.h" 3 4
# endif
# 114 "/usr/include/assert.h" 3 4

#endif /* NDEBUG.  */
# 116 "/usr/include/assert.h" 3 4


#if defined __USE_ISOC11 && !defined __cplusplus
/* Static assertion.  Requires support in the compiler.  */
# undef static_assert
# define static_assert _Static_assert
#endif
# 123 "/usr/include/assert.h" 3 4
# 2 "oski.c" 2
#if 0 /* expanded by -frewrite-includes */
#include <stdio.h>
#endif /* expanded by -frewrite-includes */
# 2 "oski.c"
# 1 "/usr/include/stdio.h" 1 3 4
/* Define ISO C stdio on top of C++ iostreams.
   Copyright (C) 1991-2014 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */

/*
 *	ISO C99 Standard: 7.19 Input/output	<stdio.h>
 */

#ifndef _STDIO_H

#if !defined __need_FILE && !defined __need___FILE
# define _STDIO_H	1
#if 0 /* expanded by -frewrite-includes */
# include <features.h>
#endif /* expanded by -frewrite-includes */
# 27 "/usr/include/stdio.h" 3 4
# 28 "/usr/include/stdio.h" 3 4

__BEGIN_DECLS

# define __need_size_t
# define __need_NULL
#if 0 /* expanded by -frewrite-includes */
# include <stddef.h>
#endif /* expanded by -frewrite-includes */
# 33 "/usr/include/stdio.h" 3 4
# 1 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 1 3 4
/*===---- stddef.h - Basic type definitions --------------------------------===
 *
 * Copyright (c) 2008 Eli Friedman
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *===-----------------------------------------------------------------------===
 */

#if !defined(__STDDEF_H) || defined(__need_ptrdiff_t) ||                       \
    defined(__need_size_t) || defined(__need_wchar_t) ||                       \
    defined(__need_NULL) || defined(__need_wint_t)

#if !defined(__need_ptrdiff_t) && !defined(__need_size_t) &&                   \
    !defined(__need_wchar_t) && !defined(__need_NULL) &&                       \
    !defined(__need_wint_t)
/* Always define miscellaneous pieces when modules are available. */
#if !__has_feature(modules)
#define __STDDEF_H
#endif
# 37 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#define __need_ptrdiff_t
#define __need_size_t
#define __need_wchar_t
#define __need_NULL
#define __need_STDDEF_H_misc
/* __need_wint_t is intentionally not defined here. */
#endif
# 44 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#if defined(__need_ptrdiff_t)
#if !defined(_PTRDIFF_T) || __has_feature(modules)
/* Always define ptrdiff_t when modules are available. */
#if !__has_feature(modules)
#define _PTRDIFF_T
#endif
# 51 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
typedef __PTRDIFF_TYPE__ ptrdiff_t;
#endif
# 53 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#undef __need_ptrdiff_t
#endif /* defined(__need_ptrdiff_t) */
# 55 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#if defined(__need_size_t)
#if !defined(_SIZE_T) || __has_feature(modules)
/* Always define size_t when modules are available. */
#if !__has_feature(modules)
#define _SIZE_T
#endif
# 62 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
typedef __SIZE_TYPE__ size_t;
#endif
# 64 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#undef __need_size_t
#endif /*defined(__need_size_t) */
# 66 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#if defined(__need_STDDEF_H_misc)
/* ISO9899:2011 7.20 (C11 Annex K): Define rsize_t if __STDC_WANT_LIB_EXT1__ is
 * enabled. */
#if (defined(__STDC_WANT_LIB_EXT1__) && __STDC_WANT_LIB_EXT1__ >= 1 && \
     !defined(_RSIZE_T)) || __has_feature(modules)
/* Always define rsize_t when modules are available. */
#if !__has_feature(modules)
#define _RSIZE_T
#endif
# 76 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
typedef __SIZE_TYPE__ rsize_t;
#endif
# 78 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#endif /* defined(__need_STDDEF_H_misc) */
# 79 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#if defined(__need_wchar_t)
#ifndef __cplusplus
/* Always define wchar_t when modules are available. */
#if !defined(_WCHAR_T) || __has_feature(modules)
#if !__has_feature(modules)
#define _WCHAR_T
#if defined(_MSC_EXTENSIONS)
#define _WCHAR_T_DEFINED
#endif
# 89 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#endif
# 90 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
typedef __WCHAR_TYPE__ wchar_t;
#endif
# 92 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#endif
# 93 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#undef __need_wchar_t
#endif /* defined(__need_wchar_t) */
# 95 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#if defined(__need_NULL)
#undef NULL
#ifdef __cplusplus
#  if !defined(__MINGW32__) && !defined(_MSC_VER)
#    define NULL __null
#  else
# 102 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#    define NULL 0
#  endif
# 104 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#else
# 105 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#  define NULL ((void*)0)
#endif
# 107 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#ifdef __cplusplus
#if defined(_MSC_EXTENSIONS) && defined(_NATIVE_NULLPTR_SUPPORTED)
namespace std { typedef decltype(nullptr) nullptr_t; }
using ::std::nullptr_t;
#endif
# 112 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#endif
# 113 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#undef __need_NULL
#endif /* defined(__need_NULL) */
# 115 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#if defined(__need_STDDEF_H_misc)
#if __STDC_VERSION__ >= 201112L || __cplusplus >= 201103L
#if 0 /* expanded by -frewrite-includes */
#include "__stddef_max_align_t.h"
#endif /* expanded by -frewrite-includes */
# 118 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
# 119 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#endif
# 120 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#define offsetof(t, d) __builtin_offsetof(t, d)
#undef __need_STDDEF_H_misc
#endif  /* defined(__need_STDDEF_H_misc) */
# 123 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

/* Some C libraries expect to see a wint_t here. Others (notably MinGW) will use
__WINT_TYPE__ directly; accommodate both by requiring __need_wint_t */
#if defined(__need_wint_t)
/* Always define wint_t when modules are available. */
#if !defined(_WINT_T) || __has_feature(modules)
#if !__has_feature(modules)
#define _WINT_T
#endif
# 132 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
typedef __WINT_TYPE__ wint_t;
#endif
# 134 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#undef __need_wint_t
#endif /* __need_wint_t */
# 136 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#endif
# 138 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
# 34 "/usr/include/stdio.h" 2 3 4

#if 0 /* expanded by -frewrite-includes */
# include <bits/types.h>
#endif /* expanded by -frewrite-includes */
# 35 "/usr/include/stdio.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/types.h" 1 3 4
/* bits/types.h -- definitions of __*_t types underlying *_t types.
   Copyright (C) 2002-2014 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */

/*
 * Never include this file directly; use <sys/types.h> instead.
 */

#ifndef	_BITS_TYPES_H
#define	_BITS_TYPES_H	1

#if 0 /* expanded by -frewrite-includes */
#include <features.h>
#endif /* expanded by -frewrite-includes */
# 26 "/usr/include/x86_64-linux-gnu/bits/types.h" 3 4
# 27 "/usr/include/x86_64-linux-gnu/bits/types.h" 3 4
#if 0 /* expanded by -frewrite-includes */
#include <bits/wordsize.h>
#endif /* expanded by -frewrite-includes */
# 27 "/usr/include/x86_64-linux-gnu/bits/types.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/wordsize.h" 1 3 4
/* Determine the wordsize from the preprocessor defines.  */

#if defined __x86_64__ && !defined __ILP32__
# define __WORDSIZE	64
#else
# 6 "/usr/include/x86_64-linux-gnu/bits/wordsize.h" 3 4
# define __WORDSIZE	32
#endif
# 8 "/usr/include/x86_64-linux-gnu/bits/wordsize.h" 3 4

#ifdef __x86_64__
# define __WORDSIZE_TIME64_COMPAT32	1
/* Both x86-64 and x32 use the 64-bit system call interface.  */
# define __SYSCALL_WORDSIZE		64
#endif
# 14 "/usr/include/x86_64-linux-gnu/bits/wordsize.h" 3 4
# 28 "/usr/include/x86_64-linux-gnu/bits/types.h" 2 3 4

/* Convenience types.  */
typedef unsigned char __u_char;
typedef unsigned short int __u_short;
typedef unsigned int __u_int;
typedef unsigned long int __u_long;

/* Fixed-size types, underlying types depend on word size and compiler.  */
typedef signed char __int8_t;
typedef unsigned char __uint8_t;
typedef signed short int __int16_t;
typedef unsigned short int __uint16_t;
typedef signed int __int32_t;
typedef unsigned int __uint32_t;
#if __WORDSIZE == 64
typedef signed long int __int64_t;
typedef unsigned long int __uint64_t;
#else
# 46 "/usr/include/x86_64-linux-gnu/bits/types.h" 3 4
__extension__ typedef signed long long int __int64_t;
__extension__ typedef unsigned long long int __uint64_t;
#endif
# 49 "/usr/include/x86_64-linux-gnu/bits/types.h" 3 4

/* quad_t is also 64 bits.  */
#if __WORDSIZE == 64
typedef long int __quad_t;
typedef unsigned long int __u_quad_t;
#else
# 55 "/usr/include/x86_64-linux-gnu/bits/types.h" 3 4
__extension__ typedef long long int __quad_t;
__extension__ typedef unsigned long long int __u_quad_t;
#endif
# 58 "/usr/include/x86_64-linux-gnu/bits/types.h" 3 4


/* The machine-dependent file <bits/typesizes.h> defines __*_T_TYPE
   macros for each of the OS types we define below.  The definitions
   of those macros must use the following macros for underlying types.
   We define __S<SIZE>_TYPE and __U<SIZE>_TYPE for the signed and unsigned
   variants of each of the following integer types on this machine.

	16		-- "natural" 16-bit type (always short)
	32		-- "natural" 32-bit type (always int)
	64		-- "natural" 64-bit type (long or long long)
	LONG32		-- 32-bit type, traditionally long
	QUAD		-- 64-bit type, always long long
	WORD		-- natural type of __WORDSIZE bits (int or long)
	LONGWORD	-- type of __WORDSIZE bits, traditionally long

   We distinguish WORD/LONGWORD, 32/LONG32, and 64/QUAD so that the
   conventional uses of `long' or `long long' type modifiers match the
   types we define, even when a less-adorned type would be the same size.
   This matters for (somewhat) portably writing printf/scanf formats for
   these types, where using the appropriate l or ll format modifiers can
   make the typedefs and the formats match up across all GNU platforms.  If
   we used `long' when it's 64 bits where `long long' is expected, then the
   compiler would warn about the formats not matching the argument types,
   and the programmer changing them to shut up the compiler would break the
   program's portability.

   Here we assume what is presently the case in all the GCC configurations
   we support: long long is always 64 bits, long is always word/address size,
   and int is always 32 bits.  */

#define	__S16_TYPE		short int
#define __U16_TYPE		unsigned short int
#define	__S32_TYPE		int
#define __U32_TYPE		unsigned int
#define __SLONGWORD_TYPE	long int
#define __ULONGWORD_TYPE	unsigned long int
#if __WORDSIZE == 32
# define __SQUAD_TYPE		__quad_t
# define __UQUAD_TYPE		__u_quad_t
# define __SWORD_TYPE		int
# define __UWORD_TYPE		unsigned int
# define __SLONG32_TYPE		long int
# define __ULONG32_TYPE		unsigned long int
# define __S64_TYPE		__quad_t
# define __U64_TYPE		__u_quad_t
/* We want __extension__ before typedef's that use nonstandard base types
   such as `long long' in C89 mode.  */
# define __STD_TYPE		__extension__ typedef
#elif __WORDSIZE == 64
# 108 "/usr/include/x86_64-linux-gnu/bits/types.h" 3 4
# define __SQUAD_TYPE		long int
# define __UQUAD_TYPE		unsigned long int
# define __SWORD_TYPE		long int
# define __UWORD_TYPE		unsigned long int
# define __SLONG32_TYPE		int
# define __ULONG32_TYPE		unsigned int
# define __S64_TYPE		long int
# define __U64_TYPE		unsigned long int
/* No need to mark the typedef with __extension__.   */
# define __STD_TYPE		typedef
#else
# 119 "/usr/include/x86_64-linux-gnu/bits/types.h" 3 4
# error
#endif
# 121 "/usr/include/x86_64-linux-gnu/bits/types.h" 3 4
#if 0 /* expanded by -frewrite-includes */
#include <bits/typesizes.h>	/* Defines __*_T_TYPE macros.  */
#endif /* expanded by -frewrite-includes */
# 121 "/usr/include/x86_64-linux-gnu/bits/types.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/typesizes.h" 1 3 4
/* bits/typesizes.h -- underlying types for *_t.  Linux/x86-64 version.
   Copyright (C) 2012-2014 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */

#ifndef _BITS_TYPES_H
# error "Never include <bits/typesizes.h> directly; use <sys/types.h> instead."
#endif
# 22 "/usr/include/x86_64-linux-gnu/bits/typesizes.h" 3 4

#ifndef	_BITS_TYPESIZES_H
#define	_BITS_TYPESIZES_H	1

/* See <bits/types.h> for the meaning of these macros.  This file exists so
   that <bits/types.h> need not vary across different GNU platforms.  */

/* X32 kernel interface is 64-bit.  */
#if defined __x86_64__ && defined __ILP32__
# define __SYSCALL_SLONG_TYPE	__SQUAD_TYPE
# define __SYSCALL_ULONG_TYPE	__UQUAD_TYPE
#else
# 34 "/usr/include/x86_64-linux-gnu/bits/typesizes.h" 3 4
# define __SYSCALL_SLONG_TYPE	__SLONGWORD_TYPE
# define __SYSCALL_ULONG_TYPE	__ULONGWORD_TYPE
#endif
# 37 "/usr/include/x86_64-linux-gnu/bits/typesizes.h" 3 4

#define __DEV_T_TYPE		__UQUAD_TYPE
#define __UID_T_TYPE		__U32_TYPE
#define __GID_T_TYPE		__U32_TYPE
#define __INO_T_TYPE		__SYSCALL_ULONG_TYPE
#define __INO64_T_TYPE		__UQUAD_TYPE
#define __MODE_T_TYPE		__U32_TYPE
#ifdef __x86_64__
# define __NLINK_T_TYPE		__SYSCALL_ULONG_TYPE
# define __FSWORD_T_TYPE	__SYSCALL_SLONG_TYPE
#else
# 48 "/usr/include/x86_64-linux-gnu/bits/typesizes.h" 3 4
# define __NLINK_T_TYPE		__UWORD_TYPE
# define __FSWORD_T_TYPE	__SWORD_TYPE
#endif
# 51 "/usr/include/x86_64-linux-gnu/bits/typesizes.h" 3 4
#define __OFF_T_TYPE		__SYSCALL_SLONG_TYPE
#define __OFF64_T_TYPE		__SQUAD_TYPE
#define __PID_T_TYPE		__S32_TYPE
#define __RLIM_T_TYPE		__SYSCALL_ULONG_TYPE
#define __RLIM64_T_TYPE		__UQUAD_TYPE
#define __BLKCNT_T_TYPE		__SYSCALL_SLONG_TYPE
#define __BLKCNT64_T_TYPE	__SQUAD_TYPE
#define __FSBLKCNT_T_TYPE	__SYSCALL_ULONG_TYPE
#define __FSBLKCNT64_T_TYPE	__UQUAD_TYPE
#define __FSFILCNT_T_TYPE	__SYSCALL_ULONG_TYPE
#define __FSFILCNT64_T_TYPE	__UQUAD_TYPE
#define __ID_T_TYPE		__U32_TYPE
#define __CLOCK_T_TYPE		__SYSCALL_SLONG_TYPE
#define __TIME_T_TYPE		__SYSCALL_SLONG_TYPE
#define __USECONDS_T_TYPE	__U32_TYPE
#define __SUSECONDS_T_TYPE	__SYSCALL_SLONG_TYPE
#define __DADDR_T_TYPE		__S32_TYPE
#define __KEY_T_TYPE		__S32_TYPE
#define __CLOCKID_T_TYPE	__S32_TYPE
#define __TIMER_T_TYPE		void *
#define __BLKSIZE_T_TYPE	__SYSCALL_SLONG_TYPE
#define __FSID_T_TYPE		struct { int __val[2]; }
#define __SSIZE_T_TYPE		__SWORD_TYPE

#ifdef __x86_64__
/* Tell the libc code that off_t and off64_t are actually the same type
   for all ABI purposes, even if possibly expressed as different base types
   for C type-checking purposes.  */
# define __OFF_T_MATCHES_OFF64_T	1

/* Same for ino_t and ino64_t.  */
# define __INO_T_MATCHES_INO64_T	1
#endif
# 84 "/usr/include/x86_64-linux-gnu/bits/typesizes.h" 3 4

/* Number of descriptors that can fit in an `fd_set'.  */
#define __FD_SETSIZE		1024


#endif /* bits/typesizes.h */
# 90 "/usr/include/x86_64-linux-gnu/bits/typesizes.h" 3 4
# 122 "/usr/include/x86_64-linux-gnu/bits/types.h" 2 3 4


__STD_TYPE __DEV_T_TYPE __dev_t;	/* Type of device numbers.  */
__STD_TYPE __UID_T_TYPE __uid_t;	/* Type of user identifications.  */
__STD_TYPE __GID_T_TYPE __gid_t;	/* Type of group identifications.  */
__STD_TYPE __INO_T_TYPE __ino_t;	/* Type of file serial numbers.  */
__STD_TYPE __INO64_T_TYPE __ino64_t;	/* Type of file serial numbers (LFS).*/
__STD_TYPE __MODE_T_TYPE __mode_t;	/* Type of file attribute bitmasks.  */
__STD_TYPE __NLINK_T_TYPE __nlink_t;	/* Type of file link counts.  */
__STD_TYPE __OFF_T_TYPE __off_t;	/* Type of file sizes and offsets.  */
__STD_TYPE __OFF64_T_TYPE __off64_t;	/* Type of file sizes and offsets (LFS).  */
__STD_TYPE __PID_T_TYPE __pid_t;	/* Type of process identifications.  */
__STD_TYPE __FSID_T_TYPE __fsid_t;	/* Type of file system IDs.  */
__STD_TYPE __CLOCK_T_TYPE __clock_t;	/* Type of CPU usage counts.  */
__STD_TYPE __RLIM_T_TYPE __rlim_t;	/* Type for resource measurement.  */
__STD_TYPE __RLIM64_T_TYPE __rlim64_t;	/* Type for resource measurement (LFS).  */
__STD_TYPE __ID_T_TYPE __id_t;		/* General type for IDs.  */
__STD_TYPE __TIME_T_TYPE __time_t;	/* Seconds since the Epoch.  */
__STD_TYPE __USECONDS_T_TYPE __useconds_t; /* Count of microseconds.  */
__STD_TYPE __SUSECONDS_T_TYPE __suseconds_t; /* Signed count of microseconds.  */

__STD_TYPE __DADDR_T_TYPE __daddr_t;	/* The type of a disk address.  */
__STD_TYPE __KEY_T_TYPE __key_t;	/* Type of an IPC key.  */

/* Clock ID used in clock and timer functions.  */
__STD_TYPE __CLOCKID_T_TYPE __clockid_t;

/* Timer ID returned by `timer_create'.  */
__STD_TYPE __TIMER_T_TYPE __timer_t;

/* Type to represent block size.  */
__STD_TYPE __BLKSIZE_T_TYPE __blksize_t;

/* Types from the Large File Support interface.  */

/* Type to count number of disk blocks.  */
__STD_TYPE __BLKCNT_T_TYPE __blkcnt_t;
__STD_TYPE __BLKCNT64_T_TYPE __blkcnt64_t;

/* Type to count file system blocks.  */
__STD_TYPE __FSBLKCNT_T_TYPE __fsblkcnt_t;
__STD_TYPE __FSBLKCNT64_T_TYPE __fsblkcnt64_t;

/* Type to count file system nodes.  */
__STD_TYPE __FSFILCNT_T_TYPE __fsfilcnt_t;
__STD_TYPE __FSFILCNT64_T_TYPE __fsfilcnt64_t;

/* Type of miscellaneous file system fields.  */
__STD_TYPE __FSWORD_T_TYPE __fsword_t;

__STD_TYPE __SSIZE_T_TYPE __ssize_t; /* Type of a byte count, or error.  */

/* Signed long type used in system calls.  */
__STD_TYPE __SYSCALL_SLONG_TYPE __syscall_slong_t;
/* Unsigned long type used in system calls.  */
__STD_TYPE __SYSCALL_ULONG_TYPE __syscall_ulong_t;

/* These few don't really vary by system, they always correspond
   to one of the other defined types.  */
typedef __off64_t __loff_t;	/* Type of file sizes and offsets (LFS).  */
typedef __quad_t *__qaddr_t;
typedef char *__caddr_t;

/* Duplicates info from stdint.h but this is used in unistd.h.  */
__STD_TYPE __SWORD_TYPE __intptr_t;

/* Duplicate info from sys/socket.h.  */
__STD_TYPE __U32_TYPE __socklen_t;


#undef __STD_TYPE

#endif /* bits/types.h */
# 195 "/usr/include/x86_64-linux-gnu/bits/types.h" 3 4
# 36 "/usr/include/stdio.h" 2 3 4
# define __need_FILE
# define __need___FILE
#endif /* Don't need FILE.  */
# 39 "/usr/include/stdio.h" 3 4


#if !defined __FILE_defined && defined __need_FILE

/* Define outside of namespace so the C++ is happy.  */
struct _IO_FILE;

__BEGIN_NAMESPACE_STD
/* The opaque type of streams.  This is the definition used elsewhere.  */
typedef struct _IO_FILE FILE;
__END_NAMESPACE_STD
#if defined __USE_LARGEFILE64 || defined __USE_SVID || defined __USE_POSIX \
    || defined __USE_BSD || defined __USE_ISOC99 || defined __USE_XOPEN \
    || defined __USE_POSIX2
__USING_NAMESPACE_STD(FILE)
#endif
# 55 "/usr/include/stdio.h" 3 4

# define __FILE_defined	1
#endif /* FILE not defined.  */
# 58 "/usr/include/stdio.h" 3 4
#undef	__need_FILE


#if !defined ____FILE_defined && defined __need___FILE

/* The opaque type of streams.  This is the definition used elsewhere.  */
typedef struct _IO_FILE __FILE;

# define ____FILE_defined	1
#endif /* __FILE not defined.  */
# 68 "/usr/include/stdio.h" 3 4
#undef	__need___FILE


#ifdef	_STDIO_H
#define _STDIO_USES_IOSTREAM

#if 0 /* expanded by -frewrite-includes */
#include <libio.h>
#endif /* expanded by -frewrite-includes */
# 74 "/usr/include/stdio.h" 3 4
# 1 "/usr/include/libio.h" 1 3 4
/* Copyright (C) 1991-2014 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written by Per Bothner <bothner@cygnus.com>.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.

   As a special exception, if you link the code in this file with
   files compiled with a GNU compiler to produce an executable,
   that does not cause the resulting executable to be covered by
   the GNU Lesser General Public License.  This exception does not
   however invalidate any other reasons why the executable file
   might be covered by the GNU Lesser General Public License.
   This exception applies to code released by its copyright holders
   in files containing the exception.  */

#ifndef _IO_STDIO_H
#define _IO_STDIO_H

#if 0 /* expanded by -frewrite-includes */
#include <_G_config.h>
#endif /* expanded by -frewrite-includes */
# 31 "/usr/include/libio.h" 3 4
# 1 "/usr/include/_G_config.h" 1 3 4
/* This file is needed by libio to define various configuration parameters.
   These are always the same in the GNU C library.  */

#ifndef _G_config_h
#define _G_config_h 1

/* Define types for libio in terms of the standard internal type names.  */

#if 0 /* expanded by -frewrite-includes */
#include <bits/types.h>
#endif /* expanded by -frewrite-includes */
# 9 "/usr/include/_G_config.h" 3 4
# 10 "/usr/include/_G_config.h" 3 4
#define __need_size_t
#if defined _LIBC || defined _GLIBCPP_USE_WCHAR_T
# define __need_wchar_t
#endif
# 14 "/usr/include/_G_config.h" 3 4
#define __need_NULL
#if 0 /* expanded by -frewrite-includes */
#include <stddef.h>
#endif /* expanded by -frewrite-includes */
# 15 "/usr/include/_G_config.h" 3 4
# 1 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 1 3 4
/*===---- stddef.h - Basic type definitions --------------------------------===
 *
 * Copyright (c) 2008 Eli Friedman
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *===-----------------------------------------------------------------------===
 */

#if !defined(__STDDEF_H) || defined(__need_ptrdiff_t) ||                       \
    defined(__need_size_t) || defined(__need_wchar_t) ||                       \
    defined(__need_NULL) || defined(__need_wint_t)

#if !defined(__need_ptrdiff_t) && !defined(__need_size_t) &&                   \
    !defined(__need_wchar_t) && !defined(__need_NULL) &&                       \
    !defined(__need_wint_t)
/* Always define miscellaneous pieces when modules are available. */
#if !__has_feature(modules)
#define __STDDEF_H
#endif
# 37 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#define __need_ptrdiff_t
#define __need_size_t
#define __need_wchar_t
#define __need_NULL
#define __need_STDDEF_H_misc
/* __need_wint_t is intentionally not defined here. */
#endif
# 44 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#if defined(__need_ptrdiff_t)
#if !defined(_PTRDIFF_T) || __has_feature(modules)
/* Always define ptrdiff_t when modules are available. */
#if !__has_feature(modules)
#define _PTRDIFF_T
#endif
# 51 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
typedef __PTRDIFF_TYPE__ ptrdiff_t;
#endif
# 53 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#undef __need_ptrdiff_t
#endif /* defined(__need_ptrdiff_t) */
# 55 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#if defined(__need_size_t)
#if !defined(_SIZE_T) || __has_feature(modules)
/* Always define size_t when modules are available. */
#if !__has_feature(modules)
#define _SIZE_T
#endif
# 62 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
typedef __SIZE_TYPE__ size_t;
#endif
# 64 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#undef __need_size_t
#endif /*defined(__need_size_t) */
# 66 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#if defined(__need_STDDEF_H_misc)
/* ISO9899:2011 7.20 (C11 Annex K): Define rsize_t if __STDC_WANT_LIB_EXT1__ is
 * enabled. */
#if (defined(__STDC_WANT_LIB_EXT1__) && __STDC_WANT_LIB_EXT1__ >= 1 && \
     !defined(_RSIZE_T)) || __has_feature(modules)
/* Always define rsize_t when modules are available. */
#if !__has_feature(modules)
#define _RSIZE_T
#endif
# 76 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
typedef __SIZE_TYPE__ rsize_t;
#endif
# 78 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#endif /* defined(__need_STDDEF_H_misc) */
# 79 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#if defined(__need_wchar_t)
#ifndef __cplusplus
/* Always define wchar_t when modules are available. */
#if !defined(_WCHAR_T) || __has_feature(modules)
#if !__has_feature(modules)
#define _WCHAR_T
#if defined(_MSC_EXTENSIONS)
#define _WCHAR_T_DEFINED
#endif
# 89 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#endif
# 90 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
typedef __WCHAR_TYPE__ wchar_t;
#endif
# 92 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#endif
# 93 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#undef __need_wchar_t
#endif /* defined(__need_wchar_t) */
# 95 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#if defined(__need_NULL)
#undef NULL
#ifdef __cplusplus
#  if !defined(__MINGW32__) && !defined(_MSC_VER)
#    define NULL __null
#  else
# 102 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#    define NULL 0
#  endif
# 104 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#else
# 105 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#  define NULL ((void*)0)
#endif
# 107 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#ifdef __cplusplus
#if defined(_MSC_EXTENSIONS) && defined(_NATIVE_NULLPTR_SUPPORTED)
namespace std { typedef decltype(nullptr) nullptr_t; }
using ::std::nullptr_t;
#endif
# 112 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#endif
# 113 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#undef __need_NULL
#endif /* defined(__need_NULL) */
# 115 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#if defined(__need_STDDEF_H_misc)
#if __STDC_VERSION__ >= 201112L || __cplusplus >= 201103L
#if 0 /* expanded by -frewrite-includes */
#include "__stddef_max_align_t.h"
#endif /* expanded by -frewrite-includes */
# 118 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
# 119 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#endif
# 120 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#define offsetof(t, d) __builtin_offsetof(t, d)
#undef __need_STDDEF_H_misc
#endif  /* defined(__need_STDDEF_H_misc) */
# 123 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

/* Some C libraries expect to see a wint_t here. Others (notably MinGW) will use
__WINT_TYPE__ directly; accommodate both by requiring __need_wint_t */
#if defined(__need_wint_t)
/* Always define wint_t when modules are available. */
#if !defined(_WINT_T) || __has_feature(modules)
#if !__has_feature(modules)
#define _WINT_T
#endif
# 132 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
typedef __WINT_TYPE__ wint_t;
#endif
# 134 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#undef __need_wint_t
#endif /* __need_wint_t */
# 136 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#endif
# 138 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
# 16 "/usr/include/_G_config.h" 2 3 4
#define __need_mbstate_t
#if defined _LIBC || defined _GLIBCPP_USE_WCHAR_T
# define __need_wint_t
#endif
# 20 "/usr/include/_G_config.h" 3 4
#if 0 /* expanded by -frewrite-includes */
#include <wchar.h>
#endif /* expanded by -frewrite-includes */
# 20 "/usr/include/_G_config.h" 3 4
# 1 "/usr/include/wchar.h" 1 3 4
/* Copyright (C) 1995-2014 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */

/*
 *      ISO C99 Standard: 7.24
 *	Extended multibyte and wide character utilities	<wchar.h>
 */

#ifndef _WCHAR_H

#if !defined __need_mbstate_t && !defined __need_wint_t
# define _WCHAR_H 1
#if 0 /* expanded by -frewrite-includes */
# include <features.h>
#endif /* expanded by -frewrite-includes */
# 27 "/usr/include/wchar.h" 3 4
# 28 "/usr/include/wchar.h" 3 4
#endif
# 29 "/usr/include/wchar.h" 3 4

#ifdef _WCHAR_H
/* Get FILE definition.  */
# define __need___FILE
# if defined __USE_UNIX98 || defined __USE_XOPEN2K
#  define __need_FILE
# endif
# 36 "/usr/include/wchar.h" 3 4
#if 0 /* expanded by -frewrite-includes */
# include <stdio.h>
#endif /* expanded by -frewrite-includes */
# 36 "/usr/include/wchar.h" 3 4
# 37 "/usr/include/wchar.h" 3 4
/* Get va_list definition.  */
# define __need___va_list
#if 0 /* expanded by -frewrite-includes */
# include <stdarg.h>
#endif /* expanded by -frewrite-includes */
# 39 "/usr/include/wchar.h" 3 4
# 40 "/usr/include/wchar.h" 3 4

#if 0 /* expanded by -frewrite-includes */
# include <bits/wchar.h>
#endif /* expanded by -frewrite-includes */
# 41 "/usr/include/wchar.h" 3 4
# 42 "/usr/include/wchar.h" 3 4

/* Get size_t, wchar_t, wint_t and NULL from <stddef.h>.  */
# define __need_size_t
# define __need_wchar_t
# define __need_NULL
#endif
# 48 "/usr/include/wchar.h" 3 4
#if defined _WCHAR_H || defined __need_wint_t || !defined __WINT_TYPE__
# undef __need_wint_t
# define __need_wint_t
#if 0 /* expanded by -frewrite-includes */
# include <stddef.h>
#endif /* expanded by -frewrite-includes */
# 51 "/usr/include/wchar.h" 3 4
# 52 "/usr/include/wchar.h" 3 4

/* We try to get wint_t from <stddef.h>, but not all GCC versions define it
   there.  So define it ourselves if it remains undefined.  */
# ifndef _WINT_T
/* Integral type unchanged by default argument promotions that can
   hold any value corresponding to members of the extended character
   set, as well as at least one value that does not correspond to any
   member of the extended character set.  */
#  define _WINT_T
typedef unsigned int wint_t;
# else
# 63 "/usr/include/wchar.h" 3 4
/* Work around problems with the <stddef.h> file which doesn't put
   wint_t in the std namespace.  */
#  if defined __cplusplus && defined _GLIBCPP_USE_NAMESPACES \
      && defined __WINT_TYPE__
__BEGIN_NAMESPACE_STD
typedef __WINT_TYPE__ wint_t;
__END_NAMESPACE_STD
#  endif
# 71 "/usr/include/wchar.h" 3 4
# endif
# 72 "/usr/include/wchar.h" 3 4

/* Tell the caller that we provide correct C++ prototypes.  */
# if defined __cplusplus && __GNUC_PREREQ (4, 4)
#  define __CORRECT_ISO_CPP_WCHAR_H_PROTO
# endif
# 77 "/usr/include/wchar.h" 3 4
#endif
# 78 "/usr/include/wchar.h" 3 4

#if (defined _WCHAR_H || defined __need_mbstate_t) && !defined ____mbstate_t_defined
# define ____mbstate_t_defined	1
/* Conversion state information.  */
typedef struct
{
  int __count;
  union
  {
# ifdef __WINT_TYPE__
    __WINT_TYPE__ __wch;
# else
# 90 "/usr/include/wchar.h" 3 4
    wint_t __wch;
# endif
# 92 "/usr/include/wchar.h" 3 4
    char __wchb[4];
  } __value;		/* Value so far.  */
} __mbstate_t;
#endif
# 96 "/usr/include/wchar.h" 3 4
#undef __need_mbstate_t


/* The rest of the file is only used if used if __need_mbstate_t is not
   defined.  */
#ifdef _WCHAR_H

# ifndef __mbstate_t_defined
__BEGIN_NAMESPACE_C99
/* Public type.  */
typedef __mbstate_t mbstate_t;
__END_NAMESPACE_C99
#  define __mbstate_t_defined 1
# endif
# 110 "/usr/include/wchar.h" 3 4

#ifdef __USE_GNU
__USING_NAMESPACE_C99(mbstate_t)
#endif
# 114 "/usr/include/wchar.h" 3 4

#ifndef WCHAR_MIN
/* These constants might also be defined in <inttypes.h>.  */
# define WCHAR_MIN __WCHAR_MIN
# define WCHAR_MAX __WCHAR_MAX
#endif
# 120 "/usr/include/wchar.h" 3 4

#ifndef WEOF
# define WEOF (0xffffffffu)
#endif
# 124 "/usr/include/wchar.h" 3 4

/* For XPG4 compliance we have to define the stuff from <wctype.h> here
   as well.  */
#if defined __USE_XOPEN && !defined __USE_UNIX98
#if 0 /* expanded by -frewrite-includes */
# include <wctype.h>
#endif /* expanded by -frewrite-includes */
# 128 "/usr/include/wchar.h" 3 4
# 129 "/usr/include/wchar.h" 3 4
#endif
# 130 "/usr/include/wchar.h" 3 4


__BEGIN_DECLS

__BEGIN_NAMESPACE_STD
/* This incomplete type is defined in <time.h> but needed here because
   of `wcsftime'.  */
struct tm;
__END_NAMESPACE_STD
/* XXX We have to clean this up at some point.  Since tm is in the std
   namespace but wcsftime is in __c99 the type wouldn't be found
   without inserting it in the global namespace.  */
__USING_NAMESPACE_STD(tm)


__BEGIN_NAMESPACE_STD
/* Copy SRC to DEST.  */
extern wchar_t *wcscpy (wchar_t *__restrict __dest,
			const wchar_t *__restrict __src) __THROW;
/* Copy no more than N wide-characters of SRC to DEST.  */
extern wchar_t *wcsncpy (wchar_t *__restrict __dest,
			 const wchar_t *__restrict __src, size_t __n)
     __THROW;

/* Append SRC onto DEST.  */
extern wchar_t *wcscat (wchar_t *__restrict __dest,
			const wchar_t *__restrict __src) __THROW;
/* Append no more than N wide-characters of SRC onto DEST.  */
extern wchar_t *wcsncat (wchar_t *__restrict __dest,
			 const wchar_t *__restrict __src, size_t __n)
     __THROW;

/* Compare S1 and S2.  */
extern int wcscmp (const wchar_t *__s1, const wchar_t *__s2)
     __THROW __attribute_pure__;
/* Compare N wide-characters of S1 and S2.  */
extern int wcsncmp (const wchar_t *__s1, const wchar_t *__s2, size_t __n)
     __THROW __attribute_pure__;
__END_NAMESPACE_STD

#ifdef __USE_XOPEN2K8
/* Compare S1 and S2, ignoring case.  */
extern int wcscasecmp (const wchar_t *__s1, const wchar_t *__s2) __THROW;

/* Compare no more than N chars of S1 and S2, ignoring case.  */
extern int wcsncasecmp (const wchar_t *__s1, const wchar_t *__s2,
			size_t __n) __THROW;

/* Similar to the two functions above but take the information from
   the provided locale and not the global locale.  */
#if 0 /* expanded by -frewrite-includes */
# include <xlocale.h>
#endif /* expanded by -frewrite-includes */
# 180 "/usr/include/wchar.h" 3 4
# 181 "/usr/include/wchar.h" 3 4

extern int wcscasecmp_l (const wchar_t *__s1, const wchar_t *__s2,
			 __locale_t __loc) __THROW;

extern int wcsncasecmp_l (const wchar_t *__s1, const wchar_t *__s2,
			  size_t __n, __locale_t __loc) __THROW;
#endif
# 188 "/usr/include/wchar.h" 3 4

__BEGIN_NAMESPACE_STD
/* Compare S1 and S2, both interpreted as appropriate to the
   LC_COLLATE category of the current locale.  */
extern int wcscoll (const wchar_t *__s1, const wchar_t *__s2) __THROW;
/* Transform S2 into array pointed to by S1 such that if wcscmp is
   applied to two transformed strings the result is the as applying
   `wcscoll' to the original strings.  */
extern size_t wcsxfrm (wchar_t *__restrict __s1,
		       const wchar_t *__restrict __s2, size_t __n) __THROW;
__END_NAMESPACE_STD

#ifdef __USE_XOPEN2K8
/* Similar to the two functions above but take the information from
   the provided locale and not the global locale.  */

/* Compare S1 and S2, both interpreted as appropriate to the
   LC_COLLATE category of the given locale.  */
extern int wcscoll_l (const wchar_t *__s1, const wchar_t *__s2,
		      __locale_t __loc) __THROW;

/* Transform S2 into array pointed to by S1 such that if wcscmp is
   applied to two transformed strings the result is the as applying
   `wcscoll' to the original strings.  */
extern size_t wcsxfrm_l (wchar_t *__s1, const wchar_t *__s2,
			 size_t __n, __locale_t __loc) __THROW;

/* Duplicate S, returning an identical malloc'd string.  */
extern wchar_t *wcsdup (const wchar_t *__s) __THROW __attribute_malloc__;
#endif
# 218 "/usr/include/wchar.h" 3 4

__BEGIN_NAMESPACE_STD
/* Find the first occurrence of WC in WCS.  */
#ifdef __CORRECT_ISO_CPP_WCHAR_H_PROTO
extern "C++" wchar_t *wcschr (wchar_t *__wcs, wchar_t __wc)
     __THROW __asm ("wcschr") __attribute_pure__;
extern "C++" const wchar_t *wcschr (const wchar_t *__wcs, wchar_t __wc)
     __THROW __asm ("wcschr") __attribute_pure__;
#else
# 227 "/usr/include/wchar.h" 3 4
extern wchar_t *wcschr (const wchar_t *__wcs, wchar_t __wc)
     __THROW __attribute_pure__;
#endif
# 230 "/usr/include/wchar.h" 3 4
/* Find the last occurrence of WC in WCS.  */
#ifdef __CORRECT_ISO_CPP_WCHAR_H_PROTO
extern "C++" wchar_t *wcsrchr (wchar_t *__wcs, wchar_t __wc)
     __THROW __asm ("wcsrchr") __attribute_pure__;
extern "C++" const wchar_t *wcsrchr (const wchar_t *__wcs, wchar_t __wc)
     __THROW __asm ("wcsrchr") __attribute_pure__;
#else
# 237 "/usr/include/wchar.h" 3 4
extern wchar_t *wcsrchr (const wchar_t *__wcs, wchar_t __wc)
     __THROW __attribute_pure__;
#endif
# 240 "/usr/include/wchar.h" 3 4
__END_NAMESPACE_STD

#ifdef __USE_GNU
/* This function is similar to `wcschr'.  But it returns a pointer to
   the closing NUL wide character in case C is not found in S.  */
extern wchar_t *wcschrnul (const wchar_t *__s, wchar_t __wc)
     __THROW __attribute_pure__;
#endif
# 248 "/usr/include/wchar.h" 3 4

__BEGIN_NAMESPACE_STD
/* Return the length of the initial segmet of WCS which
   consists entirely of wide characters not in REJECT.  */
extern size_t wcscspn (const wchar_t *__wcs, const wchar_t *__reject)
     __THROW __attribute_pure__;
/* Return the length of the initial segmet of WCS which
   consists entirely of wide characters in  ACCEPT.  */
extern size_t wcsspn (const wchar_t *__wcs, const wchar_t *__accept)
     __THROW __attribute_pure__;
/* Find the first occurrence in WCS of any character in ACCEPT.  */
#ifdef __CORRECT_ISO_CPP_WCHAR_H_PROTO
extern "C++" wchar_t *wcspbrk (wchar_t *__wcs, const wchar_t *__accept)
     __THROW __asm ("wcspbrk") __attribute_pure__;
extern "C++" const wchar_t *wcspbrk (const wchar_t *__wcs,
				     const wchar_t *__accept)
     __THROW __asm ("wcspbrk") __attribute_pure__;
#else
# 266 "/usr/include/wchar.h" 3 4
extern wchar_t *wcspbrk (const wchar_t *__wcs, const wchar_t *__accept)
     __THROW __attribute_pure__;
#endif
# 269 "/usr/include/wchar.h" 3 4
/* Find the first occurrence of NEEDLE in HAYSTACK.  */
#ifdef __CORRECT_ISO_CPP_WCHAR_H_PROTO
extern "C++" wchar_t *wcsstr (wchar_t *__haystack, const wchar_t *__needle)
     __THROW __asm ("wcsstr") __attribute_pure__;
extern "C++" const wchar_t *wcsstr (const wchar_t *__haystack,
				    const wchar_t *__needle)
     __THROW __asm ("wcsstr") __attribute_pure__;
#else
# 277 "/usr/include/wchar.h" 3 4
extern wchar_t *wcsstr (const wchar_t *__haystack, const wchar_t *__needle)
     __THROW __attribute_pure__;
#endif
# 280 "/usr/include/wchar.h" 3 4

/* Divide WCS into tokens separated by characters in DELIM.  */
extern wchar_t *wcstok (wchar_t *__restrict __s,
			const wchar_t *__restrict __delim,
			wchar_t **__restrict __ptr) __THROW;

/* Return the number of wide characters in S.  */
extern size_t wcslen (const wchar_t *__s) __THROW __attribute_pure__;
__END_NAMESPACE_STD

#ifdef __USE_XOPEN
/* Another name for `wcsstr' from XPG4.  */
# ifdef __CORRECT_ISO_CPP_WCHAR_H_PROTO
extern "C++" wchar_t *wcswcs (wchar_t *__haystack, const wchar_t *__needle)
     __THROW __asm ("wcswcs") __attribute_pure__;
extern "C++" const wchar_t *wcswcs (const wchar_t *__haystack,
				    const wchar_t *__needle)
     __THROW __asm ("wcswcs") __attribute_pure__;
# else
# 299 "/usr/include/wchar.h" 3 4
extern wchar_t *wcswcs (const wchar_t *__haystack, const wchar_t *__needle)
     __THROW __attribute_pure__;
# endif
# 302 "/usr/include/wchar.h" 3 4
#endif
# 303 "/usr/include/wchar.h" 3 4

#ifdef __USE_XOPEN2K8
/* Return the number of wide characters in S, but at most MAXLEN.  */
extern size_t wcsnlen (const wchar_t *__s, size_t __maxlen)
     __THROW __attribute_pure__;
#endif
# 309 "/usr/include/wchar.h" 3 4


__BEGIN_NAMESPACE_STD
/* Search N wide characters of S for C.  */
#ifdef __CORRECT_ISO_CPP_WCHAR_H_PROTO
extern "C++" wchar_t *wmemchr (wchar_t *__s, wchar_t __c, size_t __n)
     __THROW __asm ("wmemchr") __attribute_pure__;
extern "C++" const wchar_t *wmemchr (const wchar_t *__s, wchar_t __c,
				     size_t __n)
     __THROW __asm ("wmemchr") __attribute_pure__;
#else
# 320 "/usr/include/wchar.h" 3 4
extern wchar_t *wmemchr (const wchar_t *__s, wchar_t __c, size_t __n)
     __THROW __attribute_pure__;
#endif
# 323 "/usr/include/wchar.h" 3 4

/* Compare N wide characters of S1 and S2.  */
extern int wmemcmp (const wchar_t *__s1, const wchar_t *__s2, size_t __n)
     __THROW __attribute_pure__;

/* Copy N wide characters of SRC to DEST.  */
extern wchar_t *wmemcpy (wchar_t *__restrict __s1,
			 const wchar_t *__restrict __s2, size_t __n) __THROW;

/* Copy N wide characters of SRC to DEST, guaranteeing
   correct behavior for overlapping strings.  */
extern wchar_t *wmemmove (wchar_t *__s1, const wchar_t *__s2, size_t __n)
     __THROW;

/* Set N wide characters of S to C.  */
extern wchar_t *wmemset (wchar_t *__s, wchar_t __c, size_t __n) __THROW;
__END_NAMESPACE_STD

#ifdef __USE_GNU
/* Copy N wide characters of SRC to DEST and return pointer to following
   wide character.  */
extern wchar_t *wmempcpy (wchar_t *__restrict __s1,
			  const wchar_t *__restrict __s2, size_t __n)
     __THROW;
#endif
# 348 "/usr/include/wchar.h" 3 4


__BEGIN_NAMESPACE_STD
/* Determine whether C constitutes a valid (one-byte) multibyte
   character.  */
extern wint_t btowc (int __c) __THROW;

/* Determine whether C corresponds to a member of the extended
   character set whose multibyte representation is a single byte.  */
extern int wctob (wint_t __c) __THROW;

/* Determine whether PS points to an object representing the initial
   state.  */
extern int mbsinit (const mbstate_t *__ps) __THROW __attribute_pure__;

/* Write wide character representation of multibyte character pointed
   to by S to PWC.  */
extern size_t mbrtowc (wchar_t *__restrict __pwc,
		       const char *__restrict __s, size_t __n,
		       mbstate_t *__restrict __p) __THROW;

/* Write multibyte representation of wide character WC to S.  */
extern size_t wcrtomb (char *__restrict __s, wchar_t __wc,
		       mbstate_t *__restrict __ps) __THROW;

/* Return number of bytes in multibyte character pointed to by S.  */
extern size_t __mbrlen (const char *__restrict __s, size_t __n,
			mbstate_t *__restrict __ps) __THROW;
extern size_t mbrlen (const char *__restrict __s, size_t __n,
		      mbstate_t *__restrict __ps) __THROW;
__END_NAMESPACE_STD

#ifdef __USE_EXTERN_INLINES
/* Define inline function as optimization.  */

/* We can use the BTOWC and WCTOB optimizations since we know that all
   locales must use ASCII encoding for the values in the ASCII range
   and because the wchar_t encoding is always ISO 10646.  */
extern wint_t __btowc_alias (int __c) __asm ("btowc");
__extern_inline wint_t
__NTH (btowc (int __c))
{ return (__builtin_constant_p (__c) && __c >= '\0' && __c <= '\x7f'
	  ? (wint_t) __c : __btowc_alias (__c)); }

extern int __wctob_alias (wint_t __c) __asm ("wctob");
__extern_inline int
__NTH (wctob (wint_t __wc))
{ return (__builtin_constant_p (__wc) && __wc >= L'\0' && __wc <= L'\x7f'
	  ? (int) __wc : __wctob_alias (__wc)); }

__extern_inline size_t
__NTH (mbrlen (const char *__restrict __s, size_t __n,
	       mbstate_t *__restrict __ps))
{ return (__ps != NULL
	  ? mbrtowc (NULL, __s, __n, __ps) : __mbrlen (__s, __n, NULL)); }
#endif
# 404 "/usr/include/wchar.h" 3 4

__BEGIN_NAMESPACE_STD
/* Write wide character representation of multibyte character string
   SRC to DST.  */
extern size_t mbsrtowcs (wchar_t *__restrict __dst,
			 const char **__restrict __src, size_t __len,
			 mbstate_t *__restrict __ps) __THROW;

/* Write multibyte character representation of wide character string
   SRC to DST.  */
extern size_t wcsrtombs (char *__restrict __dst,
			 const wchar_t **__restrict __src, size_t __len,
			 mbstate_t *__restrict __ps) __THROW;
__END_NAMESPACE_STD


#ifdef	__USE_XOPEN2K8
/* Write wide character representation of at most NMC bytes of the
   multibyte character string SRC to DST.  */
extern size_t mbsnrtowcs (wchar_t *__restrict __dst,
			  const char **__restrict __src, size_t __nmc,
			  size_t __len, mbstate_t *__restrict __ps) __THROW;

/* Write multibyte character representation of at most NWC characters
   from the wide character string SRC to DST.  */
extern size_t wcsnrtombs (char *__restrict __dst,
			  const wchar_t **__restrict __src,
			  size_t __nwc, size_t __len,
			  mbstate_t *__restrict __ps) __THROW;
#endif	/* use POSIX 2008 */
# 434 "/usr/include/wchar.h" 3 4


/* The following functions are extensions found in X/Open CAE.  */
#ifdef __USE_XOPEN
/* Determine number of column positions required for C.  */
extern int wcwidth (wchar_t __c) __THROW;

/* Determine number of column positions required for first N wide
   characters (or fewer if S ends before this) in S.  */
extern int wcswidth (const wchar_t *__s, size_t __n) __THROW;
#endif	/* Use X/Open.  */
# 445 "/usr/include/wchar.h" 3 4


__BEGIN_NAMESPACE_STD
/* Convert initial portion of the wide string NPTR to `double'
   representation.  */
extern double wcstod (const wchar_t *__restrict __nptr,
		      wchar_t **__restrict __endptr) __THROW;
__END_NAMESPACE_STD

#ifdef __USE_ISOC99
__BEGIN_NAMESPACE_C99
/* Likewise for `float' and `long double' sizes of floating-point numbers.  */
extern float wcstof (const wchar_t *__restrict __nptr,
		     wchar_t **__restrict __endptr) __THROW;
extern long double wcstold (const wchar_t *__restrict __nptr,
			    wchar_t **__restrict __endptr) __THROW;
__END_NAMESPACE_C99
#endif /* C99 */
# 463 "/usr/include/wchar.h" 3 4


__BEGIN_NAMESPACE_STD
/* Convert initial portion of wide string NPTR to `long int'
   representation.  */
extern long int wcstol (const wchar_t *__restrict __nptr,
			wchar_t **__restrict __endptr, int __base) __THROW;

/* Convert initial portion of wide string NPTR to `unsigned long int'
   representation.  */
extern unsigned long int wcstoul (const wchar_t *__restrict __nptr,
				  wchar_t **__restrict __endptr, int __base)
     __THROW;
__END_NAMESPACE_STD

#ifdef __USE_ISOC99
__BEGIN_NAMESPACE_C99
/* Convert initial portion of wide string NPTR to `long long int'
   representation.  */
__extension__
extern long long int wcstoll (const wchar_t *__restrict __nptr,
			      wchar_t **__restrict __endptr, int __base)
     __THROW;

/* Convert initial portion of wide string NPTR to `unsigned long long int'
   representation.  */
__extension__
extern unsigned long long int wcstoull (const wchar_t *__restrict __nptr,
					wchar_t **__restrict __endptr,
					int __base) __THROW;
__END_NAMESPACE_C99
#endif /* ISO C99.  */
# 495 "/usr/include/wchar.h" 3 4

#ifdef __USE_GNU
/* Convert initial portion of wide string NPTR to `long long int'
   representation.  */
__extension__
extern long long int wcstoq (const wchar_t *__restrict __nptr,
			     wchar_t **__restrict __endptr, int __base)
     __THROW;

/* Convert initial portion of wide string NPTR to `unsigned long long int'
   representation.  */
__extension__
extern unsigned long long int wcstouq (const wchar_t *__restrict __nptr,
				       wchar_t **__restrict __endptr,
				       int __base) __THROW;
#endif /* Use GNU.  */
# 511 "/usr/include/wchar.h" 3 4

#ifdef __USE_GNU
/* The concept of one static locale per category is not very well
   thought out.  Many applications will need to process its data using
   information from several different locales.  Another application is
   the implementation of the internationalization handling in the
   upcoming ISO C++ standard library.  To support this another set of
   the functions using locale data exist which have an additional
   argument.

   Attention: all these functions are *not* standardized in any form.
   This is a proof-of-concept implementation.  */

/* Structure for reentrant locale using functions.  This is an
   (almost) opaque type for the user level programs.  */
#if 0 /* expanded by -frewrite-includes */
# include <xlocale.h>
#endif /* expanded by -frewrite-includes */
# 526 "/usr/include/wchar.h" 3 4
# 527 "/usr/include/wchar.h" 3 4

/* Special versions of the functions above which take the locale to
   use as an additional parameter.  */
extern long int wcstol_l (const wchar_t *__restrict __nptr,
			  wchar_t **__restrict __endptr, int __base,
			  __locale_t __loc) __THROW;

extern unsigned long int wcstoul_l (const wchar_t *__restrict __nptr,
				    wchar_t **__restrict __endptr,
				    int __base, __locale_t __loc) __THROW;

__extension__
extern long long int wcstoll_l (const wchar_t *__restrict __nptr,
				wchar_t **__restrict __endptr,
				int __base, __locale_t __loc) __THROW;

__extension__
extern unsigned long long int wcstoull_l (const wchar_t *__restrict __nptr,
					  wchar_t **__restrict __endptr,
					  int __base, __locale_t __loc)
     __THROW;

extern double wcstod_l (const wchar_t *__restrict __nptr,
			wchar_t **__restrict __endptr, __locale_t __loc)
     __THROW;

extern float wcstof_l (const wchar_t *__restrict __nptr,
		       wchar_t **__restrict __endptr, __locale_t __loc)
     __THROW;

extern long double wcstold_l (const wchar_t *__restrict __nptr,
			      wchar_t **__restrict __endptr,
			      __locale_t __loc) __THROW;
#endif	/* use GNU */
# 561 "/usr/include/wchar.h" 3 4


#ifdef __USE_XOPEN2K8
/* Copy SRC to DEST, returning the address of the terminating L'\0' in
   DEST.  */
extern wchar_t *wcpcpy (wchar_t *__restrict __dest,
			const wchar_t *__restrict __src) __THROW;

/* Copy no more than N characters of SRC to DEST, returning the address of
   the last character written into DEST.  */
extern wchar_t *wcpncpy (wchar_t *__restrict __dest,
			 const wchar_t *__restrict __src, size_t __n)
     __THROW;


/* Wide character I/O functions.  */

/* Like OPEN_MEMSTREAM, but the stream is wide oriented and produces
   a wide character string.  */
extern __FILE *open_wmemstream (wchar_t **__bufloc, size_t *__sizeloc) __THROW;
#endif
# 582 "/usr/include/wchar.h" 3 4

#if defined __USE_ISOC95 || defined __USE_UNIX98
__BEGIN_NAMESPACE_STD

/* Select orientation for stream.  */
extern int fwide (__FILE *__fp, int __mode) __THROW;


/* Write formatted output to STREAM.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int fwprintf (__FILE *__restrict __stream,
		     const wchar_t *__restrict __format, ...)
     /* __attribute__ ((__format__ (__wprintf__, 2, 3))) */;
/* Write formatted output to stdout.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int wprintf (const wchar_t *__restrict __format, ...)
     /* __attribute__ ((__format__ (__wprintf__, 1, 2))) */;
/* Write formatted output of at most N characters to S.  */
extern int swprintf (wchar_t *__restrict __s, size_t __n,
		     const wchar_t *__restrict __format, ...)
     __THROW /* __attribute__ ((__format__ (__wprintf__, 3, 4))) */;

/* Write formatted output to S from argument list ARG.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int vfwprintf (__FILE *__restrict __s,
		      const wchar_t *__restrict __format,
		      __gnuc_va_list __arg)
     /* __attribute__ ((__format__ (__wprintf__, 2, 0))) */;
/* Write formatted output to stdout from argument list ARG.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int vwprintf (const wchar_t *__restrict __format,
		     __gnuc_va_list __arg)
     /* __attribute__ ((__format__ (__wprintf__, 1, 0))) */;
/* Write formatted output of at most N character to S from argument
   list ARG.  */
extern int vswprintf (wchar_t *__restrict __s, size_t __n,
		      const wchar_t *__restrict __format,
		      __gnuc_va_list __arg)
     __THROW /* __attribute__ ((__format__ (__wprintf__, 3, 0))) */;


/* Read formatted input from STREAM.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int fwscanf (__FILE *__restrict __stream,
		    const wchar_t *__restrict __format, ...)
     /* __attribute__ ((__format__ (__wscanf__, 2, 3))) */;
/* Read formatted input from stdin.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int wscanf (const wchar_t *__restrict __format, ...)
     /* __attribute__ ((__format__ (__wscanf__, 1, 2))) */;
/* Read formatted input from S.  */
extern int swscanf (const wchar_t *__restrict __s,
		    const wchar_t *__restrict __format, ...)
     __THROW /* __attribute__ ((__format__ (__wscanf__, 2, 3))) */;

# if defined __USE_ISOC99 && !defined __USE_GNU \
     && (!defined __LDBL_COMPAT || !defined __REDIRECT) \
     && (defined __STRICT_ANSI__ || defined __USE_XOPEN2K)
#  ifdef __REDIRECT
/* For strict ISO C99 or POSIX compliance disallow %as, %aS and %a[
   GNU extension which conflicts with valid %a followed by letter
   s, S or [.  */
extern int __REDIRECT (fwscanf, (__FILE *__restrict __stream,
				 const wchar_t *__restrict __format, ...),
		       __isoc99_fwscanf)
     /* __attribute__ ((__format__ (__wscanf__, 2, 3))) */;
extern int __REDIRECT (wscanf, (const wchar_t *__restrict __format, ...),
		       __isoc99_wscanf)
     /* __attribute__ ((__format__ (__wscanf__, 1, 2))) */;
extern int __REDIRECT_NTH (swscanf, (const wchar_t *__restrict __s,
				     const wchar_t *__restrict __format,
				     ...), __isoc99_swscanf)
     /* __attribute__ ((__format__ (__wscanf__, 2, 3))) */;
#  else
# 668 "/usr/include/wchar.h" 3 4
extern int __isoc99_fwscanf (__FILE *__restrict __stream,
			     const wchar_t *__restrict __format, ...);
extern int __isoc99_wscanf (const wchar_t *__restrict __format, ...);
extern int __isoc99_swscanf (const wchar_t *__restrict __s,
			     const wchar_t *__restrict __format, ...)
     __THROW;
#   define fwscanf __isoc99_fwscanf
#   define wscanf __isoc99_wscanf
#   define swscanf __isoc99_swscanf
#  endif
# 678 "/usr/include/wchar.h" 3 4
# endif
# 679 "/usr/include/wchar.h" 3 4

__END_NAMESPACE_STD
#endif /* Use ISO C95, C99 and Unix98. */
# 682 "/usr/include/wchar.h" 3 4

#ifdef __USE_ISOC99
__BEGIN_NAMESPACE_C99
/* Read formatted input from S into argument list ARG.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int vfwscanf (__FILE *__restrict __s,
		     const wchar_t *__restrict __format,
		     __gnuc_va_list __arg)
     /* __attribute__ ((__format__ (__wscanf__, 2, 0))) */;
/* Read formatted input from stdin into argument list ARG.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int vwscanf (const wchar_t *__restrict __format,
		    __gnuc_va_list __arg)
     /* __attribute__ ((__format__ (__wscanf__, 1, 0))) */;
/* Read formatted input from S into argument list ARG.  */
extern int vswscanf (const wchar_t *__restrict __s,
		     const wchar_t *__restrict __format,
		     __gnuc_va_list __arg)
     __THROW /* __attribute__ ((__format__ (__wscanf__, 2, 0))) */;

# if !defined __USE_GNU \
     && (!defined __LDBL_COMPAT || !defined __REDIRECT) \
     && (defined __STRICT_ANSI__ || defined __USE_XOPEN2K)
#  ifdef __REDIRECT
extern int __REDIRECT (vfwscanf, (__FILE *__restrict __s,
				  const wchar_t *__restrict __format,
				  __gnuc_va_list __arg), __isoc99_vfwscanf)
     /* __attribute__ ((__format__ (__wscanf__, 2, 0))) */;
extern int __REDIRECT (vwscanf, (const wchar_t *__restrict __format,
				 __gnuc_va_list __arg), __isoc99_vwscanf)
     /* __attribute__ ((__format__ (__wscanf__, 1, 0))) */;
extern int __REDIRECT_NTH (vswscanf, (const wchar_t *__restrict __s,
				      const wchar_t *__restrict __format,
				      __gnuc_va_list __arg), __isoc99_vswscanf)
     /* __attribute__ ((__format__ (__wscanf__, 2, 0))) */;
#  else
# 722 "/usr/include/wchar.h" 3 4
extern int __isoc99_vfwscanf (__FILE *__restrict __s,
			      const wchar_t *__restrict __format,
			      __gnuc_va_list __arg);
extern int __isoc99_vwscanf (const wchar_t *__restrict __format,
			     __gnuc_va_list __arg);
extern int __isoc99_vswscanf (const wchar_t *__restrict __s,
			      const wchar_t *__restrict __format,
			      __gnuc_va_list __arg) __THROW;
#   define vfwscanf __isoc99_vfwscanf
#   define vwscanf __isoc99_vwscanf
#   define vswscanf __isoc99_vswscanf
#  endif
# 734 "/usr/include/wchar.h" 3 4
# endif
# 735 "/usr/include/wchar.h" 3 4

__END_NAMESPACE_C99
#endif /* Use ISO C99. */
# 738 "/usr/include/wchar.h" 3 4


__BEGIN_NAMESPACE_STD
/* Read a character from STREAM.

   These functions are possible cancellation points and therefore not
   marked with __THROW.  */
extern wint_t fgetwc (__FILE *__stream);
extern wint_t getwc (__FILE *__stream);

/* Read a character from stdin.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern wint_t getwchar (void);


/* Write a character to STREAM.

   These functions are possible cancellation points and therefore not
   marked with __THROW.  */
extern wint_t fputwc (wchar_t __wc, __FILE *__stream);
extern wint_t putwc (wchar_t __wc, __FILE *__stream);

/* Write a character to stdout.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern wint_t putwchar (wchar_t __wc);


/* Get a newline-terminated wide character string of finite length
   from STREAM.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern wchar_t *fgetws (wchar_t *__restrict __ws, int __n,
			__FILE *__restrict __stream);

/* Write a string to STREAM.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int fputws (const wchar_t *__restrict __ws,
		   __FILE *__restrict __stream);


/* Push a character back onto the input buffer of STREAM.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern wint_t ungetwc (wint_t __wc, __FILE *__stream);
__END_NAMESPACE_STD


#ifdef __USE_GNU
/* These are defined to be equivalent to the `char' functions defined
   in POSIX.1:1996.

   These functions are not part of POSIX and therefore no official
   cancellation point.  But due to similarity with an POSIX interface
   or due to the implementation they are cancellation points and
   therefore not marked with __THROW.  */
extern wint_t getwc_unlocked (__FILE *__stream);
extern wint_t getwchar_unlocked (void);

/* This is the wide character version of a GNU extension.

   This function is not part of POSIX and therefore no official
   cancellation point.  But due to similarity with an POSIX interface
   or due to the implementation it is a cancellation point and
   therefore not marked with __THROW.  */
extern wint_t fgetwc_unlocked (__FILE *__stream);

/* Faster version when locking is not necessary.

   This function is not part of POSIX and therefore no official
   cancellation point.  But due to similarity with an POSIX interface
   or due to the implementation it is a cancellation point and
   therefore not marked with __THROW.  */
extern wint_t fputwc_unlocked (wchar_t __wc, __FILE *__stream);

/* These are defined to be equivalent to the `char' functions defined
   in POSIX.1:1996.

   These functions are not part of POSIX and therefore no official
   cancellation point.  But due to similarity with an POSIX interface
   or due to the implementation they are cancellation points and
   therefore not marked with __THROW.  */
extern wint_t putwc_unlocked (wchar_t __wc, __FILE *__stream);
extern wint_t putwchar_unlocked (wchar_t __wc);


/* This function does the same as `fgetws' but does not lock the stream.

   This function is not part of POSIX and therefore no official
   cancellation point.  But due to similarity with an POSIX interface
   or due to the implementation it is a cancellation point and
   therefore not marked with __THROW.  */
extern wchar_t *fgetws_unlocked (wchar_t *__restrict __ws, int __n,
				 __FILE *__restrict __stream);

/* This function does the same as `fputws' but does not lock the stream.

   This function is not part of POSIX and therefore no official
   cancellation point.  But due to similarity with an POSIX interface
   or due to the implementation it is a cancellation point and
   therefore not marked with __THROW.  */
extern int fputws_unlocked (const wchar_t *__restrict __ws,
			    __FILE *__restrict __stream);
#endif
# 849 "/usr/include/wchar.h" 3 4


__BEGIN_NAMESPACE_C99
/* Format TP into S according to FORMAT.
   Write no more than MAXSIZE wide characters and return the number
   of wide characters written, or 0 if it would exceed MAXSIZE.  */
extern size_t wcsftime (wchar_t *__restrict __s, size_t __maxsize,
			const wchar_t *__restrict __format,
			const struct tm *__restrict __tp) __THROW;
__END_NAMESPACE_C99

# ifdef __USE_GNU
#if 0 /* expanded by -frewrite-includes */
# include <xlocale.h>
#endif /* expanded by -frewrite-includes */
# 861 "/usr/include/wchar.h" 3 4
# 862 "/usr/include/wchar.h" 3 4

/* Similar to `wcsftime' but takes the information from
   the provided locale and not the global locale.  */
extern size_t wcsftime_l (wchar_t *__restrict __s, size_t __maxsize,
			  const wchar_t *__restrict __format,
			  const struct tm *__restrict __tp,
			  __locale_t __loc) __THROW;
# endif
# 870 "/usr/include/wchar.h" 3 4

/* The X/Open standard demands that most of the functions defined in
   the <wctype.h> header must also appear here.  This is probably
   because some X/Open members wrote their implementation before the
   ISO C standard was published and introduced the better solution.
   We have to provide these definitions for compliance reasons but we
   do this nonsense only if really necessary.  */
#if defined __USE_UNIX98 && !defined __USE_GNU
# define __need_iswxxx
#if 0 /* expanded by -frewrite-includes */
# include <wctype.h>
#endif /* expanded by -frewrite-includes */
# 879 "/usr/include/wchar.h" 3 4
# 880 "/usr/include/wchar.h" 3 4
#endif
# 881 "/usr/include/wchar.h" 3 4

/* Define some macros helping to catch buffer overflows.  */
#if __USE_FORTIFY_LEVEL > 0 && defined __fortify_function
#if 0 /* expanded by -frewrite-includes */
# include <bits/wchar2.h>
#endif /* expanded by -frewrite-includes */
# 884 "/usr/include/wchar.h" 3 4
# 885 "/usr/include/wchar.h" 3 4
#endif
# 886 "/usr/include/wchar.h" 3 4

#ifdef __LDBL_COMPAT
#if 0 /* expanded by -frewrite-includes */
# include <bits/wchar-ldbl.h>
#endif /* expanded by -frewrite-includes */
# 888 "/usr/include/wchar.h" 3 4
# 889 "/usr/include/wchar.h" 3 4
#endif
# 890 "/usr/include/wchar.h" 3 4

__END_DECLS

#endif	/* _WCHAR_H defined */
# 894 "/usr/include/wchar.h" 3 4

#endif /* wchar.h  */
# 896 "/usr/include/wchar.h" 3 4

/* Undefine all __need_* constants in case we are included to get those
   constants but the whole file was already read.  */
#undef __need_mbstate_t
#undef __need_wint_t
# 21 "/usr/include/_G_config.h" 2 3 4
typedef struct
{
  __off_t __pos;
  __mbstate_t __state;
} _G_fpos_t;
typedef struct
{
  __off64_t __pos;
  __mbstate_t __state;
} _G_fpos64_t;
#if defined _LIBC || defined _GLIBCPP_USE_WCHAR_T
#if 0 /* expanded by -frewrite-includes */
# include <gconv.h>
#endif /* expanded by -frewrite-includes */
# 32 "/usr/include/_G_config.h" 3 4
# 33 "/usr/include/_G_config.h" 3 4
typedef union
{
  struct __gconv_info __cd;
  struct
  {
    struct __gconv_info __cd;
    struct __gconv_step_data __data;
  } __combined;
} _G_iconv_t;
#endif
# 43 "/usr/include/_G_config.h" 3 4


/* These library features are always available in the GNU C library.  */
#define _G_va_list __gnuc_va_list

#define _G_HAVE_MMAP 1
#define _G_HAVE_MREMAP 1

#define _G_IO_IO_FILE_VERSION 0x20001

/* This is defined by <bits/stat.h> if `st_blksize' exists.  */
#define _G_HAVE_ST_BLKSIZE defined (_STATBUF_ST_BLKSIZE)

#define _G_BUFSIZ 8192

#endif	/* _G_config.h */
# 59 "/usr/include/_G_config.h" 3 4
# 32 "/usr/include/libio.h" 2 3 4
/* ALL of these should be defined in _G_config.h */
#define _IO_fpos_t _G_fpos_t
#define _IO_fpos64_t _G_fpos64_t
#define _IO_size_t size_t
#define _IO_ssize_t __ssize_t
#define _IO_off_t __off_t
#define _IO_off64_t __off64_t
#define _IO_pid_t __pid_t
#define _IO_uid_t __uid_t
#define _IO_iconv_t _G_iconv_t
#define _IO_HAVE_ST_BLKSIZE _G_HAVE_ST_BLKSIZE
#define _IO_BUFSIZ _G_BUFSIZ
#define _IO_va_list _G_va_list
#define _IO_wint_t wint_t

/* This define avoids name pollution if we're using GNU stdarg.h */
#define __need___va_list
#if 0 /* expanded by -frewrite-includes */
#include <stdarg.h>
#endif /* expanded by -frewrite-includes */
# 49 "/usr/include/libio.h" 3 4
# 1 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stdarg.h" 1 3 4
/*===---- stdarg.h - Variable argument handling ----------------------------===
 *
 * Copyright (c) 2008 Eli Friedman
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __STDARG_H
#define __STDARG_H

#ifndef _VA_LIST
typedef __builtin_va_list va_list;
#define _VA_LIST
#endif
# 33 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stdarg.h" 3 4
#define va_start(ap, param) __builtin_va_start(ap, param)
#define va_end(ap)          __builtin_va_end(ap)
#define va_arg(ap, type)    __builtin_va_arg(ap, type)

/* GCC always defines __va_copy, but does not define va_copy unless in c99 mode
 * or -ansi is not specified, since it was not part of C90.
 */
#define __va_copy(d,s) __builtin_va_copy(d,s)

#if __STDC_VERSION__ >= 199901L || __cplusplus >= 201103L || !defined(__STRICT_ANSI__)
#define va_copy(dest, src)  __builtin_va_copy(dest, src)
#endif
# 45 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stdarg.h" 3 4

#ifndef __GNUC_VA_LIST
#define __GNUC_VA_LIST 1
typedef __builtin_va_list __gnuc_va_list;
#endif
# 50 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stdarg.h" 3 4

#endif /* __STDARG_H */
# 52 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stdarg.h" 3 4
# 50 "/usr/include/libio.h" 2 3 4
#ifdef __GNUC_VA_LIST
# undef _IO_va_list
# define _IO_va_list __gnuc_va_list
#endif /* __GNUC_VA_LIST */
# 54 "/usr/include/libio.h" 3 4

#ifndef __P
#if 0 /* expanded by -frewrite-includes */
# include <sys/cdefs.h>
#endif /* expanded by -frewrite-includes */
# 56 "/usr/include/libio.h" 3 4
# 57 "/usr/include/libio.h" 3 4
#endif /*!__P*/
# 58 "/usr/include/libio.h" 3 4

#define _IO_UNIFIED_JUMPTABLES 1

#ifndef EOF
# define EOF (-1)
#endif
# 64 "/usr/include/libio.h" 3 4
#ifndef NULL
# if defined __GNUG__ && \
    (__GNUC__ > 2 || (__GNUC__ == 2 && __GNUC_MINOR__ >= 8))
#  define NULL (__null)
# else
# 69 "/usr/include/libio.h" 3 4
#  if !defined(__cplusplus)
#   define NULL ((void*)0)
#  else
# 72 "/usr/include/libio.h" 3 4
#   define NULL (0)
#  endif
# 74 "/usr/include/libio.h" 3 4
# endif
# 75 "/usr/include/libio.h" 3 4
#endif
# 76 "/usr/include/libio.h" 3 4

#define _IOS_INPUT	1
#define _IOS_OUTPUT	2
#define _IOS_ATEND	4
#define _IOS_APPEND	8
#define _IOS_TRUNC	16
#define _IOS_NOCREATE	32
#define _IOS_NOREPLACE	64
#define _IOS_BIN	128

/* Magic numbers and bits for the _flags field.
   The magic numbers use the high-order bits of _flags;
   the remaining bits are available for variable flags.
   Note: The magic numbers must all be negative if stdio
   emulation is desired. */

#define _IO_MAGIC 0xFBAD0000 /* Magic number */
#define _OLD_STDIO_MAGIC 0xFABC0000 /* Emulate old stdio. */
#define _IO_MAGIC_MASK 0xFFFF0000
#define _IO_USER_BUF 1 /* User owns buffer; don't delete it on close. */
#define _IO_UNBUFFERED 2
#define _IO_NO_READS 4 /* Reading not allowed */
#define _IO_NO_WRITES 8 /* Writing not allowd */
#define _IO_EOF_SEEN 0x10
#define _IO_ERR_SEEN 0x20
#define _IO_DELETE_DONT_CLOSE 0x40 /* Don't call close(_fileno) on cleanup. */
#define _IO_LINKED 0x80 /* Set if linked (using _chain) to streambuf::_list_all.*/
#define _IO_IN_BACKUP 0x100
#define _IO_LINE_BUF 0x200
#define _IO_TIED_PUT_GET 0x400 /* Set if put and get pointer logicly tied. */
#define _IO_CURRENTLY_PUTTING 0x800
#define _IO_IS_APPENDING 0x1000
#define _IO_IS_FILEBUF 0x2000
#define _IO_BAD_SEEN 0x4000
#define _IO_USER_LOCK 0x8000

#define _IO_FLAGS2_MMAP 1
#define _IO_FLAGS2_NOTCANCEL 2
#ifdef _LIBC
# define _IO_FLAGS2_FORTIFY 4
#endif
# 117 "/usr/include/libio.h" 3 4
#define _IO_FLAGS2_USER_WBUF 8
#ifdef _LIBC
# define _IO_FLAGS2_SCANF_STD 16
# define _IO_FLAGS2_NOCLOSE 32
# define _IO_FLAGS2_CLOEXEC 64
#endif
# 123 "/usr/include/libio.h" 3 4

/* These are "formatting flags" matching the iostream fmtflags enum values. */
#define _IO_SKIPWS 01
#define _IO_LEFT 02
#define _IO_RIGHT 04
#define _IO_INTERNAL 010
#define _IO_DEC 020
#define _IO_OCT 040
#define _IO_HEX 0100
#define _IO_SHOWBASE 0200
#define _IO_SHOWPOINT 0400
#define _IO_UPPERCASE 01000
#define _IO_SHOWPOS 02000
#define _IO_SCIENTIFIC 04000
#define _IO_FIXED 010000
#define _IO_UNITBUF 020000
#define _IO_STDIO 040000
#define _IO_DONT_CLOSE 0100000
#define _IO_BOOLALPHA 0200000


struct _IO_jump_t;  struct _IO_FILE;

/* Handle lock.  */
#ifdef _IO_MTSAFE_IO
# if defined __GLIBC__ && __GLIBC__ >= 2
#if 0 /* expanded by -frewrite-includes */
#  include <bits/stdio-lock.h>
#endif /* expanded by -frewrite-includes */
# 149 "/usr/include/libio.h" 3 4
# 150 "/usr/include/libio.h" 3 4
# else
# 151 "/usr/include/libio.h" 3 4
/*# include <comthread.h>*/
# endif
# 153 "/usr/include/libio.h" 3 4
#else
# 154 "/usr/include/libio.h" 3 4
typedef void _IO_lock_t;
#endif
# 156 "/usr/include/libio.h" 3 4


/* A streammarker remembers a position in a buffer. */

struct _IO_marker {
  struct _IO_marker *_next;
  struct _IO_FILE *_sbuf;
  /* If _pos >= 0
 it points to _buf->Gbase()+_pos. FIXME comment */
  /* if _pos < 0, it points to _buf->eBptr()+_pos. FIXME comment */
  int _pos;
#if 0
    void set_streampos(streampos sp) { _spos = sp; }
    void set_offset(int offset) { _pos = offset; _spos = (streampos)(-2); }
  public:
    streammarker(streambuf *sb);
    ~streammarker();
    int saving() { return  _spos == -2; }
    int delta(streammarker&);
    int delta();
#endif
# 177 "/usr/include/libio.h" 3 4
};

/* This is the structure from the libstdc++ codecvt class.  */
enum __codecvt_result
{
  __codecvt_ok,
  __codecvt_partial,
  __codecvt_error,
  __codecvt_noconv
};

#if defined _LIBC || defined _GLIBCPP_USE_WCHAR_T
/* The order of the elements in the following struct must match the order
   of the virtual functions in the libstdc++ codecvt class.  */
struct _IO_codecvt
{
  void (*__codecvt_destr) (struct _IO_codecvt *);
  enum __codecvt_result (*__codecvt_do_out) (struct _IO_codecvt *,
					     __mbstate_t *,
					     const wchar_t *,
					     const wchar_t *,
					     const wchar_t **, char *,
					     char *, char **);
  enum __codecvt_result (*__codecvt_do_unshift) (struct _IO_codecvt *,
						 __mbstate_t *, char *,
						 char *, char **);
  enum __codecvt_result (*__codecvt_do_in) (struct _IO_codecvt *,
					    __mbstate_t *,
					    const char *, const char *,
					    const char **, wchar_t *,
					    wchar_t *, wchar_t **);
  int (*__codecvt_do_encoding) (struct _IO_codecvt *);
  int (*__codecvt_do_always_noconv) (struct _IO_codecvt *);
  int (*__codecvt_do_length) (struct _IO_codecvt *, __mbstate_t *,
			      const char *, const char *, _IO_size_t);
  int (*__codecvt_do_max_length) (struct _IO_codecvt *);

  _IO_iconv_t __cd_in;
  _IO_iconv_t __cd_out;
};

/* Extra data for wide character streams.  */
struct _IO_wide_data
{
  wchar_t *_IO_read_ptr;	/* Current read pointer */
  wchar_t *_IO_read_end;	/* End of get area. */
  wchar_t *_IO_read_base;	/* Start of putback+get area. */
  wchar_t *_IO_write_base;	/* Start of put area. */
  wchar_t *_IO_write_ptr;	/* Current put pointer. */
  wchar_t *_IO_write_end;	/* End of put area. */
  wchar_t *_IO_buf_base;	/* Start of reserve area. */
  wchar_t *_IO_buf_end;		/* End of reserve area. */
  /* The following fields are used to support backing up and undo. */
  wchar_t *_IO_save_base;	/* Pointer to start of non-current get area. */
  wchar_t *_IO_backup_base;	/* Pointer to first valid character of
				   backup area */
  wchar_t *_IO_save_end;	/* Pointer to end of non-current get area. */

  __mbstate_t _IO_state;
  __mbstate_t _IO_last_state;
  struct _IO_codecvt _codecvt;

  wchar_t _shortbuf[1];

  const struct _IO_jump_t *_wide_vtable;
};
#endif
# 244 "/usr/include/libio.h" 3 4

struct _IO_FILE {
  int _flags;		/* High-order word is _IO_MAGIC; rest is flags. */
#define _IO_file_flags _flags

  /* The following pointers correspond to the C++ streambuf protocol. */
  /* Note:  Tk uses the _IO_read_ptr and _IO_read_end fields directly. */
  char* _IO_read_ptr;	/* Current read pointer */
  char* _IO_read_end;	/* End of get area. */
  char* _IO_read_base;	/* Start of putback+get area. */
  char* _IO_write_base;	/* Start of put area. */
  char* _IO_write_ptr;	/* Current put pointer. */
  char* _IO_write_end;	/* End of put area. */
  char* _IO_buf_base;	/* Start of reserve area. */
  char* _IO_buf_end;	/* End of reserve area. */
  /* The following fields are used to support backing up and undo. */
  char *_IO_save_base; /* Pointer to start of non-current get area. */
  char *_IO_backup_base;  /* Pointer to first valid character of backup area */
  char *_IO_save_end; /* Pointer to end of non-current get area. */

  struct _IO_marker *_markers;

  struct _IO_FILE *_chain;

  int _fileno;
#if 0
  int _blksize;
#else
# 272 "/usr/include/libio.h" 3 4
  int _flags2;
#endif
# 274 "/usr/include/libio.h" 3 4
  _IO_off_t _old_offset; /* This used to be _offset but it's too small.  */

#define __HAVE_COLUMN /* temporary */
  /* 1+column number of pbase(); 0 is unknown. */
  unsigned short _cur_column;
  signed char _vtable_offset;
  char _shortbuf[1];

  /*  char* _save_gptr;  char* _save_egptr; */

  _IO_lock_t *_lock;
#ifdef _IO_USE_OLD_IO_FILE
};

struct _IO_FILE_complete
{
  struct _IO_FILE _file;
#endif
# 292 "/usr/include/libio.h" 3 4
#if defined _G_IO_IO_FILE_VERSION && _G_IO_IO_FILE_VERSION == 0x20001
  _IO_off64_t _offset;
# if defined _LIBC || defined _GLIBCPP_USE_WCHAR_T
  /* Wide character stream stuff.  */
  struct _IO_codecvt *_codecvt;
  struct _IO_wide_data *_wide_data;
  struct _IO_FILE *_freeres_list;
  void *_freeres_buf;
  size_t _freeres_size;
# else
# 302 "/usr/include/libio.h" 3 4
  void *__pad1;
  void *__pad2;
  void *__pad3;
  void *__pad4;
  size_t __pad5;
# endif
# 308 "/usr/include/libio.h" 3 4
  int _mode;
  /* Make sure we don't get into trouble again.  */
  char _unused2[15 * sizeof (int) - 4 * sizeof (void *) - sizeof (size_t)];
#endif
# 312 "/usr/include/libio.h" 3 4
};

#ifndef __cplusplus
typedef struct _IO_FILE _IO_FILE;
#endif
# 317 "/usr/include/libio.h" 3 4

struct _IO_FILE_plus;

extern struct _IO_FILE_plus _IO_2_1_stdin_;
extern struct _IO_FILE_plus _IO_2_1_stdout_;
extern struct _IO_FILE_plus _IO_2_1_stderr_;
#ifndef _LIBC
#define _IO_stdin ((_IO_FILE*)(&_IO_2_1_stdin_))
#define _IO_stdout ((_IO_FILE*)(&_IO_2_1_stdout_))
#define _IO_stderr ((_IO_FILE*)(&_IO_2_1_stderr_))
#else
# 328 "/usr/include/libio.h" 3 4
extern _IO_FILE *_IO_stdin attribute_hidden;
extern _IO_FILE *_IO_stdout attribute_hidden;
extern _IO_FILE *_IO_stderr attribute_hidden;
#endif
# 332 "/usr/include/libio.h" 3 4


/* Functions to do I/O and file management for a stream.  */

/* Read NBYTES bytes from COOKIE into a buffer pointed to by BUF.
   Return number of bytes read.  */
typedef __ssize_t __io_read_fn (void *__cookie, char *__buf, size_t __nbytes);

/* Write N bytes pointed to by BUF to COOKIE.  Write all N bytes
   unless there is an error.  Return number of bytes written.  If
   there is an error, return 0 and do not write anything.  If the file
   has been opened for append (__mode.__append set), then set the file
   pointer to the end of the file and then do the write; if not, just
   write at the current file pointer.  */
typedef __ssize_t __io_write_fn (void *__cookie, const char *__buf,
				 size_t __n);

/* Move COOKIE's file position to *POS bytes from the
   beginning of the file (if W is SEEK_SET),
   the current position (if W is SEEK_CUR),
   or the end of the file (if W is SEEK_END).
   Set *POS to the new file position.
   Returns zero if successful, nonzero if not.  */
typedef int __io_seek_fn (void *__cookie, _IO_off64_t *__pos, int __w);

/* Close COOKIE.  */
typedef int __io_close_fn (void *__cookie);


#ifdef _GNU_SOURCE
/* User-visible names for the above.  */
typedef __io_read_fn cookie_read_function_t;
typedef __io_write_fn cookie_write_function_t;
typedef __io_seek_fn cookie_seek_function_t;
typedef __io_close_fn cookie_close_function_t;

/* The structure with the cookie function pointers.  */
typedef struct
{
  __io_read_fn *read;		/* Read bytes.  */
  __io_write_fn *write;		/* Write bytes.  */
  __io_seek_fn *seek;		/* Seek/tell file position.  */
  __io_close_fn *close;		/* Close file.  */
} _IO_cookie_io_functions_t;
typedef _IO_cookie_io_functions_t cookie_io_functions_t;

struct _IO_cookie_file;

/* Initialize one of those.  */
extern void _IO_cookie_init (struct _IO_cookie_file *__cfile, int __read_write,
			     void *__cookie, _IO_cookie_io_functions_t __fns);
#endif
# 384 "/usr/include/libio.h" 3 4


#ifdef __cplusplus
extern "C" {
#endif
# 389 "/usr/include/libio.h" 3 4

extern int __underflow (_IO_FILE *);
extern int __uflow (_IO_FILE *);
extern int __overflow (_IO_FILE *, int);
#if defined _LIBC || defined _GLIBCPP_USE_WCHAR_T
extern _IO_wint_t __wunderflow (_IO_FILE *);
extern _IO_wint_t __wuflow (_IO_FILE *);
extern _IO_wint_t __woverflow (_IO_FILE *, _IO_wint_t);
#endif
# 398 "/usr/include/libio.h" 3 4

#if  __GNUC__ >= 3
# define _IO_BE(expr, res) __builtin_expect ((expr), res)
#else
# 402 "/usr/include/libio.h" 3 4
# define _IO_BE(expr, res) (expr)
#endif
# 404 "/usr/include/libio.h" 3 4

#define _IO_getc_unlocked(_fp) \
       (_IO_BE ((_fp)->_IO_read_ptr >= (_fp)->_IO_read_end, 0) \
	? __uflow (_fp) : *(unsigned char *) (_fp)->_IO_read_ptr++)
#define _IO_peekc_unlocked(_fp) \
       (_IO_BE ((_fp)->_IO_read_ptr >= (_fp)->_IO_read_end, 0) \
	  && __underflow (_fp) == EOF ? EOF \
	: *(unsigned char *) (_fp)->_IO_read_ptr)
#define _IO_putc_unlocked(_ch, _fp) \
   (_IO_BE ((_fp)->_IO_write_ptr >= (_fp)->_IO_write_end, 0) \
    ? __overflow (_fp, (unsigned char) (_ch)) \
    : (unsigned char) (*(_fp)->_IO_write_ptr++ = (_ch)))

#if defined _LIBC || defined _GLIBCPP_USE_WCHAR_T
# define _IO_getwc_unlocked(_fp) \
  (_IO_BE ((_fp)->_wide_data == NULL					\
	   || ((_fp)->_wide_data->_IO_read_ptr				\
	       >= (_fp)->_wide_data->_IO_read_end), 0)			\
   ? __wuflow (_fp) : (_IO_wint_t) *(_fp)->_wide_data->_IO_read_ptr++)
# define _IO_putwc_unlocked(_wch, _fp) \
  (_IO_BE ((_fp)->_wide_data == NULL					\
	   || ((_fp)->_wide_data->_IO_write_ptr				\
	       >= (_fp)->_wide_data->_IO_write_end), 0)			\
   ? __woverflow (_fp, _wch)						\
   : (_IO_wint_t) (*(_fp)->_wide_data->_IO_write_ptr++ = (_wch)))
#endif
# 430 "/usr/include/libio.h" 3 4

#define _IO_feof_unlocked(__fp) (((__fp)->_flags & _IO_EOF_SEEN) != 0)
#define _IO_ferror_unlocked(__fp) (((__fp)->_flags & _IO_ERR_SEEN) != 0)

extern int _IO_getc (_IO_FILE *__fp);
extern int _IO_putc (int __c, _IO_FILE *__fp);
extern int _IO_feof (_IO_FILE *__fp) __THROW;
extern int _IO_ferror (_IO_FILE *__fp) __THROW;

extern int _IO_peekc_locked (_IO_FILE *__fp);

/* This one is for Emacs. */
#define _IO_PENDING_OUTPUT_COUNT(_fp)	\
	((_fp)->_IO_write_ptr - (_fp)->_IO_write_base)

extern void _IO_flockfile (_IO_FILE *) __THROW;
extern void _IO_funlockfile (_IO_FILE *) __THROW;
extern int _IO_ftrylockfile (_IO_FILE *) __THROW;

#ifdef _IO_MTSAFE_IO
# define _IO_peekc(_fp) _IO_peekc_locked (_fp)
# define _IO_flockfile(_fp) \
  if (((_fp)->_flags & _IO_USER_LOCK) == 0) _IO_flockfile (_fp)
# define _IO_funlockfile(_fp) \
  if (((_fp)->_flags & _IO_USER_LOCK) == 0) _IO_funlockfile (_fp)
#else
# 456 "/usr/include/libio.h" 3 4
# define _IO_peekc(_fp) _IO_peekc_unlocked (_fp)
# define _IO_flockfile(_fp) /**/
# define _IO_funlockfile(_fp) /**/
# define _IO_ftrylockfile(_fp) /**/
# define _IO_cleanup_region_start(_fct, _fp) /**/
# define _IO_cleanup_region_end(_Doit) /**/
#endif /* !_IO_MTSAFE_IO */
# 463 "/usr/include/libio.h" 3 4

extern int _IO_vfscanf (_IO_FILE * __restrict, const char * __restrict,
			_IO_va_list, int *__restrict);
extern int _IO_vfprintf (_IO_FILE *__restrict, const char *__restrict,
			 _IO_va_list);
extern _IO_ssize_t _IO_padn (_IO_FILE *, int, _IO_ssize_t);
extern _IO_size_t _IO_sgetn (_IO_FILE *, void *, _IO_size_t);

extern _IO_off64_t _IO_seekoff (_IO_FILE *, _IO_off64_t, int, int);
extern _IO_off64_t _IO_seekpos (_IO_FILE *, _IO_off64_t, int);

extern void _IO_free_backup_area (_IO_FILE *) __THROW;

#if defined _LIBC || defined _GLIBCPP_USE_WCHAR_T
extern _IO_wint_t _IO_getwc (_IO_FILE *__fp);
extern _IO_wint_t _IO_putwc (wchar_t __wc, _IO_FILE *__fp);
extern int _IO_fwide (_IO_FILE *__fp, int __mode) __THROW;
# if __GNUC__ >= 2
/* While compiling glibc we have to handle compatibility with very old
   versions.  */
#  if defined _LIBC && defined SHARED
#if 0 /* expanded by -frewrite-includes */
#   include <shlib-compat.h>
#endif /* expanded by -frewrite-includes */
# 484 "/usr/include/libio.h" 3 4
# 485 "/usr/include/libio.h" 3 4
#   if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_1)
#    define _IO_fwide_maybe_incompatible \
  (__builtin_expect (&_IO_stdin_used == NULL, 0))
extern const int _IO_stdin_used;
weak_extern (_IO_stdin_used);
#   endif
# 491 "/usr/include/libio.h" 3 4
#  endif
# 492 "/usr/include/libio.h" 3 4
#  ifndef _IO_fwide_maybe_incompatible
#   define _IO_fwide_maybe_incompatible (0)
#  endif
# 495 "/usr/include/libio.h" 3 4
/* A special optimized version of the function above.  It optimizes the
   case of initializing an unoriented byte stream.  */
#  define _IO_fwide(__fp, __mode) \
  ({ int __result = (__mode);						      \
     if (__result < 0 && ! _IO_fwide_maybe_incompatible)		      \
       {								      \
	 if ((__fp)->_mode == 0)					      \
	   /* We know that all we have to do is to set the flag.  */	      \
	   (__fp)->_mode = -1;						      \
	 __result = (__fp)->_mode;					      \
       }								      \
     else if (__builtin_constant_p (__mode) && (__mode) == 0)		      \
       __result = _IO_fwide_maybe_incompatible ? -1 : (__fp)->_mode;	      \
     else								      \
       __result = _IO_fwide (__fp, __result);				      \
     __result; })
# endif
# 512 "/usr/include/libio.h" 3 4

extern int _IO_vfwscanf (_IO_FILE * __restrict, const wchar_t * __restrict,
			 _IO_va_list, int *__restrict);
extern int _IO_vfwprintf (_IO_FILE *__restrict, const wchar_t *__restrict,
			  _IO_va_list);
extern _IO_ssize_t _IO_wpadn (_IO_FILE *, wint_t, _IO_ssize_t);
extern void _IO_free_wbackup_area (_IO_FILE *) __THROW;
#endif
# 520 "/usr/include/libio.h" 3 4

#ifdef __LDBL_COMPAT
#if 0 /* expanded by -frewrite-includes */
# include <bits/libio-ldbl.h>
#endif /* expanded by -frewrite-includes */
# 522 "/usr/include/libio.h" 3 4
# 523 "/usr/include/libio.h" 3 4
#endif
# 524 "/usr/include/libio.h" 3 4

#ifdef __cplusplus
}
#endif
# 528 "/usr/include/libio.h" 3 4

#endif /* _IO_STDIO_H */
# 530 "/usr/include/libio.h" 3 4
# 75 "/usr/include/stdio.h" 2 3 4

#if defined __USE_XOPEN || defined __USE_XOPEN2K8
# ifdef __GNUC__
#  ifndef _VA_LIST_DEFINED
typedef _G_va_list va_list;
#   define _VA_LIST_DEFINED
#  endif
# 82 "/usr/include/stdio.h" 3 4
# else
# 83 "/usr/include/stdio.h" 3 4
#if 0 /* expanded by -frewrite-includes */
#  include <stdarg.h>
#endif /* expanded by -frewrite-includes */
# 83 "/usr/include/stdio.h" 3 4
# 84 "/usr/include/stdio.h" 3 4
# endif
# 85 "/usr/include/stdio.h" 3 4
#endif
# 86 "/usr/include/stdio.h" 3 4

#ifdef __USE_XOPEN2K8
# ifndef __off_t_defined
# ifndef __USE_FILE_OFFSET64
typedef __off_t off_t;
# else
# 92 "/usr/include/stdio.h" 3 4
typedef __off64_t off_t;
# endif
# 94 "/usr/include/stdio.h" 3 4
# define __off_t_defined
# endif
# 96 "/usr/include/stdio.h" 3 4
# if defined __USE_LARGEFILE64 && !defined __off64_t_defined
typedef __off64_t off64_t;
# define __off64_t_defined
# endif
# 100 "/usr/include/stdio.h" 3 4

# ifndef __ssize_t_defined
typedef __ssize_t ssize_t;
# define __ssize_t_defined
# endif
# 105 "/usr/include/stdio.h" 3 4
#endif
# 106 "/usr/include/stdio.h" 3 4

/* The type of the second argument to `fgetpos' and `fsetpos'.  */
__BEGIN_NAMESPACE_STD
#ifndef __USE_FILE_OFFSET64
typedef _G_fpos_t fpos_t;
#else
# 112 "/usr/include/stdio.h" 3 4
typedef _G_fpos64_t fpos_t;
#endif
# 114 "/usr/include/stdio.h" 3 4
__END_NAMESPACE_STD
#ifdef __USE_LARGEFILE64
typedef _G_fpos64_t fpos64_t;
#endif
# 118 "/usr/include/stdio.h" 3 4

/* The possibilities for the third argument to `setvbuf'.  */
#define _IOFBF 0		/* Fully buffered.  */
#define _IOLBF 1		/* Line buffered.  */
#define _IONBF 2		/* No buffering.  */


/* Default buffer size.  */
#ifndef BUFSIZ
# define BUFSIZ _IO_BUFSIZ
#endif
# 129 "/usr/include/stdio.h" 3 4


/* End of file character.
   Some things throughout the library rely on this being -1.  */
#ifndef EOF
# define EOF (-1)
#endif
# 136 "/usr/include/stdio.h" 3 4


/* The possibilities for the third argument to `fseek'.
   These values should not be changed.  */
#define SEEK_SET	0	/* Seek from beginning of file.  */
#define SEEK_CUR	1	/* Seek from current position.  */
#define SEEK_END	2	/* Seek from end of file.  */
#ifdef __USE_GNU
# define SEEK_DATA	3	/* Seek to next data.  */
# define SEEK_HOLE	4	/* Seek to next hole.  */
#endif
# 147 "/usr/include/stdio.h" 3 4


#if defined __USE_SVID || defined __USE_XOPEN
/* Default path prefix for `tempnam' and `tmpnam'.  */
# define P_tmpdir	"/tmp"
#endif
# 153 "/usr/include/stdio.h" 3 4


/* Get the values:
   L_tmpnam	How long an array of chars must be to be passed to `tmpnam'.
   TMP_MAX	The minimum number of unique filenames generated by tmpnam
		(and tempnam when it uses tmpnam's name space),
		or tempnam (the two are separate).
   L_ctermid	How long an array to pass to `ctermid'.
   L_cuserid	How long an array to pass to `cuserid'.
   FOPEN_MAX	Minimum number of files that can be open at once.
   FILENAME_MAX	Maximum length of a filename.  */
#if 0 /* expanded by -frewrite-includes */
#include <bits/stdio_lim.h>
#endif /* expanded by -frewrite-includes */
# 164 "/usr/include/stdio.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/stdio_lim.h" 1 3 4
/* Copyright (C) 1994-2014 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */

#if !defined _STDIO_H && !defined __need_FOPEN_MAX && !defined __need_IOV_MAX
# error "Never include <bits/stdio_lim.h> directly; use <stdio.h> instead."
#endif
# 21 "/usr/include/x86_64-linux-gnu/bits/stdio_lim.h" 3 4

#ifdef _STDIO_H
# define L_tmpnam 20
# define TMP_MAX 238328
# define FILENAME_MAX 4096

# ifdef __USE_POSIX
#  define L_ctermid 9
#  if !defined __USE_XOPEN2K || defined __USE_GNU
#   define L_cuserid 9
#  endif
# 32 "/usr/include/x86_64-linux-gnu/bits/stdio_lim.h" 3 4
# endif
# 33 "/usr/include/x86_64-linux-gnu/bits/stdio_lim.h" 3 4
#endif
# 34 "/usr/include/x86_64-linux-gnu/bits/stdio_lim.h" 3 4

#if defined __need_FOPEN_MAX || defined _STDIO_H
# undef  FOPEN_MAX
# define FOPEN_MAX 16
#endif
# 39 "/usr/include/x86_64-linux-gnu/bits/stdio_lim.h" 3 4

#if defined __need_IOV_MAX && !defined IOV_MAX
# define IOV_MAX 1024
#endif
# 43 "/usr/include/x86_64-linux-gnu/bits/stdio_lim.h" 3 4
# 165 "/usr/include/stdio.h" 2 3 4


/* Standard streams.  */
extern struct _IO_FILE *stdin;		/* Standard input stream.  */
extern struct _IO_FILE *stdout;		/* Standard output stream.  */
extern struct _IO_FILE *stderr;		/* Standard error output stream.  */
/* C89/C99 say they're macros.  Make them happy.  */
#define stdin stdin
#define stdout stdout
#define stderr stderr

__BEGIN_NAMESPACE_STD
/* Remove file FILENAME.  */
extern int remove (const char *__filename) __THROW;
/* Rename file OLD to NEW.  */
extern int rename (const char *__old, const char *__new) __THROW;
__END_NAMESPACE_STD

#ifdef __USE_ATFILE
/* Rename file OLD relative to OLDFD to NEW relative to NEWFD.  */
extern int renameat (int __oldfd, const char *__old, int __newfd,
		     const char *__new) __THROW;
#endif
# 188 "/usr/include/stdio.h" 3 4

__BEGIN_NAMESPACE_STD
/* Create a temporary file and open it read/write.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
#ifndef __USE_FILE_OFFSET64
extern FILE *tmpfile (void) __wur;
#else
# 197 "/usr/include/stdio.h" 3 4
# ifdef __REDIRECT
extern FILE *__REDIRECT (tmpfile, (void), tmpfile64) __wur;
# else
# 200 "/usr/include/stdio.h" 3 4
#  define tmpfile tmpfile64
# endif
# 202 "/usr/include/stdio.h" 3 4
#endif
# 203 "/usr/include/stdio.h" 3 4

#ifdef __USE_LARGEFILE64
extern FILE *tmpfile64 (void) __wur;
#endif
# 207 "/usr/include/stdio.h" 3 4

/* Generate a temporary filename.  */
extern char *tmpnam (char *__s) __THROW __wur;
__END_NAMESPACE_STD

#ifdef __USE_MISC
/* This is the reentrant variant of `tmpnam'.  The only difference is
   that it does not allow S to be NULL.  */
extern char *tmpnam_r (char *__s) __THROW __wur;
#endif
# 217 "/usr/include/stdio.h" 3 4


#if defined __USE_SVID || defined __USE_XOPEN
/* Generate a unique temporary filename using up to five characters of PFX
   if it is not NULL.  The directory to put this file in is searched for
   as follows: First the environment variable "TMPDIR" is checked.
   If it contains the name of a writable directory, that directory is used.
   If not and if DIR is not NULL, that value is checked.  If that fails,
   P_tmpdir is tried and finally "/tmp".  The storage for the filename
   is allocated by `malloc'.  */
extern char *tempnam (const char *__dir, const char *__pfx)
     __THROW __attribute_malloc__ __wur;
#endif
# 230 "/usr/include/stdio.h" 3 4


__BEGIN_NAMESPACE_STD
/* Close STREAM.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int fclose (FILE *__stream);
/* Flush STREAM, or all streams if STREAM is NULL.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int fflush (FILE *__stream);
__END_NAMESPACE_STD

#ifdef __USE_MISC
/* Faster versions when locking is not required.

   This function is not part of POSIX and therefore no official
   cancellation point.  But due to similarity with an POSIX interface
   or due to the implementation it is a cancellation point and
   therefore not marked with __THROW.  */
extern int fflush_unlocked (FILE *__stream);
#endif
# 254 "/usr/include/stdio.h" 3 4

#ifdef __USE_GNU
/* Close all streams.

   This function is not part of POSIX and therefore no official
   cancellation point.  But due to similarity with an POSIX interface
   or due to the implementation it is a cancellation point and
   therefore not marked with __THROW.  */
extern int fcloseall (void);
#endif
# 264 "/usr/include/stdio.h" 3 4


__BEGIN_NAMESPACE_STD
#ifndef __USE_FILE_OFFSET64
/* Open a file and create a new stream for it.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern FILE *fopen (const char *__restrict __filename,
		    const char *__restrict __modes) __wur;
/* Open a file, replacing an existing stream with it.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern FILE *freopen (const char *__restrict __filename,
		      const char *__restrict __modes,
		      FILE *__restrict __stream) __wur;
#else
# 282 "/usr/include/stdio.h" 3 4
# ifdef __REDIRECT
extern FILE *__REDIRECT (fopen, (const char *__restrict __filename,
				 const char *__restrict __modes), fopen64)
  __wur;
extern FILE *__REDIRECT (freopen, (const char *__restrict __filename,
				   const char *__restrict __modes,
				   FILE *__restrict __stream), freopen64)
  __wur;
# else
# 291 "/usr/include/stdio.h" 3 4
#  define fopen fopen64
#  define freopen freopen64
# endif
# 294 "/usr/include/stdio.h" 3 4
#endif
# 295 "/usr/include/stdio.h" 3 4
__END_NAMESPACE_STD
#ifdef __USE_LARGEFILE64
extern FILE *fopen64 (const char *__restrict __filename,
		      const char *__restrict __modes) __wur;
extern FILE *freopen64 (const char *__restrict __filename,
			const char *__restrict __modes,
			FILE *__restrict __stream) __wur;
#endif
# 303 "/usr/include/stdio.h" 3 4

#ifdef	__USE_POSIX
/* Create a new stream that refers to an existing system file descriptor.  */
extern FILE *fdopen (int __fd, const char *__modes) __THROW __wur;
#endif
# 308 "/usr/include/stdio.h" 3 4

#ifdef	__USE_GNU
/* Create a new stream that refers to the given magic cookie,
   and uses the given functions for input and output.  */
extern FILE *fopencookie (void *__restrict __magic_cookie,
			  const char *__restrict __modes,
			  _IO_cookie_io_functions_t __io_funcs) __THROW __wur;
#endif
# 316 "/usr/include/stdio.h" 3 4

#ifdef __USE_XOPEN2K8
/* Create a new stream that refers to a memory buffer.  */
extern FILE *fmemopen (void *__s, size_t __len, const char *__modes)
  __THROW __wur;

/* Open a stream that writes into a malloc'd buffer that is expanded as
   necessary.  *BUFLOC and *SIZELOC are updated with the buffer's location
   and the number of characters written on fflush or fclose.  */
extern FILE *open_memstream (char **__bufloc, size_t *__sizeloc) __THROW __wur;
#endif
# 327 "/usr/include/stdio.h" 3 4


__BEGIN_NAMESPACE_STD
/* If BUF is NULL, make STREAM unbuffered.
   Else make it use buffer BUF, of size BUFSIZ.  */
extern void setbuf (FILE *__restrict __stream, char *__restrict __buf) __THROW;
/* Make STREAM use buffering mode MODE.
   If BUF is not NULL, use N bytes of it for buffering;
   else allocate an internal buffer N bytes long.  */
extern int setvbuf (FILE *__restrict __stream, char *__restrict __buf,
		    int __modes, size_t __n) __THROW;
__END_NAMESPACE_STD

#ifdef	__USE_BSD
/* If BUF is NULL, make STREAM unbuffered.
   Else make it use SIZE bytes of BUF for buffering.  */
extern void setbuffer (FILE *__restrict __stream, char *__restrict __buf,
		       size_t __size) __THROW;

/* Make STREAM line-buffered.  */
extern void setlinebuf (FILE *__stream) __THROW;
#endif
# 349 "/usr/include/stdio.h" 3 4


__BEGIN_NAMESPACE_STD
/* Write formatted output to STREAM.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int fprintf (FILE *__restrict __stream,
		    const char *__restrict __format, ...);
/* Write formatted output to stdout.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int printf (const char *__restrict __format, ...);
/* Write formatted output to S.  */
extern int sprintf (char *__restrict __s,
		    const char *__restrict __format, ...) __THROWNL;

/* Write formatted output to S from argument list ARG.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int vfprintf (FILE *__restrict __s, const char *__restrict __format,
		     _G_va_list __arg);
/* Write formatted output to stdout from argument list ARG.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int vprintf (const char *__restrict __format, _G_va_list __arg);
/* Write formatted output to S from argument list ARG.  */
extern int vsprintf (char *__restrict __s, const char *__restrict __format,
		     _G_va_list __arg) __THROWNL;
__END_NAMESPACE_STD

#if defined __USE_BSD || defined __USE_ISOC99 || defined __USE_UNIX98
__BEGIN_NAMESPACE_C99
/* Maximum chars of output to write in MAXLEN.  */
extern int snprintf (char *__restrict __s, size_t __maxlen,
		     const char *__restrict __format, ...)
     __THROWNL __attribute__ ((__format__ (__printf__, 3, 4)));

extern int vsnprintf (char *__restrict __s, size_t __maxlen,
		      const char *__restrict __format, _G_va_list __arg)
     __THROWNL __attribute__ ((__format__ (__printf__, 3, 0)));
__END_NAMESPACE_C99
#endif
# 395 "/usr/include/stdio.h" 3 4

#ifdef __USE_GNU
/* Write formatted output to a string dynamically allocated with `malloc'.
   Store the address of the string in *PTR.  */
extern int vasprintf (char **__restrict __ptr, const char *__restrict __f,
		      _G_va_list __arg)
     __THROWNL __attribute__ ((__format__ (__printf__, 2, 0))) __wur;
extern int __asprintf (char **__restrict __ptr,
		       const char *__restrict __fmt, ...)
     __THROWNL __attribute__ ((__format__ (__printf__, 2, 3))) __wur;
extern int asprintf (char **__restrict __ptr,
		     const char *__restrict __fmt, ...)
     __THROWNL __attribute__ ((__format__ (__printf__, 2, 3))) __wur;
#endif
# 409 "/usr/include/stdio.h" 3 4

#ifdef __USE_XOPEN2K8
/* Write formatted output to a file descriptor.  */
extern int vdprintf (int __fd, const char *__restrict __fmt,
		     _G_va_list __arg)
     __attribute__ ((__format__ (__printf__, 2, 0)));
extern int dprintf (int __fd, const char *__restrict __fmt, ...)
     __attribute__ ((__format__ (__printf__, 2, 3)));
#endif
# 418 "/usr/include/stdio.h" 3 4


__BEGIN_NAMESPACE_STD
/* Read formatted input from STREAM.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int fscanf (FILE *__restrict __stream,
		   const char *__restrict __format, ...) __wur;
/* Read formatted input from stdin.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int scanf (const char *__restrict __format, ...) __wur;
/* Read formatted input from S.  */
extern int sscanf (const char *__restrict __s,
		   const char *__restrict __format, ...) __THROW;

#if defined __USE_ISOC99 && !defined __USE_GNU \
    && (!defined __LDBL_COMPAT || !defined __REDIRECT) \
    && (defined __STRICT_ANSI__ || defined __USE_XOPEN2K)
# ifdef __REDIRECT
/* For strict ISO C99 or POSIX compliance disallow %as, %aS and %a[
   GNU extension which conflicts with valid %a followed by letter
   s, S or [.  */
extern int __REDIRECT (fscanf, (FILE *__restrict __stream,
				const char *__restrict __format, ...),
		       __isoc99_fscanf) __wur;
extern int __REDIRECT (scanf, (const char *__restrict __format, ...),
		       __isoc99_scanf) __wur;
extern int __REDIRECT_NTH (sscanf, (const char *__restrict __s,
				    const char *__restrict __format, ...),
			   __isoc99_sscanf);
# else
# 452 "/usr/include/stdio.h" 3 4
extern int __isoc99_fscanf (FILE *__restrict __stream,
			    const char *__restrict __format, ...) __wur;
extern int __isoc99_scanf (const char *__restrict __format, ...) __wur;
extern int __isoc99_sscanf (const char *__restrict __s,
			    const char *__restrict __format, ...) __THROW;
#  define fscanf __isoc99_fscanf
#  define scanf __isoc99_scanf
#  define sscanf __isoc99_sscanf
# endif
# 461 "/usr/include/stdio.h" 3 4
#endif
# 462 "/usr/include/stdio.h" 3 4

__END_NAMESPACE_STD

#ifdef	__USE_ISOC99
__BEGIN_NAMESPACE_C99
/* Read formatted input from S into argument list ARG.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int vfscanf (FILE *__restrict __s, const char *__restrict __format,
		    _G_va_list __arg)
     __attribute__ ((__format__ (__scanf__, 2, 0))) __wur;

/* Read formatted input from stdin into argument list ARG.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int vscanf (const char *__restrict __format, _G_va_list __arg)
     __attribute__ ((__format__ (__scanf__, 1, 0))) __wur;

/* Read formatted input from S into argument list ARG.  */
extern int vsscanf (const char *__restrict __s,
		    const char *__restrict __format, _G_va_list __arg)
     __THROW __attribute__ ((__format__ (__scanf__, 2, 0)));

# if !defined __USE_GNU \
     && (!defined __LDBL_COMPAT || !defined __REDIRECT) \
     && (defined __STRICT_ANSI__ || defined __USE_XOPEN2K)
#  ifdef __REDIRECT
/* For strict ISO C99 or POSIX compliance disallow %as, %aS and %a[
   GNU extension which conflicts with valid %a followed by letter
   s, S or [.  */
extern int __REDIRECT (vfscanf,
		       (FILE *__restrict __s,
			const char *__restrict __format, _G_va_list __arg),
		       __isoc99_vfscanf)
     __attribute__ ((__format__ (__scanf__, 2, 0))) __wur;
extern int __REDIRECT (vscanf, (const char *__restrict __format,
				_G_va_list __arg), __isoc99_vscanf)
     __attribute__ ((__format__ (__scanf__, 1, 0))) __wur;
extern int __REDIRECT_NTH (vsscanf,
			   (const char *__restrict __s,
			    const char *__restrict __format,
			    _G_va_list __arg), __isoc99_vsscanf)
     __attribute__ ((__format__ (__scanf__, 2, 0)));
#  else
# 508 "/usr/include/stdio.h" 3 4
extern int __isoc99_vfscanf (FILE *__restrict __s,
			     const char *__restrict __format,
			     _G_va_list __arg) __wur;
extern int __isoc99_vscanf (const char *__restrict __format,
			    _G_va_list __arg) __wur;
extern int __isoc99_vsscanf (const char *__restrict __s,
			     const char *__restrict __format,
			     _G_va_list __arg) __THROW;
#   define vfscanf __isoc99_vfscanf
#   define vscanf __isoc99_vscanf
#   define vsscanf __isoc99_vsscanf
#  endif
# 520 "/usr/include/stdio.h" 3 4
# endif
# 521 "/usr/include/stdio.h" 3 4

__END_NAMESPACE_C99
#endif /* Use ISO C9x.  */
# 524 "/usr/include/stdio.h" 3 4


__BEGIN_NAMESPACE_STD
/* Read a character from STREAM.

   These functions are possible cancellation points and therefore not
   marked with __THROW.  */
extern int fgetc (FILE *__stream);
extern int getc (FILE *__stream);

/* Read a character from stdin.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int getchar (void);
__END_NAMESPACE_STD

/* The C standard explicitly says this is a macro, so we always do the
   optimization for it.  */
#define getc(_fp) _IO_getc (_fp)

#if defined __USE_POSIX || defined __USE_MISC
/* These are defined in POSIX.1:1996.

   These functions are possible cancellation points and therefore not
   marked with __THROW.  */
extern int getc_unlocked (FILE *__stream);
extern int getchar_unlocked (void);
#endif /* Use POSIX or MISC.  */
# 553 "/usr/include/stdio.h" 3 4

#ifdef __USE_MISC
/* Faster version when locking is not necessary.

   This function is not part of POSIX and therefore no official
   cancellation point.  But due to similarity with an POSIX interface
   or due to the implementation it is a cancellation point and
   therefore not marked with __THROW.  */
extern int fgetc_unlocked (FILE *__stream);
#endif /* Use MISC.  */
# 563 "/usr/include/stdio.h" 3 4


__BEGIN_NAMESPACE_STD
/* Write a character to STREAM.

   These functions are possible cancellation points and therefore not
   marked with __THROW.

   These functions is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int fputc (int __c, FILE *__stream);
extern int putc (int __c, FILE *__stream);

/* Write a character to stdout.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int putchar (int __c);
__END_NAMESPACE_STD

/* The C standard explicitly says this can be a macro,
   so we always do the optimization for it.  */
#define putc(_ch, _fp) _IO_putc (_ch, _fp)

#ifdef __USE_MISC
/* Faster version when locking is not necessary.

   This function is not part of POSIX and therefore no official
   cancellation point.  But due to similarity with an POSIX interface
   or due to the implementation it is a cancellation point and
   therefore not marked with __THROW.  */
extern int fputc_unlocked (int __c, FILE *__stream);
#endif /* Use MISC.  */
# 596 "/usr/include/stdio.h" 3 4

#if defined __USE_POSIX || defined __USE_MISC
/* These are defined in POSIX.1:1996.

   These functions are possible cancellation points and therefore not
   marked with __THROW.  */
extern int putc_unlocked (int __c, FILE *__stream);
extern int putchar_unlocked (int __c);
#endif /* Use POSIX or MISC.  */
# 605 "/usr/include/stdio.h" 3 4


#if defined __USE_SVID || defined __USE_MISC \
    || (defined __USE_XOPEN && !defined __USE_XOPEN2K)
/* Get a word (int) from STREAM.  */
extern int getw (FILE *__stream);

/* Write a word (int) to STREAM.  */
extern int putw (int __w, FILE *__stream);
#endif
# 615 "/usr/include/stdio.h" 3 4


__BEGIN_NAMESPACE_STD
/* Get a newline-terminated string of finite length from STREAM.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern char *fgets (char *__restrict __s, int __n, FILE *__restrict __stream)
     __wur;

#if !defined __USE_ISOC11 \
    || (defined __cplusplus && __cplusplus <= 201103L)
/* Get a newline-terminated string from stdin, removing the newline.
   DO NOT USE THIS FUNCTION!!  There is no limit on how much it will read.

   The function has been officially removed in ISO C11.  This opportunity
   is used to also remove it from the GNU feature list.  It is now only
   available when explicitly using an old ISO C, Unix, or POSIX standard.
   GCC defines _GNU_SOURCE when building C++ code and the function is still
   in C++11, so it is also available for C++.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern char *gets (char *__s) __wur __attribute_deprecated__;
#endif
# 640 "/usr/include/stdio.h" 3 4
__END_NAMESPACE_STD

#ifdef __USE_GNU
/* This function does the same as `fgets' but does not lock the stream.

   This function is not part of POSIX and therefore no official
   cancellation point.  But due to similarity with an POSIX interface
   or due to the implementation it is a cancellation point and
   therefore not marked with __THROW.  */
extern char *fgets_unlocked (char *__restrict __s, int __n,
			     FILE *__restrict __stream) __wur;
#endif
# 652 "/usr/include/stdio.h" 3 4


#ifdef	__USE_XOPEN2K8
/* Read up to (and including) a DELIMITER from STREAM into *LINEPTR
   (and null-terminate it). *LINEPTR is a pointer returned from malloc (or
   NULL), pointing to *N characters of space.  It is realloc'd as
   necessary.  Returns the number of characters read (not including the
   null terminator), or -1 on error or EOF.

   These functions are not part of POSIX and therefore no official
   cancellation point.  But due to similarity with an POSIX interface
   or due to the implementation they are cancellation points and
   therefore not marked with __THROW.  */
extern _IO_ssize_t __getdelim (char **__restrict __lineptr,
			       size_t *__restrict __n, int __delimiter,
			       FILE *__restrict __stream) __wur;
extern _IO_ssize_t getdelim (char **__restrict __lineptr,
			     size_t *__restrict __n, int __delimiter,
			     FILE *__restrict __stream) __wur;

/* Like `getdelim', but reads up to a newline.

   This function is not part of POSIX and therefore no official
   cancellation point.  But due to similarity with an POSIX interface
   or due to the implementation it is a cancellation point and
   therefore not marked with __THROW.  */
extern _IO_ssize_t getline (char **__restrict __lineptr,
			    size_t *__restrict __n,
			    FILE *__restrict __stream) __wur;
#endif
# 682 "/usr/include/stdio.h" 3 4


__BEGIN_NAMESPACE_STD
/* Write a string to STREAM.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int fputs (const char *__restrict __s, FILE *__restrict __stream);

/* Write a string, followed by a newline, to stdout.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int puts (const char *__s);


/* Push a character back onto the input buffer of STREAM.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int ungetc (int __c, FILE *__stream);


/* Read chunks of generic data from STREAM.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern size_t fread (void *__restrict __ptr, size_t __size,
		     size_t __n, FILE *__restrict __stream) __wur;
/* Write chunks of generic data to STREAM.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern size_t fwrite (const void *__restrict __ptr, size_t __size,
		      size_t __n, FILE *__restrict __s);
__END_NAMESPACE_STD

#ifdef __USE_GNU
/* This function does the same as `fputs' but does not lock the stream.

   This function is not part of POSIX and therefore no official
   cancellation point.  But due to similarity with an POSIX interface
   or due to the implementation it is a cancellation point and
   therefore not marked with __THROW.  */
extern int fputs_unlocked (const char *__restrict __s,
			   FILE *__restrict __stream);
#endif
# 729 "/usr/include/stdio.h" 3 4

#ifdef __USE_MISC
/* Faster versions when locking is not necessary.

   These functions are not part of POSIX and therefore no official
   cancellation point.  But due to similarity with an POSIX interface
   or due to the implementation they are cancellation points and
   therefore not marked with __THROW.  */
extern size_t fread_unlocked (void *__restrict __ptr, size_t __size,
			      size_t __n, FILE *__restrict __stream) __wur;
extern size_t fwrite_unlocked (const void *__restrict __ptr, size_t __size,
			       size_t __n, FILE *__restrict __stream);
#endif
# 742 "/usr/include/stdio.h" 3 4


__BEGIN_NAMESPACE_STD
/* Seek to a certain position on STREAM.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int fseek (FILE *__stream, long int __off, int __whence);
/* Return the current position of STREAM.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern long int ftell (FILE *__stream) __wur;
/* Rewind to the beginning of STREAM.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern void rewind (FILE *__stream);
__END_NAMESPACE_STD

/* The Single Unix Specification, Version 2, specifies an alternative,
   more adequate interface for the two functions above which deal with
   file offset.  `long int' is not the right type.  These definitions
   are originally defined in the Large File Support API.  */

#if defined __USE_LARGEFILE || defined __USE_XOPEN2K
# ifndef __USE_FILE_OFFSET64
/* Seek to a certain position on STREAM.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int fseeko (FILE *__stream, __off_t __off, int __whence);
/* Return the current position of STREAM.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern __off_t ftello (FILE *__stream) __wur;
# else
# 780 "/usr/include/stdio.h" 3 4
#  ifdef __REDIRECT
extern int __REDIRECT (fseeko,
		       (FILE *__stream, __off64_t __off, int __whence),
		       fseeko64);
extern __off64_t __REDIRECT (ftello, (FILE *__stream), ftello64);
#  else
# 786 "/usr/include/stdio.h" 3 4
#   define fseeko fseeko64
#   define ftello ftello64
#  endif
# 789 "/usr/include/stdio.h" 3 4
# endif
# 790 "/usr/include/stdio.h" 3 4
#endif
# 791 "/usr/include/stdio.h" 3 4

__BEGIN_NAMESPACE_STD
#ifndef __USE_FILE_OFFSET64
/* Get STREAM's position.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int fgetpos (FILE *__restrict __stream, fpos_t *__restrict __pos);
/* Set STREAM's position.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int fsetpos (FILE *__stream, const fpos_t *__pos);
#else
# 805 "/usr/include/stdio.h" 3 4
# ifdef __REDIRECT
extern int __REDIRECT (fgetpos, (FILE *__restrict __stream,
				 fpos_t *__restrict __pos), fgetpos64);
extern int __REDIRECT (fsetpos,
		       (FILE *__stream, const fpos_t *__pos), fsetpos64);
# else
# 811 "/usr/include/stdio.h" 3 4
#  define fgetpos fgetpos64
#  define fsetpos fsetpos64
# endif
# 814 "/usr/include/stdio.h" 3 4
#endif
# 815 "/usr/include/stdio.h" 3 4
__END_NAMESPACE_STD

#ifdef __USE_LARGEFILE64
extern int fseeko64 (FILE *__stream, __off64_t __off, int __whence);
extern __off64_t ftello64 (FILE *__stream) __wur;
extern int fgetpos64 (FILE *__restrict __stream, fpos64_t *__restrict __pos);
extern int fsetpos64 (FILE *__stream, const fpos64_t *__pos);
#endif
# 823 "/usr/include/stdio.h" 3 4

__BEGIN_NAMESPACE_STD
/* Clear the error and EOF indicators for STREAM.  */
extern void clearerr (FILE *__stream) __THROW;
/* Return the EOF indicator for STREAM.  */
extern int feof (FILE *__stream) __THROW __wur;
/* Return the error indicator for STREAM.  */
extern int ferror (FILE *__stream) __THROW __wur;
__END_NAMESPACE_STD

#ifdef __USE_MISC
/* Faster versions when locking is not required.  */
extern void clearerr_unlocked (FILE *__stream) __THROW;
extern int feof_unlocked (FILE *__stream) __THROW __wur;
extern int ferror_unlocked (FILE *__stream) __THROW __wur;
#endif
# 839 "/usr/include/stdio.h" 3 4


__BEGIN_NAMESPACE_STD
/* Print a message describing the meaning of the value of errno.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern void perror (const char *__s);
__END_NAMESPACE_STD

/* Provide the declarations for `sys_errlist' and `sys_nerr' if they
   are available on this system.  Even if available, these variables
   should not be used directly.  The `strerror' function provides
   all the necessary functionality.  */
#if 0 /* expanded by -frewrite-includes */
#include <bits/sys_errlist.h>
#endif /* expanded by -frewrite-includes */
# 853 "/usr/include/stdio.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/sys_errlist.h" 1 3 4
/* Declare sys_errlist and sys_nerr, or don't.  Compatibility (do) version.
   Copyright (C) 2002-2014 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */

#ifndef _STDIO_H
# error "Never include <bits/sys_errlist.h> directly; use <stdio.h> instead."
#endif
# 22 "/usr/include/x86_64-linux-gnu/bits/sys_errlist.h" 3 4

/* sys_errlist and sys_nerr are deprecated.  Use strerror instead.  */

#ifdef  __USE_BSD
extern int sys_nerr;
extern const char *const sys_errlist[];
#endif
# 29 "/usr/include/x86_64-linux-gnu/bits/sys_errlist.h" 3 4
#ifdef  __USE_GNU
extern int _sys_nerr;
extern const char *const _sys_errlist[];
#endif
# 33 "/usr/include/x86_64-linux-gnu/bits/sys_errlist.h" 3 4
# 854 "/usr/include/stdio.h" 2 3 4


#ifdef	__USE_POSIX
/* Return the system file descriptor for STREAM.  */
extern int fileno (FILE *__stream) __THROW __wur;
#endif /* Use POSIX.  */
# 860 "/usr/include/stdio.h" 3 4

#ifdef __USE_MISC
/* Faster version when locking is not required.  */
extern int fileno_unlocked (FILE *__stream) __THROW __wur;
#endif
# 865 "/usr/include/stdio.h" 3 4


#if (defined __USE_POSIX2 || defined __USE_SVID  || defined __USE_BSD || \
     defined __USE_MISC)
/* Create a new stream connected to a pipe running the given command.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern FILE *popen (const char *__command, const char *__modes) __wur;

/* Close a stream opened by popen and return the status of its child.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
extern int pclose (FILE *__stream);
#endif
# 881 "/usr/include/stdio.h" 3 4


#ifdef	__USE_POSIX
/* Return the name of the controlling terminal.  */
extern char *ctermid (char *__s) __THROW;
#endif /* Use POSIX.  */
# 887 "/usr/include/stdio.h" 3 4


#ifdef __USE_XOPEN
/* Return the name of the current user.  */
extern char *cuserid (char *__s);
#endif /* Use X/Open, but not issue 6.  */
# 893 "/usr/include/stdio.h" 3 4


#ifdef	__USE_GNU
struct obstack;			/* See <obstack.h>.  */

/* Write formatted output to an obstack.  */
extern int obstack_printf (struct obstack *__restrict __obstack,
			   const char *__restrict __format, ...)
     __THROWNL __attribute__ ((__format__ (__printf__, 2, 3)));
extern int obstack_vprintf (struct obstack *__restrict __obstack,
			    const char *__restrict __format,
			    _G_va_list __args)
     __THROWNL __attribute__ ((__format__ (__printf__, 2, 0)));
#endif /* Use GNU.  */
# 907 "/usr/include/stdio.h" 3 4


#if defined __USE_POSIX || defined __USE_MISC
/* These are defined in POSIX.1:1996.  */

/* Acquire ownership of STREAM.  */
extern void flockfile (FILE *__stream) __THROW;

/* Try to acquire ownership of STREAM but do not block if it is not
   possible.  */
extern int ftrylockfile (FILE *__stream) __THROW __wur;

/* Relinquish the ownership granted for STREAM.  */
extern void funlockfile (FILE *__stream) __THROW;
#endif /* POSIX || misc */
# 922 "/usr/include/stdio.h" 3 4

#if defined __USE_XOPEN && !defined __USE_XOPEN2K && !defined __USE_GNU
/* The X/Open standard requires some functions and variables to be
   declared here which do not belong into this header.  But we have to
   follow.  In GNU mode we don't do this nonsense.  */
# define __need_getopt
#if 0 /* expanded by -frewrite-includes */
# include <getopt.h>
#endif /* expanded by -frewrite-includes */
# 928 "/usr/include/stdio.h" 3 4
# 929 "/usr/include/stdio.h" 3 4
#endif	/* X/Open, but not issue 6 and not for GNU.  */
# 930 "/usr/include/stdio.h" 3 4

/* If we are compiling with optimizing read this file.  It contains
   several optimizing inline functions and macros.  */
#ifdef __USE_EXTERN_INLINES
#if 0 /* expanded by -frewrite-includes */
# include <bits/stdio.h>
#endif /* expanded by -frewrite-includes */
# 934 "/usr/include/stdio.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/stdio.h" 1 3 4
/* Optimizing macros and inline functions for stdio functions.
   Copyright (C) 1998-2014 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */

#ifndef _STDIO_H
# error "Never include <bits/stdio.h> directly; use <stdio.h> instead."
#endif
# 22 "/usr/include/x86_64-linux-gnu/bits/stdio.h" 3 4

#ifndef __extern_inline
# define __STDIO_INLINE inline
#else
# 26 "/usr/include/x86_64-linux-gnu/bits/stdio.h" 3 4
# define __STDIO_INLINE __extern_inline
#endif
# 28 "/usr/include/x86_64-linux-gnu/bits/stdio.h" 3 4


#ifdef __USE_EXTERN_INLINES
/* For -D_FORTIFY_SOURCE{,=2} bits/stdio2.h will define a different
   inline.  */
# if !(__USE_FORTIFY_LEVEL > 0 && defined __fortify_function)
/* Write formatted output to stdout from argument list ARG.  */
__STDIO_INLINE int
vprintf (const char *__restrict __fmt, _G_va_list __arg)
{
  return vfprintf (stdout, __fmt, __arg);
}
# endif
# 41 "/usr/include/x86_64-linux-gnu/bits/stdio.h" 3 4

/* Read a character from stdin.  */
__STDIO_INLINE int
getchar (void)
{
  return _IO_getc (stdin);
}


# ifdef __USE_MISC
/* Faster version when locking is not necessary.  */
__STDIO_INLINE int
fgetc_unlocked (FILE *__fp)
{
  return _IO_getc_unlocked (__fp);
}
# endif /* misc */
# 58 "/usr/include/x86_64-linux-gnu/bits/stdio.h" 3 4


# if defined __USE_POSIX || defined __USE_MISC
/* This is defined in POSIX.1:1996.  */
__STDIO_INLINE int
getc_unlocked (FILE *__fp)
{
  return _IO_getc_unlocked (__fp);
}

/* This is defined in POSIX.1:1996.  */
__STDIO_INLINE int
getchar_unlocked (void)
{
  return _IO_getc_unlocked (stdin);
}
# endif	/* POSIX || misc */
# 75 "/usr/include/x86_64-linux-gnu/bits/stdio.h" 3 4


/* Write a character to stdout.  */
__STDIO_INLINE int
putchar (int __c)
{
  return _IO_putc (__c, stdout);
}


# ifdef __USE_MISC
/* Faster version when locking is not necessary.  */
__STDIO_INLINE int
fputc_unlocked (int __c, FILE *__stream)
{
  return _IO_putc_unlocked (__c, __stream);
}
# endif /* misc */
# 93 "/usr/include/x86_64-linux-gnu/bits/stdio.h" 3 4


# if defined __USE_POSIX || defined __USE_MISC
/* This is defined in POSIX.1:1996.  */
__STDIO_INLINE int
putc_unlocked (int __c, FILE *__stream)
{
  return _IO_putc_unlocked (__c, __stream);
}

/* This is defined in POSIX.1:1996.  */
__STDIO_INLINE int
putchar_unlocked (int __c)
{
  return _IO_putc_unlocked (__c, stdout);
}
# endif	/* POSIX || misc */
# 110 "/usr/include/x86_64-linux-gnu/bits/stdio.h" 3 4


# ifdef	__USE_GNU
/* Like `getdelim', but reads up to a newline.  */
__STDIO_INLINE _IO_ssize_t
getline (char **__lineptr, size_t *__n, FILE *__stream)
{
  return __getdelim (__lineptr, __n, '\n', __stream);
}
# endif /* GNU */
# 120 "/usr/include/x86_64-linux-gnu/bits/stdio.h" 3 4


# ifdef __USE_MISC
/* Faster versions when locking is not required.  */
__STDIO_INLINE int
__NTH (feof_unlocked (FILE *__stream))
{
  return _IO_feof_unlocked (__stream);
}

/* Faster versions when locking is not required.  */
__STDIO_INLINE int
__NTH (ferror_unlocked (FILE *__stream))
{
  return _IO_ferror_unlocked (__stream);
}
# endif /* misc */
# 137 "/usr/include/x86_64-linux-gnu/bits/stdio.h" 3 4

#endif /* Use extern inlines.  */
# 139 "/usr/include/x86_64-linux-gnu/bits/stdio.h" 3 4


#if defined __USE_MISC && defined __GNUC__ && defined __OPTIMIZE__ \
    && !defined __cplusplus
/* Perform some simple optimizations.  */
# define fread_unlocked(ptr, size, n, stream) \
  (__extension__ ((__builtin_constant_p (size) && __builtin_constant_p (n)    \
		   && (size_t) (size) * (size_t) (n) <= 8		      \
		   && (size_t) (size) != 0)				      \
		  ? ({ char *__ptr = (char *) (ptr);			      \
		       FILE *__stream = (stream);			      \
		       size_t __cnt;					      \
		       for (__cnt = (size_t) (size) * (size_t) (n);	      \
			    __cnt > 0; --__cnt)				      \
			 {						      \
			   int __c = _IO_getc_unlocked (__stream);	      \
			   if (__c == EOF)				      \
			     break;					      \
			   *__ptr++ = __c;				      \
			 }						      \
		       ((size_t) (size) * (size_t) (n) - __cnt)		      \
			/ (size_t) (size); })				      \
		  : (((__builtin_constant_p (size) && (size_t) (size) == 0)   \
		      || (__builtin_constant_p (n) && (size_t) (n) == 0))     \
			/* Evaluate all parameters once.  */		      \
		     ? ((void) (ptr), (void) (stream), (void) (size),	      \
			(void) (n), (size_t) 0)				      \
		     : fread_unlocked (ptr, size, n, stream))))

# define fwrite_unlocked(ptr, size, n, stream) \
  (__extension__ ((__builtin_constant_p (size) && __builtin_constant_p (n)    \
		   && (size_t) (size) * (size_t) (n) <= 8		      \
		   && (size_t) (size) != 0)				      \
		  ? ({ const char *__ptr = (const char *) (ptr);	      \
		       FILE *__stream = (stream);			      \
		       size_t __cnt;					      \
		       for (__cnt = (size_t) (size) * (size_t) (n);	      \
			    __cnt > 0; --__cnt)				      \
			 if (_IO_putc_unlocked (*__ptr++, __stream) == EOF)   \
			   break;					      \
		       ((size_t) (size) * (size_t) (n) - __cnt)		      \
			/ (size_t) (size); })				      \
		  : (((__builtin_constant_p (size) && (size_t) (size) == 0)   \
		      || (__builtin_constant_p (n) && (size_t) (n) == 0))     \
			/* Evaluate all parameters once.  */		      \
		     ? ((void) (ptr), (void) (stream), (void) (size),	      \
			(void) (n), (size_t) 0)				      \
		     : fwrite_unlocked (ptr, size, n, stream))))
#endif
# 188 "/usr/include/x86_64-linux-gnu/bits/stdio.h" 3 4

/* Define helper macro.  */
#undef __STDIO_INLINE
# 935 "/usr/include/stdio.h" 2 3 4
#endif
# 936 "/usr/include/stdio.h" 3 4
#if __USE_FORTIFY_LEVEL > 0 && defined __extern_always_inline
#if 0 /* expanded by -frewrite-includes */
# include <bits/stdio2.h>
#endif /* expanded by -frewrite-includes */
# 937 "/usr/include/stdio.h" 3 4
# 938 "/usr/include/stdio.h" 3 4
#endif
# 939 "/usr/include/stdio.h" 3 4
#ifdef __LDBL_COMPAT
#if 0 /* expanded by -frewrite-includes */
# include <bits/stdio-ldbl.h>
#endif /* expanded by -frewrite-includes */
# 940 "/usr/include/stdio.h" 3 4
# 941 "/usr/include/stdio.h" 3 4
#endif
# 942 "/usr/include/stdio.h" 3 4

__END_DECLS

#endif /* <stdio.h> included.  */
# 946 "/usr/include/stdio.h" 3 4

#endif /* !_STDIO_H */
# 948 "/usr/include/stdio.h" 3 4
# 3 "oski.c" 2
#if 0 /* expanded by -frewrite-includes */
#include <stdlib.h>
#endif /* expanded by -frewrite-includes */
# 3 "oski.c"
# 1 "/usr/include/stdlib.h" 1 3 4
/* Copyright (C) 1991-2014 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */

/*
 *	ISO C99 Standard: 7.20 General utilities	<stdlib.h>
 */

#ifndef	_STDLIB_H

#if 0 /* expanded by -frewrite-includes */
#include <features.h>
#endif /* expanded by -frewrite-includes */
# 24 "/usr/include/stdlib.h" 3 4
# 25 "/usr/include/stdlib.h" 3 4

/* Get size_t, wchar_t and NULL from <stddef.h>.  */
#define		__need_size_t
#ifndef __need_malloc_and_calloc
# define	__need_wchar_t
# define	__need_NULL
#endif
# 32 "/usr/include/stdlib.h" 3 4
#if 0 /* expanded by -frewrite-includes */
#include <stddef.h>
#endif /* expanded by -frewrite-includes */
# 32 "/usr/include/stdlib.h" 3 4
# 1 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 1 3 4
/*===---- stddef.h - Basic type definitions --------------------------------===
 *
 * Copyright (c) 2008 Eli Friedman
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *===-----------------------------------------------------------------------===
 */

#if !defined(__STDDEF_H) || defined(__need_ptrdiff_t) ||                       \
    defined(__need_size_t) || defined(__need_wchar_t) ||                       \
    defined(__need_NULL) || defined(__need_wint_t)

#if !defined(__need_ptrdiff_t) && !defined(__need_size_t) &&                   \
    !defined(__need_wchar_t) && !defined(__need_NULL) &&                       \
    !defined(__need_wint_t)
/* Always define miscellaneous pieces when modules are available. */
#if !__has_feature(modules)
#define __STDDEF_H
#endif
# 37 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#define __need_ptrdiff_t
#define __need_size_t
#define __need_wchar_t
#define __need_NULL
#define __need_STDDEF_H_misc
/* __need_wint_t is intentionally not defined here. */
#endif
# 44 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#if defined(__need_ptrdiff_t)
#if !defined(_PTRDIFF_T) || __has_feature(modules)
/* Always define ptrdiff_t when modules are available. */
#if !__has_feature(modules)
#define _PTRDIFF_T
#endif
# 51 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
typedef __PTRDIFF_TYPE__ ptrdiff_t;
#endif
# 53 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#undef __need_ptrdiff_t
#endif /* defined(__need_ptrdiff_t) */
# 55 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#if defined(__need_size_t)
#if !defined(_SIZE_T) || __has_feature(modules)
/* Always define size_t when modules are available. */
#if !__has_feature(modules)
#define _SIZE_T
#endif
# 62 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
typedef __SIZE_TYPE__ size_t;
#endif
# 64 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#undef __need_size_t
#endif /*defined(__need_size_t) */
# 66 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#if defined(__need_STDDEF_H_misc)
/* ISO9899:2011 7.20 (C11 Annex K): Define rsize_t if __STDC_WANT_LIB_EXT1__ is
 * enabled. */
#if (defined(__STDC_WANT_LIB_EXT1__) && __STDC_WANT_LIB_EXT1__ >= 1 && \
     !defined(_RSIZE_T)) || __has_feature(modules)
/* Always define rsize_t when modules are available. */
#if !__has_feature(modules)
#define _RSIZE_T
#endif
# 76 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
typedef __SIZE_TYPE__ rsize_t;
#endif
# 78 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#endif /* defined(__need_STDDEF_H_misc) */
# 79 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#if defined(__need_wchar_t)
#ifndef __cplusplus
/* Always define wchar_t when modules are available. */
#if !defined(_WCHAR_T) || __has_feature(modules)
#if !__has_feature(modules)
#define _WCHAR_T
#if defined(_MSC_EXTENSIONS)
#define _WCHAR_T_DEFINED
#endif
# 89 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#endif
# 90 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
typedef __WCHAR_TYPE__ wchar_t;
#endif
# 92 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#endif
# 93 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#undef __need_wchar_t
#endif /* defined(__need_wchar_t) */
# 95 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#if defined(__need_NULL)
#undef NULL
#ifdef __cplusplus
#  if !defined(__MINGW32__) && !defined(_MSC_VER)
#    define NULL __null
#  else
# 102 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#    define NULL 0
#  endif
# 104 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#else
# 105 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#  define NULL ((void*)0)
#endif
# 107 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#ifdef __cplusplus
#if defined(_MSC_EXTENSIONS) && defined(_NATIVE_NULLPTR_SUPPORTED)
namespace std { typedef decltype(nullptr) nullptr_t; }
using ::std::nullptr_t;
#endif
# 112 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#endif
# 113 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#undef __need_NULL
#endif /* defined(__need_NULL) */
# 115 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#if defined(__need_STDDEF_H_misc)
#if __STDC_VERSION__ >= 201112L || __cplusplus >= 201103L
#if 0 /* expanded by -frewrite-includes */
#include "__stddef_max_align_t.h"
#endif /* expanded by -frewrite-includes */
# 118 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
# 119 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#endif
# 120 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#define offsetof(t, d) __builtin_offsetof(t, d)
#undef __need_STDDEF_H_misc
#endif  /* defined(__need_STDDEF_H_misc) */
# 123 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

/* Some C libraries expect to see a wint_t here. Others (notably MinGW) will use
__WINT_TYPE__ directly; accommodate both by requiring __need_wint_t */
#if defined(__need_wint_t)
/* Always define wint_t when modules are available. */
#if !defined(_WINT_T) || __has_feature(modules)
#if !__has_feature(modules)
#define _WINT_T
#endif
# 132 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
typedef __WINT_TYPE__ wint_t;
#endif
# 134 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#undef __need_wint_t
#endif /* __need_wint_t */
# 136 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#endif
# 138 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
# 33 "/usr/include/stdlib.h" 2 3 4

__BEGIN_DECLS

#ifndef __need_malloc_and_calloc
#define	_STDLIB_H	1

#if (defined __USE_XOPEN || defined __USE_XOPEN2K8) && !defined _SYS_WAIT_H
/* XPG requires a few symbols from <sys/wait.h> being defined.  */
#if 0 /* expanded by -frewrite-includes */
# include <bits/waitflags.h>
#endif /* expanded by -frewrite-includes */
# 41 "/usr/include/stdlib.h" 3 4
# 42 "/usr/include/stdlib.h" 3 4
#if 0 /* expanded by -frewrite-includes */
# include <bits/waitstatus.h>
#endif /* expanded by -frewrite-includes */
# 42 "/usr/include/stdlib.h" 3 4
# 43 "/usr/include/stdlib.h" 3 4

# ifdef __USE_BSD

/* Lots of hair to allow traditional BSD use of `union wait'
   as well as POSIX.1 use of `int' for the status word.  */

#  if defined __GNUC__ && !defined __cplusplus
#   define __WAIT_INT(status) \
  (__extension__ (((union { __typeof(status) __in; int __i; }) \
		   { .__in = (status) }).__i))
#  else
# 54 "/usr/include/stdlib.h" 3 4
#   define __WAIT_INT(status)	(*(int *) &(status))
#  endif
# 56 "/usr/include/stdlib.h" 3 4

/* This is the type of the argument to `wait'.  The funky union
   causes redeclarations with either `int *' or `union wait *' to be
   allowed without complaint.  __WAIT_STATUS_DEFN is the type used in
   the actual function definitions.  */

#  if !defined __GNUC__ || __GNUC__ < 2 || defined __cplusplus
#   define __WAIT_STATUS	void *
#   define __WAIT_STATUS_DEFN	void *
#  else
# 66 "/usr/include/stdlib.h" 3 4
/* This works in GCC 2.6.1 and later.  */
typedef union
  {
    union wait *__uptr;
    int *__iptr;
  } __WAIT_STATUS __attribute__ ((__transparent_union__));
#   define __WAIT_STATUS_DEFN	int *
#  endif
# 74 "/usr/include/stdlib.h" 3 4

# else /* Don't use BSD.  */
# 76 "/usr/include/stdlib.h" 3 4

#  define __WAIT_INT(status)	(status)
#  define __WAIT_STATUS		int *
#  define __WAIT_STATUS_DEFN	int *

# endif /* Use BSD.  */
# 82 "/usr/include/stdlib.h" 3 4

/* Define the macros <sys/wait.h> also would define this way.  */
# define WEXITSTATUS(status)	__WEXITSTATUS (__WAIT_INT (status))
# define WTERMSIG(status)	__WTERMSIG (__WAIT_INT (status))
# define WSTOPSIG(status)	__WSTOPSIG (__WAIT_INT (status))
# define WIFEXITED(status)	__WIFEXITED (__WAIT_INT (status))
# define WIFSIGNALED(status)	__WIFSIGNALED (__WAIT_INT (status))
# define WIFSTOPPED(status)	__WIFSTOPPED (__WAIT_INT (status))
# ifdef __WIFCONTINUED
#  define WIFCONTINUED(status)	__WIFCONTINUED (__WAIT_INT (status))
# endif
# 93 "/usr/include/stdlib.h" 3 4
#endif	/* X/Open or XPG7 and <sys/wait.h> not included.  */
# 94 "/usr/include/stdlib.h" 3 4

__BEGIN_NAMESPACE_STD
/* Returned by `div'.  */
typedef struct
  {
    int quot;			/* Quotient.  */
    int rem;			/* Remainder.  */
  } div_t;

/* Returned by `ldiv'.  */
#ifndef __ldiv_t_defined
typedef struct
  {
    long int quot;		/* Quotient.  */
    long int rem;		/* Remainder.  */
  } ldiv_t;
# define __ldiv_t_defined	1
#endif
# 112 "/usr/include/stdlib.h" 3 4
__END_NAMESPACE_STD

#if defined __USE_ISOC99 && !defined __lldiv_t_defined
__BEGIN_NAMESPACE_C99
/* Returned by `lldiv'.  */
__extension__ typedef struct
  {
    long long int quot;		/* Quotient.  */
    long long int rem;		/* Remainder.  */
  } lldiv_t;
# define __lldiv_t_defined	1
__END_NAMESPACE_C99
#endif
# 125 "/usr/include/stdlib.h" 3 4


/* The largest number rand will return (same as INT_MAX).  */
#define	RAND_MAX	2147483647


/* We define these the same for all machines.
   Changes from this to the outside world should be done in `_exit'.  */
#define	EXIT_FAILURE	1	/* Failing exit status.  */
#define	EXIT_SUCCESS	0	/* Successful exit status.  */


/* Maximum length of a multibyte character in the current locale.  */
#define	MB_CUR_MAX	(__ctype_get_mb_cur_max ())
extern size_t __ctype_get_mb_cur_max (void) __THROW __wur;


__BEGIN_NAMESPACE_STD
/* Convert a string to a floating-point number.  */
extern double atof (const char *__nptr)
     __THROW __attribute_pure__ __nonnull ((1)) __wur;
/* Convert a string to an integer.  */
extern int atoi (const char *__nptr)
     __THROW __attribute_pure__ __nonnull ((1)) __wur;
/* Convert a string to a long integer.  */
extern long int atol (const char *__nptr)
     __THROW __attribute_pure__ __nonnull ((1)) __wur;
__END_NAMESPACE_STD

#if defined __USE_ISOC99 || defined __USE_MISC
__BEGIN_NAMESPACE_C99
/* Convert a string to a long long integer.  */
__extension__ extern long long int atoll (const char *__nptr)
     __THROW __attribute_pure__ __nonnull ((1)) __wur;
__END_NAMESPACE_C99
#endif
# 161 "/usr/include/stdlib.h" 3 4

__BEGIN_NAMESPACE_STD
/* Convert a string to a floating-point number.  */
extern double strtod (const char *__restrict __nptr,
		      char **__restrict __endptr)
     __THROW __nonnull ((1));
__END_NAMESPACE_STD

#ifdef	__USE_ISOC99
__BEGIN_NAMESPACE_C99
/* Likewise for `float' and `long double' sizes of floating-point numbers.  */
extern float strtof (const char *__restrict __nptr,
		     char **__restrict __endptr) __THROW __nonnull ((1));

extern long double strtold (const char *__restrict __nptr,
			    char **__restrict __endptr)
     __THROW __nonnull ((1));
__END_NAMESPACE_C99
#endif
# 180 "/usr/include/stdlib.h" 3 4

__BEGIN_NAMESPACE_STD
/* Convert a string to a long integer.  */
extern long int strtol (const char *__restrict __nptr,
			char **__restrict __endptr, int __base)
     __THROW __nonnull ((1));
/* Convert a string to an unsigned long integer.  */
extern unsigned long int strtoul (const char *__restrict __nptr,
				  char **__restrict __endptr, int __base)
     __THROW __nonnull ((1));
__END_NAMESPACE_STD

#ifdef __USE_BSD
/* Convert a string to a quadword integer.  */
__extension__
extern long long int strtoq (const char *__restrict __nptr,
			     char **__restrict __endptr, int __base)
     __THROW __nonnull ((1));
/* Convert a string to an unsigned quadword integer.  */
__extension__
extern unsigned long long int strtouq (const char *__restrict __nptr,
				       char **__restrict __endptr, int __base)
     __THROW __nonnull ((1));
#endif /* Use BSD.  */
# 204 "/usr/include/stdlib.h" 3 4

#if defined __USE_ISOC99 || defined __USE_MISC
__BEGIN_NAMESPACE_C99
/* Convert a string to a quadword integer.  */
__extension__
extern long long int strtoll (const char *__restrict __nptr,
			      char **__restrict __endptr, int __base)
     __THROW __nonnull ((1));
/* Convert a string to an unsigned quadword integer.  */
__extension__
extern unsigned long long int strtoull (const char *__restrict __nptr,
					char **__restrict __endptr, int __base)
     __THROW __nonnull ((1));
__END_NAMESPACE_C99
#endif /* ISO C99 or use MISC.  */
# 219 "/usr/include/stdlib.h" 3 4


#ifdef __USE_GNU
/* The concept of one static locale per category is not very well
   thought out.  Many applications will need to process its data using
   information from several different locales.  Another problem is
   the implementation of the internationalization handling in the
   ISO C++ standard library.  To support this another set of
   the functions using locale data exist which take an additional
   argument.

   Attention: even though several *_l interfaces are part of POSIX:2008,
   these are not.  */

/* Structure for reentrant locale using functions.  This is an
   (almost) opaque type for the user level programs.  */
#if 0 /* expanded by -frewrite-includes */
# include <xlocale.h>
#endif /* expanded by -frewrite-includes */
# 235 "/usr/include/stdlib.h" 3 4
# 236 "/usr/include/stdlib.h" 3 4

/* Special versions of the functions above which take the locale to
   use as an additional parameter.  */
extern long int strtol_l (const char *__restrict __nptr,
			  char **__restrict __endptr, int __base,
			  __locale_t __loc) __THROW __nonnull ((1, 4));

extern unsigned long int strtoul_l (const char *__restrict __nptr,
				    char **__restrict __endptr,
				    int __base, __locale_t __loc)
     __THROW __nonnull ((1, 4));

__extension__
extern long long int strtoll_l (const char *__restrict __nptr,
				char **__restrict __endptr, int __base,
				__locale_t __loc)
     __THROW __nonnull ((1, 4));

__extension__
extern unsigned long long int strtoull_l (const char *__restrict __nptr,
					  char **__restrict __endptr,
					  int __base, __locale_t __loc)
     __THROW __nonnull ((1, 4));

extern double strtod_l (const char *__restrict __nptr,
			char **__restrict __endptr, __locale_t __loc)
     __THROW __nonnull ((1, 3));

extern float strtof_l (const char *__restrict __nptr,
		       char **__restrict __endptr, __locale_t __loc)
     __THROW __nonnull ((1, 3));

extern long double strtold_l (const char *__restrict __nptr,
			      char **__restrict __endptr,
			      __locale_t __loc)
     __THROW __nonnull ((1, 3));
#endif /* GNU */
# 273 "/usr/include/stdlib.h" 3 4


#ifdef __USE_EXTERN_INLINES
__BEGIN_NAMESPACE_STD
__extern_inline int
__NTH (atoi (const char *__nptr))
{
  return (int) strtol (__nptr, (char **) NULL, 10);
}
__extern_inline long int
__NTH (atol (const char *__nptr))
{
  return strtol (__nptr, (char **) NULL, 10);
}
__END_NAMESPACE_STD

# if defined __USE_MISC || defined __USE_ISOC99
__BEGIN_NAMESPACE_C99
__extension__ __extern_inline long long int
__NTH (atoll (const char *__nptr))
{
  return strtoll (__nptr, (char **) NULL, 10);
}
__END_NAMESPACE_C99
# endif
# 298 "/usr/include/stdlib.h" 3 4
#endif /* Optimizing and Inlining.  */
# 299 "/usr/include/stdlib.h" 3 4


#if defined __USE_SVID || defined __USE_XOPEN_EXTENDED
/* Convert N to base 64 using the digits "./0-9A-Za-z", least-significant
   digit first.  Returns a pointer to static storage overwritten by the
   next call.  */
extern char *l64a (long int __n) __THROW __wur;

/* Read a number from a string S in base 64 as above.  */
extern long int a64l (const char *__s)
     __THROW __attribute_pure__ __nonnull ((1)) __wur;

#endif	/* Use SVID || extended X/Open.  */
# 312 "/usr/include/stdlib.h" 3 4

#if defined __USE_SVID || defined __USE_XOPEN_EXTENDED || defined __USE_BSD
#if 0 /* expanded by -frewrite-includes */
# include <sys/types.h>	/* we need int32_t... */
#endif /* expanded by -frewrite-includes */
# 314 "/usr/include/stdlib.h" 3 4
# 315 "/usr/include/stdlib.h" 3 4

/* These are the functions that actually do things.  The `random', `srandom',
   `initstate' and `setstate' functions are those from BSD Unices.
   The `rand' and `srand' functions are required by the ANSI standard.
   We provide both interfaces to the same random number generator.  */
/* Return a random long integer between 0 and RAND_MAX inclusive.  */
extern long int random (void) __THROW;

/* Seed the random number generator with the given number.  */
extern void srandom (unsigned int __seed) __THROW;

/* Initialize the random number generator to use state buffer STATEBUF,
   of length STATELEN, and seed it with SEED.  Optimal lengths are 8, 16,
   32, 64, 128 and 256, the bigger the better; values less than 8 will
   cause an error and values greater than 256 will be rounded down.  */
extern char *initstate (unsigned int __seed, char *__statebuf,
			size_t __statelen) __THROW __nonnull ((2));

/* Switch the random number generator to state buffer STATEBUF,
   which should have been previously initialized by `initstate'.  */
extern char *setstate (char *__statebuf) __THROW __nonnull ((1));


# ifdef __USE_MISC
/* Reentrant versions of the `random' family of functions.
   These functions all use the following data structure to contain
   state, rather than global state variables.  */

struct random_data
  {
    int32_t *fptr;		/* Front pointer.  */
    int32_t *rptr;		/* Rear pointer.  */
    int32_t *state;		/* Array of state values.  */
    int rand_type;		/* Type of random number generator.  */
    int rand_deg;		/* Degree of random number generator.  */
    int rand_sep;		/* Distance between front and rear.  */
    int32_t *end_ptr;		/* Pointer behind state table.  */
  };

extern int random_r (struct random_data *__restrict __buf,
		     int32_t *__restrict __result) __THROW __nonnull ((1, 2));

extern int srandom_r (unsigned int __seed, struct random_data *__buf)
     __THROW __nonnull ((2));

extern int initstate_r (unsigned int __seed, char *__restrict __statebuf,
			size_t __statelen,
			struct random_data *__restrict __buf)
     __THROW __nonnull ((2, 4));

extern int setstate_r (char *__restrict __statebuf,
		       struct random_data *__restrict __buf)
     __THROW __nonnull ((1, 2));
# endif	/* Use misc.  */
# 369 "/usr/include/stdlib.h" 3 4
#endif	/* Use SVID || extended X/Open || BSD. */
# 370 "/usr/include/stdlib.h" 3 4


__BEGIN_NAMESPACE_STD
/* Return a random integer between 0 and RAND_MAX inclusive.  */
extern int rand (void) __THROW;
/* Seed the random number generator with the given number.  */
extern void srand (unsigned int __seed) __THROW;
__END_NAMESPACE_STD

#ifdef __USE_POSIX
/* Reentrant interface according to POSIX.1.  */
extern int rand_r (unsigned int *__seed) __THROW;
#endif
# 383 "/usr/include/stdlib.h" 3 4


#if defined __USE_SVID || defined __USE_XOPEN
/* System V style 48-bit random number generator functions.  */

/* Return non-negative, double-precision floating-point value in [0.0,1.0).  */
extern double drand48 (void) __THROW;
extern double erand48 (unsigned short int __xsubi[3]) __THROW __nonnull ((1));

/* Return non-negative, long integer in [0,2^31).  */
extern long int lrand48 (void) __THROW;
extern long int nrand48 (unsigned short int __xsubi[3])
     __THROW __nonnull ((1));

/* Return signed, long integers in [-2^31,2^31).  */
extern long int mrand48 (void) __THROW;
extern long int jrand48 (unsigned short int __xsubi[3])
     __THROW __nonnull ((1));

/* Seed random number generator.  */
extern void srand48 (long int __seedval) __THROW;
extern unsigned short int *seed48 (unsigned short int __seed16v[3])
     __THROW __nonnull ((1));
extern void lcong48 (unsigned short int __param[7]) __THROW __nonnull ((1));

# ifdef __USE_MISC
/* Data structure for communication with thread safe versions.  This
   type is to be regarded as opaque.  It's only exported because users
   have to allocate objects of this type.  */
struct drand48_data
  {
    unsigned short int __x[3];	/* Current state.  */
    unsigned short int __old_x[3]; /* Old state.  */
    unsigned short int __c;	/* Additive const. in congruential formula.  */
    unsigned short int __init;	/* Flag for initializing.  */
    __extension__ unsigned long long int __a;	/* Factor in congruential
						   formula.  */
  };

/* Return non-negative, double-precision floating-point value in [0.0,1.0).  */
extern int drand48_r (struct drand48_data *__restrict __buffer,
		      double *__restrict __result) __THROW __nonnull ((1, 2));
extern int erand48_r (unsigned short int __xsubi[3],
		      struct drand48_data *__restrict __buffer,
		      double *__restrict __result) __THROW __nonnull ((1, 2));

/* Return non-negative, long integer in [0,2^31).  */
extern int lrand48_r (struct drand48_data *__restrict __buffer,
		      long int *__restrict __result)
     __THROW __nonnull ((1, 2));
extern int nrand48_r (unsigned short int __xsubi[3],
		      struct drand48_data *__restrict __buffer,
		      long int *__restrict __result)
     __THROW __nonnull ((1, 2));

/* Return signed, long integers in [-2^31,2^31).  */
extern int mrand48_r (struct drand48_data *__restrict __buffer,
		      long int *__restrict __result)
     __THROW __nonnull ((1, 2));
extern int jrand48_r (unsigned short int __xsubi[3],
		      struct drand48_data *__restrict __buffer,
		      long int *__restrict __result)
     __THROW __nonnull ((1, 2));

/* Seed random number generator.  */
extern int srand48_r (long int __seedval, struct drand48_data *__buffer)
     __THROW __nonnull ((2));

extern int seed48_r (unsigned short int __seed16v[3],
		     struct drand48_data *__buffer) __THROW __nonnull ((1, 2));

extern int lcong48_r (unsigned short int __param[7],
		      struct drand48_data *__buffer)
     __THROW __nonnull ((1, 2));
# endif	/* Use misc.  */
# 458 "/usr/include/stdlib.h" 3 4
#endif	/* Use SVID or X/Open.  */
# 459 "/usr/include/stdlib.h" 3 4

#endif /* don't just need malloc and calloc */
# 461 "/usr/include/stdlib.h" 3 4

#ifndef __malloc_and_calloc_defined
# define __malloc_and_calloc_defined
__BEGIN_NAMESPACE_STD
/* Allocate SIZE bytes of memory.  */
extern void *malloc (size_t __size) __THROW __attribute_malloc__ __wur;
/* Allocate NMEMB elements of SIZE bytes each, all initialized to 0.  */
extern void *calloc (size_t __nmemb, size_t __size)
     __THROW __attribute_malloc__ __wur;
__END_NAMESPACE_STD
#endif
# 472 "/usr/include/stdlib.h" 3 4

#ifndef __need_malloc_and_calloc
__BEGIN_NAMESPACE_STD
/* Re-allocate the previously allocated block
   in PTR, making the new block SIZE bytes long.  */
/* __attribute_malloc__ is not used, because if realloc returns
   the same pointer that was passed to it, aliasing needs to be allowed
   between objects pointed by the old and new pointers.  */
extern void *realloc (void *__ptr, size_t __size)
     __THROW __attribute_warn_unused_result__;
/* Free a block allocated by `malloc', `realloc' or `calloc'.  */
extern void free (void *__ptr) __THROW;
__END_NAMESPACE_STD

#ifdef	__USE_MISC
/* Free a block.  An alias for `free'.	(Sun Unices).  */
extern void cfree (void *__ptr) __THROW;
#endif /* Use misc.  */
# 490 "/usr/include/stdlib.h" 3 4

#if defined __USE_GNU || defined __USE_BSD || defined __USE_MISC
#if 0 /* expanded by -frewrite-includes */
# include <alloca.h>
#endif /* expanded by -frewrite-includes */
# 492 "/usr/include/stdlib.h" 3 4
# 493 "/usr/include/stdlib.h" 3 4
#endif /* Use GNU, BSD, or misc.  */
# 494 "/usr/include/stdlib.h" 3 4

#if (defined __USE_XOPEN_EXTENDED && !defined __USE_XOPEN2K) \
    || defined __USE_BSD
/* Allocate SIZE bytes on a page boundary.  The storage cannot be freed.  */
extern void *valloc (size_t __size) __THROW __attribute_malloc__ __wur;
#endif
# 500 "/usr/include/stdlib.h" 3 4

#ifdef __USE_XOPEN2K
/* Allocate memory of SIZE bytes with an alignment of ALIGNMENT.  */
extern int posix_memalign (void **__memptr, size_t __alignment, size_t __size)
     __THROW __nonnull ((1)) __wur;
#endif
# 506 "/usr/include/stdlib.h" 3 4

#ifdef __USE_ISOC11
/* ISO C variant of aligned allocation.  */
extern void *aligned_alloc (size_t __alignment, size_t __size)
     __THROW __attribute_malloc__ __attribute_alloc_size__ ((2)) __wur;
#endif
# 512 "/usr/include/stdlib.h" 3 4

__BEGIN_NAMESPACE_STD
/* Abort execution and generate a core-dump.  */
extern void abort (void) __THROW __attribute__ ((__noreturn__));


/* Register a function to be called when `exit' is called.  */
extern int atexit (void (*__func) (void)) __THROW __nonnull ((1));

#if defined __USE_ISOC11 || defined __USE_ISOCXX11
/* Register a function to be called when `quick_exit' is called.  */
# ifdef __cplusplus
extern "C++" int at_quick_exit (void (*__func) (void))
     __THROW __asm ("at_quick_exit") __nonnull ((1));
# else
# 527 "/usr/include/stdlib.h" 3 4
extern int at_quick_exit (void (*__func) (void)) __THROW __nonnull ((1));
# endif
# 529 "/usr/include/stdlib.h" 3 4
#endif
# 530 "/usr/include/stdlib.h" 3 4
__END_NAMESPACE_STD

#ifdef	__USE_MISC
/* Register a function to be called with the status
   given to `exit' and the given argument.  */
extern int on_exit (void (*__func) (int __status, void *__arg), void *__arg)
     __THROW __nonnull ((1));
#endif
# 538 "/usr/include/stdlib.h" 3 4

__BEGIN_NAMESPACE_STD
/* Call all functions registered with `atexit' and `on_exit',
   in the reverse of the order in which they were registered,
   perform stdio cleanup, and terminate program execution with STATUS.  */
extern void exit (int __status) __THROW __attribute__ ((__noreturn__));

#if defined __USE_ISOC11 || defined __USE_ISOCXX11
/* Call all functions registered with `at_quick_exit' in the reverse
   of the order in which they were registered and terminate program
   execution with STATUS.  */
extern void quick_exit (int __status) __THROW __attribute__ ((__noreturn__));
#endif
# 551 "/usr/include/stdlib.h" 3 4
__END_NAMESPACE_STD

#ifdef __USE_ISOC99
__BEGIN_NAMESPACE_C99
/* Terminate the program with STATUS without calling any of the
   functions registered with `atexit' or `on_exit'.  */
extern void _Exit (int __status) __THROW __attribute__ ((__noreturn__));
__END_NAMESPACE_C99
#endif
# 560 "/usr/include/stdlib.h" 3 4


__BEGIN_NAMESPACE_STD
/* Return the value of envariable NAME, or NULL if it doesn't exist.  */
extern char *getenv (const char *__name) __THROW __nonnull ((1)) __wur;
__END_NAMESPACE_STD

#ifdef __USE_GNU
/* This function is similar to the above but returns NULL if the
   programs is running with SUID or SGID enabled.  */
extern char *secure_getenv (const char *__name)
     __THROW __nonnull ((1)) __wur;
#endif
# 573 "/usr/include/stdlib.h" 3 4

#if defined __USE_SVID || defined __USE_XOPEN
/* The SVID says this is in <stdio.h>, but this seems a better place.	*/
/* Put STRING, which is of the form "NAME=VALUE", in the environment.
   If there is no `=', remove NAME from the environment.  */
extern int putenv (char *__string) __THROW __nonnull ((1));
#endif
# 580 "/usr/include/stdlib.h" 3 4

#if defined __USE_BSD || defined __USE_XOPEN2K
/* Set NAME to VALUE in the environment.
   If REPLACE is nonzero, overwrite an existing value.  */
extern int setenv (const char *__name, const char *__value, int __replace)
     __THROW __nonnull ((2));

/* Remove the variable NAME from the environment.  */
extern int unsetenv (const char *__name) __THROW __nonnull ((1));
#endif
# 590 "/usr/include/stdlib.h" 3 4

#ifdef	__USE_MISC
/* The `clearenv' was planned to be added to POSIX.1 but probably
   never made it.  Nevertheless the POSIX.9 standard (POSIX bindings
   for Fortran 77) requires this function.  */
extern int clearenv (void) __THROW;
#endif
# 597 "/usr/include/stdlib.h" 3 4


#if defined __USE_MISC \
    || (defined __USE_XOPEN_EXTENDED && !defined __USE_XOPEN2K8)
/* Generate a unique temporary file name from TEMPLATE.
   The last six characters of TEMPLATE must be "XXXXXX";
   they are replaced with a string that makes the file name unique.
   Always returns TEMPLATE, it's either a temporary file name or a null
   string if it cannot get a unique file name.  */
extern char *mktemp (char *__template) __THROW __nonnull ((1));
#endif
# 608 "/usr/include/stdlib.h" 3 4

#if defined __USE_MISC || defined __USE_XOPEN_EXTENDED \
    || defined __USE_XOPEN2K8
/* Generate a unique temporary file name from TEMPLATE.
   The last six characters of TEMPLATE must be "XXXXXX";
   they are replaced with a string that makes the filename unique.
   Returns a file descriptor open on the file for reading and writing,
   or -1 if it cannot create a uniquely-named file.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
# ifndef __USE_FILE_OFFSET64
extern int mkstemp (char *__template) __nonnull ((1)) __wur;
# else
# 622 "/usr/include/stdlib.h" 3 4
#  ifdef __REDIRECT
extern int __REDIRECT (mkstemp, (char *__template), mkstemp64)
     __nonnull ((1)) __wur;
#  else
# 626 "/usr/include/stdlib.h" 3 4
#   define mkstemp mkstemp64
#  endif
# 628 "/usr/include/stdlib.h" 3 4
# endif
# 629 "/usr/include/stdlib.h" 3 4
# ifdef __USE_LARGEFILE64
extern int mkstemp64 (char *__template) __nonnull ((1)) __wur;
# endif
# 632 "/usr/include/stdlib.h" 3 4
#endif
# 633 "/usr/include/stdlib.h" 3 4

#ifdef __USE_MISC
/* Similar to mkstemp, but the template can have a suffix after the
   XXXXXX.  The length of the suffix is specified in the second
   parameter.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
# ifndef __USE_FILE_OFFSET64
extern int mkstemps (char *__template, int __suffixlen) __nonnull ((1)) __wur;
# else
# 644 "/usr/include/stdlib.h" 3 4
#  ifdef __REDIRECT
extern int __REDIRECT (mkstemps, (char *__template, int __suffixlen),
		       mkstemps64) __nonnull ((1)) __wur;
#  else
# 648 "/usr/include/stdlib.h" 3 4
#   define mkstemps mkstemps64
#  endif
# 650 "/usr/include/stdlib.h" 3 4
# endif
# 651 "/usr/include/stdlib.h" 3 4
# ifdef __USE_LARGEFILE64
extern int mkstemps64 (char *__template, int __suffixlen)
     __nonnull ((1)) __wur;
# endif
# 655 "/usr/include/stdlib.h" 3 4
#endif
# 656 "/usr/include/stdlib.h" 3 4

#if defined __USE_BSD || defined __USE_XOPEN2K8
/* Create a unique temporary directory from TEMPLATE.
   The last six characters of TEMPLATE must be "XXXXXX";
   they are replaced with a string that makes the directory name unique.
   Returns TEMPLATE, or a null pointer if it cannot get a unique name.
   The directory is created mode 700.  */
extern char *mkdtemp (char *__template) __THROW __nonnull ((1)) __wur;
#endif
# 665 "/usr/include/stdlib.h" 3 4

#ifdef __USE_GNU
/* Generate a unique temporary file name from TEMPLATE similar to
   mkstemp.  But allow the caller to pass additional flags which are
   used in the open call to create the file..

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
# ifndef __USE_FILE_OFFSET64
extern int mkostemp (char *__template, int __flags) __nonnull ((1)) __wur;
# else
# 676 "/usr/include/stdlib.h" 3 4
#  ifdef __REDIRECT
extern int __REDIRECT (mkostemp, (char *__template, int __flags), mkostemp64)
     __nonnull ((1)) __wur;
#  else
# 680 "/usr/include/stdlib.h" 3 4
#   define mkostemp mkostemp64
#  endif
# 682 "/usr/include/stdlib.h" 3 4
# endif
# 683 "/usr/include/stdlib.h" 3 4
# ifdef __USE_LARGEFILE64
extern int mkostemp64 (char *__template, int __flags) __nonnull ((1)) __wur;
# endif
# 686 "/usr/include/stdlib.h" 3 4

/* Similar to mkostemp, but the template can have a suffix after the
   XXXXXX.  The length of the suffix is specified in the second
   parameter.

   This function is a possible cancellation point and therefore not
   marked with __THROW.  */
# ifndef __USE_FILE_OFFSET64
extern int mkostemps (char *__template, int __suffixlen, int __flags)
     __nonnull ((1)) __wur;
# else
# 697 "/usr/include/stdlib.h" 3 4
#  ifdef __REDIRECT
extern int __REDIRECT (mkostemps, (char *__template, int __suffixlen,
				   int __flags), mkostemps64)
     __nonnull ((1)) __wur;
#  else
# 702 "/usr/include/stdlib.h" 3 4
#   define mkostemps mkostemps64
#  endif
# 704 "/usr/include/stdlib.h" 3 4
# endif
# 705 "/usr/include/stdlib.h" 3 4
# ifdef __USE_LARGEFILE64
extern int mkostemps64 (char *__template, int __suffixlen, int __flags)
     __nonnull ((1)) __wur;
# endif
# 709 "/usr/include/stdlib.h" 3 4
#endif
# 710 "/usr/include/stdlib.h" 3 4


__BEGIN_NAMESPACE_STD
/* Execute the given line as a shell command.

   This function is a cancellation point and therefore not marked with
   __THROW.  */
extern int system (const char *__command) __wur;
__END_NAMESPACE_STD


#ifdef	__USE_GNU
/* Return a malloc'd string containing the canonical absolute name of the
   existing named file.  */
extern char *canonicalize_file_name (const char *__name)
     __THROW __nonnull ((1)) __wur;
#endif
# 727 "/usr/include/stdlib.h" 3 4

#if defined __USE_BSD || defined __USE_XOPEN_EXTENDED
/* Return the canonical absolute name of file NAME.  If RESOLVED is
   null, the result is malloc'd; otherwise, if the canonical name is
   PATH_MAX chars or more, returns null with `errno' set to
   ENAMETOOLONG; if the name fits in fewer than PATH_MAX chars,
   returns the name in RESOLVED.  */
extern char *realpath (const char *__restrict __name,
		       char *__restrict __resolved) __THROW __wur;
#endif
# 737 "/usr/include/stdlib.h" 3 4


/* Shorthand for type of comparison functions.  */
#ifndef __COMPAR_FN_T
# define __COMPAR_FN_T
typedef int (*__compar_fn_t) (const void *, const void *);

# ifdef	__USE_GNU
typedef __compar_fn_t comparison_fn_t;
# endif
# 747 "/usr/include/stdlib.h" 3 4
#endif
# 748 "/usr/include/stdlib.h" 3 4
#ifdef __USE_GNU
typedef int (*__compar_d_fn_t) (const void *, const void *, void *);
#endif
# 751 "/usr/include/stdlib.h" 3 4

__BEGIN_NAMESPACE_STD
/* Do a binary search for KEY in BASE, which consists of NMEMB elements
   of SIZE bytes each, using COMPAR to perform the comparisons.  */
extern void *bsearch (const void *__key, const void *__base,
		      size_t __nmemb, size_t __size, __compar_fn_t __compar)
     __nonnull ((1, 2, 5)) __wur;

#ifdef __USE_EXTERN_INLINES
#if 0 /* expanded by -frewrite-includes */
# include <bits/stdlib-bsearch.h>
#endif /* expanded by -frewrite-includes */
# 760 "/usr/include/stdlib.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/stdlib-bsearch.h" 1 3 4
/* Perform binary search - inline version.
   Copyright (C) 1991-2014 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */

__extern_inline void *
bsearch (const void *__key, const void *__base, size_t __nmemb, size_t __size,
	 __compar_fn_t __compar)
{
  size_t __l, __u, __idx;
  const void *__p;
  int __comparison;

  __l = 0;
  __u = __nmemb;
  while (__l < __u)
    {
      __idx = (__l + __u) / 2;
      __p = (void *) (((const char *) __base) + (__idx * __size));
      __comparison = (*__compar) (__key, __p);
      if (__comparison < 0)
	__u = __idx;
      else if (__comparison > 0)
	__l = __idx + 1;
      else
	return (void *) __p;
    }

  return NULL;
}
# 761 "/usr/include/stdlib.h" 2 3 4
#endif
# 762 "/usr/include/stdlib.h" 3 4

/* Sort NMEMB elements of BASE, of SIZE bytes each,
   using COMPAR to perform the comparisons.  */
extern void qsort (void *__base, size_t __nmemb, size_t __size,
		   __compar_fn_t __compar) __nonnull ((1, 4));
#ifdef __USE_GNU
extern void qsort_r (void *__base, size_t __nmemb, size_t __size,
		     __compar_d_fn_t __compar, void *__arg)
  __nonnull ((1, 4));
#endif
# 772 "/usr/include/stdlib.h" 3 4


/* Return the absolute value of X.  */
extern int abs (int __x) __THROW __attribute__ ((__const__)) __wur;
extern long int labs (long int __x) __THROW __attribute__ ((__const__)) __wur;
__END_NAMESPACE_STD

#ifdef __USE_ISOC99
__extension__ extern long long int llabs (long long int __x)
     __THROW __attribute__ ((__const__)) __wur;
#endif
# 783 "/usr/include/stdlib.h" 3 4


__BEGIN_NAMESPACE_STD
/* Return the `div_t', `ldiv_t' or `lldiv_t' representation
   of the value of NUMER over DENOM. */
/* GCC may have built-ins for these someday.  */
extern div_t div (int __numer, int __denom)
     __THROW __attribute__ ((__const__)) __wur;
extern ldiv_t ldiv (long int __numer, long int __denom)
     __THROW __attribute__ ((__const__)) __wur;
__END_NAMESPACE_STD

#ifdef __USE_ISOC99
__BEGIN_NAMESPACE_C99
__extension__ extern lldiv_t lldiv (long long int __numer,
				    long long int __denom)
     __THROW __attribute__ ((__const__)) __wur;
__END_NAMESPACE_C99
#endif
# 802 "/usr/include/stdlib.h" 3 4


#if (defined __USE_XOPEN_EXTENDED && !defined __USE_XOPEN2K8) \
    || defined __USE_SVID
/* Convert floating point numbers to strings.  The returned values are
   valid only until another call to the same function.  */

/* Convert VALUE to a string with NDIGIT digits and return a pointer to
   this.  Set *DECPT with the position of the decimal character and *SIGN
   with the sign of the number.  */
extern char *ecvt (double __value, int __ndigit, int *__restrict __decpt,
		   int *__restrict __sign) __THROW __nonnull ((3, 4)) __wur;

/* Convert VALUE to a string rounded to NDIGIT decimal digits.  Set *DECPT
   with the position of the decimal character and *SIGN with the sign of
   the number.  */
extern char *fcvt (double __value, int __ndigit, int *__restrict __decpt,
		   int *__restrict __sign) __THROW __nonnull ((3, 4)) __wur;

/* If possible convert VALUE to a string with NDIGIT significant digits.
   Otherwise use exponential representation.  The resulting string will
   be written to BUF.  */
extern char *gcvt (double __value, int __ndigit, char *__buf)
     __THROW __nonnull ((3)) __wur;
#endif
# 827 "/usr/include/stdlib.h" 3 4

#ifdef __USE_MISC
/* Long double versions of above functions.  */
extern char *qecvt (long double __value, int __ndigit,
		    int *__restrict __decpt, int *__restrict __sign)
     __THROW __nonnull ((3, 4)) __wur;
extern char *qfcvt (long double __value, int __ndigit,
		    int *__restrict __decpt, int *__restrict __sign)
     __THROW __nonnull ((3, 4)) __wur;
extern char *qgcvt (long double __value, int __ndigit, char *__buf)
     __THROW __nonnull ((3)) __wur;


/* Reentrant version of the functions above which provide their own
   buffers.  */
extern int ecvt_r (double __value, int __ndigit, int *__restrict __decpt,
		   int *__restrict __sign, char *__restrict __buf,
		   size_t __len) __THROW __nonnull ((3, 4, 5));
extern int fcvt_r (double __value, int __ndigit, int *__restrict __decpt,
		   int *__restrict __sign, char *__restrict __buf,
		   size_t __len) __THROW __nonnull ((3, 4, 5));

extern int qecvt_r (long double __value, int __ndigit,
		    int *__restrict __decpt, int *__restrict __sign,
		    char *__restrict __buf, size_t __len)
     __THROW __nonnull ((3, 4, 5));
extern int qfcvt_r (long double __value, int __ndigit,
		    int *__restrict __decpt, int *__restrict __sign,
		    char *__restrict __buf, size_t __len)
     __THROW __nonnull ((3, 4, 5));
#endif	/* misc */
# 858 "/usr/include/stdlib.h" 3 4


__BEGIN_NAMESPACE_STD
/* Return the length of the multibyte character
   in S, which is no longer than N.  */
extern int mblen (const char *__s, size_t __n) __THROW;
/* Return the length of the given multibyte character,
   putting its `wchar_t' representation in *PWC.  */
extern int mbtowc (wchar_t *__restrict __pwc,
		   const char *__restrict __s, size_t __n) __THROW;
/* Put the multibyte character represented
   by WCHAR in S, returning its length.  */
extern int wctomb (char *__s, wchar_t __wchar) __THROW;


/* Convert a multibyte string to a wide char string.  */
extern size_t mbstowcs (wchar_t *__restrict  __pwcs,
			const char *__restrict __s, size_t __n) __THROW;
/* Convert a wide char string to multibyte string.  */
extern size_t wcstombs (char *__restrict __s,
			const wchar_t *__restrict __pwcs, size_t __n)
     __THROW;
__END_NAMESPACE_STD


#ifdef __USE_SVID
/* Determine whether the string value of RESPONSE matches the affirmation
   or negative response expression as specified by the LC_MESSAGES category
   in the program's current locale.  Returns 1 if affirmative, 0 if
   negative, and -1 if not matching.  */
extern int rpmatch (const char *__response) __THROW __nonnull ((1)) __wur;
#endif
# 890 "/usr/include/stdlib.h" 3 4


#if defined __USE_XOPEN_EXTENDED || defined __USE_XOPEN2K8
/* Parse comma separated suboption from *OPTIONP and match against
   strings in TOKENS.  If found return index and set *VALUEP to
   optional value introduced by an equal sign.  If the suboption is
   not part of TOKENS return in *VALUEP beginning of unknown
   suboption.  On exit *OPTIONP is set to the beginning of the next
   token or at the terminating NUL character.  */
extern int getsubopt (char **__restrict __optionp,
		      char *const *__restrict __tokens,
		      char **__restrict __valuep)
     __THROW __nonnull ((1, 2, 3)) __wur;
#endif
# 904 "/usr/include/stdlib.h" 3 4


#ifdef __USE_XOPEN
/* Setup DES tables according KEY.  */
extern void setkey (const char *__key) __THROW __nonnull ((1));
#endif
# 910 "/usr/include/stdlib.h" 3 4


/* X/Open pseudo terminal handling.  */

#ifdef __USE_XOPEN2KXSI
/* Return a master pseudo-terminal handle.  */
extern int posix_openpt (int __oflag) __wur;
#endif
# 918 "/usr/include/stdlib.h" 3 4

#ifdef __USE_XOPEN
/* The next four functions all take a master pseudo-tty fd and
   perform an operation on the associated slave:  */

/* Chown the slave to the calling user.  */
extern int grantpt (int __fd) __THROW;

/* Release an internal lock so the slave can be opened.
   Call after grantpt().  */
extern int unlockpt (int __fd) __THROW;

/* Return the pathname of the pseudo terminal slave associated with
   the master FD is open on, or NULL on errors.
   The returned storage is good until the next call to this function.  */
extern char *ptsname (int __fd) __THROW __wur;
#endif
# 935 "/usr/include/stdlib.h" 3 4

#ifdef __USE_GNU
/* Store at most BUFLEN characters of the pathname of the slave pseudo
   terminal associated with the master FD is open on in BUF.
   Return 0 on success, otherwise an error number.  */
extern int ptsname_r (int __fd, char *__buf, size_t __buflen)
     __THROW __nonnull ((2));

/* Open a master pseudo terminal and return its file descriptor.  */
extern int getpt (void);
#endif
# 946 "/usr/include/stdlib.h" 3 4

#ifdef __USE_BSD
/* Put the 1 minute, 5 minute and 15 minute load averages into the first
   NELEM elements of LOADAVG.  Return the number written (never more than
   three, but may be less than NELEM), or -1 if an error occurred.  */
extern int getloadavg (double __loadavg[], int __nelem)
     __THROW __nonnull ((1));
#endif
# 954 "/usr/include/stdlib.h" 3 4

#if 0 /* expanded by -frewrite-includes */
#include <bits/stdlib-float.h>
#endif /* expanded by -frewrite-includes */
# 955 "/usr/include/stdlib.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/stdlib-float.h" 1 3 4
/* Floating-point inline functions for stdlib.h.
   Copyright (C) 2012-2014 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */

#ifndef _STDLIB_H
# error "Never use <bits/stdlib-float.h> directly; include <stdlib.h> instead."
#endif
# 22 "/usr/include/x86_64-linux-gnu/bits/stdlib-float.h" 3 4

#ifdef __USE_EXTERN_INLINES
__BEGIN_NAMESPACE_STD
__extern_inline double
__NTH (atof (const char *__nptr))
{
  return strtod (__nptr, (char **) NULL);
}
__END_NAMESPACE_STD
#endif /* Optimizing and Inlining.  */
# 32 "/usr/include/x86_64-linux-gnu/bits/stdlib-float.h" 3 4
# 956 "/usr/include/stdlib.h" 2 3 4

/* Define some macros helping to catch buffer overflows.  */
#if __USE_FORTIFY_LEVEL > 0 && defined __fortify_function
#if 0 /* expanded by -frewrite-includes */
# include <bits/stdlib.h>
#endif /* expanded by -frewrite-includes */
# 959 "/usr/include/stdlib.h" 3 4
# 960 "/usr/include/stdlib.h" 3 4
#endif
# 961 "/usr/include/stdlib.h" 3 4
#ifdef __LDBL_COMPAT
#if 0 /* expanded by -frewrite-includes */
# include <bits/stdlib-ldbl.h>
#endif /* expanded by -frewrite-includes */
# 962 "/usr/include/stdlib.h" 3 4
# 963 "/usr/include/stdlib.h" 3 4
#endif
# 964 "/usr/include/stdlib.h" 3 4

#endif /* don't just need malloc and calloc */
# 966 "/usr/include/stdlib.h" 3 4
#undef __need_malloc_and_calloc

__END_DECLS

#endif /* stdlib.h  */
# 971 "/usr/include/stdlib.h" 3 4
# 4 "oski.c" 2
#if 0 /* expanded by -frewrite-includes */
#include <string.h>
#endif /* expanded by -frewrite-includes */
# 4 "oski.c"
# 1 "/usr/include/string.h" 1 3 4
/* Copyright (C) 1991-2014 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */

/*
 *	ISO C99 Standard: 7.21 String handling	<string.h>
 */

#ifndef	_STRING_H
#define	_STRING_H	1

#if 0 /* expanded by -frewrite-includes */
#include <features.h>
#endif /* expanded by -frewrite-includes */
# 25 "/usr/include/string.h" 3 4
# 26 "/usr/include/string.h" 3 4

__BEGIN_DECLS

/* Get size_t and NULL from <stddef.h>.  */
#define	__need_size_t
#define	__need_NULL
#if 0 /* expanded by -frewrite-includes */
#include <stddef.h>
#endif /* expanded by -frewrite-includes */
# 32 "/usr/include/string.h" 3 4
# 1 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 1 3 4
/*===---- stddef.h - Basic type definitions --------------------------------===
 *
 * Copyright (c) 2008 Eli Friedman
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *===-----------------------------------------------------------------------===
 */

#if !defined(__STDDEF_H) || defined(__need_ptrdiff_t) ||                       \
    defined(__need_size_t) || defined(__need_wchar_t) ||                       \
    defined(__need_NULL) || defined(__need_wint_t)

#if !defined(__need_ptrdiff_t) && !defined(__need_size_t) &&                   \
    !defined(__need_wchar_t) && !defined(__need_NULL) &&                       \
    !defined(__need_wint_t)
/* Always define miscellaneous pieces when modules are available. */
#if !__has_feature(modules)
#define __STDDEF_H
#endif
# 37 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#define __need_ptrdiff_t
#define __need_size_t
#define __need_wchar_t
#define __need_NULL
#define __need_STDDEF_H_misc
/* __need_wint_t is intentionally not defined here. */
#endif
# 44 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#if defined(__need_ptrdiff_t)
#if !defined(_PTRDIFF_T) || __has_feature(modules)
/* Always define ptrdiff_t when modules are available. */
#if !__has_feature(modules)
#define _PTRDIFF_T
#endif
# 51 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
typedef __PTRDIFF_TYPE__ ptrdiff_t;
#endif
# 53 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#undef __need_ptrdiff_t
#endif /* defined(__need_ptrdiff_t) */
# 55 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#if defined(__need_size_t)
#if !defined(_SIZE_T) || __has_feature(modules)
/* Always define size_t when modules are available. */
#if !__has_feature(modules)
#define _SIZE_T
#endif
# 62 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
typedef __SIZE_TYPE__ size_t;
#endif
# 64 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#undef __need_size_t
#endif /*defined(__need_size_t) */
# 66 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#if defined(__need_STDDEF_H_misc)
/* ISO9899:2011 7.20 (C11 Annex K): Define rsize_t if __STDC_WANT_LIB_EXT1__ is
 * enabled. */
#if (defined(__STDC_WANT_LIB_EXT1__) && __STDC_WANT_LIB_EXT1__ >= 1 && \
     !defined(_RSIZE_T)) || __has_feature(modules)
/* Always define rsize_t when modules are available. */
#if !__has_feature(modules)
#define _RSIZE_T
#endif
# 76 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
typedef __SIZE_TYPE__ rsize_t;
#endif
# 78 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#endif /* defined(__need_STDDEF_H_misc) */
# 79 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#if defined(__need_wchar_t)
#ifndef __cplusplus
/* Always define wchar_t when modules are available. */
#if !defined(_WCHAR_T) || __has_feature(modules)
#if !__has_feature(modules)
#define _WCHAR_T
#if defined(_MSC_EXTENSIONS)
#define _WCHAR_T_DEFINED
#endif
# 89 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#endif
# 90 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
typedef __WCHAR_TYPE__ wchar_t;
#endif
# 92 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#endif
# 93 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#undef __need_wchar_t
#endif /* defined(__need_wchar_t) */
# 95 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#if defined(__need_NULL)
#undef NULL
#ifdef __cplusplus
#  if !defined(__MINGW32__) && !defined(_MSC_VER)
#    define NULL __null
#  else
# 102 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#    define NULL 0
#  endif
# 104 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#else
# 105 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#  define NULL ((void*)0)
#endif
# 107 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#ifdef __cplusplus
#if defined(_MSC_EXTENSIONS) && defined(_NATIVE_NULLPTR_SUPPORTED)
namespace std { typedef decltype(nullptr) nullptr_t; }
using ::std::nullptr_t;
#endif
# 112 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#endif
# 113 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#undef __need_NULL
#endif /* defined(__need_NULL) */
# 115 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#if defined(__need_STDDEF_H_misc)
#if __STDC_VERSION__ >= 201112L || __cplusplus >= 201103L
#if 0 /* expanded by -frewrite-includes */
#include "__stddef_max_align_t.h"
#endif /* expanded by -frewrite-includes */
# 118 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
# 119 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#endif
# 120 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#define offsetof(t, d) __builtin_offsetof(t, d)
#undef __need_STDDEF_H_misc
#endif  /* defined(__need_STDDEF_H_misc) */
# 123 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

/* Some C libraries expect to see a wint_t here. Others (notably MinGW) will use
__WINT_TYPE__ directly; accommodate both by requiring __need_wint_t */
#if defined(__need_wint_t)
/* Always define wint_t when modules are available. */
#if !defined(_WINT_T) || __has_feature(modules)
#if !__has_feature(modules)
#define _WINT_T
#endif
# 132 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
typedef __WINT_TYPE__ wint_t;
#endif
# 134 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
#undef __need_wint_t
#endif /* __need_wint_t */
# 136 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4

#endif
# 138 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/stddef.h" 3 4
# 33 "/usr/include/string.h" 2 3 4

/* Provide correct C++ prototypes, and indicate this to the caller.  This
   requires a compatible C++ standard library.  As a heuristic, we provide
   these when the compiler indicates full conformance with C++98 or later,
   and for older GCC versions that are known to provide a compatible
   libstdc++.  */
#if defined __cplusplus && (__cplusplus >= 199711L || __GNUC_PREREQ (4, 4))
# define __CORRECT_ISO_CPP_STRING_H_PROTO
#endif
# 42 "/usr/include/string.h" 3 4


__BEGIN_NAMESPACE_STD
/* Copy N bytes of SRC to DEST.  */
extern void *memcpy (void *__restrict __dest, const void *__restrict __src,
		     size_t __n) __THROW __nonnull ((1, 2));
/* Copy N bytes of SRC to DEST, guaranteeing
   correct behavior for overlapping strings.  */
extern void *memmove (void *__dest, const void *__src, size_t __n)
     __THROW __nonnull ((1, 2));
__END_NAMESPACE_STD

/* Copy no more than N bytes of SRC to DEST, stopping when C is found.
   Return the position in DEST one byte past where C was copied,
   or NULL if C was not found in the first N bytes of SRC.  */
#if defined __USE_SVID || defined __USE_BSD || defined __USE_XOPEN
extern void *memccpy (void *__restrict __dest, const void *__restrict __src,
		      int __c, size_t __n)
     __THROW __nonnull ((1, 2));
#endif /* SVID.  */
# 62 "/usr/include/string.h" 3 4


__BEGIN_NAMESPACE_STD
/* Set N bytes of S to C.  */
extern void *memset (void *__s, int __c, size_t __n) __THROW __nonnull ((1));

/* Compare N bytes of S1 and S2.  */
extern int memcmp (const void *__s1, const void *__s2, size_t __n)
     __THROW __attribute_pure__ __nonnull ((1, 2));

/* Search N bytes of S for C.  */
#ifdef __CORRECT_ISO_CPP_STRING_H_PROTO
extern "C++"
{
extern void *memchr (void *__s, int __c, size_t __n)
      __THROW __asm ("memchr") __attribute_pure__ __nonnull ((1));
extern const void *memchr (const void *__s, int __c, size_t __n)
      __THROW __asm ("memchr") __attribute_pure__ __nonnull ((1));

# ifdef __OPTIMIZE__
__extern_always_inline void *
memchr (void *__s, int __c, size_t __n) __THROW
{
  return __builtin_memchr (__s, __c, __n);
}

__extern_always_inline const void *
memchr (const void *__s, int __c, size_t __n) __THROW
{
  return __builtin_memchr (__s, __c, __n);
}
# endif
# 94 "/usr/include/string.h" 3 4
}
#else
# 96 "/usr/include/string.h" 3 4
extern void *memchr (const void *__s, int __c, size_t __n)
      __THROW __attribute_pure__ __nonnull ((1));
#endif
# 99 "/usr/include/string.h" 3 4
__END_NAMESPACE_STD

#ifdef __USE_GNU
/* Search in S for C.  This is similar to `memchr' but there is no
   length limit.  */
# ifdef __CORRECT_ISO_CPP_STRING_H_PROTO
extern "C++" void *rawmemchr (void *__s, int __c)
     __THROW __asm ("rawmemchr") __attribute_pure__ __nonnull ((1));
extern "C++" const void *rawmemchr (const void *__s, int __c)
     __THROW __asm ("rawmemchr") __attribute_pure__ __nonnull ((1));
# else
# 110 "/usr/include/string.h" 3 4
extern void *rawmemchr (const void *__s, int __c)
     __THROW __attribute_pure__ __nonnull ((1));
# endif
# 113 "/usr/include/string.h" 3 4

/* Search N bytes of S for the final occurrence of C.  */
# ifdef __CORRECT_ISO_CPP_STRING_H_PROTO
extern "C++" void *memrchr (void *__s, int __c, size_t __n)
      __THROW __asm ("memrchr") __attribute_pure__ __nonnull ((1));
extern "C++" const void *memrchr (const void *__s, int __c, size_t __n)
      __THROW __asm ("memrchr") __attribute_pure__ __nonnull ((1));
# else
# 121 "/usr/include/string.h" 3 4
extern void *memrchr (const void *__s, int __c, size_t __n)
      __THROW __attribute_pure__ __nonnull ((1));
# endif
# 124 "/usr/include/string.h" 3 4
#endif
# 125 "/usr/include/string.h" 3 4


__BEGIN_NAMESPACE_STD
/* Copy SRC to DEST.  */
extern char *strcpy (char *__restrict __dest, const char *__restrict __src)
     __THROW __nonnull ((1, 2));
/* Copy no more than N characters of SRC to DEST.  */
extern char *strncpy (char *__restrict __dest,
		      const char *__restrict __src, size_t __n)
     __THROW __nonnull ((1, 2));

/* Append SRC onto DEST.  */
extern char *strcat (char *__restrict __dest, const char *__restrict __src)
     __THROW __nonnull ((1, 2));
/* Append no more than N characters from SRC onto DEST.  */
extern char *strncat (char *__restrict __dest, const char *__restrict __src,
		      size_t __n) __THROW __nonnull ((1, 2));

/* Compare S1 and S2.  */
extern int strcmp (const char *__s1, const char *__s2)
     __THROW __attribute_pure__ __nonnull ((1, 2));
/* Compare N characters of S1 and S2.  */
extern int strncmp (const char *__s1, const char *__s2, size_t __n)
     __THROW __attribute_pure__ __nonnull ((1, 2));

/* Compare the collated forms of S1 and S2.  */
extern int strcoll (const char *__s1, const char *__s2)
     __THROW __attribute_pure__ __nonnull ((1, 2));
/* Put a transformation of SRC into no more than N bytes of DEST.  */
extern size_t strxfrm (char *__restrict __dest,
		       const char *__restrict __src, size_t __n)
     __THROW __nonnull ((2));
__END_NAMESPACE_STD

#ifdef __USE_XOPEN2K8
/* The following functions are equivalent to the both above but they
   take the locale they use for the collation as an extra argument.
   This is not standardsized but something like will come.  */
#if 0 /* expanded by -frewrite-includes */
# include <xlocale.h>
#endif /* expanded by -frewrite-includes */
# 163 "/usr/include/string.h" 3 4
# 164 "/usr/include/string.h" 3 4

/* Compare the collated forms of S1 and S2 using rules from L.  */
extern int strcoll_l (const char *__s1, const char *__s2, __locale_t __l)
     __THROW __attribute_pure__ __nonnull ((1, 2, 3));
/* Put a transformation of SRC into no more than N bytes of DEST.  */
extern size_t strxfrm_l (char *__dest, const char *__src, size_t __n,
			 __locale_t __l) __THROW __nonnull ((2, 4));
#endif
# 172 "/usr/include/string.h" 3 4

#if defined __USE_SVID || defined __USE_BSD || defined __USE_XOPEN_EXTENDED \
    || defined __USE_XOPEN2K8
/* Duplicate S, returning an identical malloc'd string.  */
extern char *strdup (const char *__s)
     __THROW __attribute_malloc__ __nonnull ((1));
#endif
# 179 "/usr/include/string.h" 3 4

/* Return a malloc'd copy of at most N bytes of STRING.  The
   resultant string is terminated even if no null terminator
   appears before STRING[N].  */
#if defined __USE_XOPEN2K8
extern char *strndup (const char *__string, size_t __n)
     __THROW __attribute_malloc__ __nonnull ((1));
#endif
# 187 "/usr/include/string.h" 3 4

#if defined __USE_GNU && defined __GNUC__
/* Duplicate S, returning an identical alloca'd string.  */
# define strdupa(s)							      \
  (__extension__							      \
    ({									      \
      const char *__old = (s);						      \
      size_t __len = strlen (__old) + 1;				      \
      char *__new = (char *) __builtin_alloca (__len);			      \
      (char *) memcpy (__new, __old, __len);				      \
    }))

/* Return an alloca'd copy of at most N bytes of string.  */
# define strndupa(s, n)							      \
  (__extension__							      \
    ({									      \
      const char *__old = (s);						      \
      size_t __len = strnlen (__old, (n));				      \
      char *__new = (char *) __builtin_alloca (__len + 1);		      \
      __new[__len] = '\0';						      \
      (char *) memcpy (__new, __old, __len);				      \
    }))
#endif
# 210 "/usr/include/string.h" 3 4

__BEGIN_NAMESPACE_STD
/* Find the first occurrence of C in S.  */
#ifdef __CORRECT_ISO_CPP_STRING_H_PROTO
extern "C++"
{
extern char *strchr (char *__s, int __c)
     __THROW __asm ("strchr") __attribute_pure__ __nonnull ((1));
extern const char *strchr (const char *__s, int __c)
     __THROW __asm ("strchr") __attribute_pure__ __nonnull ((1));

# ifdef __OPTIMIZE__
__extern_always_inline char *
strchr (char *__s, int __c) __THROW
{
  return __builtin_strchr (__s, __c);
}

__extern_always_inline const char *
strchr (const char *__s, int __c) __THROW
{
  return __builtin_strchr (__s, __c);
}
# endif
# 234 "/usr/include/string.h" 3 4
}
#else
# 236 "/usr/include/string.h" 3 4
extern char *strchr (const char *__s, int __c)
     __THROW __attribute_pure__ __nonnull ((1));
#endif
# 239 "/usr/include/string.h" 3 4
/* Find the last occurrence of C in S.  */
#ifdef __CORRECT_ISO_CPP_STRING_H_PROTO
extern "C++"
{
extern char *strrchr (char *__s, int __c)
     __THROW __asm ("strrchr") __attribute_pure__ __nonnull ((1));
extern const char *strrchr (const char *__s, int __c)
     __THROW __asm ("strrchr") __attribute_pure__ __nonnull ((1));

# ifdef __OPTIMIZE__
__extern_always_inline char *
strrchr (char *__s, int __c) __THROW
{
  return __builtin_strrchr (__s, __c);
}

__extern_always_inline const char *
strrchr (const char *__s, int __c) __THROW
{
  return __builtin_strrchr (__s, __c);
}
# endif
# 261 "/usr/include/string.h" 3 4
}
#else
# 263 "/usr/include/string.h" 3 4
extern char *strrchr (const char *__s, int __c)
     __THROW __attribute_pure__ __nonnull ((1));
#endif
# 266 "/usr/include/string.h" 3 4
__END_NAMESPACE_STD

#ifdef __USE_GNU
/* This function is similar to `strchr'.  But it returns a pointer to
   the closing NUL byte in case C is not found in S.  */
# ifdef __CORRECT_ISO_CPP_STRING_H_PROTO
extern "C++" char *strchrnul (char *__s, int __c)
     __THROW __asm ("strchrnul") __attribute_pure__ __nonnull ((1));
extern "C++" const char *strchrnul (const char *__s, int __c)
     __THROW __asm ("strchrnul") __attribute_pure__ __nonnull ((1));
# else
# 277 "/usr/include/string.h" 3 4
extern char *strchrnul (const char *__s, int __c)
     __THROW __attribute_pure__ __nonnull ((1));
# endif
# 280 "/usr/include/string.h" 3 4
#endif
# 281 "/usr/include/string.h" 3 4

__BEGIN_NAMESPACE_STD
/* Return the length of the initial segment of S which
   consists entirely of characters not in REJECT.  */
extern size_t strcspn (const char *__s, const char *__reject)
     __THROW __attribute_pure__ __nonnull ((1, 2));
/* Return the length of the initial segment of S which
   consists entirely of characters in ACCEPT.  */
extern size_t strspn (const char *__s, const char *__accept)
     __THROW __attribute_pure__ __nonnull ((1, 2));
/* Find the first occurrence in S of any character in ACCEPT.  */
#ifdef __CORRECT_ISO_CPP_STRING_H_PROTO
extern "C++"
{
extern char *strpbrk (char *__s, const char *__accept)
     __THROW __asm ("strpbrk") __attribute_pure__ __nonnull ((1, 2));
extern const char *strpbrk (const char *__s, const char *__accept)
     __THROW __asm ("strpbrk") __attribute_pure__ __nonnull ((1, 2));

# ifdef __OPTIMIZE__
__extern_always_inline char *
strpbrk (char *__s, const char *__accept) __THROW
{
  return __builtin_strpbrk (__s, __accept);
}

__extern_always_inline const char *
strpbrk (const char *__s, const char *__accept) __THROW
{
  return __builtin_strpbrk (__s, __accept);
}
# endif
# 313 "/usr/include/string.h" 3 4
}
#else
# 315 "/usr/include/string.h" 3 4
extern char *strpbrk (const char *__s, const char *__accept)
     __THROW __attribute_pure__ __nonnull ((1, 2));
#endif
# 318 "/usr/include/string.h" 3 4
/* Find the first occurrence of NEEDLE in HAYSTACK.  */
#ifdef __CORRECT_ISO_CPP_STRING_H_PROTO
extern "C++"
{
extern char *strstr (char *__haystack, const char *__needle)
     __THROW __asm ("strstr") __attribute_pure__ __nonnull ((1, 2));
extern const char *strstr (const char *__haystack, const char *__needle)
     __THROW __asm ("strstr") __attribute_pure__ __nonnull ((1, 2));

# ifdef __OPTIMIZE__
__extern_always_inline char *
strstr (char *__haystack, const char *__needle) __THROW
{
  return __builtin_strstr (__haystack, __needle);
}

__extern_always_inline const char *
strstr (const char *__haystack, const char *__needle) __THROW
{
  return __builtin_strstr (__haystack, __needle);
}
# endif
# 340 "/usr/include/string.h" 3 4
}
#else
# 342 "/usr/include/string.h" 3 4
extern char *strstr (const char *__haystack, const char *__needle)
     __THROW __attribute_pure__ __nonnull ((1, 2));
#endif
# 345 "/usr/include/string.h" 3 4


/* Divide S into tokens separated by characters in DELIM.  */
extern char *strtok (char *__restrict __s, const char *__restrict __delim)
     __THROW __nonnull ((2));
__END_NAMESPACE_STD

/* Divide S into tokens separated by characters in DELIM.  Information
   passed between calls are stored in SAVE_PTR.  */
extern char *__strtok_r (char *__restrict __s,
			 const char *__restrict __delim,
			 char **__restrict __save_ptr)
     __THROW __nonnull ((2, 3));
#if defined __USE_POSIX || defined __USE_MISC
extern char *strtok_r (char *__restrict __s, const char *__restrict __delim,
		       char **__restrict __save_ptr)
     __THROW __nonnull ((2, 3));
#endif
# 363 "/usr/include/string.h" 3 4

#ifdef __USE_GNU
/* Similar to `strstr' but this function ignores the case of both strings.  */
# ifdef __CORRECT_ISO_CPP_STRING_H_PROTO
extern "C++" char *strcasestr (char *__haystack, const char *__needle)
     __THROW __asm ("strcasestr") __attribute_pure__ __nonnull ((1, 2));
extern "C++" const char *strcasestr (const char *__haystack,
				     const char *__needle)
     __THROW __asm ("strcasestr") __attribute_pure__ __nonnull ((1, 2));
# else
# 373 "/usr/include/string.h" 3 4
extern char *strcasestr (const char *__haystack, const char *__needle)
     __THROW __attribute_pure__ __nonnull ((1, 2));
# endif
# 376 "/usr/include/string.h" 3 4
#endif
# 377 "/usr/include/string.h" 3 4

#ifdef __USE_GNU
/* Find the first occurrence of NEEDLE in HAYSTACK.
   NEEDLE is NEEDLELEN bytes long;
   HAYSTACK is HAYSTACKLEN bytes long.  */
extern void *memmem (const void *__haystack, size_t __haystacklen,
		     const void *__needle, size_t __needlelen)
     __THROW __attribute_pure__ __nonnull ((1, 3));

/* Copy N bytes of SRC to DEST, return pointer to bytes after the
   last written byte.  */
extern void *__mempcpy (void *__restrict __dest,
			const void *__restrict __src, size_t __n)
     __THROW __nonnull ((1, 2));
extern void *mempcpy (void *__restrict __dest,
		      const void *__restrict __src, size_t __n)
     __THROW __nonnull ((1, 2));
#endif
# 395 "/usr/include/string.h" 3 4


__BEGIN_NAMESPACE_STD
/* Return the length of S.  */
extern size_t strlen (const char *__s)
     __THROW __attribute_pure__ __nonnull ((1));
__END_NAMESPACE_STD

#ifdef	__USE_XOPEN2K8
/* Find the length of STRING, but scan at most MAXLEN characters.
   If no '\0' terminator is found in that many characters, return MAXLEN.  */
extern size_t strnlen (const char *__string, size_t __maxlen)
     __THROW __attribute_pure__ __nonnull ((1));
#endif
# 409 "/usr/include/string.h" 3 4


__BEGIN_NAMESPACE_STD
/* Return a string describing the meaning of the `errno' code in ERRNUM.  */
extern char *strerror (int __errnum) __THROW;
__END_NAMESPACE_STD
#if defined __USE_XOPEN2K || defined __USE_MISC
/* Reentrant version of `strerror'.
   There are 2 flavors of `strerror_r', GNU which returns the string
   and may or may not use the supplied temporary buffer and POSIX one
   which fills the string into the buffer.
   To use the POSIX version, -D_XOPEN_SOURCE=600 or -D_POSIX_C_SOURCE=200112L
   without -D_GNU_SOURCE is needed, otherwise the GNU version is
   preferred.  */
# if defined __USE_XOPEN2K && !defined __USE_GNU
/* Fill BUF with a string describing the meaning of the `errno' code in
   ERRNUM.  */
#  ifdef __REDIRECT_NTH
extern int __REDIRECT_NTH (strerror_r,
			   (int __errnum, char *__buf, size_t __buflen),
			   __xpg_strerror_r) __nonnull ((2));
#  else
# 431 "/usr/include/string.h" 3 4
extern int __xpg_strerror_r (int __errnum, char *__buf, size_t __buflen)
     __THROW __nonnull ((2));
#   define strerror_r __xpg_strerror_r
#  endif
# 435 "/usr/include/string.h" 3 4
# else
# 436 "/usr/include/string.h" 3 4
/* If a temporary buffer is required, at most BUFLEN bytes of BUF will be
   used.  */
extern char *strerror_r (int __errnum, char *__buf, size_t __buflen)
     __THROW __nonnull ((2)) __wur;
# endif
# 441 "/usr/include/string.h" 3 4
#endif
# 442 "/usr/include/string.h" 3 4

#ifdef __USE_XOPEN2K8
/* Translate error number to string according to the locale L.  */
extern char *strerror_l (int __errnum, __locale_t __l) __THROW;
#endif
# 447 "/usr/include/string.h" 3 4


/* We define this function always since `bzero' is sometimes needed when
   the namespace rules does not allow this.  */
extern void __bzero (void *__s, size_t __n) __THROW __nonnull ((1));

#ifdef __USE_BSD
/* Copy N bytes of SRC to DEST (like memmove, but args reversed).  */
extern void bcopy (const void *__src, void *__dest, size_t __n)
     __THROW __nonnull ((1, 2));

/* Set N bytes of S to 0.  */
extern void bzero (void *__s, size_t __n) __THROW __nonnull ((1));

/* Compare N bytes of S1 and S2 (same as memcmp).  */
extern int bcmp (const void *__s1, const void *__s2, size_t __n)
     __THROW __attribute_pure__ __nonnull ((1, 2));

/* Find the first occurrence of C in S (same as strchr).  */
# ifdef __CORRECT_ISO_CPP_STRING_H_PROTO
extern "C++"
{
extern char *index (char *__s, int __c)
     __THROW __asm ("index") __attribute_pure__ __nonnull ((1));
extern const char *index (const char *__s, int __c)
     __THROW __asm ("index") __attribute_pure__ __nonnull ((1));

#  if defined __OPTIMIZE__ && !defined __CORRECT_ISO_CPP_STRINGS_H_PROTO
__extern_always_inline char *
index (char *__s, int __c) __THROW
{
  return __builtin_index (__s, __c);
}

__extern_always_inline const char *
index (const char *__s, int __c) __THROW
{
  return __builtin_index (__s, __c);
}
#  endif
# 487 "/usr/include/string.h" 3 4
}
# else
# 489 "/usr/include/string.h" 3 4
extern char *index (const char *__s, int __c)
     __THROW __attribute_pure__ __nonnull ((1));
# endif
# 492 "/usr/include/string.h" 3 4

/* Find the last occurrence of C in S (same as strrchr).  */
# ifdef __CORRECT_ISO_CPP_STRING_H_PROTO
extern "C++"
{
extern char *rindex (char *__s, int __c)
     __THROW __asm ("rindex") __attribute_pure__ __nonnull ((1));
extern const char *rindex (const char *__s, int __c)
     __THROW __asm ("rindex") __attribute_pure__ __nonnull ((1));

#  if defined __OPTIMIZE__ && !defined __CORRECT_ISO_CPP_STRINGS_H_PROTO
__extern_always_inline char *
rindex (char *__s, int __c) __THROW
{
  return __builtin_rindex (__s, __c);
}

__extern_always_inline const char *
rindex (const char *__s, int __c) __THROW
{
  return __builtin_rindex (__s, __c);
}
#endif
# 515 "/usr/include/string.h" 3 4
}
# else
# 517 "/usr/include/string.h" 3 4
extern char *rindex (const char *__s, int __c)
     __THROW __attribute_pure__ __nonnull ((1));
# endif
# 520 "/usr/include/string.h" 3 4

/* Return the position of the first bit set in I, or 0 if none are set.
   The least-significant bit is position 1, the most-significant 32.  */
extern int ffs (int __i) __THROW __attribute__ ((__const__));

/* The following two functions are non-standard but necessary for non-32 bit
   platforms.  */
# ifdef	__USE_GNU
extern int ffsl (long int __l) __THROW __attribute__ ((__const__));
__extension__ extern int ffsll (long long int __ll)
     __THROW __attribute__ ((__const__));
# endif
# 532 "/usr/include/string.h" 3 4

/* Compare S1 and S2, ignoring case.  */
extern int strcasecmp (const char *__s1, const char *__s2)
     __THROW __attribute_pure__ __nonnull ((1, 2));

/* Compare no more than N chars of S1 and S2, ignoring case.  */
extern int strncasecmp (const char *__s1, const char *__s2, size_t __n)
     __THROW __attribute_pure__ __nonnull ((1, 2));
#endif /* Use BSD.  */
# 541 "/usr/include/string.h" 3 4

#ifdef	__USE_GNU
/* Again versions of a few functions which use the given locale instead
   of the global one.  */
extern int strcasecmp_l (const char *__s1, const char *__s2,
			 __locale_t __loc)
     __THROW __attribute_pure__ __nonnull ((1, 2, 3));

extern int strncasecmp_l (const char *__s1, const char *__s2,
			  size_t __n, __locale_t __loc)
     __THROW __attribute_pure__ __nonnull ((1, 2, 4));
#endif
# 553 "/usr/include/string.h" 3 4

#ifdef	__USE_BSD
/* Return the next DELIM-delimited token from *STRINGP,
   terminating it with a '\0', and update *STRINGP to point past it.  */
extern char *strsep (char **__restrict __stringp,
		     const char *__restrict __delim)
     __THROW __nonnull ((1, 2));
#endif
# 561 "/usr/include/string.h" 3 4

#ifdef	__USE_XOPEN2K8
/* Return a string describing the meaning of the signal number in SIG.  */
extern char *strsignal (int __sig) __THROW;

/* Copy SRC to DEST, returning the address of the terminating '\0' in DEST.  */
extern char *__stpcpy (char *__restrict __dest, const char *__restrict __src)
     __THROW __nonnull ((1, 2));
extern char *stpcpy (char *__restrict __dest, const char *__restrict __src)
     __THROW __nonnull ((1, 2));

/* Copy no more than N characters of SRC to DEST, returning the address of
   the last character written into DEST.  */
extern char *__stpncpy (char *__restrict __dest,
			const char *__restrict __src, size_t __n)
     __THROW __nonnull ((1, 2));
extern char *stpncpy (char *__restrict __dest,
		      const char *__restrict __src, size_t __n)
     __THROW __nonnull ((1, 2));
#endif
# 581 "/usr/include/string.h" 3 4

#ifdef	__USE_GNU
/* Compare S1 and S2 as strings holding name & indices/version numbers.  */
extern int strverscmp (const char *__s1, const char *__s2)
     __THROW __attribute_pure__ __nonnull ((1, 2));

/* Sautee STRING briskly.  */
extern char *strfry (char *__string) __THROW __nonnull ((1));

/* Frobnicate N bytes of S.  */
extern void *memfrob (void *__s, size_t __n) __THROW __nonnull ((1));

# ifndef basename
/* Return the file name within directory of FILENAME.  We don't
   declare the function if the `basename' macro is available (defined
   in <libgen.h>) which makes the XPG version of this function
   available.  */
#  ifdef __CORRECT_ISO_CPP_STRING_H_PROTO
extern "C++" char *basename (char *__filename)
     __THROW __asm ("basename") __nonnull ((1));
extern "C++" const char *basename (const char *__filename)
     __THROW __asm ("basename") __nonnull ((1));
#  else
# 604 "/usr/include/string.h" 3 4
extern char *basename (const char *__filename) __THROW __nonnull ((1));
#  endif
# 606 "/usr/include/string.h" 3 4
# endif
# 607 "/usr/include/string.h" 3 4
#endif
# 608 "/usr/include/string.h" 3 4


#if defined __GNUC__ && __GNUC__ >= 2
# if defined __OPTIMIZE__ && !defined __OPTIMIZE_SIZE__ \
     && !defined __NO_INLINE__ && !defined __cplusplus
/* When using GNU CC we provide some optimized versions of selected
   functions from this header.  There are two kinds of optimizations:

   - machine-dependent optimizations, most probably using inline
     assembler code; these might be quite expensive since the code
     size can increase significantly.
     These optimizations are not used unless the symbol
	__USE_STRING_INLINES
     is defined before including this header.

   - machine-independent optimizations which do not increase the
     code size significantly and which optimize mainly situations
     where one or more arguments are compile-time constants.
     These optimizations are used always when the compiler is
     taught to optimize.

   One can inhibit all optimizations by defining __NO_STRING_INLINES.  */

/* Get the machine-dependent optimizations (if any).  */
#if 0 /* expanded by -frewrite-includes */
#  include <bits/string.h>
#endif /* expanded by -frewrite-includes */
# 632 "/usr/include/string.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/string.h" 1 3 4
/* Optimized, inlined string functions.  i486/x86-64 version.
   Copyright (C) 2001-2014 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */

#ifndef _STRING_H
# error "Never use <bits/string.h> directly; include <string.h> instead."
#endif
# 22 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4

/* The ix86 processors can access unaligned multi-byte variables.  */
#define _STRING_ARCH_unaligned	1

/* Enable inline functions only for i486 or better when compiling for
   ia32.  */
#if !defined __x86_64__ && (defined __i486__ || defined __pentium__	      \
			    || defined __pentiumpro__ || defined __pentium4__ \
			    || defined __nocona__ || defined __atom__ 	      \
			    || defined __core2__ || defined __corei7__	      \
			    || defined __k6__ || defined __geode__	      \
			    || defined __k8__ || defined __athlon__	      \
			    || defined __amdfam10__)

/* We only provide optimizations if the user selects them and if
   GNU CC is used.  */
# if !defined __NO_STRING_INLINES && defined __USE_STRING_INLINES \
    && defined __GNUC__ && __GNUC__ >= 2

# ifndef __STRING_INLINE
#  ifndef __extern_inline
#   define __STRING_INLINE inline
#  else
# 45 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4
#   define __STRING_INLINE __extern_inline
#  endif
# 47 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4
# endif
# 48 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4

/* The macros are used in some of the optimized implementations below.  */
# define __STRING_SMALL_GET16(src, idx) \
  ((((const unsigned char *) (src))[idx + 1] << 8)			      \
   | ((const unsigned char *) (src))[idx])
# define __STRING_SMALL_GET32(src, idx) \
  (((((const unsigned char *) (src))[idx + 3] << 8			      \
     | ((const unsigned char *) (src))[idx + 2]) << 8			      \
    | ((const unsigned char *) (src))[idx + 1]) << 8			      \
   | ((const unsigned char *) (src))[idx])


/* Copy N bytes of SRC to DEST.  */
# define _HAVE_STRING_ARCH_memcpy 1
# define memcpy(dest, src, n) \
  (__extension__ (__builtin_constant_p (n)				      \
		  ? __memcpy_c ((dest), (src), (n))			      \
		  : __memcpy_g ((dest), (src), (n))))
# define __memcpy_c(dest, src, n) \
  ((n) == 0								      \
   ? (dest)								      \
   : (((n) % 4 == 0)							      \
      ? __memcpy_by4 (dest, src, n)					      \
      : (((n) % 2 == 0)							      \
	 ? __memcpy_by2 (dest, src, n)					      \
	 : __memcpy_g (dest, src, n))))

__STRING_INLINE void *__memcpy_by4 (void *__dest, const void *__src,
				    size_t __n);

__STRING_INLINE void *
__memcpy_by4 (void *__dest, const void *__src, size_t __n)
{
  register unsigned long int __d0, __d1;
  register void *__tmp = __dest;
  __asm__ __volatile__
    ("1:\n\t"
     "movl	(%2),%0\n\t"
     "leal	4(%2),%2\n\t"
     "movl	%0,(%1)\n\t"
     "leal	4(%1),%1\n\t"
     "decl	%3\n\t"
     "jnz	1b"
     : "=&r" (__d0), "=&r" (__tmp), "=&r" (__src), "=&r" (__d1)
     : "1" (__tmp), "2" (__src), "3" (__n / 4)
     : "memory", "cc");
  return __dest;
}

__STRING_INLINE void *__memcpy_by2 (void *__dest, const void *__src,
				    size_t __n);

__STRING_INLINE void *
__memcpy_by2 (void *__dest, const void *__src, size_t __n)
{
  register unsigned long int __d0, __d1;
  register void *__tmp = __dest;
  __asm__ __volatile__
    ("shrl	$1,%3\n\t"
     "jz	2f\n"                 /* only a word */
     "1:\n\t"
     "movl	(%2),%0\n\t"
     "leal	4(%2),%2\n\t"
     "movl	%0,(%1)\n\t"
     "leal	4(%1),%1\n\t"
     "decl	%3\n\t"
     "jnz	1b\n"
     "2:\n\t"
     "movw	(%2),%w0\n\t"
     "movw	%w0,(%1)"
     : "=&q" (__d0), "=&r" (__tmp), "=&r" (__src), "=&r" (__d1)
     : "1" (__tmp), "2" (__src), "3" (__n / 2)
     : "memory", "cc");
  return __dest;
}

__STRING_INLINE void *__memcpy_g (void *__dest, const void *__src, size_t __n);

__STRING_INLINE void *
__memcpy_g (void *__dest, const void *__src, size_t __n)
{
  register unsigned long int __d0, __d1, __d2;
  register void *__tmp = __dest;
  __asm__ __volatile__
    ("cld\n\t"
     "shrl	$1,%%ecx\n\t"
     "jnc	1f\n\t"
     "movsb\n"
     "1:\n\t"
     "shrl	$1,%%ecx\n\t"
     "jnc	2f\n\t"
     "movsw\n"
     "2:\n\t"
     "rep; movsl"
     : "=&c" (__d0), "=&D" (__d1), "=&S" (__d2),
       "=m" ( *(struct { __extension__ char __x[__n]; } *)__dest)
     : "0" (__n), "1" (__tmp), "2" (__src),
       "m" ( *(struct { __extension__ char __x[__n]; } *)__src)
     : "cc");
  return __dest;
}

# define _HAVE_STRING_ARCH_memmove 1
# ifndef _FORCE_INLINES
/* Copy N bytes of SRC to DEST, guaranteeing
   correct behavior for overlapping strings.  */
#  define memmove(dest, src, n) __memmove_g (dest, src, n)

__STRING_INLINE void *__memmove_g (void *, const void *, size_t)
     __asm__ ("memmove");

__STRING_INLINE void *
__memmove_g (void *__dest, const void *__src, size_t __n)
{
  register unsigned long int __d0, __d1, __d2;
  register void *__tmp = __dest;
  if (__dest < __src)
    __asm__ __volatile__
      ("cld\n\t"
       "rep; movsb"
       : "=&c" (__d0), "=&S" (__d1), "=&D" (__d2),
	 "=m" ( *(struct { __extension__ char __x[__n]; } *)__dest)
       : "0" (__n), "1" (__src), "2" (__tmp),
	 "m" ( *(struct { __extension__ char __x[__n]; } *)__src));
  else
    __asm__ __volatile__
      ("std\n\t"
       "rep; movsb\n\t"
       "cld"
       : "=&c" (__d0), "=&S" (__d1), "=&D" (__d2),
	 "=m" ( *(struct { __extension__ char __x[__n]; } *)__dest)
       : "0" (__n), "1" (__n - 1 + (const char *) __src),
	 "2" (__n - 1 + (char *) __tmp),
	 "m" ( *(struct { __extension__ char __x[__n]; } *)__src));
  return __dest;
}
# endif
# 185 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4

/* Compare N bytes of S1 and S2.  */
# define _HAVE_STRING_ARCH_memcmp 1
# ifndef _FORCE_INLINES
#  ifndef __PIC__
/* gcc has problems to spill registers when using PIC.  */
__STRING_INLINE int
memcmp (const void *__s1, const void *__s2, size_t __n)
{
  register unsigned long int __d0, __d1, __d2;
  register int __res;
  __asm__ __volatile__
    ("cld\n\t"
     "testl %3,%3\n\t"
     "repe; cmpsb\n\t"
     "je	1f\n\t"
     "sbbl	%0,%0\n\t"
     "orl	$1,%0\n"
     "1:"
     : "=&a" (__res), "=&S" (__d0), "=&D" (__d1), "=&c" (__d2)
     : "0" (0), "1" (__s1), "2" (__s2), "3" (__n),
       "m" ( *(struct { __extension__ char __x[__n]; } *)__s1),
       "m" ( *(struct { __extension__ char __x[__n]; } *)__s2)
     : "cc");
  return __res;
}
#  endif
# 212 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4
# endif
# 213 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4

/* Set N bytes of S to C.  */
# define _HAVE_STRING_ARCH_memset 1
# define _USE_STRING_ARCH_memset 1
# define memset(s, c, n) \
  (__extension__ (__builtin_constant_p (n) && (n) <= 16			      \
		  ? ((n) == 1						      \
		     ? __memset_c1 ((s), (c))				      \
		     : __memset_gc ((s), (c), (n)))			      \
		  : (__builtin_constant_p (c)				      \
		     ? (__builtin_constant_p (n)			      \
			? __memset_ccn ((s), (c), (n))			      \
			: memset ((s), (c), (n)))			      \
		     : (__builtin_constant_p (n)			      \
			? __memset_gcn ((s), (c), (n))			      \
			: memset ((s), (c), (n))))))

# define __memset_c1(s, c) ({ void *__s = (s);				      \
			      *((unsigned char *) __s) = (unsigned char) (c); \
			      __s; })

# define __memset_gc(s, c, n) \
  ({ void *__s = (s);							      \
     union {								      \
       unsigned int __ui;						      \
       unsigned short int __usi;					      \
       unsigned char __uc;						      \
     } *__u = __s;							      \
     unsigned int __c = ((unsigned int) ((unsigned char) (c))) * 0x01010101;  \
									      \
     /* We apply a trick here.  `gcc' would implement the following	      \
	assignments using immediate operands.  But this uses to much	      \
	memory (7, instead of 4 bytes).  So we force the value in a	      \
	registers.  */							      \
     if ((n) == 3 || (n) >= 5)						      \
       __asm__ __volatile__ ("" : "=r" (__c) : "0" (__c));		      \
									      \
     /* This `switch' statement will be removed at compile-time.  */	      \
     switch (n)								      \
       {								      \
       case 15:								      \
	 __u->__ui = __c;						      \
	 __u = __extension__ ((void *) __u + 4);			      \
       case 11:								      \
	 __u->__ui = __c;						      \
	 __u = __extension__ ((void *) __u + 4);			      \
       case 7:								      \
	 __u->__ui = __c;						      \
	 __u = __extension__ ((void *) __u + 4);			      \
       case 3:								      \
	 __u->__usi = (unsigned short int) __c;				      \
	 __u = __extension__ ((void *) __u + 2);			      \
	 __u->__uc = (unsigned char) __c;				      \
	 break;								      \
									      \
       case 14:								      \
	 __u->__ui = __c;						      \
	 __u = __extension__ ((void *) __u + 4);			      \
       case 10:								      \
	 __u->__ui = __c;						      \
	 __u = __extension__ ((void *) __u + 4);			      \
       case 6:								      \
	 __u->__ui = __c;						      \
	 __u = __extension__ ((void *) __u + 4);			      \
       case 2:								      \
	 __u->__usi = (unsigned short int) __c;				      \
	 break;								      \
									      \
       case 13:								      \
	 __u->__ui = __c;						      \
	 __u = __extension__ ((void *) __u + 4);			      \
       case 9:								      \
	 __u->__ui = __c;						      \
	 __u = __extension__ ((void *) __u + 4);			      \
       case 5:								      \
	 __u->__ui = __c;						      \
	 __u = __extension__ ((void *) __u + 4);			      \
       case 1:								      \
	 __u->__uc = (unsigned char) __c;				      \
	 break;								      \
									      \
       case 16:								      \
	 __u->__ui = __c;						      \
	 __u = __extension__ ((void *) __u + 4);			      \
       case 12:								      \
	 __u->__ui = __c;						      \
	 __u = __extension__ ((void *) __u + 4);			      \
       case 8:								      \
	 __u->__ui = __c;						      \
	 __u = __extension__ ((void *) __u + 4);			      \
       case 4:								      \
	 __u->__ui = __c;						      \
       case 0:								      \
	 break;								      \
       }								      \
									      \
     __s; })

# define __memset_ccn(s, c, n) \
  (((n) % 4 == 0)							      \
   ? __memset_ccn_by4 (s, ((unsigned int) ((unsigned char) (c))) * 0x01010101,\
		       n)						      \
   : (((n) % 2 == 0)							      \
      ? __memset_ccn_by2 (s,						      \
			  ((unsigned int) ((unsigned char) (c))) * 0x01010101,\
			   n)						      \
      : memset (s, c, n)))

__STRING_INLINE void *__memset_ccn_by4 (void *__s, unsigned int __c,
					size_t __n);

__STRING_INLINE void *
__memset_ccn_by4 (void *__s, unsigned int __c, size_t __n)
{
  register void *__tmp = __s;
  register unsigned long int __d0;
# ifdef __i686__
  __asm__ __volatile__
    ("cld\n\t"
     "rep; stosl"
     : "=&a" (__c), "=&D" (__tmp), "=&c" (__d0),
       "=m" ( *(struct { __extension__ char __x[__n]; } *)__s)
     : "0" ((unsigned int) __c), "1" (__tmp), "2" (__n / 4)
     : "cc");
# else
# 338 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4
  __asm__ __volatile__
    ("1:\n\t"
     "movl	%0,(%1)\n\t"
     "addl	$4,%1\n\t"
     "decl	%2\n\t"
     "jnz	1b\n"
     : "=&r" (__c), "=&r" (__tmp), "=&r" (__d0),
       "=m" ( *(struct { __extension__ char __x[__n]; } *)__s)
     : "0" ((unsigned int) __c), "1" (__tmp), "2" (__n / 4)
     : "cc");
# endif
# 349 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4
  return __s;
}

__STRING_INLINE void *__memset_ccn_by2 (void *__s, unsigned int __c,
					size_t __n);

__STRING_INLINE void *
__memset_ccn_by2 (void *__s, unsigned int __c, size_t __n)
{
  register unsigned long int __d0, __d1;
  register void *__tmp = __s;
# ifdef __i686__
  __asm__ __volatile__
    ("cld\n\t"
     "rep; stosl\n"
     "stosw"
     : "=&a" (__d0), "=&D" (__tmp), "=&c" (__d1),
       "=m" ( *(struct { __extension__ char __x[__n]; } *)__s)
     : "0" ((unsigned int) __c), "1" (__tmp), "2" (__n / 4)
     : "cc");
# else
# 370 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4
  __asm__ __volatile__
    ("1:\tmovl	%0,(%1)\n\t"
     "leal	4(%1),%1\n\t"
     "decl	%2\n\t"
     "jnz	1b\n"
     "movw	%w0,(%1)"
     : "=&q" (__d0), "=&r" (__tmp), "=&r" (__d1),
       "=m" ( *(struct { __extension__ char __x[__n]; } *)__s)
     : "0" ((unsigned int) __c), "1" (__tmp), "2" (__n / 4)
     : "cc");
#endif
# 381 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4
  return __s;
}

# define __memset_gcn(s, c, n) \
  (((n) % 4 == 0)							      \
   ? __memset_gcn_by4 (s, c, n)						      \
   : (((n) % 2 == 0)							      \
      ? __memset_gcn_by2 (s, c, n)					      \
      : memset (s, c, n)))

__STRING_INLINE void *__memset_gcn_by4 (void *__s, int __c, size_t __n);

__STRING_INLINE void *
__memset_gcn_by4 (void *__s, int __c, size_t __n)
{
  register void *__tmp = __s;
  register unsigned long int __d0;
  __asm__ __volatile__
    ("movb	%b0,%h0\n"
     "pushw	%w0\n\t"
     "shll	$16,%0\n\t"
     "popw	%w0\n"
     "1:\n\t"
     "movl	%0,(%1)\n\t"
     "addl	$4,%1\n\t"
     "decl	%2\n\t"
     "jnz	1b\n"
     : "=&q" (__c), "=&r" (__tmp), "=&r" (__d0),
       "=m" ( *(struct { __extension__ char __x[__n]; } *)__s)
     : "0" ((unsigned int) __c), "1" (__tmp), "2" (__n / 4)
     : "cc");
  return __s;
}

__STRING_INLINE void *__memset_gcn_by2 (void *__s, int __c, size_t __n);

__STRING_INLINE void *
__memset_gcn_by2 (void *__s, int __c, size_t __n)
{
  register unsigned long int __d0, __d1;
  register void *__tmp = __s;
  __asm__ __volatile__
    ("movb	%b0,%h0\n\t"
     "pushw	%w0\n\t"
     "shll	$16,%0\n\t"
     "popw	%w0\n"
     "1:\n\t"
     "movl	%0,(%1)\n\t"
     "leal	4(%1),%1\n\t"
     "decl	%2\n\t"
     "jnz	1b\n"
     "movw	%w0,(%1)"
     : "=&q" (__d0), "=&r" (__tmp), "=&r" (__d1),
       "=m" ( *(struct { __extension__ char __x[__n]; } *)__s)
     : "0" ((unsigned int) __c), "1" (__tmp), "2" (__n / 4)
     : "cc");
  return __s;
}


/* Search N bytes of S for C.  */
# define _HAVE_STRING_ARCH_memchr 1
# ifndef _FORCE_INLINES
__STRING_INLINE void *
memchr (const void *__s, int __c, size_t __n)
{
  register unsigned long int __d0;
#  ifdef __i686__
  register unsigned long int __d1;
#  endif
# 451 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4
  register unsigned char *__res;
  if (__n == 0)
    return NULL;
#  ifdef __i686__
  __asm__ __volatile__
    ("cld\n\t"
     "repne; scasb\n\t"
     "cmovne %2,%0"
     : "=D" (__res), "=&c" (__d0), "=&r" (__d1)
     : "a" (__c), "0" (__s), "1" (__n), "2" (1),
       "m" ( *(struct { __extension__ char __x[__n]; } *)__s)
     : "cc");
#  else
# 464 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4
  __asm__ __volatile__
    ("cld\n\t"
     "repne; scasb\n\t"
     "je	1f\n\t"
     "movl	$1,%0\n"
     "1:"
     : "=D" (__res), "=&c" (__d0)
     : "a" (__c), "0" (__s), "1" (__n),
       "m" ( *(struct { __extension__ char __x[__n]; } *)__s)
     : "cc");
#  endif
# 475 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4
  return __res - 1;
}
# endif
# 478 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4

# define _HAVE_STRING_ARCH_memrchr 1
# ifndef _FORCE_INLINES
__STRING_INLINE void *__memrchr (const void *__s, int __c, size_t __n);

__STRING_INLINE void *
__memrchr (const void *__s, int __c, size_t __n)
{
  register unsigned long int __d0;
#  ifdef __i686__
  register unsigned long int __d1;
#  endif
# 490 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4
  register void *__res;
  if (__n == 0)
    return NULL;
#  ifdef __i686__
  __asm__ __volatile__
    ("std\n\t"
     "repne; scasb\n\t"
     "cmovne %2,%0\n\t"
     "cld\n\t"
     "incl %0"
     : "=D" (__res), "=&c" (__d0), "=&r" (__d1)
     : "a" (__c), "0" (__s + __n - 1), "1" (__n), "2" (-1),
       "m" ( *(struct { __extension__ char __x[__n]; } *)__s)
     : "cc");
#  else
# 505 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4
  __asm__ __volatile__
    ("std\n\t"
     "repne; scasb\n\t"
     "je 1f\n\t"
     "orl $-1,%0\n"
     "1:\tcld\n\t"
     "incl %0"
     : "=D" (__res), "=&c" (__d0)
     : "a" (__c), "0" (__s + __n - 1), "1" (__n),
       "m" ( *(struct { __extension__ char __x[__n]; } *)__s)
     : "cc");
#  endif
# 517 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4
  return __res;
}
#  ifdef __USE_GNU
#   define memrchr(s, c, n) __memrchr ((s), (c), (n))
#  endif
# 522 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4
# endif
# 523 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4

/* Return pointer to C in S.  */
# define _HAVE_STRING_ARCH_rawmemchr 1
__STRING_INLINE void *__rawmemchr (const void *__s, int __c);

# ifndef _FORCE_INLINES
__STRING_INLINE void *
__rawmemchr (const void *__s, int __c)
{
  register unsigned long int __d0;
  register unsigned char *__res;
  __asm__ __volatile__
    ("cld\n\t"
     "repne; scasb\n\t"
     : "=D" (__res), "=&c" (__d0)
     : "a" (__c), "0" (__s), "1" (0xffffffff),
       "m" ( *(struct { char __x[0xfffffff]; } *)__s)
     : "cc");
  return __res - 1;
}
#  ifdef __USE_GNU
__STRING_INLINE void *
rawmemchr (const void *__s, int __c)
{
  return __rawmemchr (__s, __c);
}
#  endif /* use GNU */
# 550 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4
# endif
# 551 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4


/* Return the length of S.  */
# define _HAVE_STRING_ARCH_strlen 1
# define strlen(str) \
  (__extension__ (__builtin_constant_p (str)				      \
		  ? __builtin_strlen (str)				      \
		  : __strlen_g (str)))
__STRING_INLINE size_t __strlen_g (const char *__str);

__STRING_INLINE size_t
__strlen_g (const char *__str)
{
  register char __dummy;
  register const char *__tmp = __str;
  __asm__ __volatile__
    ("1:\n\t"
     "movb	(%0),%b1\n\t"
     "leal	1(%0),%0\n\t"
     "testb	%b1,%b1\n\t"
     "jne	1b"
     : "=r" (__tmp), "=&q" (__dummy)
     : "0" (__str),
       "m" ( *(struct { char __x[0xfffffff]; } *)__str)
     : "cc" );
  return __tmp - __str - 1;
}


/* Copy SRC to DEST.  */
# define _HAVE_STRING_ARCH_strcpy 1
# define strcpy(dest, src) \
  (__extension__ (__builtin_constant_p (src)				      \
		  ? (sizeof ((src)[0]) == 1 && strlen (src) + 1 <= 8	      \
		     ? __strcpy_a_small ((dest), (src), strlen (src) + 1)     \
		     : (char *) memcpy ((char *) (dest),		      \
					(const char *) (src),		      \
					strlen (src) + 1))		      \
		  : __strcpy_g ((dest), (src))))

# define __strcpy_a_small(dest, src, srclen) \
  (__extension__ ({ char *__dest = (dest);				      \
		    union {						      \
		      unsigned int __ui;				      \
		      unsigned short int __usi;				      \
		      unsigned char __uc;				      \
		      char __c;						      \
		    } *__u = (void *) __dest;				      \
		    switch (srclen)					      \
		      {							      \
		      case 1:						      \
			__u->__uc = '\0';				      \
			break;						      \
		      case 2:						      \
			__u->__usi = __STRING_SMALL_GET16 (src, 0);	      \
			break;						      \
		      case 3:						      \
			__u->__usi = __STRING_SMALL_GET16 (src, 0);	      \
			__u = __extension__ ((void *) __u + 2);		      \
			__u->__uc = '\0';				      \
			break;						      \
		      case 4:						      \
			__u->__ui = __STRING_SMALL_GET32 (src, 0);	      \
			break;						      \
		      case 5:						      \
			__u->__ui = __STRING_SMALL_GET32 (src, 0);	      \
			__u = __extension__ ((void *) __u + 4);		      \
			__u->__uc = '\0';				      \
			break;						      \
		      case 6:						      \
			__u->__ui = __STRING_SMALL_GET32 (src, 0);	      \
			__u = __extension__ ((void *) __u + 4);		      \
			__u->__usi = __STRING_SMALL_GET16 (src, 4);	      \
			break;						      \
		      case 7:						      \
			__u->__ui = __STRING_SMALL_GET32 (src, 0);	      \
			__u = __extension__ ((void *) __u + 4);		      \
			__u->__usi = __STRING_SMALL_GET16 (src, 4);	      \
			__u = __extension__ ((void *) __u + 2);		      \
			__u->__uc = '\0';				      \
			break;						      \
		      case 8:						      \
			__u->__ui = __STRING_SMALL_GET32 (src, 0);	      \
			__u = __extension__ ((void *) __u + 4);		      \
			__u->__ui = __STRING_SMALL_GET32 (src, 4);	      \
			break;						      \
		      }							      \
		    (char *) __dest; }))

__STRING_INLINE char *__strcpy_g (char *__dest, const char *__src);

__STRING_INLINE char *
__strcpy_g (char *__dest, const char *__src)
{
  register char *__tmp = __dest;
  register char __dummy;
  __asm__ __volatile__
    (
     "1:\n\t"
     "movb	(%0),%b2\n\t"
     "leal	1(%0),%0\n\t"
     "movb	%b2,(%1)\n\t"
     "leal	1(%1),%1\n\t"
     "testb	%b2,%b2\n\t"
     "jne	1b"
     : "=&r" (__src), "=&r" (__tmp), "=&q" (__dummy),
       "=m" ( *(struct { char __x[0xfffffff]; } *)__dest)
     : "0" (__src), "1" (__tmp),
       "m" ( *(struct { char __x[0xfffffff]; } *)__src)
     : "cc");
  return __dest;
}


# ifdef __USE_GNU
#  define _HAVE_STRING_ARCH_stpcpy 1
/* Copy SRC to DEST.  */
#  define __stpcpy(dest, src) \
  (__extension__ (__builtin_constant_p (src)				      \
		  ? (strlen (src) + 1 <= 8				      \
		     ? __stpcpy_a_small ((dest), (src), strlen (src) + 1)     \
		     : __stpcpy_c ((dest), (src), strlen (src) + 1))	      \
		  : __stpcpy_g ((dest), (src))))
#  define __stpcpy_c(dest, src, srclen) \
  ((srclen) % 4 == 0							      \
   ? __mempcpy_by4 (dest, src, srclen) - 1				      \
   : ((srclen) % 2 == 0							      \
      ? __mempcpy_by2 (dest, src, srclen) - 1				      \
      : __mempcpy_byn (dest, src, srclen) - 1))

/* In glibc itself we use this symbol for namespace reasons.  */
#  define stpcpy(dest, src) __stpcpy ((dest), (src))

#  define __stpcpy_a_small(dest, src, srclen) \
  (__extension__ ({ union {						      \
		      unsigned int __ui;				      \
		      unsigned short int __usi;				      \
		      unsigned char __uc;				      \
		      char __c;						      \
		    } *__u = (void *) (dest);				      \
		    switch (srclen)					      \
		      {							      \
		      case 1:						      \
			__u->__uc = '\0';				      \
			break;						      \
		      case 2:						      \
			__u->__usi = __STRING_SMALL_GET16 (src, 0);	      \
			__u = __extension__ ((void *) __u + 1);		      \
			break;						      \
		      case 3:						      \
			__u->__usi = __STRING_SMALL_GET16 (src, 0);	      \
			__u = __extension__ ((void *) __u + 2);		      \
			__u->__uc = '\0';				      \
			break;						      \
		      case 4:						      \
			__u->__ui = __STRING_SMALL_GET32 (src, 0);	      \
			__u = __extension__ ((void *) __u + 3);		      \
			break;						      \
		      case 5:						      \
			__u->__ui = __STRING_SMALL_GET32 (src, 0);	      \
			__u = __extension__ ((void *) __u + 4);		      \
			__u->__uc = '\0';				      \
			break;						      \
		      case 6:						      \
			__u->__ui = __STRING_SMALL_GET32 (src, 0);	      \
			__u = __extension__ ((void *) __u + 4);		      \
			__u->__usi = __STRING_SMALL_GET16 (src, 4);	      \
			__u = __extension__ ((void *) __u + 1);		      \
			break;						      \
		      case 7:						      \
			__u->__ui = __STRING_SMALL_GET32 (src, 0);	      \
			__u = __extension__ ((void *) __u + 4);		      \
			__u->__usi = __STRING_SMALL_GET16 (src, 4);	      \
			__u = __extension__ ((void *) __u + 2);		      \
			__u->__uc = '\0';				      \
			break;						      \
		      case 8:						      \
			__u->__ui = __STRING_SMALL_GET32 (src, 0);	      \
			__u = __extension__ ((void *) __u + 4);		      \
			__u->__ui = __STRING_SMALL_GET32 (src, 4);	      \
			__u = __extension__ ((void *) __u + 3);		      \
			break;						      \
		      }							      \
		    (char *) __u; }))

__STRING_INLINE char *__mempcpy_by4 (char *__dest, const char *__src,
				     size_t __srclen);

__STRING_INLINE char *
__mempcpy_by4 (char *__dest, const char *__src, size_t __srclen)
{
  register char *__tmp = __dest;
  register unsigned long int __d0, __d1;
  __asm__ __volatile__
    ("1:\n\t"
     "movl	(%2),%0\n\t"
     "leal	4(%2),%2\n\t"
     "movl	%0,(%1)\n\t"
     "leal	4(%1),%1\n\t"
     "decl	%3\n\t"
     "jnz	1b"
     : "=&r" (__d0), "=r" (__tmp), "=&r" (__src), "=&r" (__d1)
     : "1" (__tmp), "2" (__src), "3" (__srclen / 4)
     : "memory", "cc");
  return __tmp;
}

__STRING_INLINE char *__mempcpy_by2 (char *__dest, const char *__src,
				     size_t __srclen);

__STRING_INLINE char *
__mempcpy_by2 (char *__dest, const char *__src, size_t __srclen)
{
  register char *__tmp = __dest;
  register unsigned long int __d0, __d1;
  __asm__ __volatile__
    ("shrl	$1,%3\n\t"
     "jz	2f\n"                 /* only a word */
     "1:\n\t"
     "movl	(%2),%0\n\t"
     "leal	4(%2),%2\n\t"
     "movl	%0,(%1)\n\t"
     "leal	4(%1),%1\n\t"
     "decl	%3\n\t"
     "jnz	1b\n"
     "2:\n\t"
     "movw	(%2),%w0\n\t"
     "movw	%w0,(%1)"
     : "=&q" (__d0), "=r" (__tmp), "=&r" (__src), "=&r" (__d1),
       "=m" ( *(struct { __extension__ char __x[__srclen]; } *)__dest)
     : "1" (__tmp), "2" (__src), "3" (__srclen / 2),
       "m" ( *(struct { __extension__ char __x[__srclen]; } *)__src)
     : "cc");
  return __tmp + 2;
}

__STRING_INLINE char *__mempcpy_byn (char *__dest, const char *__src,
				     size_t __srclen);

__STRING_INLINE char *
__mempcpy_byn (char *__dest, const char *__src, size_t __srclen)
{
  register unsigned long __d0, __d1;
  register char *__tmp = __dest;
  __asm__ __volatile__
    ("cld\n\t"
     "shrl	$1,%%ecx\n\t"
     "jnc	1f\n\t"
     "movsb\n"
     "1:\n\t"
     "shrl	$1,%%ecx\n\t"
     "jnc	2f\n\t"
     "movsw\n"
     "2:\n\t"
     "rep; movsl"
     : "=D" (__tmp), "=&c" (__d0), "=&S" (__d1),
       "=m" ( *(struct { __extension__ char __x[__srclen]; } *)__dest)
     : "0" (__tmp), "1" (__srclen), "2" (__src),
       "m" ( *(struct { __extension__ char __x[__srclen]; } *)__src)
     : "cc");
  return __tmp;
}

__STRING_INLINE char *__stpcpy_g (char *__dest, const char *__src);

__STRING_INLINE char *
__stpcpy_g (char *__dest, const char *__src)
{
  register char *__tmp = __dest;
  register char __dummy;
  __asm__ __volatile__
    (
     "1:\n\t"
     "movb	(%0),%b2\n\t"
     "leal	1(%0),%0\n\t"
     "movb	%b2,(%1)\n\t"
     "leal	1(%1),%1\n\t"
     "testb	%b2,%b2\n\t"
     "jne	1b"
     : "=&r" (__src), "=r" (__tmp), "=&q" (__dummy),
       "=m" ( *(struct { char __x[0xfffffff]; } *)__dest)
     : "0" (__src), "1" (__tmp),
       "m" ( *(struct { char __x[0xfffffff]; } *)__src)
     : "cc");
  return __tmp - 1;
}
# endif
# 838 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4


/* Copy no more than N characters of SRC to DEST.  */
# define _HAVE_STRING_ARCH_strncpy 1
# define strncpy(dest, src, n) \
  (__extension__ (__builtin_constant_p (src)				      \
		  ? ((strlen (src) + 1 >= ((size_t) (n))		      \
		      ? (char *) memcpy ((char *) (dest),		      \
					 (const char *) (src), n)	      \
		      : __strncpy_cg ((dest), (src), strlen (src) + 1, n)))   \
		  : __strncpy_gg ((dest), (src), n)))
# define __strncpy_cg(dest, src, srclen, n) \
  (((srclen) % 4 == 0)							      \
   ? __strncpy_by4 (dest, src, srclen, n)				      \
   : (((srclen) % 2 == 0)						      \
      ? __strncpy_by2 (dest, src, srclen, n)				      \
      : __strncpy_byn (dest, src, srclen, n)))

__STRING_INLINE char *__strncpy_by4 (char *__dest, const char __src[],
				     size_t __srclen, size_t __n);

__STRING_INLINE char *
__strncpy_by4 (char *__dest, const char __src[], size_t __srclen, size_t __n)
{
  register char *__tmp = __dest;
  register int __dummy1, __dummy2;
  __asm__ __volatile__
    ("1:\n\t"
     "movl	(%2),%0\n\t"
     "leal	4(%2),%2\n\t"
     "movl	%0,(%1)\n\t"
     "leal	4(%1),%1\n\t"
     "decl	%3\n\t"
     "jnz	1b"
     : "=&r" (__dummy1), "=r" (__tmp), "=&r" (__src), "=&r" (__dummy2),
       "=m" ( *(struct { __extension__ char __x[__srclen]; } *)__dest)
     : "1" (__tmp), "2" (__src), "3" (__srclen / 4),
       "m" ( *(struct { __extension__ char __x[__srclen]; } *)__src)
     : "cc");
  (void) memset (__tmp, '\0', __n - __srclen);
  return __dest;
}

__STRING_INLINE char *__strncpy_by2 (char *__dest, const char __src[],
				     size_t __srclen, size_t __n);

__STRING_INLINE char *
__strncpy_by2 (char *__dest, const char __src[], size_t __srclen, size_t __n)
{
  register char *__tmp = __dest;
  register int __dummy1, __dummy2;
  __asm__ __volatile__
    ("shrl	$1,%3\n\t"
     "jz	2f\n"                 /* only a word */
     "1:\n\t"
     "movl	(%2),%0\n\t"
     "leal	4(%2),%2\n\t"
     "movl	%0,(%1)\n\t"
     "leal	4(%1),%1\n\t"
     "decl	%3\n\t"
     "jnz	1b\n"
     "2:\n\t"
     "movw	(%2),%w0\n\t"
     "movw	%w0,(%1)\n\t"
     : "=&q" (__dummy1), "=r" (__tmp), "=&r" (__src), "=&r" (__dummy2),
       "=m" ( *(struct { __extension__ char __x[__srclen]; } *)__dest)
     : "1" (__tmp), "2" (__src), "3" (__srclen / 2),
       "m" ( *(struct { __extension__ char __x[__srclen]; } *)__src)
     : "cc");
  (void) memset (__tmp + 2, '\0', __n - __srclen);
  return __dest;
}

__STRING_INLINE char *__strncpy_byn (char *__dest, const char __src[],
				     size_t __srclen, size_t __n);

__STRING_INLINE char *
__strncpy_byn (char *__dest, const char __src[], size_t __srclen, size_t __n)
{
  register unsigned long int __d0, __d1;
  register char *__tmp = __dest;
  __asm__ __volatile__
    ("cld\n\t"
     "shrl	$1,%1\n\t"
     "jnc	1f\n\t"
     "movsb\n"
     "1:\n\t"
     "shrl	$1,%1\n\t"
     "jnc	2f\n\t"
     "movsw\n"
     "2:\n\t"
     "rep; movsl"
     : "=D" (__tmp), "=&c" (__d0), "=&S" (__d1),
       "=m" ( *(struct { __extension__ char __x[__srclen]; } *)__dest)
     : "1" (__srclen), "0" (__tmp),"2" (__src),
       "m" ( *(struct { __extension__ char __x[__srclen]; } *)__src)
     : "cc");
  (void) memset (__tmp, '\0', __n - __srclen);
  return __dest;
}

__STRING_INLINE char *__strncpy_gg (char *__dest, const char *__src,
				    size_t __n);

__STRING_INLINE char *
__strncpy_gg (char *__dest, const char *__src, size_t __n)
{
  register char *__tmp = __dest;
  register char __dummy;
  if (__n > 0)
    __asm__ __volatile__
      ("1:\n\t"
       "movb	(%0),%2\n\t"
       "incl	%0\n\t"
       "movb	%2,(%1)\n\t"
       "incl	%1\n\t"
       "decl	%3\n\t"
       "je	3f\n\t"
       "testb	%2,%2\n\t"
       "jne	1b\n\t"
       "2:\n\t"
       "movb	%2,(%1)\n\t"
       "incl	%1\n\t"
       "decl	%3\n\t"
       "jne	2b\n\t"
       "3:"
       : "=&r" (__src), "=&r" (__tmp), "=&q" (__dummy), "=&r" (__n)
       : "0" (__src), "1" (__tmp), "3" (__n)
       : "memory", "cc");

  return __dest;
}


/* Append SRC onto DEST.  */
# define _HAVE_STRING_ARCH_strcat 1
# define strcat(dest, src) \
  (__extension__ (__builtin_constant_p (src)				      \
		  ? __strcat_c ((dest), (src), strlen (src) + 1)	      \
		  : __strcat_g ((dest), (src))))

__STRING_INLINE char *__strcat_c (char *__dest, const char __src[],
				  size_t __srclen);

__STRING_INLINE char *
__strcat_c (char *__dest, const char __src[], size_t __srclen)
{
# ifdef __i686__
  register unsigned long int __d0;
  register char *__tmp;
  __asm__ __volatile__
    ("repne; scasb"
     : "=D" (__tmp), "=&c" (__d0),
       "=m" ( *(struct { char __x[0xfffffff]; } *)__dest)
     : "0" (__dest), "1" (0xffffffff), "a" (0),
       "m" ( *(struct { __extension__ char __x[__srclen]; } *)__src)
     : "cc");
  --__tmp;
# else
# 997 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4
  register char *__tmp = __dest - 1;
  __asm__ __volatile__
    ("1:\n\t"
     "incl	%0\n\t"
     "cmpb	$0,(%0)\n\t"
     "jne	1b\n"
     : "=r" (__tmp),
       "=m" ( *(struct { char __x[0xfffffff]; } *)__dest)
     : "0" (__tmp),
       "m" ( *(struct { __extension__ char __x[__srclen]; } *)__src)
     : "cc");
# endif
# 1009 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4
  (void) memcpy (__tmp, __src, __srclen);
  return __dest;
}

__STRING_INLINE char *__strcat_g (char *__dest, const char *__src);

__STRING_INLINE char *
__strcat_g (char *__dest, const char *__src)
{
  register char *__tmp = __dest - 1;
  register char __dummy;
  __asm__ __volatile__
    ("1:\n\t"
     "incl	%1\n\t"
     "cmpb	$0,(%1)\n\t"
     "jne	1b\n"
     "2:\n\t"
     "movb	(%2),%b0\n\t"
     "incl	%2\n\t"
     "movb	%b0,(%1)\n\t"
     "incl	%1\n\t"
     "testb	%b0,%b0\n\t"
     "jne	2b\n"
     : "=&q" (__dummy), "=&r" (__tmp), "=&r" (__src),
       "=m" ( *(struct { char __x[0xfffffff]; } *)__dest)
     : "1"  (__tmp), "2"  (__src),
       "m" ( *(struct { char __x[0xfffffff]; } *)__src)
     : "memory", "cc");
  return __dest;
}


/* Append no more than N characters from SRC onto DEST.  */
# define _HAVE_STRING_ARCH_strncat 1
# define strncat(dest, src, n) \
  (__extension__ ({ char *__dest = (dest);				      \
		    __builtin_constant_p (src) && __builtin_constant_p (n)    \
		    ? (strlen (src) < ((size_t) (n))			      \
		       ? strcat (__dest, (src))				      \
		       : (*(char *)__mempcpy (strchr (__dest, '\0'),	      \
					       (const char *) (src),	      \
					      (n)) = 0, __dest))	      \
		    : __strncat_g (__dest, (src), (n)); }))

__STRING_INLINE char *__strncat_g (char *__dest, const char __src[],
				   size_t __n);

__STRING_INLINE char *
__strncat_g (char *__dest, const char __src[], size_t __n)
{
  register char *__tmp = __dest;
  register char __dummy;
# ifdef __i686__
  __asm__ __volatile__
    ("repne; scasb\n"
     "movl %4, %3\n\t"
     "decl %1\n\t"
     "1:\n\t"
     "subl	$1,%3\n\t"
     "jc	2f\n\t"
     "movb	(%2),%b0\n\t"
     "movsb\n\t"
     "testb	%b0,%b0\n\t"
     "jne	1b\n\t"
     "decl	%1\n"
     "2:\n\t"
     "movb	$0,(%1)"
     : "=&a" (__dummy), "=&D" (__tmp), "=&S" (__src), "=&c" (__n)
     :  "g" (__n), "0" (0), "1" (__tmp), "2" (__src), "3" (0xffffffff)
     : "memory", "cc");
# else
# 1080 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4
  --__tmp;
  __asm__ __volatile__
    ("1:\n\t"
     "cmpb	$0,1(%1)\n\t"
     "leal	1(%1),%1\n\t"
     "jne	1b\n"
     "2:\n\t"
     "subl	$1,%3\n\t"
     "jc	3f\n\t"
     "movb	(%2),%b0\n\t"
     "leal	1(%2),%2\n\t"
     "movb	%b0,(%1)\n\t"
     "leal	1(%1),%1\n\t"
     "testb	%b0,%b0\n\t"
     "jne	2b\n\t"
     "decl	%1\n"
     "3:\n\t"
     "movb	$0,(%1)"
     : "=&q" (__dummy), "=&r" (__tmp), "=&r" (__src), "=&r" (__n)
     : "1" (__tmp), "2" (__src), "3" (__n)
     : "memory", "cc");
#endif
# 1102 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4
  return __dest;
}


/* Compare S1 and S2.  */
# define _HAVE_STRING_ARCH_strcmp 1
# define strcmp(s1, s2) \
  (__extension__ (__builtin_constant_p (s1) && __builtin_constant_p (s2)      \
		  && (sizeof ((s1)[0]) != 1 || strlen (s1) >= 4)	      \
		  && (sizeof ((s2)[0]) != 1 || strlen (s2) >= 4)	      \
		  ? memcmp ((const char *) (s1), (const char *) (s2),	      \
			    (strlen (s1) < strlen (s2)			      \
			     ? strlen (s1) : strlen (s2)) + 1)		      \
		  : (__builtin_constant_p (s1) && sizeof ((s1)[0]) == 1	      \
		     && sizeof ((s2)[0]) == 1 && strlen (s1) < 4	      \
		     ? (__builtin_constant_p (s2) && sizeof ((s2)[0]) == 1    \
			? __strcmp_cc ((const unsigned char *) (s1),	      \
				       (const unsigned char *) (s2),	      \
				       strlen (s1))			      \
			: __strcmp_cg ((const unsigned char *) (s1),	      \
				       (const unsigned char *) (s2),	      \
				       strlen (s1)))			      \
		     : (__builtin_constant_p (s2) && sizeof ((s1)[0]) == 1    \
			&& sizeof ((s2)[0]) == 1 && strlen (s2) < 4	      \
			? (__builtin_constant_p (s1)			      \
			   ? __strcmp_cc ((const unsigned char *) (s1),	      \
					  (const unsigned char *) (s2),	      \
					  strlen (s2))			      \
			   : __strcmp_gc ((const unsigned char *) (s1),	      \
					  (const unsigned char *) (s2),	      \
					  strlen (s2)))			      \
			: __strcmp_gg ((s1), (s2))))))

# define __strcmp_cc(s1, s2, l) \
  (__extension__ ({ register int __result = (s1)[0] - (s2)[0];		      \
		    if (l > 0 && __result == 0)				      \
		      {							      \
			__result = (s1)[1] - (s2)[1];			      \
			if (l > 1 && __result == 0)			      \
			  {						      \
			    __result = (s1)[2] - (s2)[2];		      \
			    if (l > 2 && __result == 0)			      \
			      __result = (s1)[3] - (s2)[3];		      \
			  }						      \
		      }							      \
		    __result; }))

# define __strcmp_cg(s1, s2, l1) \
  (__extension__ ({ const unsigned char *__s2 = (s2);			      \
		    register int __result = (s1)[0] - __s2[0];		      \
		    if (l1 > 0 && __result == 0)			      \
		      {							      \
			__result = (s1)[1] - __s2[1];			      \
			if (l1 > 1 && __result == 0)			      \
			  {						      \
			    __result = (s1)[2] - __s2[2];		      \
			    if (l1 > 2 && __result == 0)		      \
			      __result = (s1)[3] - __s2[3];		      \
			  }						      \
		      }							      \
		    __result; }))

# define __strcmp_gc(s1, s2, l2) \
  (__extension__ ({ const unsigned char *__s1 = (s1);			      \
		    register int __result = __s1[0] - (s2)[0];		      \
		    if (l2 > 0 && __result == 0)			      \
		      {							      \
			__result = __s1[1] - (s2)[1];			      \
			if (l2 > 1 && __result == 0)			      \
			  {						      \
			    __result = __s1[2] - (s2)[2];		      \
			    if (l2 > 2 && __result == 0)		      \
			      __result = __s1[3] - (s2)[3];		      \
			  }						      \
		      }							      \
		    __result; }))

__STRING_INLINE int __strcmp_gg (const char *__s1, const char *__s2);

__STRING_INLINE int
__strcmp_gg (const char *__s1, const char *__s2)
{
  register int __res;
  __asm__ __volatile__
    ("1:\n\t"
     "movb	(%1),%b0\n\t"
     "leal	1(%1),%1\n\t"
     "cmpb	%b0,(%2)\n\t"
     "jne	2f\n\t"
     "leal	1(%2),%2\n\t"
     "testb	%b0,%b0\n\t"
     "jne	1b\n\t"
     "xorl	%0,%0\n\t"
     "jmp	3f\n"
     "2:\n\t"
     "movl	$1,%0\n\t"
     "jb	3f\n\t"
     "negl	%0\n"
     "3:"
     : "=q" (__res), "=&r" (__s1), "=&r" (__s2)
     : "1" (__s1), "2" (__s2),
       "m" ( *(struct { char __x[0xfffffff]; } *)__s1),
       "m" ( *(struct { char __x[0xfffffff]; } *)__s2)
     : "cc");
  return __res;
}


/* Compare N characters of S1 and S2.  */
# define _HAVE_STRING_ARCH_strncmp 1
# define strncmp(s1, s2, n) \
  (__extension__ (__builtin_constant_p (s1) && strlen (s1) < ((size_t) (n))   \
		  ? strcmp ((s1), (s2))					      \
		  : (__builtin_constant_p (s2) && strlen (s2) < ((size_t) (n))\
		     ? strcmp ((s1), (s2))				      \
		     : __strncmp_g ((s1), (s2), (n)))))

__STRING_INLINE int __strncmp_g (const char *__s1, const char *__s2,
				 size_t __n);

__STRING_INLINE int
__strncmp_g (const char *__s1, const char *__s2, size_t __n)
{
  register int __res;
  __asm__ __volatile__
    ("1:\n\t"
     "subl	$1,%3\n\t"
     "jc	2f\n\t"
     "movb	(%1),%b0\n\t"
     "incl	%1\n\t"
     "cmpb	%b0,(%2)\n\t"
     "jne	3f\n\t"
     "incl	%2\n\t"
     "testb	%b0,%b0\n\t"
     "jne	1b\n"
     "2:\n\t"
     "xorl	%0,%0\n\t"
     "jmp	4f\n"
     "3:\n\t"
     "movl	$1,%0\n\t"
     "jb	4f\n\t"
     "negl	%0\n"
     "4:"
     : "=q" (__res), "=&r" (__s1), "=&r" (__s2), "=&r" (__n)
     : "1"  (__s1), "2"  (__s2),  "3" (__n),
       "m" ( *(struct { __extension__ char __x[__n]; } *)__s1),
       "m" ( *(struct { __extension__ char __x[__n]; } *)__s2)
     : "cc");
  return __res;
}


/* Find the first occurrence of C in S.  */
# define _HAVE_STRING_ARCH_strchr 1
# define _USE_STRING_ARCH_strchr 1
# define strchr(s, c) \
  (__extension__ (__builtin_constant_p (c)				      \
		  ? ((c) == '\0'					      \
		     ? (char *) __rawmemchr ((s), (c))			      \
		     : __strchr_c ((s), ((c) & 0xff) << 8))		      \
		  : __strchr_g ((s), (c))))

__STRING_INLINE char *__strchr_c (const char *__s, int __c);

__STRING_INLINE char *
__strchr_c (const char *__s, int __c)
{
  register unsigned long int __d0;
  register char *__res;
  __asm__ __volatile__
    ("1:\n\t"
     "movb	(%0),%%al\n\t"
     "cmpb	%%ah,%%al\n\t"
     "je	2f\n\t"
     "leal	1(%0),%0\n\t"
     "testb	%%al,%%al\n\t"
     "jne	1b\n\t"
     "xorl	%0,%0\n"
     "2:"
     : "=r" (__res), "=&a" (__d0)
     : "0" (__s), "1" (__c),
       "m" ( *(struct { char __x[0xfffffff]; } *)__s)
     : "cc");
  return __res;
}

__STRING_INLINE char *__strchr_g (const char *__s, int __c);

__STRING_INLINE char *
__strchr_g (const char *__s, int __c)
{
  register unsigned long int __d0;
  register char *__res;
  __asm__ __volatile__
    ("movb	%%al,%%ah\n"
     "1:\n\t"
     "movb	(%0),%%al\n\t"
     "cmpb	%%ah,%%al\n\t"
     "je	2f\n\t"
     "leal	1(%0),%0\n\t"
     "testb	%%al,%%al\n\t"
     "jne	1b\n\t"
     "xorl	%0,%0\n"
     "2:"
     : "=r" (__res), "=&a" (__d0)
     : "0" (__s), "1" (__c),
       "m" ( *(struct { char __x[0xfffffff]; } *)__s)
     : "cc");
  return __res;
}


/* Find the first occurrence of C in S or the final NUL byte.  */
# define _HAVE_STRING_ARCH_strchrnul 1
# define __strchrnul(s, c) \
  (__extension__ (__builtin_constant_p (c)				      \
		  ? ((c) == '\0'					      \
		     ? (char *) __rawmemchr ((s), c)			      \
		     : __strchrnul_c ((s), ((c) & 0xff) << 8))		      \
		  : __strchrnul_g ((s), c)))

__STRING_INLINE char *__strchrnul_c (const char *__s, int __c);

__STRING_INLINE char *
__strchrnul_c (const char *__s, int __c)
{
  register unsigned long int __d0;
  register char *__res;
  __asm__ __volatile__
    ("1:\n\t"
     "movb	(%0),%%al\n\t"
     "cmpb	%%ah,%%al\n\t"
     "je	2f\n\t"
     "leal	1(%0),%0\n\t"
     "testb	%%al,%%al\n\t"
     "jne	1b\n\t"
     "decl	%0\n"
     "2:"
     : "=r" (__res), "=&a" (__d0)
     : "0" (__s), "1" (__c),
       "m" ( *(struct { char __x[0xfffffff]; } *)__s)
     : "cc");
  return __res;
}

__STRING_INLINE char *__strchrnul_g (const char *__s, int __c);

__STRING_INLINE char *
__strchrnul_g (const char *__s, int __c)
{
  register unsigned long int __d0;
  register char *__res;
  __asm__ __volatile__
    ("movb	%%al,%%ah\n"
     "1:\n\t"
     "movb	(%0),%%al\n\t"
     "cmpb	%%ah,%%al\n\t"
     "je	2f\n\t"
     "leal	1(%0),%0\n\t"
     "testb	%%al,%%al\n\t"
     "jne	1b\n\t"
     "decl	%0\n"
     "2:"
     : "=r" (__res), "=&a" (__d0)
     : "0" (__s), "1" (__c),
       "m" ( *(struct { char __x[0xfffffff]; } *)__s)
     : "cc");
  return __res;
}
# ifdef __USE_GNU
#  define strchrnul(s, c) __strchrnul ((s), (c))
# endif
# 1374 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4


# if defined __USE_BSD || defined __USE_XOPEN_EXTENDED
/* Find the first occurrence of C in S.  This is the BSD name.  */
#  define _HAVE_STRING_ARCH_index 1
#  define index(s, c) \
  (__extension__ (__builtin_constant_p (c)				      \
		  ? __strchr_c ((s), ((c) & 0xff) << 8)			      \
		  : __strchr_g ((s), (c))))
# endif
# 1384 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4


/* Find the last occurrence of C in S.  */
# define _HAVE_STRING_ARCH_strrchr 1
# define strrchr(s, c) \
  (__extension__ (__builtin_constant_p (c)				      \
		  ? __strrchr_c ((s), ((c) & 0xff) << 8)		      \
		  : __strrchr_g ((s), (c))))

# ifdef __i686__
__STRING_INLINE char *__strrchr_c (const char *__s, int __c);

__STRING_INLINE char *
__strrchr_c (const char *__s, int __c)
{
  register unsigned long int __d0, __d1;
  register char *__res;
  __asm__ __volatile__
    ("cld\n"
     "1:\n\t"
     "lodsb\n\t"
     "cmpb	%h2,%b2\n\t"
     "cmove	%1,%0\n\t"
     "testb	%b2,%b2\n\t"
     "jne 1b"
     : "=d" (__res), "=&S" (__d0), "=&a" (__d1)
     : "0" (1), "1" (__s), "2" (__c),
       "m" ( *(struct { char __x[0xfffffff]; } *)__s)
     : "cc");
  return __res - 1;
}

__STRING_INLINE char *__strrchr_g (const char *__s, int __c);

__STRING_INLINE char *
__strrchr_g (const char *__s, int __c)
{
  register unsigned long int __d0, __d1;
  register char *__res;
  __asm__ __volatile__
    ("movb	%b2,%h2\n"
     "cld\n\t"
     "1:\n\t"
     "lodsb\n\t"
     "cmpb	%h2,%b2\n\t"
     "cmove	%1,%0\n\t"
     "testb	%b2,%b2\n\t"
     "jne 1b"
     : "=d" (__res), "=&S" (__d0), "=&a" (__d1)
     : "0" (1), "1" (__s), "2" (__c),
       "m" ( *(struct { char __x[0xfffffff]; } *)__s)
     : "cc");
  return __res - 1;
}
# else
# 1439 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4
__STRING_INLINE char *__strrchr_c (const char *__s, int __c);

__STRING_INLINE char *
__strrchr_c (const char *__s, int __c)
{
  register unsigned long int __d0, __d1;
  register char *__res;
  __asm__ __volatile__
    ("cld\n"
     "1:\n\t"
     "lodsb\n\t"
     "cmpb	%%ah,%%al\n\t"
     "jne	2f\n\t"
     "leal	-1(%%esi),%0\n"
     "2:\n\t"
     "testb	%%al,%%al\n\t"
     "jne 1b"
     : "=d" (__res), "=&S" (__d0), "=&a" (__d1)
     : "0" (0), "1" (__s), "2" (__c),
       "m" ( *(struct { char __x[0xfffffff]; } *)__s)
     : "cc");
  return __res;
}

__STRING_INLINE char *__strrchr_g (const char *__s, int __c);

__STRING_INLINE char *
__strrchr_g (const char *__s, int __c)
{
  register unsigned long int __d0, __d1;
  register char *__res;
  __asm__ __volatile__
    ("movb	%%al,%%ah\n"
     "cld\n\t"
     "1:\n\t"
     "lodsb\n\t"
     "cmpb	%%ah,%%al\n\t"
     "jne	2f\n\t"
     "leal	-1(%%esi),%0\n"
     "2:\n\t"
     "testb	%%al,%%al\n\t"
     "jne 1b"
     : "=r" (__res), "=&S" (__d0), "=&a" (__d1)
     : "0" (0), "1" (__s), "2" (__c),
       "m" ( *(struct { char __x[0xfffffff]; } *)__s)
     : "cc");
  return __res;
}
# endif
# 1488 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4


# if defined __USE_BSD || defined __USE_XOPEN_EXTENDED
/* Find the last occurrence of C in S.  This is the BSD name.  */
#  define _HAVE_STRING_ARCH_rindex 1
#  define rindex(s, c) \
  (__extension__ (__builtin_constant_p (c)				      \
		  ? __strrchr_c ((s), ((c) & 0xff) << 8)		      \
		  : __strrchr_g ((s), (c))))
# endif
# 1498 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4


/* Return the length of the initial segment of S which
   consists entirely of characters not in REJECT.  */
# define _HAVE_STRING_ARCH_strcspn 1
# define strcspn(s, reject) \
  (__extension__ (__builtin_constant_p (reject) && sizeof ((reject)[0]) == 1  \
		  ? ((reject)[0] == '\0'				      \
		     ? strlen (s)					      \
		     : ((reject)[1] == '\0'				      \
			? __strcspn_c1 ((s), (((reject)[0] << 8) & 0xff00))   \
			: __strcspn_cg ((s), (reject), strlen (reject))))     \
		  : __strcspn_g ((s), (reject))))

__STRING_INLINE size_t __strcspn_c1 (const char *__s, int __reject);

# ifndef _FORCE_INLINES
__STRING_INLINE size_t
__strcspn_c1 (const char *__s, int __reject)
{
  register unsigned long int __d0;
  register char *__res;
  __asm__ __volatile__
    ("1:\n\t"
     "movb	(%0),%%al\n\t"
     "leal	1(%0),%0\n\t"
     "cmpb	%%ah,%%al\n\t"
     "je	2f\n\t"
     "testb	%%al,%%al\n\t"
     "jne	1b\n"
     "2:"
     : "=r" (__res), "=&a" (__d0)
     : "0" (__s), "1" (__reject),
       "m" ( *(struct { char __x[0xfffffff]; } *)__s)
     : "cc");
  return (__res - 1) - __s;
}
# endif
# 1536 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4

__STRING_INLINE size_t __strcspn_cg (const char *__s, const char __reject[],
				     size_t __reject_len);

__STRING_INLINE size_t
__strcspn_cg (const char *__s, const char __reject[], size_t __reject_len)
{
  register unsigned long int __d0, __d1, __d2;
  register const char *__res;
  __asm__ __volatile__
    ("cld\n"
     "1:\n\t"
     "lodsb\n\t"
     "testb	%%al,%%al\n\t"
     "je	2f\n\t"
     "movl	%5,%%edi\n\t"
     "movl	%6,%%ecx\n\t"
     "repne; scasb\n\t"
     "jne	1b\n"
     "2:"
     : "=S" (__res), "=&a" (__d0), "=&c" (__d1), "=&D" (__d2)
     : "0" (__s), "d" (__reject), "g" (__reject_len)
     : "memory", "cc");
  return (__res - 1) - __s;
}

__STRING_INLINE size_t __strcspn_g (const char *__s, const char *__reject);
# ifdef __PIC__

__STRING_INLINE size_t
__strcspn_g (const char *__s, const char *__reject)
{
  register unsigned long int __d0, __d1, __d2;
  register const char *__res;
  __asm__ __volatile__
    ("pushl	%%ebx\n\t"
     "movl	%4,%%edi\n\t"
     "cld\n\t"
     "repne; scasb\n\t"
     "notl	%%ecx\n\t"
     "leal	-1(%%ecx),%%ebx\n"
     "1:\n\t"
     "lodsb\n\t"
     "testb	%%al,%%al\n\t"
     "je	2f\n\t"
     "movl	%4,%%edi\n\t"
     "movl	%%ebx,%%ecx\n\t"
     "repne; scasb\n\t"
     "jne	1b\n"
     "2:\n\t"
     "popl	%%ebx"
     : "=S" (__res), "=&a" (__d0), "=&c" (__d1), "=&D" (__d2)
     : "r" (__reject), "0" (__s), "1" (0), "2" (0xffffffff)
     : "memory", "cc");
  return (__res - 1) - __s;
}
# else
# 1593 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4
__STRING_INLINE size_t
__strcspn_g (const char *__s, const char *__reject)
{
  register unsigned long int __d0, __d1, __d2, __d3;
  register const char *__res;
  __asm__ __volatile__
    ("cld\n\t"
     "repne; scasb\n\t"
     "notl	%%ecx\n\t"
     "leal	-1(%%ecx),%%edx\n"
     "1:\n\t"
     "lodsb\n\t"
     "testb	%%al,%%al\n\t"
     "je	2f\n\t"
     "movl	%%ebx,%%edi\n\t"
     "movl	%%edx,%%ecx\n\t"
     "repne; scasb\n\t"
     "jne	1b\n"
     "2:"
     : "=S" (__res), "=&a" (__d0), "=&c" (__d1), "=&D" (__d2), "=&d" (__d3)
     : "0" (__s), "1" (0), "2" (0xffffffff), "3" (__reject), "b" (__reject)
     /* Clobber memory, otherwise GCC cannot handle this.  */
     : "memory", "cc");
  return (__res - 1) - __s;
}
# endif
# 1619 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4


/* Return the length of the initial segment of S which
   consists entirely of characters in ACCEPT.  */
# define _HAVE_STRING_ARCH_strspn 1
# define strspn(s, accept) \
  (__extension__ (__builtin_constant_p (accept) && sizeof ((accept)[0]) == 1  \
		  ? ((accept)[0] == '\0'				      \
		     ? ((void) (s), 0)					      \
		     : ((accept)[1] == '\0'				      \
			? __strspn_c1 ((s), (((accept)[0] << 8 ) & 0xff00))   \
			: __strspn_cg ((s), (accept), strlen (accept))))      \
		  : __strspn_g ((s), (accept))))

# ifndef _FORCE_INLINES
__STRING_INLINE size_t __strspn_c1 (const char *__s, int __accept);

__STRING_INLINE size_t
__strspn_c1 (const char *__s, int __accept)
{
  register unsigned long int __d0;
  register char *__res;
  /* Please note that __accept never can be '\0'.  */
  __asm__ __volatile__
    ("1:\n\t"
     "movb	(%0),%b1\n\t"
     "leal	1(%0),%0\n\t"
     "cmpb	%h1,%b1\n\t"
     "je	1b"
     : "=r" (__res), "=&q" (__d0)
     : "0" (__s), "1" (__accept),
       "m" ( *(struct { char __x[0xfffffff]; } *)__s)
     : "cc");
  return (__res - 1) - __s;
}
# endif
# 1655 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4

__STRING_INLINE size_t __strspn_cg (const char *__s, const char __accept[],
				    size_t __accept_len);

__STRING_INLINE size_t
__strspn_cg (const char *__s, const char __accept[], size_t __accept_len)
{
  register unsigned long int __d0, __d1, __d2;
  register const char *__res;
  __asm__ __volatile__
    ("cld\n"
     "1:\n\t"
     "lodsb\n\t"
     "testb	%%al,%%al\n\t"
     "je	2f\n\t"
     "movl	%5,%%edi\n\t"
     "movl	%6,%%ecx\n\t"
     "repne; scasb\n\t"
     "je	1b\n"
     "2:"
     : "=S" (__res), "=&a" (__d0), "=&c" (__d1), "=&D" (__d2)
     : "0" (__s), "g" (__accept), "g" (__accept_len),
       /* Since we do not know how large the memory we access it, use a
	  really large amount.  */
       "m" ( *(struct { char __x[0xfffffff]; } *)__s),
       "m" ( *(struct { __extension__ char __x[__accept_len]; } *)__accept)
     : "cc");
  return (__res - 1) - __s;
}

__STRING_INLINE size_t __strspn_g (const char *__s, const char *__accept);
# ifdef __PIC__

__STRING_INLINE size_t
__strspn_g (const char *__s, const char *__accept)
{
  register unsigned long int __d0, __d1, __d2;
  register const char *__res;
  __asm__ __volatile__
    ("pushl	%%ebx\n\t"
     "cld\n\t"
     "repne; scasb\n\t"
     "notl	%%ecx\n\t"
     "leal	-1(%%ecx),%%ebx\n"
     "1:\n\t"
     "lodsb\n\t"
     "testb	%%al,%%al\n\t"
     "je	2f\n\t"
     "movl	%%edx,%%edi\n\t"
     "movl	%%ebx,%%ecx\n\t"
     "repne; scasb\n\t"
     "je	1b\n"
     "2:\n\t"
     "popl	%%ebx"
     : "=S" (__res), "=&a" (__d0), "=&c" (__d1), "=&D" (__d2)
     : "d" (__accept), "0" (__s), "1" (0), "2" (0xffffffff), "3" (__accept)
     : "memory", "cc");
  return (__res - 1) - __s;
}
# else
# 1715 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4
__STRING_INLINE size_t
__strspn_g (const char *__s, const char *__accept)
{
  register unsigned long int __d0, __d1, __d2, __d3;
  register const char *__res;
  __asm__ __volatile__
    ("cld\n\t"
     "repne; scasb\n\t"
     "notl	%%ecx\n\t"
     "leal	-1(%%ecx),%%edx\n"
     "1:\n\t"
     "lodsb\n\t"
     "testb	%%al,%%al\n\t"
     "je	2f\n\t"
     "movl	%%ebx,%%edi\n\t"
     "movl	%%edx,%%ecx\n\t"
     "repne; scasb\n\t"
     "je	1b\n"
     "2:"
     : "=S" (__res), "=&a" (__d0), "=&c" (__d1), "=&D" (__d2), "=&d" (__d3)
     : "0" (__s), "1" (0), "2" (0xffffffff), "3" (__accept), "b" (__accept)
     : "memory", "cc");
  return (__res - 1) - __s;
}
# endif
# 1740 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4


/* Find the first occurrence in S of any character in ACCEPT.  */
# define _HAVE_STRING_ARCH_strpbrk 1
# define strpbrk(s, accept) \
  (__extension__ (__builtin_constant_p (accept) && sizeof ((accept)[0]) == 1  \
		  ? ((accept)[0] == '\0'				      \
		     ? ((void) (s), (char *) 0)				      \
		     : ((accept)[1] == '\0'				      \
			? strchr ((s), (accept)[0])			      \
			: __strpbrk_cg ((s), (accept), strlen (accept))))     \
		  : __strpbrk_g ((s), (accept))))

__STRING_INLINE char *__strpbrk_cg (const char *__s, const char __accept[],
				    size_t __accept_len);

__STRING_INLINE char *
__strpbrk_cg (const char *__s, const char __accept[], size_t __accept_len)
{
  register unsigned long int __d0, __d1, __d2;
  register char *__res;
  __asm__ __volatile__
    ("cld\n"
     "1:\n\t"
     "lodsb\n\t"
     "testb	%%al,%%al\n\t"
     "je	2f\n\t"
     "movl	%5,%%edi\n\t"
     "movl	%6,%%ecx\n\t"
     "repne; scasb\n\t"
     "jne	1b\n\t"
     "decl	%0\n\t"
     "jmp	3f\n"
     "2:\n\t"
     "xorl	%0,%0\n"
     "3:"
     : "=S" (__res), "=&a" (__d0), "=&c" (__d1), "=&D" (__d2)
     : "0" (__s), "d" (__accept), "g" (__accept_len)
     : "memory", "cc");
  return __res;
}

__STRING_INLINE char *__strpbrk_g (const char *__s, const char *__accept);
# ifdef __PIC__

__STRING_INLINE char *
__strpbrk_g (const char *__s, const char *__accept)
{
  register unsigned long int __d0, __d1, __d2;
  register char *__res;
  __asm__ __volatile__
    ("pushl	%%ebx\n\t"
     "movl	%%edx,%%edi\n\t"
     "cld\n\t"
     "repne; scasb\n\t"
     "notl	%%ecx\n\t"
     "leal	-1(%%ecx),%%ebx\n"
     "1:\n\t"
     "lodsb\n\t"
     "testb	%%al,%%al\n\t"
     "je	2f\n\t"
     "movl	%%edx,%%edi\n\t"
     "movl	%%ebx,%%ecx\n\t"
     "repne; scasb\n\t"
     "jne	1b\n\t"
     "decl	%0\n\t"
     "jmp	3f\n"
     "2:\n\t"
     "xorl	%0,%0\n"
     "3:\n\t"
     "popl	%%ebx"
     : "=S" (__res), "=&a" (__d0), "=&c" (__d1), "=&D" (__d2)
     : "d" (__accept), "0" (__s), "1" (0), "2" (0xffffffff)
     : "memory", "cc");
  return __res;
}
# else
# 1817 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4
__STRING_INLINE char *
__strpbrk_g (const char *__s, const char *__accept)
{
  register unsigned long int __d0, __d1, __d2, __d3;
  register char *__res;
  __asm__ __volatile__
    ("movl	%%ebx,%%edi\n\t"
     "cld\n\t"
     "repne; scasb\n\t"
     "notl	%%ecx\n\t"
     "leal	-1(%%ecx),%%edx\n"
     "1:\n\t"
     "lodsb\n\t"
     "testb	%%al,%%al\n\t"
     "je	2f\n\t"
     "movl	%%ebx,%%edi\n\t"
     "movl	%%edx,%%ecx\n\t"
     "repne; scasb\n\t"
     "jne	1b\n\t"
     "decl	%0\n\t"
     "jmp	3f\n"
     "2:\n\t"
     "xorl	%0,%0\n"
     "3:"
     : "=S" (__res), "=&a" (__d0), "=&c" (__d1), "=&d" (__d2), "=&D" (__d3)
     : "0" (__s), "1" (0), "2" (0xffffffff), "b" (__accept)
     : "memory", "cc");
  return __res;
}
# endif
# 1847 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4


/* Find the first occurrence of NEEDLE in HAYSTACK.  */
# define _HAVE_STRING_ARCH_strstr 1
# define strstr(haystack, needle) \
  (__extension__ (__builtin_constant_p (needle) && sizeof ((needle)[0]) == 1  \
		  ? ((needle)[0] == '\0'				      \
		     ? (haystack)					      \
		     : ((needle)[1] == '\0'				      \
			? strchr ((haystack), (needle)[0])		      \
			: __strstr_cg ((haystack), (needle),		      \
				       strlen (needle))))		      \
		  : __strstr_g ((haystack), (needle))))

/* Please note that this function need not handle NEEDLEs with a
   length shorter than two.  */
__STRING_INLINE char *__strstr_cg (const char *__haystack,
				   const char __needle[],
				   size_t __needle_len);

__STRING_INLINE char *
__strstr_cg (const char *__haystack, const char __needle[],
	     size_t __needle_len)
{
  register unsigned long int __d0, __d1, __d2;
  register char *__res;
  __asm__ __volatile__
    ("cld\n" \
     "1:\n\t"
     "movl	%6,%%edi\n\t"
     "movl	%5,%%eax\n\t"
     "movl	%4,%%ecx\n\t"
     "repe; cmpsb\n\t"
     "je	2f\n\t"
     "cmpb	$0,-1(%%esi)\n\t"
     "leal	1(%%eax),%5\n\t"
     "jne	1b\n\t"
     "xorl	%%eax,%%eax\n"
     "2:"
     : "=&a" (__res), "=&S" (__d0), "=&D" (__d1), "=&c" (__d2)
     : "g" (__needle_len), "1" (__haystack), "d" (__needle)
     : "memory", "cc");
  return __res;
}

__STRING_INLINE char *__strstr_g (const char *__haystack,
				  const char *__needle);
# ifdef __PIC__

__STRING_INLINE char *
__strstr_g (const char *__haystack, const char *__needle)
{
  register unsigned long int __d0, __d1, __d2;
  register char *__res;
  __asm__ __volatile__
    ("cld\n\t"
     "repne; scasb\n\t"
     "notl	%%ecx\n\t"
     "pushl	%%ebx\n\t"
     "decl	%%ecx\n\t"	/* NOTE! This also sets Z if searchstring='' */
     "movl	%%ecx,%%ebx\n"
     "1:\n\t"
     "movl	%%edx,%%edi\n\t"
     "movl	%%esi,%%eax\n\t"
     "movl	%%ebx,%%ecx\n\t"
     "repe; cmpsb\n\t"
     "je	2f\n\t"		/* also works for empty string, see above */
     "cmpb	$0,-1(%%esi)\n\t"
     "leal	1(%%eax),%%esi\n\t"
     "jne	1b\n\t"
     "xorl	%%eax,%%eax\n"
     "2:\n\t"
     "popl	%%ebx"
     : "=&a" (__res), "=&c" (__d0), "=&S" (__d1), "=&D" (__d2)
     : "0" (0), "1" (0xffffffff), "2" (__haystack), "3" (__needle),
       "d" (__needle)
     : "memory", "cc");
  return __res;
}
# else
# 1927 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4
__STRING_INLINE char *
__strstr_g (const char *__haystack, const char *__needle)
{
  register unsigned long int __d0, __d1, __d2, __d3;
  register char *__res;
  __asm__ __volatile__
    ("cld\n\t"
     "repne; scasb\n\t"
     "notl	%%ecx\n\t"
     "decl	%%ecx\n\t"	/* NOTE! This also sets Z if searchstring='' */
     "movl	%%ecx,%%edx\n"
     "1:\n\t"
     "movl	%%ebx,%%edi\n\t"
     "movl	%%esi,%%eax\n\t"
     "movl	%%edx,%%ecx\n\t"
     "repe; cmpsb\n\t"
     "je	2f\n\t"		/* also works for empty string, see above */
     "cmpb	$0,-1(%%esi)\n\t"
     "leal	1(%%eax),%%esi\n\t"
     "jne	1b\n\t"
     "xorl	%%eax,%%eax\n"
     "2:"
     : "=&a" (__res), "=&c" (__d0), "=&S" (__d1), "=&D" (__d2), "=&d" (__d3)
     : "0" (0), "1" (0xffffffff), "2" (__haystack), "3" (__needle),
       "b" (__needle)
     : "memory", "cc");
  return __res;
}
# endif
# 1956 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4


/* Bit find functions.  We define only the i686 version since for the other
   processors gcc generates good code.  */
# if defined __USE_BSD || defined __USE_XOPEN_EXTENDED
#  ifdef __i686__
#   define _HAVE_STRING_ARCH_ffs 1
#   define ffs(word) (__builtin_constant_p (word)			      \
		      ? __builtin_ffs (word)				      \
		      : ({ int __cnt, __tmp;				      \
			   __asm__ __volatile__				      \
			     ("bsfl %2,%0\n\t"				      \
			      "cmovel %1,%0"				      \
			      : "=&r" (__cnt), "=r" (__tmp)		      \
			      : "rm" (word), "1" (-1));			      \
			   __cnt + 1; }))

#   ifndef ffsl
#    define ffsl(word) ffs(word)
#   endif
# 1976 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4
#  endif /* i686 */
# 1977 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4
# endif	/* BSD || X/Open */
# 1978 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4

# ifndef _FORCE_INLINES
#  undef __STRING_INLINE
# endif
# 1982 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4

# endif	/* use string inlines && GNU CC */
# 1984 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4

#endif
# 1986 "/usr/include/x86_64-linux-gnu/bits/string.h" 3 4
# 633 "/usr/include/string.h" 2 3 4

/* These are generic optimizations which do not add too much inline code.  */
#if 0 /* expanded by -frewrite-includes */
#  include <bits/string2.h>
#endif /* expanded by -frewrite-includes */
# 635 "/usr/include/string.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/string2.h" 1 3 4
/* Machine-independant string function optimizations.
   Copyright (C) 1997-2014 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */

#ifndef _STRING_H
# error "Never use <bits/string2.h> directly; include <string.h> instead."
#endif
# 23 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4

#ifndef __NO_STRING_INLINES

/* Unlike the definitions in the header <bits/string.h> the
   definitions contained here are not optimized down to assembler
   level.  Those optimizations are not always a good idea since this
   means the code size increases a lot.  Instead the definitions here
   optimize some functions in a way which do not dramatically
   increase the code size and which do not use assembler.  The main
   trick is to use GCC's `__builtin_constant_p' function.

   Every function XXX which has a defined version in
   <bits/string.h> must be accompanied by a symbol _HAVE_STRING_ARCH_XXX
   to make sure we don't get redefinitions.

   We must use here macros instead of inline functions since the
   trick won't work with the latter.  */

#ifndef __STRING_INLINE
# ifdef __cplusplus
#  define __STRING_INLINE inline
# else
# 45 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#  define __STRING_INLINE __extern_inline
# endif
# 47 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#endif
# 48 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4

#if _STRING_ARCH_unaligned
/* If we can do unaligned memory accesses we must know the endianess.  */
#if 0 /* expanded by -frewrite-includes */
# include <endian.h>
#endif /* expanded by -frewrite-includes */
# 51 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
# 1 "/usr/include/endian.h" 1 3 4
/* Copyright (C) 1992-2014 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */

#ifndef	_ENDIAN_H
#define	_ENDIAN_H	1

#if 0 /* expanded by -frewrite-includes */
#include <features.h>
#endif /* expanded by -frewrite-includes */
# 21 "/usr/include/endian.h" 3 4
# 22 "/usr/include/endian.h" 3 4

/* Definitions for byte order, according to significance of bytes,
   from low addresses to high addresses.  The value is what you get by
   putting '4' in the most significant byte, '3' in the second most
   significant byte, '2' in the second least significant byte, and '1'
   in the least significant byte, and then writing down one digit for
   each byte, starting with the byte at the lowest address at the left,
   and proceeding to the byte with the highest address at the right.  */

#define	__LITTLE_ENDIAN	1234
#define	__BIG_ENDIAN	4321
#define	__PDP_ENDIAN	3412

/* This file defines `__BYTE_ORDER' for the particular machine.  */
#if 0 /* expanded by -frewrite-includes */
#include <bits/endian.h>
#endif /* expanded by -frewrite-includes */
# 36 "/usr/include/endian.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/endian.h" 1 3 4
/* i386/x86_64 are little-endian.  */

#ifndef _ENDIAN_H
# error "Never use <bits/endian.h> directly; include <endian.h> instead."
#endif
# 6 "/usr/include/x86_64-linux-gnu/bits/endian.h" 3 4

#define __BYTE_ORDER __LITTLE_ENDIAN
# 37 "/usr/include/endian.h" 2 3 4

/* Some machines may need to use a different endianness for floating point
   values.  */
#ifndef __FLOAT_WORD_ORDER
# define __FLOAT_WORD_ORDER __BYTE_ORDER
#endif
# 43 "/usr/include/endian.h" 3 4

#ifdef	__USE_BSD
# define LITTLE_ENDIAN	__LITTLE_ENDIAN
# define BIG_ENDIAN	__BIG_ENDIAN
# define PDP_ENDIAN	__PDP_ENDIAN
# define BYTE_ORDER	__BYTE_ORDER
#endif
# 50 "/usr/include/endian.h" 3 4

#if __BYTE_ORDER == __LITTLE_ENDIAN
# define __LONG_LONG_PAIR(HI, LO) LO, HI
#elif __BYTE_ORDER == __BIG_ENDIAN
# 54 "/usr/include/endian.h" 3 4
# define __LONG_LONG_PAIR(HI, LO) HI, LO
#endif
# 56 "/usr/include/endian.h" 3 4


#if defined __USE_BSD && !defined __ASSEMBLER__
/* Conversion interfaces.  */
#if 0 /* expanded by -frewrite-includes */
# include <bits/byteswap.h>
#endif /* expanded by -frewrite-includes */
# 60 "/usr/include/endian.h" 3 4
# 61 "/usr/include/endian.h" 3 4

# if __BYTE_ORDER == __LITTLE_ENDIAN
#  define htobe16(x) __bswap_16 (x)
#  define htole16(x) (x)
#  define be16toh(x) __bswap_16 (x)
#  define le16toh(x) (x)

#  define htobe32(x) __bswap_32 (x)
#  define htole32(x) (x)
#  define be32toh(x) __bswap_32 (x)
#  define le32toh(x) (x)

#  define htobe64(x) __bswap_64 (x)
#  define htole64(x) (x)
#  define be64toh(x) __bswap_64 (x)
#  define le64toh(x) (x)

# else
# 79 "/usr/include/endian.h" 3 4
#  define htobe16(x) (x)
#  define htole16(x) __bswap_16 (x)
#  define be16toh(x) (x)
#  define le16toh(x) __bswap_16 (x)

#  define htobe32(x) (x)
#  define htole32(x) __bswap_32 (x)
#  define be32toh(x) (x)
#  define le32toh(x) __bswap_32 (x)

#  define htobe64(x) (x)
#  define htole64(x) __bswap_64 (x)
#  define be64toh(x) (x)
#  define le64toh(x) __bswap_64 (x)
# endif
# 94 "/usr/include/endian.h" 3 4
#endif
# 95 "/usr/include/endian.h" 3 4

#endif	/* endian.h */
# 97 "/usr/include/endian.h" 3 4
# 52 "/usr/include/x86_64-linux-gnu/bits/string2.h" 2 3 4
#if 0 /* expanded by -frewrite-includes */
# include <bits/types.h>
#endif /* expanded by -frewrite-includes */
# 52 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
# 53 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4

# if __BYTE_ORDER == __LITTLE_ENDIAN
#  define __STRING2_SMALL_GET16(src, idx) \
     (((const unsigned char *) (const char *) (src))[idx + 1] << 8	      \
      | ((const unsigned char *) (const char *) (src))[idx])
#  define __STRING2_SMALL_GET32(src, idx) \
     (((((const unsigned char *) (const char *) (src))[idx + 3] << 8	      \
	| ((const unsigned char *) (const char *) (src))[idx + 2]) << 8	      \
       | ((const unsigned char *) (const char *) (src))[idx + 1]) << 8	      \
      | ((const unsigned char *) (const char *) (src))[idx])
# else
# 64 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#  define __STRING2_SMALL_GET16(src, idx) \
     (((const unsigned char *) (const char *) (src))[idx] << 8		      \
      | ((const unsigned char *) (const char *) (src))[idx + 1])
#  define __STRING2_SMALL_GET32(src, idx) \
     (((((const unsigned char *) (const char *) (src))[idx] << 8	      \
	| ((const unsigned char *) (const char *) (src))[idx + 1]) << 8	      \
       | ((const unsigned char *) (const char *) (src))[idx + 2]) << 8	      \
      | ((const unsigned char *) (const char *) (src))[idx + 3])
# endif
# 73 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#else
# 74 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
/* These are a few types we need for the optimizations if we cannot
   use unaligned memory accesses.  */
# define __STRING2_COPY_TYPE(N) \
  typedef struct { unsigned char __arr[N]; }				      \
    __attribute__ ((__packed__)) __STRING2_COPY_ARR##N
__STRING2_COPY_TYPE (2);
__STRING2_COPY_TYPE (3);
__STRING2_COPY_TYPE (4);
__STRING2_COPY_TYPE (5);
__STRING2_COPY_TYPE (6);
__STRING2_COPY_TYPE (7);
__STRING2_COPY_TYPE (8);
# undef __STRING2_COPY_TYPE
#endif
# 88 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4

/* Dereferencing a pointer arg to run sizeof on it fails for the void
   pointer case, so we use this instead.
   Note that __x is evaluated twice. */
#define __string2_1bptr_p(__x) \
  ((size_t)(const void *)((__x) + 1) - (size_t)(const void *)(__x) == 1)

/* Set N bytes of S to C.  */
#if !defined _HAVE_STRING_ARCH_memset
# if !__GNUC_PREREQ (3, 0)
#  if _STRING_ARCH_unaligned
#   define memset(s, c, n) \
  (__extension__ (__builtin_constant_p (n) && (n) <= 16			      \
		  ? ((n) == 1						      \
		     ? __memset_1 (s, c)				      \
		     : __memset_gc (s, c, n))				      \
		  : (__builtin_constant_p (c) && (c) == '\0'		      \
		     ? ({ void *__s = (s); __bzero (__s, n); __s; })	      \
		     : memset (s, c, n))))

#   define __memset_1(s, c) ({ void *__s = (s);				      \
			    *((__uint8_t *) __s) = (__uint8_t) c; __s; })

#   define __memset_gc(s, c, n) \
  ({ void *__s = (s);							      \
     union {								      \
       unsigned int __ui;						      \
       unsigned short int __usi;					      \
       unsigned char __uc;						      \
     } *__u = __s;							      \
     __uint8_t __c = (__uint8_t) (c);					      \
									      \
     /* This `switch' statement will be removed at compile-time.  */	      \
     switch ((unsigned int) (n))					      \
       {								      \
       case 15:								      \
	 __u->__ui = __c * 0x01010101;					      \
	 __u = __extension__ ((void *) __u + 4);			      \
       case 11:								      \
	 __u->__ui = __c * 0x01010101;					      \
	 __u = __extension__ ((void *) __u + 4);			      \
       case 7:								      \
	 __u->__ui = __c * 0x01010101;					      \
	 __u = __extension__ ((void *) __u + 4);			      \
       case 3:								      \
	 __u->__usi = (unsigned short int) __c * 0x0101;		      \
	 __u = __extension__ ((void *) __u + 2);			      \
	 __u->__uc = (unsigned char) __c;				      \
	 break;								      \
									      \
       case 14:								      \
	 __u->__ui = __c * 0x01010101;					      \
	 __u = __extension__ ((void *) __u + 4);			      \
       case 10:								      \
	 __u->__ui = __c * 0x01010101;					      \
	 __u = __extension__ ((void *) __u + 4);			      \
       case 6:								      \
	 __u->__ui = __c * 0x01010101;					      \
	 __u = __extension__ ((void *) __u + 4);			      \
       case 2:								      \
	 __u->__usi = (unsigned short int) __c * 0x0101;		      \
	 break;								      \
									      \
       case 13:								      \
	 __u->__ui = __c * 0x01010101;					      \
	 __u = __extension__ ((void *) __u + 4);			      \
       case 9:								      \
	 __u->__ui = __c * 0x01010101;					      \
	 __u = __extension__ ((void *) __u + 4);			      \
       case 5:								      \
	 __u->__ui = __c * 0x01010101;					      \
	 __u = __extension__ ((void *) __u + 4);			      \
       case 1:								      \
	 __u->__uc = (unsigned char) __c;				      \
	 break;								      \
									      \
       case 16:								      \
	 __u->__ui = __c * 0x01010101;					      \
	 __u = __extension__ ((void *) __u + 4);			      \
       case 12:								      \
	 __u->__ui = __c * 0x01010101;					      \
	 __u = __extension__ ((void *) __u + 4);			      \
       case 8:								      \
	 __u->__ui = __c * 0x01010101;					      \
	 __u = __extension__ ((void *) __u + 4);			      \
       case 4:								      \
	 __u->__ui = __c * 0x01010101;					      \
       case 0:								      \
	 break;								      \
       }								      \
									      \
     __s; })
#  else
# 181 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#   define memset(s, c, n) \
  (__extension__ (__builtin_constant_p (c) && (c) == '\0'		      \
		  ? ({ void *__s = (s); __bzero (__s, n); __s; })	      \
		  : memset (s, c, n)))
#  endif
# 186 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
# endif
# 187 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4

/* GCC < 3.0 optimizes memset(s, 0, n) but not bzero(s, n).
   The optimization is broken before EGCS 1.1.
   GCC 3.0+ has __builtin_bzero as well, but at least till GCC 3.4
   if it decides to call the library function, it calls memset
   and not bzero.  */
# if __GNUC_PREREQ (2, 91)
#  define __bzero(s, n) __builtin_memset (s, '\0', n)
# endif
# 196 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4

#endif
# 198 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4


/* Copy N bytes from SRC to DEST, returning pointer to byte following the
   last copied.  */
#ifdef __USE_GNU
# if !defined _HAVE_STRING_ARCH_mempcpy || defined _FORCE_INLINES
#  ifndef _HAVE_STRING_ARCH_mempcpy
#   if __GNUC_PREREQ (3, 4)
#    define __mempcpy(dest, src, n) __builtin_mempcpy (dest, src, n)
#   elif __GNUC_PREREQ (3, 0)
# 208 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#    define __mempcpy(dest, src, n) \
  (__extension__ (__builtin_constant_p (src) && __builtin_constant_p (n)      \
		  && __string2_1bptr_p (src) && n <= 8			      \
		  ? __builtin_memcpy (dest, src, n) + (n)		      \
		  : __mempcpy (dest, src, n)))
#   else
# 214 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#    define __mempcpy(dest, src, n) \
  (__extension__ (__builtin_constant_p (src) && __builtin_constant_p (n)      \
		  && __string2_1bptr_p (src) && n <= 8			      \
		  ? __mempcpy_small (dest, __mempcpy_args (src), n)	      \
		  : __mempcpy (dest, src, n)))
#   endif
# 220 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
/* In glibc we use this function frequently but for namespace reasons
   we have to use the name `__mempcpy'.  */
#   define mempcpy(dest, src, n) __mempcpy (dest, src, n)
#  endif
# 224 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4

#  if !__GNUC_PREREQ (3, 0) || defined _FORCE_INLINES
#   if _STRING_ARCH_unaligned
#    ifndef _FORCE_INLINES
#     define __mempcpy_args(src) \
     ((const char *) (src))[0], ((const char *) (src))[2],		      \
     ((const char *) (src))[4], ((const char *) (src))[6],		      \
     __extension__ __STRING2_SMALL_GET16 (src, 0),			      \
     __extension__ __STRING2_SMALL_GET16 (src, 4),			      \
     __extension__ __STRING2_SMALL_GET32 (src, 0),			      \
     __extension__ __STRING2_SMALL_GET32 (src, 4)
#    endif
# 236 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
__STRING_INLINE void *__mempcpy_small (void *, char, char, char, char,
				       __uint16_t, __uint16_t, __uint32_t,
				       __uint32_t, size_t);
__STRING_INLINE void *
__mempcpy_small (void *__dest1,
		 char __src0_1, char __src2_1, char __src4_1, char __src6_1,
		 __uint16_t __src0_2, __uint16_t __src4_2,
		 __uint32_t __src0_4, __uint32_t __src4_4,
		 size_t __srclen)
{
  union {
    __uint32_t __ui;
    __uint16_t __usi;
    unsigned char __uc;
    unsigned char __c;
  } *__u = __dest1;
  switch ((unsigned int) __srclen)
    {
    case 1:
      __u->__c = __src0_1;
      __u = __extension__ ((void *) __u + 1);
      break;
    case 2:
      __u->__usi = __src0_2;
      __u = __extension__ ((void *) __u + 2);
      break;
    case 3:
      __u->__usi = __src0_2;
      __u = __extension__ ((void *) __u + 2);
      __u->__c = __src2_1;
      __u = __extension__ ((void *) __u + 1);
      break;
    case 4:
      __u->__ui = __src0_4;
      __u = __extension__ ((void *) __u + 4);
      break;
    case 5:
      __u->__ui = __src0_4;
      __u = __extension__ ((void *) __u + 4);
      __u->__c = __src4_1;
      __u = __extension__ ((void *) __u + 1);
      break;
    case 6:
      __u->__ui = __src0_4;
      __u = __extension__ ((void *) __u + 4);
      __u->__usi = __src4_2;
      __u = __extension__ ((void *) __u + 2);
      break;
    case 7:
      __u->__ui = __src0_4;
      __u = __extension__ ((void *) __u + 4);
      __u->__usi = __src4_2;
      __u = __extension__ ((void *) __u + 2);
      __u->__c = __src6_1;
      __u = __extension__ ((void *) __u + 1);
      break;
    case 8:
      __u->__ui = __src0_4;
      __u = __extension__ ((void *) __u + 4);
      __u->__ui = __src4_4;
      __u = __extension__ ((void *) __u + 4);
      break;
    }
  return (void *) __u;
}
#   else
# 302 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#    ifndef _FORCE_INLINES
#     define __mempcpy_args(src) \
     ((const char *) (src))[0],						      \
     __extension__ ((__STRING2_COPY_ARR2)				      \
      { { ((const char *) (src))[0], ((const char *) (src))[1] } }),	      \
     __extension__ ((__STRING2_COPY_ARR3)				      \
      { { ((const char *) (src))[0], ((const char *) (src))[1],		      \
	  ((const char *) (src))[2] } }),				      \
     __extension__ ((__STRING2_COPY_ARR4)				      \
      { { ((const char *) (src))[0], ((const char *) (src))[1],		      \
	  ((const char *) (src))[2], ((const char *) (src))[3] } }),	      \
     __extension__ ((__STRING2_COPY_ARR5)				      \
      { { ((const char *) (src))[0], ((const char *) (src))[1],		      \
	  ((const char *) (src))[2], ((const char *) (src))[3],		      \
	  ((const char *) (src))[4] } }),				      \
     __extension__ ((__STRING2_COPY_ARR6)				      \
      { { ((const char *) (src))[0], ((const char *) (src))[1],		      \
	  ((const char *) (src))[2], ((const char *) (src))[3],		      \
	  ((const char *) (src))[4], ((const char *) (src))[5] } }),	      \
     __extension__ ((__STRING2_COPY_ARR7)				      \
      { { ((const char *) (src))[0], ((const char *) (src))[1],		      \
	  ((const char *) (src))[2], ((const char *) (src))[3],		      \
	  ((const char *) (src))[4], ((const char *) (src))[5],		      \
	  ((const char *) (src))[6] } }),				      \
     __extension__ ((__STRING2_COPY_ARR8)				      \
      { { ((const char *) (src))[0], ((const char *) (src))[1],		      \
	  ((const char *) (src))[2], ((const char *) (src))[3],		      \
	  ((const char *) (src))[4], ((const char *) (src))[5],		      \
	  ((const char *) (src))[6], ((const char *) (src))[7] } })
#    endif
# 332 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
__STRING_INLINE void *__mempcpy_small (void *, char, __STRING2_COPY_ARR2,
				       __STRING2_COPY_ARR3,
				       __STRING2_COPY_ARR4,
				       __STRING2_COPY_ARR5,
				       __STRING2_COPY_ARR6,
				       __STRING2_COPY_ARR7,
				       __STRING2_COPY_ARR8, size_t);
__STRING_INLINE void *
__mempcpy_small (void *__dest, char __src1,
		 __STRING2_COPY_ARR2 __src2, __STRING2_COPY_ARR3 __src3,
		 __STRING2_COPY_ARR4 __src4, __STRING2_COPY_ARR5 __src5,
		 __STRING2_COPY_ARR6 __src6, __STRING2_COPY_ARR7 __src7,
		 __STRING2_COPY_ARR8 __src8, size_t __srclen)
{
  union {
    char __c;
    __STRING2_COPY_ARR2 __sca2;
    __STRING2_COPY_ARR3 __sca3;
    __STRING2_COPY_ARR4 __sca4;
    __STRING2_COPY_ARR5 __sca5;
    __STRING2_COPY_ARR6 __sca6;
    __STRING2_COPY_ARR7 __sca7;
    __STRING2_COPY_ARR8 __sca8;
  } *__u = __dest;
  switch ((unsigned int) __srclen)
    {
    case 1:
      __u->__c = __src1;
      break;
    case 2:
      __extension__ __u->__sca2 = __src2;
      break;
    case 3:
      __extension__ __u->__sca3 = __src3;
      break;
    case 4:
      __extension__ __u->__sca4 = __src4;
      break;
    case 5:
      __extension__ __u->__sca5 = __src5;
      break;
    case 6:
      __extension__ __u->__sca6 = __src6;
      break;
    case 7:
      __extension__ __u->__sca7 = __src7;
      break;
    case 8:
      __extension__ __u->__sca8 = __src8;
      break;
    }
  return __extension__ ((void *) __u + __srclen);
}
#   endif
# 386 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#  endif
# 387 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
# endif
# 388 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#endif
# 389 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4


/* Return pointer to C in S.  */
#ifndef _HAVE_STRING_ARCH_strchr
extern void *__rawmemchr (const void *__s, int __c);
# if __GNUC_PREREQ (3, 2)
#  define strchr(s, c) \
  (__extension__ (__builtin_constant_p (c) && !__builtin_constant_p (s)	      \
		  && (c) == '\0'					      \
		  ? (char *) __rawmemchr (s, c)				      \
		  : __builtin_strchr (s, c)))
# else
# 401 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#  define strchr(s, c) \
  (__extension__ (__builtin_constant_p (c) && (c) == '\0'		      \
		  ? (char *) __rawmemchr (s, c)				      \
		  : strchr (s, c)))
# endif
# 406 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#endif
# 407 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4


/* Copy SRC to DEST.  */
#if (!defined _HAVE_STRING_ARCH_strcpy && !__GNUC_PREREQ (3, 0)) \
    || defined _FORCE_INLINES
# if !defined _HAVE_STRING_ARCH_strcpy && !__GNUC_PREREQ (3, 0)
#  define strcpy(dest, src) \
  (__extension__ (__builtin_constant_p (src)				      \
		  ? (__string2_1bptr_p (src) && strlen (src) + 1 <= 8	      \
		     ? __strcpy_small (dest, __strcpy_args (src),	      \
				       strlen (src) + 1)		      \
		     : (char *) memcpy (dest, src, strlen (src) + 1))	      \
		  : strcpy (dest, src)))
# endif
# 421 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4

# if _STRING_ARCH_unaligned
#  ifndef _FORCE_INLINES
#   define __strcpy_args(src) \
     __extension__ __STRING2_SMALL_GET16 (src, 0),			      \
     __extension__ __STRING2_SMALL_GET16 (src, 4),			      \
     __extension__ __STRING2_SMALL_GET32 (src, 0),			      \
     __extension__ __STRING2_SMALL_GET32 (src, 4)
#  endif
# 430 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
__STRING_INLINE char *__strcpy_small (char *, __uint16_t, __uint16_t,
				      __uint32_t, __uint32_t, size_t);
__STRING_INLINE char *
__strcpy_small (char *__dest,
		__uint16_t __src0_2, __uint16_t __src4_2,
		__uint32_t __src0_4, __uint32_t __src4_4,
		size_t __srclen)
{
  union {
    __uint32_t __ui;
    __uint16_t __usi;
    unsigned char __uc;
  } *__u = (void *) __dest;
  switch ((unsigned int) __srclen)
    {
    case 1:
      __u->__uc = '\0';
      break;
    case 2:
      __u->__usi = __src0_2;
      break;
    case 3:
      __u->__usi = __src0_2;
      __u = __extension__ ((void *) __u + 2);
      __u->__uc = '\0';
      break;
    case 4:
      __u->__ui = __src0_4;
      break;
    case 5:
      __u->__ui = __src0_4;
      __u = __extension__ ((void *) __u + 4);
      __u->__uc = '\0';
      break;
    case 6:
      __u->__ui = __src0_4;
      __u = __extension__ ((void *) __u + 4);
      __u->__usi = __src4_2;
      break;
    case 7:
      __u->__ui = __src0_4;
      __u = __extension__ ((void *) __u + 4);
      __u->__usi = __src4_2;
      __u = __extension__ ((void *) __u + 2);
      __u->__uc = '\0';
      break;
    case 8:
      __u->__ui = __src0_4;
      __u = __extension__ ((void *) __u + 4);
      __u->__ui = __src4_4;
      break;
    }
  return __dest;
}
# else
# 485 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#  ifndef _FORCE_INLINES
#   define __strcpy_args(src) \
     __extension__ ((__STRING2_COPY_ARR2)				      \
      { { ((const char *) (src))[0], '\0' } }),				      \
     __extension__ ((__STRING2_COPY_ARR3)				      \
      { { ((const char *) (src))[0], ((const char *) (src))[1],		      \
	  '\0' } }),							      \
     __extension__ ((__STRING2_COPY_ARR4)				      \
      { { ((const char *) (src))[0], ((const char *) (src))[1],		      \
	  ((const char *) (src))[2], '\0' } }),				      \
     __extension__ ((__STRING2_COPY_ARR5)				      \
      { { ((const char *) (src))[0], ((const char *) (src))[1],		      \
	  ((const char *) (src))[2], ((const char *) (src))[3],		      \
	  '\0' } }),							      \
     __extension__ ((__STRING2_COPY_ARR6)				      \
      { { ((const char *) (src))[0], ((const char *) (src))[1],		      \
	  ((const char *) (src))[2], ((const char *) (src))[3],		      \
	  ((const char *) (src))[4], '\0' } }),				      \
     __extension__ ((__STRING2_COPY_ARR7)				      \
      { { ((const char *) (src))[0], ((const char *) (src))[1],		      \
	  ((const char *) (src))[2], ((const char *) (src))[3],		      \
	  ((const char *) (src))[4], ((const char *) (src))[5],		      \
	  '\0' } }),							      \
     __extension__ ((__STRING2_COPY_ARR8)				      \
      { { ((const char *) (src))[0], ((const char *) (src))[1],		      \
	  ((const char *) (src))[2], ((const char *) (src))[3],		      \
	  ((const char *) (src))[4], ((const char *) (src))[5],		      \
	  ((const char *) (src))[6], '\0' } })
#  endif
# 514 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
__STRING_INLINE char *__strcpy_small (char *, __STRING2_COPY_ARR2,
				      __STRING2_COPY_ARR3,
				      __STRING2_COPY_ARR4,
				      __STRING2_COPY_ARR5,
				      __STRING2_COPY_ARR6,
				      __STRING2_COPY_ARR7,
				      __STRING2_COPY_ARR8, size_t);
__STRING_INLINE char *
__strcpy_small (char *__dest,
		__STRING2_COPY_ARR2 __src2, __STRING2_COPY_ARR3 __src3,
		__STRING2_COPY_ARR4 __src4, __STRING2_COPY_ARR5 __src5,
		__STRING2_COPY_ARR6 __src6, __STRING2_COPY_ARR7 __src7,
		__STRING2_COPY_ARR8 __src8, size_t __srclen)
{
  union {
    char __c;
    __STRING2_COPY_ARR2 __sca2;
    __STRING2_COPY_ARR3 __sca3;
    __STRING2_COPY_ARR4 __sca4;
    __STRING2_COPY_ARR5 __sca5;
    __STRING2_COPY_ARR6 __sca6;
    __STRING2_COPY_ARR7 __sca7;
    __STRING2_COPY_ARR8 __sca8;
  } *__u = (void *) __dest;
  switch ((unsigned int) __srclen)
    {
    case 1:
      __u->__c = '\0';
      break;
    case 2:
      __extension__ __u->__sca2 = __src2;
      break;
    case 3:
      __extension__ __u->__sca3 = __src3;
      break;
    case 4:
      __extension__ __u->__sca4 = __src4;
      break;
    case 5:
      __extension__ __u->__sca5 = __src5;
      break;
    case 6:
      __extension__ __u->__sca6 = __src6;
      break;
    case 7:
      __extension__ __u->__sca7 = __src7;
      break;
    case 8:
      __extension__ __u->__sca8 = __src8;
      break;
  }
  return __dest;
}
# endif
# 568 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#endif
# 569 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4


/* Copy SRC to DEST, returning pointer to final NUL byte.  */
#ifdef __USE_GNU
# if !defined _HAVE_STRING_ARCH_stpcpy || defined _FORCE_INLINES
#  ifndef _HAVE_STRING_ARCH_stpcpy
#   if __GNUC_PREREQ (3, 4)
#    define __stpcpy(dest, src) __builtin_stpcpy (dest, src)
#   elif __GNUC_PREREQ (3, 0)
# 578 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#    define __stpcpy(dest, src) \
  (__extension__ (__builtin_constant_p (src)				      \
		  ? (__string2_1bptr_p (src) && strlen (src) + 1 <= 8	      \
		     ? __builtin_strcpy (dest, src) + strlen (src)	      \
		     : ((char *) (__mempcpy) (dest, src, strlen (src) + 1)    \
			- 1))						      \
		  : __stpcpy (dest, src)))
#   else
# 586 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#    define __stpcpy(dest, src) \
  (__extension__ (__builtin_constant_p (src)				      \
		  ? (__string2_1bptr_p (src) && strlen (src) + 1 <= 8	      \
		     ? __stpcpy_small (dest, __stpcpy_args (src),	      \
				       strlen (src) + 1)		      \
		     : ((char *) (__mempcpy) (dest, src, strlen (src) + 1)    \
			- 1))						      \
		  : __stpcpy (dest, src)))
#   endif
# 595 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
/* In glibc we use this function frequently but for namespace reasons
   we have to use the name `__stpcpy'.  */
#   define stpcpy(dest, src) __stpcpy (dest, src)
#  endif
# 599 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4

#  if !__GNUC_PREREQ (3, 0) || defined _FORCE_INLINES
#   if _STRING_ARCH_unaligned
#    ifndef _FORCE_INLINES
#     define __stpcpy_args(src) \
     __extension__ __STRING2_SMALL_GET16 (src, 0),			      \
     __extension__ __STRING2_SMALL_GET16 (src, 4),			      \
     __extension__ __STRING2_SMALL_GET32 (src, 0),			      \
     __extension__ __STRING2_SMALL_GET32 (src, 4)
#    endif
# 609 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
__STRING_INLINE char *__stpcpy_small (char *, __uint16_t, __uint16_t,
				      __uint32_t, __uint32_t, size_t);
__STRING_INLINE char *
__stpcpy_small (char *__dest,
		__uint16_t __src0_2, __uint16_t __src4_2,
		__uint32_t __src0_4, __uint32_t __src4_4,
		size_t __srclen)
{
  union {
    unsigned int __ui;
    unsigned short int __usi;
    unsigned char __uc;
    char __c;
  } *__u = (void *) __dest;
  switch ((unsigned int) __srclen)
    {
    case 1:
      __u->__uc = '\0';
      break;
    case 2:
      __u->__usi = __src0_2;
      __u = __extension__ ((void *) __u + 1);
      break;
    case 3:
      __u->__usi = __src0_2;
      __u = __extension__ ((void *) __u + 2);
      __u->__uc = '\0';
      break;
    case 4:
      __u->__ui = __src0_4;
      __u = __extension__ ((void *) __u + 3);
      break;
    case 5:
      __u->__ui = __src0_4;
      __u = __extension__ ((void *) __u + 4);
      __u->__uc = '\0';
      break;
    case 6:
      __u->__ui = __src0_4;
      __u = __extension__ ((void *) __u + 4);
      __u->__usi = __src4_2;
      __u = __extension__ ((void *) __u + 1);
      break;
    case 7:
      __u->__ui = __src0_4;
      __u = __extension__ ((void *) __u + 4);
      __u->__usi = __src4_2;
      __u = __extension__ ((void *) __u + 2);
      __u->__uc = '\0';
      break;
    case 8:
      __u->__ui = __src0_4;
      __u = __extension__ ((void *) __u + 4);
      __u->__ui = __src4_4;
      __u = __extension__ ((void *) __u + 3);
      break;
    }
  return &__u->__c;
}
#   else
# 669 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#    ifndef _FORCE_INLINES
#     define __stpcpy_args(src) \
     __extension__ ((__STRING2_COPY_ARR2)				      \
      { { ((const char *) (src))[0], '\0' } }),				      \
     __extension__ ((__STRING2_COPY_ARR3)				      \
      { { ((const char *) (src))[0], ((const char *) (src))[1],		      \
	  '\0' } }),							      \
     __extension__ ((__STRING2_COPY_ARR4)				      \
      { { ((const char *) (src))[0], ((const char *) (src))[1],		      \
	  ((const char *) (src))[2], '\0' } }),				      \
     __extension__ ((__STRING2_COPY_ARR5)				      \
      { { ((const char *) (src))[0], ((const char *) (src))[1],		      \
	  ((const char *) (src))[2], ((const char *) (src))[3],		      \
	  '\0' } }),							      \
     __extension__ ((__STRING2_COPY_ARR6)				      \
      { { ((const char *) (src))[0], ((const char *) (src))[1],		      \
	  ((const char *) (src))[2], ((const char *) (src))[3],		      \
	  ((const char *) (src))[4], '\0' } }),				      \
     __extension__ ((__STRING2_COPY_ARR7)				      \
      { { ((const char *) (src))[0], ((const char *) (src))[1],		      \
	  ((const char *) (src))[2], ((const char *) (src))[3],		      \
	  ((const char *) (src))[4], ((const char *) (src))[5],		      \
	  '\0' } }),							      \
     __extension__ ((__STRING2_COPY_ARR8)				      \
      { { ((const char *) (src))[0], ((const char *) (src))[1],		      \
	  ((const char *) (src))[2], ((const char *) (src))[3],		      \
	  ((const char *) (src))[4], ((const char *) (src))[5],		      \
	  ((const char *) (src))[6], '\0' } })
#    endif
# 698 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
__STRING_INLINE char *__stpcpy_small (char *, __STRING2_COPY_ARR2,
				      __STRING2_COPY_ARR3,
				      __STRING2_COPY_ARR4,
				      __STRING2_COPY_ARR5,
				      __STRING2_COPY_ARR6,
				      __STRING2_COPY_ARR7,
				      __STRING2_COPY_ARR8, size_t);
__STRING_INLINE char *
__stpcpy_small (char *__dest,
		__STRING2_COPY_ARR2 __src2, __STRING2_COPY_ARR3 __src3,
		__STRING2_COPY_ARR4 __src4, __STRING2_COPY_ARR5 __src5,
		__STRING2_COPY_ARR6 __src6, __STRING2_COPY_ARR7 __src7,
		__STRING2_COPY_ARR8 __src8, size_t __srclen)
{
  union {
    char __c;
    __STRING2_COPY_ARR2 __sca2;
    __STRING2_COPY_ARR3 __sca3;
    __STRING2_COPY_ARR4 __sca4;
    __STRING2_COPY_ARR5 __sca5;
    __STRING2_COPY_ARR6 __sca6;
    __STRING2_COPY_ARR7 __sca7;
    __STRING2_COPY_ARR8 __sca8;
  } *__u = (void *) __dest;
  switch ((unsigned int) __srclen)
    {
    case 1:
      __u->__c = '\0';
      break;
    case 2:
      __extension__ __u->__sca2 = __src2;
      break;
    case 3:
      __extension__ __u->__sca3 = __src3;
      break;
    case 4:
      __extension__ __u->__sca4 = __src4;
      break;
    case 5:
      __extension__ __u->__sca5 = __src5;
      break;
    case 6:
      __extension__ __u->__sca6 = __src6;
      break;
    case 7:
      __extension__ __u->__sca7 = __src7;
      break;
    case 8:
      __extension__ __u->__sca8 = __src8;
      break;
  }
  return __dest + __srclen - 1;
}
#   endif
# 752 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#  endif
# 753 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
# endif
# 754 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#endif
# 755 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4


/* Copy no more than N characters of SRC to DEST.  */
#ifndef _HAVE_STRING_ARCH_strncpy
# if __GNUC_PREREQ (3, 2)
#  define strncpy(dest, src, n) __builtin_strncpy (dest, src, n)
# else
# 762 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#  define strncpy(dest, src, n) \
  (__extension__ (__builtin_constant_p (src) && __builtin_constant_p (n)      \
		  ? (strlen (src) + 1 >= ((size_t) (n))			      \
		     ? (char *) memcpy (dest, src, n)			      \
		     : strncpy (dest, src, n))				      \
		  : strncpy (dest, src, n)))
# endif
# 769 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#endif
# 770 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4


/* Append no more than N characters from SRC onto DEST.  */
#ifndef _HAVE_STRING_ARCH_strncat
# ifdef _USE_STRING_ARCH_strchr
#  define strncat(dest, src, n) \
  (__extension__ ({ char *__dest = (dest);				      \
		    __builtin_constant_p (src) && __builtin_constant_p (n)    \
		    ? (strlen (src) < ((size_t) (n))			      \
		       ? strcat (__dest, src)				      \
		       : (*((char *) __mempcpy (strchr (__dest, '\0'),	      \
						src, n)) = '\0', __dest))     \
		    : strncat (dest, src, n); }))
# elif __GNUC_PREREQ (3, 2)
# 784 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#  define strncat(dest, src, n) __builtin_strncat (dest, src, n)
# else
# 786 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#  define strncat(dest, src, n) \
  (__extension__ (__builtin_constant_p (src) && __builtin_constant_p (n)      \
		  ? (strlen (src) < ((size_t) (n))			      \
		     ? strcat (dest, src)				      \
		     : strncat (dest, src, n))				      \
		  : strncat (dest, src, n)))
# endif
# 793 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#endif
# 794 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4


/* Compare characters of S1 and S2.  */
#ifndef _HAVE_STRING_ARCH_strcmp
# if __GNUC_PREREQ (3, 2)
#  define strcmp(s1, s2) \
  __extension__								      \
  ({ size_t __s1_len, __s2_len;						      \
     (__builtin_constant_p (s1) && __builtin_constant_p (s2)		      \
      && (__s1_len = __builtin_strlen (s1), __s2_len = __builtin_strlen (s2), \
	  (!__string2_1bptr_p (s1) || __s1_len >= 4)			      \
	  && (!__string2_1bptr_p (s2) || __s2_len >= 4))		      \
      ? __builtin_strcmp (s1, s2)					      \
      : (__builtin_constant_p (s1) && __string2_1bptr_p (s1)		      \
	 && (__s1_len = __builtin_strlen (s1), __s1_len < 4)		      \
	 ? (__builtin_constant_p (s2) && __string2_1bptr_p (s2)		      \
	    ? __builtin_strcmp (s1, s2)					      \
	    : __strcmp_cg (s1, s2, __s1_len))				      \
	 : (__builtin_constant_p (s2) && __string2_1bptr_p (s2)		      \
	    && (__s2_len = __builtin_strlen (s2), __s2_len < 4)		      \
	    ? (__builtin_constant_p (s1) && __string2_1bptr_p (s1)	      \
	       ? __builtin_strcmp (s1, s2)				      \
	       : __strcmp_gc (s1, s2, __s2_len))			      \
	    : __builtin_strcmp (s1, s2)))); })
# else
# 819 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#  define strcmp(s1, s2) \
  __extension__								      \
  ({ size_t __s1_len, __s2_len;						      \
     (__builtin_constant_p (s1) && __builtin_constant_p (s2)		      \
      && (__s1_len = strlen (s1), __s2_len = strlen (s2),		      \
	  (!__string2_1bptr_p (s1) || __s1_len >= 4)			      \
	  && (!__string2_1bptr_p (s2) || __s2_len >= 4))		      \
      ? memcmp ((const char *) (s1), (const char *) (s2),		      \
		(__s1_len < __s2_len ? __s1_len : __s2_len) + 1)	      \
      : (__builtin_constant_p (s1) && __string2_1bptr_p (s1)		      \
	 && (__s1_len = strlen (s1), __s1_len < 4)			      \
	 ? (__builtin_constant_p (s2) && __string2_1bptr_p (s2)		      \
	    ? __strcmp_cc (s1, s2, __s1_len)				      \
	    : __strcmp_cg (s1, s2, __s1_len))				      \
	 : (__builtin_constant_p (s2) && __string2_1bptr_p (s2)		      \
	    && (__s2_len = strlen (s2), __s2_len < 4)			      \
	    ? (__builtin_constant_p (s1) && __string2_1bptr_p (s1)	      \
	       ? __strcmp_cc (s1, s2, __s2_len)				      \
	       : __strcmp_gc (s1, s2, __s2_len))			      \
	    : strcmp (s1, s2)))); })
# endif
# 840 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4

# define __strcmp_cc(s1, s2, l) \
  (__extension__ ({ int __result =					      \
		      (((const unsigned char *) (const char *) (s1))[0]	      \
		       - ((const unsigned char *) (const char *)(s2))[0]);    \
		    if (l > 0 && __result == 0)				      \
		      {							      \
			__result = (((const unsigned char *)		      \
				     (const char *) (s1))[1]		      \
				    - ((const unsigned char *)		      \
				       (const char *) (s2))[1]);	      \
			if (l > 1 && __result == 0)			      \
			  {						      \
			    __result =					      \
			      (((const unsigned char *)			      \
				(const char *) (s1))[2]			      \
			       - ((const unsigned char *)		      \
				  (const char *) (s2))[2]);		      \
			    if (l > 2 && __result == 0)			      \
			      __result =				      \
				(((const unsigned char *)		      \
				  (const char *) (s1))[3]		      \
				 - ((const unsigned char *)		      \
				    (const char *) (s2))[3]);		      \
			  }						      \
		      }							      \
		    __result; }))

# define __strcmp_cg(s1, s2, l1) \
  (__extension__ ({ const unsigned char *__s2 =				      \
		      (const unsigned char *) (const char *) (s2);	      \
		    int __result =					      \
		      (((const unsigned char *) (const char *) (s1))[0]	      \
		       - __s2[0]);					      \
		    if (l1 > 0 && __result == 0)			      \
		      {							      \
			__result = (((const unsigned char *)		      \
				     (const char *) (s1))[1] - __s2[1]);      \
			if (l1 > 1 && __result == 0)			      \
			  {						      \
			    __result = (((const unsigned char *)	      \
					 (const char *) (s1))[2] - __s2[2]);  \
			    if (l1 > 2 && __result == 0)		      \
			      __result = (((const unsigned char *)	      \
					  (const char *)  (s1))[3]	      \
					  - __s2[3]);			      \
			  }						      \
		      }							      \
		    __result; }))

# define __strcmp_gc(s1, s2, l2) (- __strcmp_cg (s2, s1, l2))
#endif
# 892 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4


/* Compare N characters of S1 and S2.  */
#ifndef _HAVE_STRING_ARCH_strncmp
# define strncmp(s1, s2, n)						      \
  (__extension__ (__builtin_constant_p (n)				      \
		  && ((__builtin_constant_p (s1)			      \
		       && strlen (s1) < ((size_t) (n)))			      \
		      || (__builtin_constant_p (s2)			      \
			  && strlen (s2) < ((size_t) (n))))		      \
		  ? strcmp (s1, s2) : strncmp (s1, s2, n)))
#endif
# 904 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4


/* Return the length of the initial segment of S which
   consists entirely of characters not in REJECT.  */
#if !defined _HAVE_STRING_ARCH_strcspn || defined _FORCE_INLINES
# ifndef _HAVE_STRING_ARCH_strcspn
#  if __GNUC_PREREQ (3, 2)
#   define strcspn(s, reject) \
  __extension__								      \
  ({ char __r0, __r1, __r2;						      \
     (__builtin_constant_p (reject) && __string2_1bptr_p (reject)	      \
      ? ((__builtin_constant_p (s) && __string2_1bptr_p (s))		      \
	 ? __builtin_strcspn (s, reject)				      \
	 : ((__r0 = ((const char *) (reject))[0], __r0 == '\0')		      \
	    ? strlen (s)						      \
	    : ((__r1 = ((const char *) (reject))[1], __r1 == '\0')	      \
	       ? __strcspn_c1 (s, __r0)					      \
	       : ((__r2 = ((const char *) (reject))[2], __r2 == '\0')	      \
		  ? __strcspn_c2 (s, __r0, __r1)			      \
		  : (((const char *) (reject))[3] == '\0'		      \
		     ? __strcspn_c3 (s, __r0, __r1, __r2)		      \
		     : __builtin_strcspn (s, reject))))))		      \
      : __builtin_strcspn (s, reject)); })
#  else
# 928 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#   define strcspn(s, reject) \
  __extension__								      \
  ({ char __r0, __r1, __r2;						      \
     (__builtin_constant_p (reject) && __string2_1bptr_p (reject)	      \
      ? ((__r0 = ((const char *) (reject))[0], __r0 == '\0')		      \
	 ? strlen (s)							      \
	 : ((__r1 = ((const char *) (reject))[1], __r1 == '\0')		      \
	    ? __strcspn_c1 (s, __r0)					      \
	    : ((__r2 = ((const char *) (reject))[2], __r2 == '\0')	      \
	       ? __strcspn_c2 (s, __r0, __r1)				      \
	       : (((const char *) (reject))[3] == '\0'			      \
		  ? __strcspn_c3 (s, __r0, __r1, __r2)			      \
		  : strcspn (s, reject)))))				      \
      : strcspn (s, reject)); })
#  endif
# 943 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
# endif
# 944 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4

__STRING_INLINE size_t __strcspn_c1 (const char *__s, int __reject);
__STRING_INLINE size_t
__strcspn_c1 (const char *__s, int __reject)
{
  size_t __result = 0;
  while (__s[__result] != '\0' && __s[__result] != __reject)
    ++__result;
  return __result;
}

__STRING_INLINE size_t __strcspn_c2 (const char *__s, int __reject1,
				     int __reject2);
__STRING_INLINE size_t
__strcspn_c2 (const char *__s, int __reject1, int __reject2)
{
  size_t __result = 0;
  while (__s[__result] != '\0' && __s[__result] != __reject1
	 && __s[__result] != __reject2)
    ++__result;
  return __result;
}

__STRING_INLINE size_t __strcspn_c3 (const char *__s, int __reject1,
				     int __reject2, int __reject3);
__STRING_INLINE size_t
__strcspn_c3 (const char *__s, int __reject1, int __reject2,
	      int __reject3)
{
  size_t __result = 0;
  while (__s[__result] != '\0' && __s[__result] != __reject1
	 && __s[__result] != __reject2 && __s[__result] != __reject3)
    ++__result;
  return __result;
}
#endif
# 980 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4


/* Return the length of the initial segment of S which
   consists entirely of characters in ACCEPT.  */
#if !defined _HAVE_STRING_ARCH_strspn || defined _FORCE_INLINES
# ifndef _HAVE_STRING_ARCH_strspn
#  if __GNUC_PREREQ (3, 2)
#   define strspn(s, accept) \
  __extension__								      \
  ({ char __a0, __a1, __a2;						      \
     (__builtin_constant_p (accept) && __string2_1bptr_p (accept)	      \
      ? ((__builtin_constant_p (s) && __string2_1bptr_p (s))		      \
	 ? __builtin_strspn (s, accept)					      \
	 : ((__a0 = ((const char *) (accept))[0], __a0 == '\0')		      \
	    ? ((void) (s), (size_t) 0)					      \
	    : ((__a1 = ((const char *) (accept))[1], __a1 == '\0')	      \
	       ? __strspn_c1 (s, __a0)					      \
	       : ((__a2 = ((const char *) (accept))[2], __a2 == '\0')	      \
		  ? __strspn_c2 (s, __a0, __a1)				      \
		  : (((const char *) (accept))[3] == '\0'		      \
		     ? __strspn_c3 (s, __a0, __a1, __a2)		      \
		     : __builtin_strspn (s, accept))))))		      \
      : __builtin_strspn (s, accept)); })
#  else
# 1004 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#   define strspn(s, accept) \
  __extension__								      \
  ({ char __a0, __a1, __a2;						      \
     (__builtin_constant_p (accept) && __string2_1bptr_p (accept)	      \
      ? ((__a0 = ((const char *) (accept))[0], __a0 == '\0')		      \
	 ? ((void) (s), (size_t) 0)					      \
	 : ((__a1 = ((const char *) (accept))[1], __a1 == '\0')		      \
	    ? __strspn_c1 (s, __a0)					      \
	    : ((__a2 = ((const char *) (accept))[2], __a2 == '\0')	      \
	       ? __strspn_c2 (s, __a0, __a1)				      \
	       : (((const char *) (accept))[3] == '\0'			      \
		  ? __strspn_c3 (s, __a0, __a1, __a2)			      \
		  : strspn (s, accept)))))				      \
      : strspn (s, accept)); })
#  endif
# 1019 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
# endif
# 1020 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4

__STRING_INLINE size_t __strspn_c1 (const char *__s, int __accept);
__STRING_INLINE size_t
__strspn_c1 (const char *__s, int __accept)
{
  size_t __result = 0;
  /* Please note that __accept never can be '\0'.  */
  while (__s[__result] == __accept)
    ++__result;
  return __result;
}

__STRING_INLINE size_t __strspn_c2 (const char *__s, int __accept1,
				    int __accept2);
__STRING_INLINE size_t
__strspn_c2 (const char *__s, int __accept1, int __accept2)
{
  size_t __result = 0;
  /* Please note that __accept1 and __accept2 never can be '\0'.  */
  while (__s[__result] == __accept1 || __s[__result] == __accept2)
    ++__result;
  return __result;
}

__STRING_INLINE size_t __strspn_c3 (const char *__s, int __accept1,
				    int __accept2, int __accept3);
__STRING_INLINE size_t
__strspn_c3 (const char *__s, int __accept1, int __accept2, int __accept3)
{
  size_t __result = 0;
  /* Please note that __accept1 to __accept3 never can be '\0'.  */
  while (__s[__result] == __accept1 || __s[__result] == __accept2
	 || __s[__result] == __accept3)
    ++__result;
  return __result;
}
#endif
# 1057 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4


/* Find the first occurrence in S of any character in ACCEPT.  */
#if !defined _HAVE_STRING_ARCH_strpbrk || defined _FORCE_INLINES
# ifndef _HAVE_STRING_ARCH_strpbrk
#  if __GNUC_PREREQ (3, 2)
#   define strpbrk(s, accept) \
  __extension__								      \
  ({ char __a0, __a1, __a2;						      \
     (__builtin_constant_p (accept) && __string2_1bptr_p (accept)	      \
      ? ((__builtin_constant_p (s) && __string2_1bptr_p (s))		      \
	 ? __builtin_strpbrk (s, accept)				      \
	 : ((__a0 = ((const char  *) (accept))[0], __a0 == '\0')	      \
	    ? ((void) (s), (char *) NULL)				      \
	    : ((__a1 = ((const char *) (accept))[1], __a1 == '\0')	      \
	       ? __builtin_strchr (s, __a0)				      \
	       : ((__a2 = ((const char *) (accept))[2], __a2 == '\0')	      \
		  ? __strpbrk_c2 (s, __a0, __a1)			      \
		  : (((const char *) (accept))[3] == '\0'		      \
		     ? __strpbrk_c3 (s, __a0, __a1, __a2)		      \
		     : __builtin_strpbrk (s, accept))))))		      \
      : __builtin_strpbrk (s, accept)); })
#  else
# 1080 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#   define strpbrk(s, accept) \
  __extension__								      \
  ({ char __a0, __a1, __a2;						      \
     (__builtin_constant_p (accept) && __string2_1bptr_p (accept)	      \
      ? ((__a0 = ((const char  *) (accept))[0], __a0 == '\0')		      \
	 ? ((void) (s), (char *) NULL)					      \
	 : ((__a1 = ((const char *) (accept))[1], __a1 == '\0')		      \
	    ? strchr (s, __a0)						      \
	    : ((__a2 = ((const char *) (accept))[2], __a2 == '\0')	      \
	       ? __strpbrk_c2 (s, __a0, __a1)				      \
	       : (((const char *) (accept))[3] == '\0'			      \
		  ? __strpbrk_c3 (s, __a0, __a1, __a2)			      \
		  : strpbrk (s, accept)))))				      \
      : strpbrk (s, accept)); })
#  endif
# 1095 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
# endif
# 1096 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4

__STRING_INLINE char *__strpbrk_c2 (const char *__s, int __accept1,
				    int __accept2);
__STRING_INLINE char *
__strpbrk_c2 (const char *__s, int __accept1, int __accept2)
{
  /* Please note that __accept1 and __accept2 never can be '\0'.  */
  while (*__s != '\0' && *__s != __accept1 && *__s != __accept2)
    ++__s;
  return *__s == '\0' ? NULL : (char *) (size_t) __s;
}

__STRING_INLINE char *__strpbrk_c3 (const char *__s, int __accept1,
				    int __accept2, int __accept3);
__STRING_INLINE char *
__strpbrk_c3 (const char *__s, int __accept1, int __accept2, int __accept3)
{
  /* Please note that __accept1 to __accept3 never can be '\0'.  */
  while (*__s != '\0' && *__s != __accept1 && *__s != __accept2
	 && *__s != __accept3)
    ++__s;
  return *__s == '\0' ? NULL : (char *) (size_t) __s;
}
#endif
# 1120 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4


/* Find the first occurrence of NEEDLE in HAYSTACK.  Newer gcc versions
   do this itself.  */
#if !defined _HAVE_STRING_ARCH_strstr && !__GNUC_PREREQ (2, 97)
# define strstr(haystack, needle) \
  (__extension__ (__builtin_constant_p (needle) && __string2_1bptr_p (needle) \
		  ? (((const char *) (needle))[0] == '\0'		      \
		     ? (char *) (size_t) (haystack)			      \
		     : (((const char *) (needle))[1] == '\0'		      \
			? strchr (haystack,				      \
				  ((const char *) (needle))[0]) 	      \
			: strstr (haystack, needle)))			      \
		  : strstr (haystack, needle)))
#endif
# 1135 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4


#if !defined _HAVE_STRING_ARCH_strtok_r || defined _FORCE_INLINES
# ifndef _HAVE_STRING_ARCH_strtok_r
#  define __strtok_r(s, sep, nextp) \
  (__extension__ (__builtin_constant_p (sep) && __string2_1bptr_p (sep)	      \
		  && ((const char *) (sep))[0] != '\0'			      \
		  && ((const char *) (sep))[1] == '\0'			      \
		  ? __strtok_r_1c (s, ((const char *) (sep))[0], nextp)       \
		  : __strtok_r (s, sep, nextp)))
# endif
# 1146 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4

__STRING_INLINE char *__strtok_r_1c (char *__s, char __sep, char **__nextp);
__STRING_INLINE char *
__strtok_r_1c (char *__s, char __sep, char **__nextp)
{
  char *__result;
  if (__s == NULL)
    __s = *__nextp;
  while (*__s == __sep)
    ++__s;
  __result = NULL;
  if (*__s != '\0')
    {
      __result = __s++;
      while (*__s != '\0')
	if (*__s++ == __sep)
	  {
	    __s[-1] = '\0';
	    break;
	  }
    }
  *__nextp = __s;
  return __result;
}
# if defined __USE_POSIX || defined __USE_MISC
#  define strtok_r(s, sep, nextp) __strtok_r (s, sep, nextp)
# endif
# 1173 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#endif
# 1174 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4


#if !defined _HAVE_STRING_ARCH_strsep || defined _FORCE_INLINES
# ifndef _HAVE_STRING_ARCH_strsep

extern char *__strsep_g (char **__stringp, const char *__delim);
#  define __strsep(s, reject) \
  __extension__								      \
  ({ char __r0, __r1, __r2;						      \
     (__builtin_constant_p (reject) && __string2_1bptr_p (reject)	      \
      && (__r0 = ((const char *) (reject))[0],				      \
	  ((const char *) (reject))[0] != '\0')				      \
      ? ((__r1 = ((const char *) (reject))[1],				      \
	 ((const char *) (reject))[1] == '\0')				      \
	 ? __strsep_1c (s, __r0)					      \
	 : ((__r2 = ((const char *) (reject))[2], __r2 == '\0')		      \
	    ? __strsep_2c (s, __r0, __r1)				      \
	    : (((const char *) (reject))[3] == '\0'			      \
	       ? __strsep_3c (s, __r0, __r1, __r2)			      \
	       : __strsep_g (s, reject))))				      \
      : __strsep_g (s, reject)); })
# endif
# 1196 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4

__STRING_INLINE char *__strsep_1c (char **__s, char __reject);
__STRING_INLINE char *
__strsep_1c (char **__s, char __reject)
{
  char *__retval = *__s;
  if (__retval != NULL && (*__s = strchr (__retval, __reject)) != NULL)
    *(*__s)++ = '\0';
  return __retval;
}

__STRING_INLINE char *__strsep_2c (char **__s, char __reject1, char __reject2);
__STRING_INLINE char *
__strsep_2c (char **__s, char __reject1, char __reject2)
{
  char *__retval = *__s;
  if (__retval != NULL)
    {
      char *__cp = __retval;
      while (1)
	{
	  if (*__cp == '\0')
	    {
	      __cp = NULL;
	  break;
	    }
	  if (*__cp == __reject1 || *__cp == __reject2)
	    {
	      *__cp++ = '\0';
	      break;
	    }
	  ++__cp;
	}
      *__s = __cp;
    }
  return __retval;
}

__STRING_INLINE char *__strsep_3c (char **__s, char __reject1, char __reject2,
				   char __reject3);
__STRING_INLINE char *
__strsep_3c (char **__s, char __reject1, char __reject2, char __reject3)
{
  char *__retval = *__s;
  if (__retval != NULL)
    {
      char *__cp = __retval;
      while (1)
	{
	  if (*__cp == '\0')
	    {
	      __cp = NULL;
	  break;
	    }
	  if (*__cp == __reject1 || *__cp == __reject2 || *__cp == __reject3)
	    {
	      *__cp++ = '\0';
	      break;
	    }
	  ++__cp;
	}
      *__s = __cp;
    }
  return __retval;
}
# ifdef __USE_BSD
#  define strsep(s, reject) __strsep (s, reject)
# endif
# 1264 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
#endif
# 1265 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4

/* We need the memory allocation functions for inline strdup().
   Referring to stdlib.h (even minimally) is not allowed
   in any of the tight standards compliant modes.  */
#ifdef __USE_MISC

# if !defined _HAVE_STRING_ARCH_strdup || !defined _HAVE_STRING_ARCH_strndup
#  define __need_malloc_and_calloc
#if 0 /* expanded by -frewrite-includes */
#  include <stdlib.h>
#endif /* expanded by -frewrite-includes */
# 1273 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
# 1274 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
# endif
# 1275 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4

# ifndef _HAVE_STRING_ARCH_strdup

extern char *__strdup (const char *__string) __THROW __attribute_malloc__;
#  define __strdup(s) \
  (__extension__ (__builtin_constant_p (s) && __string2_1bptr_p (s)	      \
		  ? (((const char *) (s))[0] == '\0'			      \
		     ? (char *) calloc ((size_t) 1, (size_t) 1)		      \
		     : ({ size_t __len = strlen (s) + 1;		      \
			  char *__retval = (char *) malloc (__len);	      \
			  if (__retval != NULL)				      \
			    __retval = (char *) memcpy (__retval, s, __len);  \
			  __retval; }))					      \
		  : __strdup (s)))

#  if defined __USE_SVID || defined __USE_BSD || defined __USE_XOPEN_EXTENDED
#   define strdup(s) __strdup (s)
#  endif
# 1293 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
# endif
# 1294 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4

# ifndef _HAVE_STRING_ARCH_strndup

extern char *__strndup (const char *__string, size_t __n)
     __THROW __attribute_malloc__;
#  define __strndup(s, n) \
  (__extension__ (__builtin_constant_p (s) && __string2_1bptr_p (s)	      \
		  ? (((const char *) (s))[0] == '\0'			      \
		     ? (char *) calloc ((size_t) 1, (size_t) 1)		      \
		     : ({ size_t __len = strlen (s) + 1;		      \
			  size_t __n = (n);				      \
			  char *__retval;				      \
			  if (__n < __len)				      \
			    __len = __n + 1;				      \
			  __retval = (char *) malloc (__len);		      \
			  if (__retval != NULL)				      \
			    {						      \
			      __retval[__len - 1] = '\0';		      \
			      __retval = (char *) memcpy (__retval, s,	      \
							  __len - 1);	      \
			    }						      \
			  __retval; }))					      \
		  : __strndup (s, n)))

#  ifdef __USE_GNU
#   define strndup(s, n) __strndup (s, n)
#  endif
# 1321 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
# endif
# 1322 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4

#endif /* Use misc. or use GNU.  */
# 1324 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4

#ifndef _FORCE_INLINES
# undef __STRING_INLINE
#endif
# 1328 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4

#endif /* No string inlines.  */
# 1330 "/usr/include/x86_64-linux-gnu/bits/string2.h" 3 4
# 636 "/usr/include/string.h" 2 3 4
# endif
# 637 "/usr/include/string.h" 3 4

# if __USE_FORTIFY_LEVEL > 0 && defined __fortify_function
/* Functions with security checks.  */
#if 0 /* expanded by -frewrite-includes */
#  include <bits/string3.h>
#endif /* expanded by -frewrite-includes */
# 640 "/usr/include/string.h" 3 4
# 641 "/usr/include/string.h" 3 4
# endif
# 642 "/usr/include/string.h" 3 4
#endif
# 643 "/usr/include/string.h" 3 4

__END_DECLS

#endif /* string.h  */
# 647 "/usr/include/string.h" 3 4
# 5 "oski.c" 2
#if 0 /* expanded by -frewrite-includes */
#include "util.h"
#endif /* expanded by -frewrite-includes */
# 5 "oski.c"
# 1 "./util.h" 1
/**
 * BSD 3-Clause License
 *
 * Copyright (c) 2017, Peter Ahrens All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef UTIL_H
#define UTIL_H
//lo is inclusive, hi is exclusive

static inline int max(int a, int b) {return a > b ? a : b;};
static inline int min(int a, int b) {return a < b ? a : b;};
void random_seed (unsigned long seed);
int random_range (int *stuff, int n, int lo, int hi);
double random_uniform ();
void sort (int *stuff, int n);
int search (const int *stuff, int lo, int hi, int key);
int search_strict (const int *stuff, int lo, int hi, int key);
#endif
# 46 "./util.h"
# 6 "oski.c" 2

#if 0 /* expanded by -frewrite-includes */
#include <cilk/cilk.h>
#endif /* expanded by -frewrite-includes */
# 7 "oski.c"
# 1 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/cilk/cilk.h" 1 3
/*  cilk.h                  -*-C++-*-
 *
 *  @copyright
 *  Copyright (C) 2010-2013, Intel Corporation
 *  All rights reserved.
 *  
 *  @copyright
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *  
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in
 *      the documentation and/or other materials provided with the
 *      distribution.
 *    * Neither the name of Intel Corporation nor the names of its
 *      contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *  
 *  @copyright
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 *  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
 *  WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */
 
/** @file cilk.h
 *
 *  @brief Provides convenient aliases for the Cilk language keywords.
 *
 *  @details
 *  Since Cilk is a nonstandard extension to both C and C++, the Cilk
 *  language keywords all begin with “`_Cilk_`”, which guarantees that they
 *  will not conflict with user-defined identifiers in properly written 
 *  programs, so that “standard” C and C++ programs can safely be
 *  compiled a Cilk-enabled C or C++ compiler.
 *
 *  However, this means that the keywords _look_ like something grafted on to
 *  the base language. Therefore, you can include this header:
 *
 *      #include "cilk/cilk.h"
 *
 *  and then write the Cilk keywords with a “`cilk_`” prefix instead of
 *  “`_Cilk_`”.
 *
 *  @ingroup language
 */
 
 
/** @defgroup language Language Keywords
 *  Definitions having to do with the Cilk language.
 *  @{
 */
 
#ifndef cilk_spawn
# define cilk_spawn _Cilk_spawn ///< Spawn a task that can execute in parallel.
# define cilk_sync  _Cilk_sync  ///< Wait for spawned tasks to complete.
# define cilk_for   _Cilk_for   ///< Execute iterations of a for loop in parallel.
#endif
# 70 "/data/scratch/hjxu/tapir/src/build/lib/clang/6.0.0/include/cilk/cilk.h" 3

/// @}
# 8 "oski.c" 2

char * name() {
  return "oski";
}

/**
 *  Given an m by n CSR matrix A, estimates the fill ratio if the matrix were
 *  converted into b_r by b_c BCSR format. The fill ratio is b_r times b_c times
 *  the number of nonzero blocks in the BCSR format divided by the number of
 *  nonzeros. For each setting of b_r, block rows are completely examined with
 *  probability sigma.
 *
 *  The caller supplies this routine with a maximum row and column block size B,
 *  and this routine returns the estimated fill ratios for all
 *  1 <= b_r, b_c <= B.
 *
 *  This routine assumes the CSR matrix uses full storage, and assumes that
 *  column indicies are sorted.
 *
 *  \param[in] m Logical number of matrix rows
 *  \param[in] n Logical number of matrix columns
 *  \param[in] nnz Logical number of matrix nonzeros
 *  \param[in] *ptr CSR row pointers.
 *  \param[in] *ind CSR column indices.
 *  \param[in] B Maximum desired block size
 *  \param[in] epsilon Epsilon
 *  \param[in] delta Delta
 *  \param[in] sigma Sigma
 *  \param[out] *fill Fill ratios for all specified b_r, b_c in order
 *  \param[in] verbose 0 if you should be quiet
 *
 *  Note that the fill ratios should be stored according to the following order:
 *  int fill_index = 0;
 *  for (int b_r = 1; b_r <= B; b_r++) {
 *    for (int b_c = 1; b_c <= B; b_c++) {
 *      fill[fill_index] = fill for b_r, b_c
 *      fill_index++;
 *    }
 *  }
 *
 *  \returns On success, returns 0. On error, returns an error code.
 */
int estimate_fill (int m,
                   int n,
                   int nnz,
                   const int *ptr,
                   const int *ind,
                   int B,
                   double epsilon,
                   double delta,
                   double sigma,
                   double *fill,
                   int verbose){
  assert(n >= 1);
  assert(m >= 1);

  /* blocks + (c - 1) * n stores previously seen column block indicies in the
   * current block row when b_c = c.
   */
  int *blocks = (int*)malloc(sizeof(int) * B * n);
  assert(blocks != NULL);
  memset(blocks, 0, sizeof(int) * B * n);

  /* K[(c - 1)] counts distinct column block indicies in the current block row
   * when b_c = c.
   */
  int K[B];

  /* see above note about fill order */
  int fill_index = 0;

  for (int r = 1; r <= B; r++) {

    /* M is the number of block rows */
    int M = m / r;

    /* stores the number of examined nonzeros */
    int S = 0;

    for (int c = 1; c <= B; c++){
      K[c - 1] = 0;
    }

    // CHECK: %[[SYNCREGION:.+]] = {{.+}}call token @llvm.syncregion.start()
    /* loop over block rows */
    cilk_for (int I = 0; I < M; I++) {
      // CHECK: detach within %[[SYNCREGION]], label %[[PFORBODY:.+]], label %[[PFORINC:.+]]
      // CHECK: [[PFORBODY]]:
      // CHECK: br i1 %{{.+}}, label %[[PFORPREATTACH:.+]], label

      /* examine the block row with probability sigma */
      if (random_uniform() > sigma) {
        continue;
      } else {

        /* Count the blocks in block row I, using "blocks" to remember the
         * blocks that have been seen so far for each block column width "c".
         */
        for (int i = I * r; i < (I + 1) * r; i++) {
          for (int t = ptr[i]; t < ptr[i + 1]; t++) {
            int j = ind[t];

            for (int c = 1; c <= B; c++) {
              /* "J" is the block column index */
              int J = j / c;

              /* if the block has not yet been seen, count it */
              if (blocks[(c - 1) * n + J] == 0) {
                blocks[(c - 1) * n + J] = 1;
                K[c - 1]++;
              }
            }
          }
        }
      }
      S += ptr[(I + 1) * r] - ptr[I * r];

      /*
       * Reset "blocks" for the next block row. We loop over the nonzeros
       * instead of calling "memset" in order to keep the complexity to O(nnz).
       */
      for (int i = I * r; i < (I + 1) * r; i++) {
        for (int t = ptr[i]; t < ptr[i + 1]; t++) {
          int j = ind[t];

          for (int c = 1; c <= B; c++) {
            /* "J" is the block column index */
            int J = j / c;
            blocks[(c - 1) * n + J] = 0;
          }
        }
      }
    }
    // CHECK: [[PFORPREATTACH]]:
    // CHECK: reattach within %[[SYNCREGION]]

    // CHECK: [[PFORINC]]:
    // CHECK: br i1

    // CHECK-NOT: [[PFORPREATTACH]]
    // CHECK: return

    /*
     * Compute the fill from the number of blocks and nonzeros that have been
     * seen in the sample.
     */
    for (int c = 1; c <= B; c++) {
      if (!S)
        fill[fill_index] = K[c - 1] ? (1.0 / 0.0) : 1.0;
      else
        fill[fill_index] = ((double)K[c - 1] * r * c) / S;
      fill_index++;
    }
  }

  free(blocks);
  return 0;
}
