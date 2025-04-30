/* Declaration of functions and data types used for MD5 sum computing
   library functions.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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
   <https://www.gnu.org/licenses/>.  */

#ifndef _MD5_H
#define _MD5_H 1

#include <stdio.h>

#if defined HAVE_LIMITS_H || defined _LIBC
# include <limits.h>
#endif

#define MD5_DIGEST_SIZE 16
#define MD5_BLOCK_SIZE 64

/* The following contortions are an attempt to use the C preprocessor
   to determine an unsigned integral type that is 32 bits wide.  An
   alternative approach is to use autoconf's AC_CHECK_SIZEOF macro, but
   doing that would require that the configure script compile and *run*
   the resulting executable.  Locally running cross-compiled executables
   is usually not possible.  */

#ifdef _LIBC
# include <stdint.h>
typedef uint32_t md5_uint32;
typedef uintptr_t md5_uintptr;
#else
# define UINT_MAX_32_BITS 4294967295U

/* If UINT_MAX isn't defined, assume it's a 32-bit type.
   This should be valid for all systems GNU cares about because
   that doesn't include 16-bit systems, and only modern systems
   (that certainly have <limits.h>) have 64+-bit integral types.  */

# ifndef UINT_MAX
#  define UINT_MAX UINT_MAX_32_BITS
# endif

# if UINT_MAX == UINT_MAX_32_BITS
   typedef unsigned int md5_uint32;
# else
#  if USHRT_MAX == UINT_MAX_32_BITS
    typedef unsigned short md5_uint32;
#  else
#   if ULONG_MAX == UINT_MAX_32_BITS
     typedef unsigned long md5_uint32;
#   else
     /* The following line is intended to evoke an error.
	Using #error is not portable enough.  */
     "Cannot determine unsigned 32-bit data type."
#   endif
#  endif
# endif
/* We have to make a guess about the integer type equivalent in size
   to pointers which should always be correct.  */
typedef unsigned long int md5_uintptr;
#endif

/* Structure to save state of computation between the single steps.  */
struct md5_ctx
{
  md5_uint32 A;
  md5_uint32 B;
  md5_uint32 C;
  md5_uint32 D;

  md5_uint32 total[2];
  md5_uint32 buflen;
  union
  {
    char buffer[128];
    md5_uint32 buffer32[32];
  };
};

/*
 * The following three functions are build up the low level used in
 * the functions `md5_stream' and `md5_buffer'.
 */

/* Initialize structure containing state of computation.
   (RFC 1321, 3.3: Step 3)  */
extern void __md5_init_ctx (struct md5_ctx *ctx) __THROW;

/* Starting with the result of former calls of this function (or the
   initialization function update the context for the next LEN bytes
   starting at BUFFER.
   It is necessary that LEN is a multiple of 64!!! */
extern void __md5_process_block (const void *buffer, size_t len,
				 struct md5_ctx *ctx) __THROW;

/* Starting with the result of former calls of this function (or the
   initialization function update the context for the next LEN bytes
   starting at BUFFER.
   It is NOT required that LEN is a multiple of 64.  */
extern void __md5_process_bytes (const void *buffer, size_t len,
				 struct md5_ctx *ctx) __THROW;

/* Process the remaining bytes in the buffer and put result from CTX
   in first 16 bytes following RESBUF.  The result is always in little
   endian byte order, so that a byte-wise output yields to the wanted
   ASCII representation of the message digest.

   IMPORTANT: On some systems it is required that RESBUF is correctly
   aligned for a 32 bits value.  */
extern void *__md5_finish_ctx (struct md5_ctx *ctx, void *resbuf) __THROW;


/* Put result from CTX in first 16 bytes following RESBUF.  The result is
   always in little endian byte order, so that a byte-wise output yields
   to the wanted ASCII representation of the message digest.

   IMPORTANT: On some systems it is required that RESBUF is correctly
   aligned for a 32 bits value.  */
extern void *__md5_read_ctx (const struct md5_ctx *ctx, void *resbuf) __THROW;


/* Compute MD5 message digest for bytes read from STREAM.  The
   resulting message digest number will be written into the 16 bytes
   beginning at RESBLOCK.  */
extern int __md5_stream (FILE *stream, void *resblock) __THROW;

/* Compute MD5 message digest for LEN bytes beginning at BUFFER.  The
   result is always in little endian byte order, so that a byte-wise
   output yields to the wanted ASCII representation of the message
   digest.  */
extern void *__md5_buffer (const char *buffer, size_t len,
			   void *resblock) __THROW;

#endif /* md5.h */
