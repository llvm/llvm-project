/* Declaration of functions and data types used for SHA512 sum computing
   library functions.
   Copyright (C) 2007-2021 Free Software Foundation, Inc.
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

#ifndef _SHA512_H
#define _SHA512_H 1

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <endian.h>
#include <bits/wordsize.h>


/* Structure to save state of computation between the single steps.  */
struct sha512_ctx
{
  uint64_t H[8];

  union
  {
#if defined __GNUC__ && __WORDSIZE == 64
# define USE_TOTAL128
    unsigned int total128 __attribute__ ((__mode__ (TI)));
#endif
#define TOTAL128_low (1 - (BYTE_ORDER == LITTLE_ENDIAN))
#define TOTAL128_high (BYTE_ORDER == LITTLE_ENDIAN)
    uint64_t total[2];
  };
  uint64_t buflen;
  union
  {
    char buffer[256];
    uint64_t buffer64[32];
  };
};

/* Initialize structure containing state of computation.
   (FIPS 180-2: 5.3.3)  */
extern void __sha512_init_ctx (struct sha512_ctx *ctx) __THROW;

/* Starting with the result of former calls of this function (or the
   initialization function update the context for the next LEN bytes
   starting at BUFFER.
   It is NOT required that LEN is a multiple of 128.  */
extern void __sha512_process_bytes (const void *buffer, size_t len,
				    struct sha512_ctx *ctx) __THROW;

/* Process the remaining bytes in the buffer and put result from CTX
   in first 64 bytes following RESBUF.

   IMPORTANT: On some systems it is required that RESBUF is correctly
   aligned for a 64 bits value.  */
extern void *__sha512_finish_ctx (struct sha512_ctx *ctx, void *resbuf)
  __THROW;

#endif /* sha512.h */
