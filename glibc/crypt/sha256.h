/* Declaration of functions and data types used for SHA256 sum computing
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

#ifndef _SHA256_H
#define _SHA256_H 1

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <endian.h>


/* Structure to save state of computation between the single steps.  */
struct sha256_ctx
{
  uint32_t H[8];

  union
  {
    uint64_t total64;
#define TOTAL64_low (1 - (BYTE_ORDER == LITTLE_ENDIAN))
#define TOTAL64_high (BYTE_ORDER == LITTLE_ENDIAN)
    uint32_t total[2];
  };
  uint32_t buflen;
  union
  {
    char buffer[128];
    uint32_t buffer32[32];
    uint64_t buffer64[16];
  };
};

/* Initialize structure containing state of computation.
   (FIPS 180-2: 5.3.2)  */
extern void __sha256_init_ctx (struct sha256_ctx *ctx) __THROW;

/* Starting with the result of former calls of this function (or the
   initialization function update the context for the next LEN bytes
   starting at BUFFER.
   It is NOT required that LEN is a multiple of 64.  */
extern void __sha256_process_bytes (const void *buffer, size_t len,
				    struct sha256_ctx *ctx) __THROW;

/* Process the remaining bytes in the buffer and put result from CTX
   in first 32 bytes following RESBUF.

   IMPORTANT: On some systems it is required that RESBUF is correctly
   aligned for a 32 bits value.  */
extern void *__sha256_finish_ctx (struct sha256_ctx *ctx, void *resbuf)
  __THROW;

#endif /* sha256.h */
