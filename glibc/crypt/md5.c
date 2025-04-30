/* Functions to compute MD5 message digest of files or memory blocks.
   according to the definition of MD5 in RFC 1321 from April 1992.
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

/* Written by Ulrich Drepper <drepper@gnu.ai.mit.edu>, 1995.  */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <sys/types.h>

#if STDC_HEADERS || defined _LIBC
# include <stdlib.h>
# include <string.h>
#else
# ifndef HAVE_MEMCPY
#  define memcpy(d, s, n) (bcopy ((s), (d), (n)), (d))
# endif
#endif

#include "md5.h"

#ifdef _LIBC
# include <endian.h>
# if __BYTE_ORDER == __BIG_ENDIAN
#  define WORDS_BIGENDIAN 1
# endif
/* We need to keep the namespace clean so define the MD5 function
   protected using leading __ .  */
# define md5_init_ctx __md5_init_ctx
# define md5_process_bytes __md5_process_bytes
# define md5_finish_ctx __md5_finish_ctx
# define md5_read_ctx __md5_read_ctx
# define md5_stream __md5_stream
# define md5_buffer __md5_buffer
#endif

#ifdef WORDS_BIGENDIAN
# define SWAP(n)							\
    (((n) << 24) | (((n) & 0xff00) << 8) | (((n) >> 8) & 0xff00) | ((n) >> 24))
#else
# define SWAP(n) (n)
#endif


/* This array contains the bytes used to pad the buffer to the next
   64-byte boundary.  (RFC 1321, 3.1: Step 1)  */
static const unsigned char fillbuf[64] = { 0x80, 0 /* , 0, 0, ...  */ };


/* Initialize structure containing state of computation.
   (RFC 1321, 3.3: Step 3)  */
void
md5_init_ctx (struct md5_ctx *ctx)
{
  ctx->A = 0x67452301;
  ctx->B = 0xefcdab89;
  ctx->C = 0x98badcfe;
  ctx->D = 0x10325476;

  ctx->total[0] = ctx->total[1] = 0;
  ctx->buflen = 0;
}

/* Put result from CTX in first 16 bytes following RESBUF.  The result
   must be in little endian byte order.

   IMPORTANT: On some systems it is required that RESBUF is correctly
   aligned for a 32 bits value.  */
void *
md5_read_ctx (const struct md5_ctx *ctx, void *resbuf)
{
  ((md5_uint32 *) resbuf)[0] = SWAP (ctx->A);
  ((md5_uint32 *) resbuf)[1] = SWAP (ctx->B);
  ((md5_uint32 *) resbuf)[2] = SWAP (ctx->C);
  ((md5_uint32 *) resbuf)[3] = SWAP (ctx->D);

  return resbuf;
}

/* Process the remaining bytes in the internal buffer and the usual
   prolog according to the standard and write the result to RESBUF.

   IMPORTANT: On some systems it is required that RESBUF is correctly
   aligned for a 32 bits value.  */
void *
md5_finish_ctx (struct md5_ctx *ctx, void *resbuf)
{
  /* Take yet unprocessed bytes into account.  */
  md5_uint32 bytes = ctx->buflen;
  size_t pad;

  /* Now count remaining bytes.  */
  ctx->total[0] += bytes;
  if (ctx->total[0] < bytes)
    ++ctx->total[1];

  pad = bytes >= 56 ? 64 + 56 - bytes : 56 - bytes;
  memcpy (&ctx->buffer[bytes], fillbuf, pad);

  /* Put the 64-bit file length in *bits* at the end of the buffer.  */
  ctx->buffer32[(bytes + pad) / 4] = SWAP (ctx->total[0] << 3);
  ctx->buffer32[(bytes + pad + 4) / 4] = SWAP ((ctx->total[1] << 3)
					       | (ctx->total[0] >> 29));

  /* Process last bytes.  */
  __md5_process_block (ctx->buffer, bytes + pad + 8, ctx);

  return md5_read_ctx (ctx, resbuf);
}

/* Compute MD5 message digest for bytes read from STREAM.  The
   resulting message digest number will be written into the 16 bytes
   beginning at RESBLOCK.  */
int
md5_stream (FILE *stream, void *resblock)
{
  /* Important: BLOCKSIZE must be a multiple of 64.  */
#define BLOCKSIZE 4096
  struct md5_ctx ctx;
  char buffer[BLOCKSIZE + 72];
  size_t sum;

  /* Initialize the computation context.  */
  md5_init_ctx (&ctx);

  /* Iterate over full file contents.  */
  while (1)
    {
      /* We read the file in blocks of BLOCKSIZE bytes.  One call of the
	 computation function processes the whole buffer so that with the
	 next round of the loop another block can be read.  */
      size_t n;
      sum = 0;

      /* Read block.  Take care for partial reads.  */
      do
	{
	  n = fread (buffer + sum, 1, BLOCKSIZE - sum, stream);

	  sum += n;
	}
      while (sum < BLOCKSIZE && n != 0);
      if (n == 0 && ferror (stream))
	return 1;

      /* If end of file is reached, end the loop.  */
      if (n == 0)
	break;

      /* Process buffer with BLOCKSIZE bytes.  Note that
			BLOCKSIZE % 64 == 0
       */
      __md5_process_block (buffer, BLOCKSIZE, &ctx);
    }

  /* Add the last bytes if necessary.  */
  if (sum > 0)
    md5_process_bytes (buffer, sum, &ctx);

  /* Construct result in desired memory.  */
  md5_finish_ctx (&ctx, resblock);
  return 0;
}

/* Compute MD5 message digest for LEN bytes beginning at BUFFER.  The
   result is always in little endian byte order, so that a byte-wise
   output yields to the wanted ASCII representation of the message
   digest.  */
void *
md5_buffer (const char *buffer, size_t len, void *resblock)
{
  struct md5_ctx ctx;

  /* Initialize the computation context.  */
  md5_init_ctx (&ctx);

  /* Process whole buffer but last len % 64 bytes.  */
  md5_process_bytes (buffer, len, &ctx);

  /* Put result in desired memory area.  */
  return md5_finish_ctx (&ctx, resblock);
}


void
md5_process_bytes (const void *buffer, size_t len, struct md5_ctx *ctx)
{
  /* When we already have some bits in our internal buffer concatenate
     both inputs first.  */
  if (ctx->buflen != 0)
    {
      size_t left_over = ctx->buflen;
      size_t add = 128 - left_over > len ? len : 128 - left_over;

      memcpy (&ctx->buffer[left_over], buffer, add);
      ctx->buflen += add;

      if (ctx->buflen > 64)
	{
	  __md5_process_block (ctx->buffer, ctx->buflen & ~63, ctx);

	  ctx->buflen &= 63;
	  /* The regions in the following copy operation cannot overlap.  */
	  memcpy (ctx->buffer, &ctx->buffer[(left_over + add) & ~63],
		  ctx->buflen);
	}

      buffer = (const char *) buffer + add;
      len -= add;
    }

  /* Process available complete blocks.  */
  if (len >= 64)
    {
#if !_STRING_ARCH_unaligned
/* To check alignment gcc has an appropriate operator.  Other
   compilers don't.  */
# if __GNUC__ >= 2
#  define UNALIGNED_P(p) (((md5_uintptr) p) % __alignof__ (md5_uint32) != 0)
# else
#  define UNALIGNED_P(p) (((md5_uintptr) p) % sizeof (md5_uint32) != 0)
# endif
      if (UNALIGNED_P (buffer))
	while (len > 64)
	  {
	    __md5_process_block (memcpy (ctx->buffer, buffer, 64), 64, ctx);
	    buffer = (const char *) buffer + 64;
	    len -= 64;
	  }
      else
#endif
	{
	  __md5_process_block (buffer, len & ~63, ctx);
	  buffer = (const char *) buffer + (len & ~63);
	  len &= 63;
	}
    }

  /* Move remaining bytes in internal buffer.  */
  if (len > 0)
    {
      size_t left_over = ctx->buflen;

      memcpy (&ctx->buffer[left_over], buffer, len);
      left_over += len;
      if (left_over >= 64)
	{
	  __md5_process_block (ctx->buffer, 64, ctx);
	  left_over -= 64;
	  memcpy (ctx->buffer, &ctx->buffer[64], left_over);
	}
      ctx->buflen = left_over;
    }
}

#include <md5-block.c>
