/*
 * des_crypt.c, DES encryption library routines
 * Copyright (c) 2010, Oracle America, Inc.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the "Oracle America, Inc." nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *   FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *   COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 *   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 *   GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <sys/types.h>
#include <rpc/des_crypt.h>
#include <shlib-compat.h>
#include "des.h"

extern int _des_crypt (char *, unsigned, struct desparams *);

/*
 * Copy 8 bytes
 */
#define COPY8(src, dst) { \
	register char *a = (char *) dst; \
	register char *b = (char *) src; \
	*a++ = *b++; *a++ = *b++; *a++ = *b++; *a++ = *b++; \
	*a++ = *b++; *a++ = *b++; *a++ = *b++; *a++ = *b++; \
}

/*
 * Copy multiple of 8 bytes
 */
#define DESCOPY(src, dst, len) { \
	register char *a = (char *) dst; \
	register char *b = (char *) src; \
	register int i; \
	for (i = (int) len; i > 0; i -= 8) { \
		*a++ = *b++; *a++ = *b++; *a++ = *b++; *a++ = *b++; \
		*a++ = *b++; *a++ = *b++; *a++ = *b++; *a++ = *b++; \
	} \
}

/*
 * Common code to cbc_crypt() & ecb_crypt()
 */
static int
common_crypt (char *key, char *buf, register unsigned len,
	      unsigned mode, register struct desparams *desp)
{
  register int desdev;

  if ((len % 8) != 0 || len > DES_MAXDATA)
    return DESERR_BADPARAM;

  desp->des_dir =
    ((mode & DES_DIRMASK) == DES_ENCRYPT) ? ENCRYPT : DECRYPT;

  desdev = mode & DES_DEVMASK;
  COPY8 (key, desp->des_key);
  /*
   * software
   */
  if (!_des_crypt (buf, len, desp))
    return DESERR_HWERROR;

  return desdev == DES_SW ? DESERR_NONE : DESERR_NOHWDEVICE;
}

/* Note: these cannot be excluded from the build yet, because they are
   still used internally.  */

/*
 * CBC mode encryption
 */
int
cbc_crypt (char *key, char *buf, unsigned int len, unsigned int mode,
	   char *ivec)
{
  int err;
  struct desparams dp;

  dp.des_mode = CBC;
  COPY8 (ivec, dp.des_ivec);
  err = common_crypt (key, buf, len, mode, &dp);
  COPY8 (dp.des_ivec, ivec);
  return err;
}
hidden_nolink (cbc_crypt, libc, GLIBC_2_1)

/*
 * ECB mode encryption
 */
int
ecb_crypt (char *key, char *buf, unsigned int len, unsigned int mode)
{
  struct desparams dp;

  dp.des_mode = ECB;
  return common_crypt (key, buf, len, mode, &dp);
}
hidden_nolink (ecb_crypt, libc, GLIBC_2_1)
