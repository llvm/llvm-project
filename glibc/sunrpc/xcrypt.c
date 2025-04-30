/*
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

#if 0
#ident	"@(#)xcrypt.c	1.11	94/08/23 SMI"
#endif

#if !defined(lint) && defined(SCCSIDS)
static char sccsid[] = "@(#)xcrypt.c 1.3 89/03/24 Copyr 1986 Sun Micro";
#endif

/*
 * xcrypt.c: Hex encryption/decryption and utility routines
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <rpc/des_crypt.h>
#include <shlib-compat.h>

static const char hex[16] =
{
  '0', '1', '2', '3', '4', '5', '6', '7',
  '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
};


#ifdef _LIBC
# define hexval(c) \
  (c >= '0' && c <= '9'							      \
   ? c - '0'								      \
   : ({	int upp = toupper (c);						      \
	upp >= 'A' && upp <= 'Z' ? upp - 'A' + 10 : -1; }))
#else
static char hexval (char);
#endif

static void hex2bin (int, char *, char *);
static void bin2hex (int, unsigned char *, char *);
void passwd2des_internal (char *pw, char *key);
#ifdef _LIBC
libc_hidden_proto (passwd2des_internal)
#endif

/*
 * Turn password into DES key
 */
void
passwd2des_internal (char *pw, char *key)
{
  int i;

  memset (key, 0, 8);
  for (i = 0; *pw && i < 8; ++i)
    key[i] ^= *pw++ << 1;

  des_setparity (key);
}

#ifdef _LIBC
libc_hidden_def (passwd2des_internal)
libc_sunrpc_symbol(passwd2des_internal, passwd2des, GLIBC_2_1)
#else
void passwd2des (char *pw, char *key)
{
  return passwd2des_internal (pw, key);
}
#endif

/*
 * Encrypt a secret key given passwd
 * The secret key is passed and returned in hex notation.
 * Its length must be a multiple of 16 hex digits (64 bits).
 */
int
xencrypt (char *secret, char *passwd)
{
  char key[8];
  char ivec[8];
  char *buf;
  int err;
  int len;

  len = strlen (secret) / 2;
  buf = malloc ((unsigned) len);
  hex2bin (len, secret, buf);
  passwd2des_internal (passwd, key);
  memset (ivec, 0, 8);

  err = cbc_crypt (key, buf, len, DES_ENCRYPT | DES_HW, ivec);
  if (DES_FAILED (err))
    {
      free (buf);
      return 0;
    }
  bin2hex (len, (unsigned char *) buf, secret);
  free (buf);
  return 1;
}
libc_hidden_nolink_sunrpc (xencrypt, GLIBC_2_0)

/*
 * Decrypt secret key using passwd
 * The secret key is passed and returned in hex notation.
 * Once again, the length is a multiple of 16 hex digits
 */
int
xdecrypt (char *secret, char *passwd)
{
  char key[8];
  char ivec[8];
  char *buf;
  int err;
  int len;

  len = strlen (secret) / 2;
  buf = malloc ((unsigned) len);

  hex2bin (len, secret, buf);
  passwd2des_internal (passwd, key);
  memset (ivec, 0, 8);

  err = cbc_crypt (key, buf, len, DES_DECRYPT | DES_HW, ivec);
  if (DES_FAILED (err))
    {
      free (buf);
      return 0;
    }
  bin2hex (len, (unsigned char *) buf, secret);
  free (buf);
  return 1;
}
#ifdef EXPORT_RPC_SYMBOLS
libc_hidden_def (xdecrypt)
#else
libc_hidden_nolink_sunrpc (xdecrypt, GLIBC_2_1)
#endif

/*
 * Hex to binary conversion
 */
static void
hex2bin (int len, char *hexnum, char *binnum)
{
  int i;

  for (i = 0; i < len; i++)
    *binnum++ = 16 * hexval (hexnum[2 * i]) + hexval (hexnum[2 * i + 1]);
}

/*
 * Binary to hex conversion
 */
static void
bin2hex (int len, unsigned char *binnum, char *hexnum)
{
  int i;
  unsigned val;

  for (i = 0; i < len; i++)
    {
      val = binnum[i];
      hexnum[i * 2] = hex[val >> 4];
      hexnum[i * 2 + 1] = hex[val & 0xf];
    }
  hexnum[len * 2] = 0;
}

#ifndef _LIBC
static char
hexval (char c)
{
  if (c >= '0' && c <= '9')
    return (c - '0');
  else if (c >= 'a' && c <= 'z')
    return (c - 'a' + 10);
  else if (c >= 'A' && c <= 'Z')
    return (c - 'A' + 10);
  else
    return -1;
}
#endif
