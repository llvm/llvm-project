/* One way encryption based on MD5 sum.
   Compatible with the behavior of MD5 crypt introduced in FreeBSD 2.0.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1996.

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

#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>

#include "md5.h"
#include "crypt-private.h"


#ifdef USE_NSS
typedef int PRBool;
# include <hasht.h>
# include <nsslowhash.h>

# define md5_init_ctx(ctxp, nss_ctxp) \
  do									      \
    {									      \
      if (((nss_ctxp = NSSLOWHASH_NewContext (nss_ictx, HASH_AlgMD5))	      \
	   == NULL))							      \
	{								      \
	  if (nss_ctx != NULL)						      \
	    NSSLOWHASH_Destroy (nss_ctx);				      \
	  if (nss_alt_ctx != NULL)					      \
	    NSSLOWHASH_Destroy (nss_alt_ctx);				      \
	  return NULL;							      \
	}								      \
      NSSLOWHASH_Begin (nss_ctxp);					      \
    }									      \
  while (0)

# define md5_process_bytes(buf, len, ctxp, nss_ctxp) \
  NSSLOWHASH_Update (nss_ctxp, (const unsigned char *) buf, len)

# define md5_finish_ctx(ctxp, nss_ctxp, result) \
  do									      \
    {									      \
      unsigned int ret;							      \
      NSSLOWHASH_End (nss_ctxp, result, &ret, sizeof (result));		      \
      assert (ret == sizeof (result));					      \
      NSSLOWHASH_Destroy (nss_ctxp);					      \
      nss_ctxp = NULL;							      \
    }									      \
  while (0)
#else
# define md5_init_ctx(ctxp, nss_ctxp) \
  __md5_init_ctx (ctxp)

# define md5_process_bytes(buf, len, ctxp, nss_ctxp) \
  __md5_process_bytes(buf, len, ctxp)

# define md5_finish_ctx(ctxp, nss_ctxp, result) \
  __md5_finish_ctx (ctxp, result)
#endif


/* Define our magic string to mark salt for MD5 "encryption"
   replacement.  This is meant to be the same as for other MD5 based
   encryption implementations.  */
static const char md5_salt_prefix[] = "$1$";


/* Prototypes for local functions.  */
extern char *__md5_crypt_r (const char *key, const char *salt,
			    char *buffer, int buflen);
extern char *__md5_crypt (const char *key, const char *salt);


/* This entry point is equivalent to the `crypt' function in Unix
   libcs.  */
char *
__md5_crypt_r (const char *key, const char *salt, char *buffer, int buflen)
{
  unsigned char alt_result[16]
    __attribute__ ((__aligned__ (__alignof__ (md5_uint32))));
  size_t salt_len;
  size_t key_len;
  size_t cnt;
  char *cp;
  char *copied_key = NULL;
  char *copied_salt = NULL;
  char *free_key = NULL;
  size_t alloca_used = 0;

  /* Find beginning of salt string.  The prefix should normally always
     be present.  Just in case it is not.  */
  if (strncmp (md5_salt_prefix, salt, sizeof (md5_salt_prefix) - 1) == 0)
    /* Skip salt prefix.  */
    salt += sizeof (md5_salt_prefix) - 1;

  salt_len = MIN (strcspn (salt, "$"), 8);
  key_len = strlen (key);

  if ((key - (char *) 0) % __alignof__ (md5_uint32) != 0)
    {
      char *tmp;

      if (__libc_use_alloca (alloca_used + key_len + __alignof__ (md5_uint32)))
	tmp = (char *) alloca (key_len + __alignof__ (md5_uint32));
      else
	{
	  free_key = tmp = (char *) malloc (key_len + __alignof__ (md5_uint32));
	  if (tmp == NULL)
	    return NULL;
	}

      key = copied_key =
	memcpy (tmp + __alignof__ (md5_uint32)
		- (tmp - (char *) 0) % __alignof__ (md5_uint32),
		key, key_len);
      assert ((key - (char *) 0) % __alignof__ (md5_uint32) == 0);
    }

  if ((salt - (char *) 0) % __alignof__ (md5_uint32) != 0)
    {
      char *tmp = (char *) alloca (salt_len + __alignof__ (md5_uint32));
      salt = copied_salt =
	memcpy (tmp + __alignof__ (md5_uint32)
		- (tmp - (char *) 0) % __alignof__ (md5_uint32),
		salt, salt_len);
      assert ((salt - (char *) 0) % __alignof__ (md5_uint32) == 0);
    }

#ifdef USE_NSS
  /* Initialize libfreebl3.  */
  NSSLOWInitContext *nss_ictx = NSSLOW_Init ();
  if (nss_ictx == NULL)
    {
      free (free_key);
      return NULL;
    }
  NSSLOWHASHContext *nss_ctx = NULL;
  NSSLOWHASHContext *nss_alt_ctx = NULL;
#else
  struct md5_ctx ctx;
  struct md5_ctx alt_ctx;
#endif

  /* Prepare for the real work.  */
  md5_init_ctx (&ctx, nss_ctx);

  /* Add the key string.  */
  md5_process_bytes (key, key_len, &ctx, nss_ctx);

  /* Because the SALT argument need not always have the salt prefix we
     add it separately.  */
  md5_process_bytes (md5_salt_prefix, sizeof (md5_salt_prefix) - 1,
		     &ctx, nss_ctx);

  /* The last part is the salt string.  This must be at most 8
     characters and it ends at the first `$' character (for
     compatibility with existing implementations).  */
  md5_process_bytes (salt, salt_len, &ctx, nss_ctx);


  /* Compute alternate MD5 sum with input KEY, SALT, and KEY.  The
     final result will be added to the first context.  */
  md5_init_ctx (&alt_ctx, nss_alt_ctx);

  /* Add key.  */
  md5_process_bytes (key, key_len, &alt_ctx, nss_alt_ctx);

  /* Add salt.  */
  md5_process_bytes (salt, salt_len, &alt_ctx, nss_alt_ctx);

  /* Add key again.  */
  md5_process_bytes (key, key_len, &alt_ctx, nss_alt_ctx);

  /* Now get result of this (16 bytes) and add it to the other
     context.  */
  md5_finish_ctx (&alt_ctx, nss_alt_ctx, alt_result);

  /* Add for any character in the key one byte of the alternate sum.  */
  for (cnt = key_len; cnt > 16; cnt -= 16)
    md5_process_bytes (alt_result, 16, &ctx, nss_ctx);
  md5_process_bytes (alt_result, cnt, &ctx, nss_ctx);

  /* For the following code we need a NUL byte.  */
  *alt_result = '\0';

  /* The original implementation now does something weird: for every 1
     bit in the key the first 0 is added to the buffer, for every 0
     bit the first character of the key.  This does not seem to be
     what was intended but we have to follow this to be compatible.  */
  for (cnt = key_len; cnt > 0; cnt >>= 1)
    md5_process_bytes ((cnt & 1) != 0
		       ? (const void *) alt_result : (const void *) key, 1,
		       &ctx, nss_ctx);

  /* Create intermediate result.  */
  md5_finish_ctx (&ctx, nss_ctx, alt_result);

  /* Now comes another weirdness.  In fear of password crackers here
     comes a quite long loop which just processes the output of the
     previous round again.  We cannot ignore this here.  */
  for (cnt = 0; cnt < 1000; ++cnt)
    {
      /* New context.  */
      md5_init_ctx (&ctx, nss_ctx);

      /* Add key or last result.  */
      if ((cnt & 1) != 0)
	md5_process_bytes (key, key_len, &ctx, nss_ctx);
      else
	md5_process_bytes (alt_result, 16, &ctx, nss_ctx);

      /* Add salt for numbers not divisible by 3.  */
      if (cnt % 3 != 0)
	md5_process_bytes (salt, salt_len, &ctx, nss_ctx);

      /* Add key for numbers not divisible by 7.  */
      if (cnt % 7 != 0)
	md5_process_bytes (key, key_len, &ctx, nss_ctx);

      /* Add key or last result.  */
      if ((cnt & 1) != 0)
	md5_process_bytes (alt_result, 16, &ctx, nss_ctx);
      else
	md5_process_bytes (key, key_len, &ctx, nss_ctx);

      /* Create intermediate result.  */
      md5_finish_ctx (&ctx, nss_ctx, alt_result);
    }

#ifdef USE_NSS
  /* Free libfreebl3 resources. */
  NSSLOW_Shutdown (nss_ictx);
#endif

  /* Now we can construct the result string.  It consists of three
     parts.  */
  cp = __stpncpy (buffer, md5_salt_prefix, MAX (0, buflen));
  buflen -= sizeof (md5_salt_prefix) - 1;

  cp = __stpncpy (cp, salt, MIN ((size_t) MAX (0, buflen), salt_len));
  buflen -= MIN ((size_t) MAX (0, buflen), salt_len);

  if (buflen > 0)
    {
      *cp++ = '$';
      --buflen;
    }

  __b64_from_24bit (&cp, &buflen,
		    alt_result[0], alt_result[6], alt_result[12], 4);
  __b64_from_24bit (&cp, &buflen,
		    alt_result[1], alt_result[7], alt_result[13], 4);
  __b64_from_24bit (&cp, &buflen,
		    alt_result[2], alt_result[8], alt_result[14], 4);
  __b64_from_24bit (&cp, &buflen,
		    alt_result[3], alt_result[9], alt_result[15], 4);
  __b64_from_24bit (&cp, &buflen,
		    alt_result[4], alt_result[10], alt_result[5], 4);
  __b64_from_24bit (&cp, &buflen,
		    0, 0, alt_result[11], 2);
  if (buflen <= 0)
    {
      __set_errno (ERANGE);
      buffer = NULL;
    }
  else
    *cp = '\0';		/* Terminate the string.  */

  /* Clear the buffer for the intermediate result so that people
     attaching to processes or reading core dumps cannot get any
     information.  We do it in this way to clear correct_words[]
     inside the MD5 implementation as well.  */
#ifndef USE_NSS
  __md5_init_ctx (&ctx);
  __md5_finish_ctx (&ctx, alt_result);
  explicit_bzero (&ctx, sizeof (ctx));
  explicit_bzero (&alt_ctx, sizeof (alt_ctx));
#endif
  if (copied_key != NULL)
    explicit_bzero (copied_key, key_len);
  if (copied_salt != NULL)
    explicit_bzero (copied_salt, salt_len);

  free (free_key);
  return buffer;
}

#ifndef _LIBC
# define libc_freeres_ptr(decl) decl
#endif
libc_freeres_ptr (static char *buffer);

char *
__md5_crypt (const char *key, const char *salt)
{
  /* We don't want to have an arbitrary limit in the size of the
     password.  We can compute the size of the result in advance and
     so we can prepare the buffer we pass to `md5_crypt_r'.  */
  static int buflen;
  int needed = 3 + strlen (salt) + 1 + 26 + 1;

  if (buflen < needed)
    {
      char *new_buffer = (char *) realloc (buffer, needed);
      if (new_buffer == NULL)
	return NULL;

      buffer = new_buffer;
      buflen = needed;
    }

  return __md5_crypt_r (key, salt, buffer, buflen);
}

#ifndef _LIBC
static void
__attribute__ ((__destructor__))
free_mem (void)
{
  free (buffer);
}
#endif
