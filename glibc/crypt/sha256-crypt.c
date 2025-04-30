/* One way encryption based on SHA256 sum.
   Copyright (C) 2007-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2007.

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
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/param.h>

#include "sha256.h"
#include "crypt-private.h"


#ifdef USE_NSS
typedef int PRBool;
# include <hasht.h>
# include <nsslowhash.h>

# define sha256_init_ctx(ctxp, nss_ctxp) \
  do									      \
    {									      \
      if (((nss_ctxp = NSSLOWHASH_NewContext (nss_ictx, HASH_AlgSHA256))      \
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

# define sha256_process_bytes(buf, len, ctxp, nss_ctxp) \
  NSSLOWHASH_Update (nss_ctxp, (const unsigned char *) buf, len)

# define sha256_finish_ctx(ctxp, nss_ctxp, result) \
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
# define sha256_init_ctx(ctxp, nss_ctxp) \
  __sha256_init_ctx (ctxp)

# define sha256_process_bytes(buf, len, ctxp, nss_ctxp) \
  __sha256_process_bytes(buf, len, ctxp)

# define sha256_finish_ctx(ctxp, nss_ctxp, result) \
  __sha256_finish_ctx (ctxp, result)
#endif


/* Define our magic string to mark salt for SHA256 "encryption"
   replacement.  */
static const char sha256_salt_prefix[] = "$5$";

/* Prefix for optional rounds specification.  */
static const char sha256_rounds_prefix[] = "rounds=";

/* Maximum salt string length.  */
#define SALT_LEN_MAX 16
/* Default number of rounds if not explicitly specified.  */
#define ROUNDS_DEFAULT 5000
/* Minimum number of rounds.  */
#define ROUNDS_MIN 1000
/* Maximum number of rounds.  */
#define ROUNDS_MAX 999999999


/* Prototypes for local functions.  */
extern char *__sha256_crypt_r (const char *key, const char *salt,
			       char *buffer, int buflen);
extern char *__sha256_crypt (const char *key, const char *salt);


char *
__sha256_crypt_r (const char *key, const char *salt, char *buffer, int buflen)
{
  unsigned char alt_result[32]
    __attribute__ ((__aligned__ (__alignof__ (uint32_t))));
  unsigned char temp_result[32]
    __attribute__ ((__aligned__ (__alignof__ (uint32_t))));
  size_t salt_len;
  size_t key_len;
  size_t cnt;
  char *cp;
  char *copied_key = NULL;
  char *copied_salt = NULL;
  char *p_bytes;
  char *s_bytes;
  /* Default number of rounds.  */
  size_t rounds = ROUNDS_DEFAULT;
  bool rounds_custom = false;
  size_t alloca_used = 0;
  char *free_key = NULL;
  char *free_pbytes = NULL;

  /* Find beginning of salt string.  The prefix should normally always
     be present.  Just in case it is not.  */
  if (strncmp (sha256_salt_prefix, salt, sizeof (sha256_salt_prefix) - 1) == 0)
    /* Skip salt prefix.  */
    salt += sizeof (sha256_salt_prefix) - 1;

  if (strncmp (salt, sha256_rounds_prefix, sizeof (sha256_rounds_prefix) - 1)
      == 0)
    {
      const char *num = salt + sizeof (sha256_rounds_prefix) - 1;
      char *endp;
      unsigned long int srounds = strtoul (num, &endp, 10);
      if (*endp == '$')
	{
	  salt = endp + 1;
	  rounds = MAX (ROUNDS_MIN, MIN (srounds, ROUNDS_MAX));
	  rounds_custom = true;
	}
    }

  salt_len = MIN (strcspn (salt, "$"), SALT_LEN_MAX);
  key_len = strlen (key);

  if ((key - (char *) 0) % __alignof__ (uint32_t) != 0)
    {
      char *tmp;

      if (__libc_use_alloca (alloca_used + key_len + __alignof__ (uint32_t)))
	tmp = alloca_account (key_len + __alignof__ (uint32_t), alloca_used);
      else
	{
	  free_key = tmp = (char *) malloc (key_len + __alignof__ (uint32_t));
	  if (tmp == NULL)
	    return NULL;
	}

      key = copied_key =
	memcpy (tmp + __alignof__ (uint32_t)
		- (tmp - (char *) 0) % __alignof__ (uint32_t),
		key, key_len);
      assert ((key - (char *) 0) % __alignof__ (uint32_t) == 0);
    }

  if ((salt - (char *) 0) % __alignof__ (uint32_t) != 0)
    {
      char *tmp = (char *) alloca (salt_len + __alignof__ (uint32_t));
      alloca_used += salt_len + __alignof__ (uint32_t);
      salt = copied_salt =
	memcpy (tmp + __alignof__ (uint32_t)
		- (tmp - (char *) 0) % __alignof__ (uint32_t),
		salt, salt_len);
      assert ((salt - (char *) 0) % __alignof__ (uint32_t) == 0);
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
  struct sha256_ctx ctx;
  struct sha256_ctx alt_ctx;
#endif

  /* Prepare for the real work.  */
  sha256_init_ctx (&ctx, nss_ctx);

  /* Add the key string.  */
  sha256_process_bytes (key, key_len, &ctx, nss_ctx);

  /* The last part is the salt string.  This must be at most 16
     characters and it ends at the first `$' character.  */
  sha256_process_bytes (salt, salt_len, &ctx, nss_ctx);


  /* Compute alternate SHA256 sum with input KEY, SALT, and KEY.  The
     final result will be added to the first context.  */
  sha256_init_ctx (&alt_ctx, nss_alt_ctx);

  /* Add key.  */
  sha256_process_bytes (key, key_len, &alt_ctx, nss_alt_ctx);

  /* Add salt.  */
  sha256_process_bytes (salt, salt_len, &alt_ctx, nss_alt_ctx);

  /* Add key again.  */
  sha256_process_bytes (key, key_len, &alt_ctx, nss_alt_ctx);

  /* Now get result of this (32 bytes) and add it to the other
     context.  */
  sha256_finish_ctx (&alt_ctx, nss_alt_ctx, alt_result);

  /* Add for any character in the key one byte of the alternate sum.  */
  for (cnt = key_len; cnt > 32; cnt -= 32)
    sha256_process_bytes (alt_result, 32, &ctx, nss_ctx);
  sha256_process_bytes (alt_result, cnt, &ctx, nss_ctx);

  /* Take the binary representation of the length of the key and for every
     1 add the alternate sum, for every 0 the key.  */
  for (cnt = key_len; cnt > 0; cnt >>= 1)
    if ((cnt & 1) != 0)
      sha256_process_bytes (alt_result, 32, &ctx, nss_ctx);
    else
      sha256_process_bytes (key, key_len, &ctx, nss_ctx);

  /* Create intermediate result.  */
  sha256_finish_ctx (&ctx, nss_ctx, alt_result);

  /* Start computation of P byte sequence.  */
  sha256_init_ctx (&alt_ctx, nss_alt_ctx);

  /* For every character in the password add the entire password.  */
  for (cnt = 0; cnt < key_len; ++cnt)
    sha256_process_bytes (key, key_len, &alt_ctx, nss_alt_ctx);

  /* Finish the digest.  */
  sha256_finish_ctx (&alt_ctx, nss_alt_ctx, temp_result);

  /* Create byte sequence P.  */
  if (__libc_use_alloca (alloca_used + key_len))
    cp = p_bytes = (char *) alloca (key_len);
  else
    {
      free_pbytes = cp = p_bytes = (char *)malloc (key_len);
      if (free_pbytes == NULL)
	{
	  free (free_key);
	  return NULL;
	}
    }

  for (cnt = key_len; cnt >= 32; cnt -= 32)
    cp = mempcpy (cp, temp_result, 32);
  memcpy (cp, temp_result, cnt);

  /* Start computation of S byte sequence.  */
  sha256_init_ctx (&alt_ctx, nss_alt_ctx);

  /* For every character in the password add the entire password.  */
  for (cnt = 0; cnt < 16 + alt_result[0]; ++cnt)
    sha256_process_bytes (salt, salt_len, &alt_ctx, nss_alt_ctx);

  /* Finish the digest.  */
  sha256_finish_ctx (&alt_ctx, nss_alt_ctx, temp_result);

  /* Create byte sequence S.  */
  cp = s_bytes = alloca (salt_len);
  for (cnt = salt_len; cnt >= 32; cnt -= 32)
    cp = mempcpy (cp, temp_result, 32);
  memcpy (cp, temp_result, cnt);

  /* Repeatedly run the collected hash value through SHA256 to burn
     CPU cycles.  */
  for (cnt = 0; cnt < rounds; ++cnt)
    {
      /* New context.  */
      sha256_init_ctx (&ctx, nss_ctx);

      /* Add key or last result.  */
      if ((cnt & 1) != 0)
	sha256_process_bytes (p_bytes, key_len, &ctx, nss_ctx);
      else
	sha256_process_bytes (alt_result, 32, &ctx, nss_ctx);

      /* Add salt for numbers not divisible by 3.  */
      if (cnt % 3 != 0)
	sha256_process_bytes (s_bytes, salt_len, &ctx, nss_ctx);

      /* Add key for numbers not divisible by 7.  */
      if (cnt % 7 != 0)
	sha256_process_bytes (p_bytes, key_len, &ctx, nss_ctx);

      /* Add key or last result.  */
      if ((cnt & 1) != 0)
	sha256_process_bytes (alt_result, 32, &ctx, nss_ctx);
      else
	sha256_process_bytes (p_bytes, key_len, &ctx, nss_ctx);

      /* Create intermediate result.  */
      sha256_finish_ctx (&ctx, nss_ctx, alt_result);
    }

#ifdef USE_NSS
  /* Free libfreebl3 resources. */
  NSSLOW_Shutdown (nss_ictx);
#endif

  /* Now we can construct the result string.  It consists of three
     parts.  */
  cp = __stpncpy (buffer, sha256_salt_prefix, MAX (0, buflen));
  buflen -= sizeof (sha256_salt_prefix) - 1;

  if (rounds_custom)
    {
      int n = __snprintf (cp, MAX (0, buflen), "%s%zu$",
			  sha256_rounds_prefix, rounds);
      cp += n;
      buflen -= n;
    }

  cp = __stpncpy (cp, salt, MIN ((size_t) MAX (0, buflen), salt_len));
  buflen -= MIN ((size_t) MAX (0, buflen), salt_len);

  if (buflen > 0)
    {
      *cp++ = '$';
      --buflen;
    }

  __b64_from_24bit (&cp, &buflen,
		    alt_result[0], alt_result[10], alt_result[20], 4);
  __b64_from_24bit (&cp, &buflen,
		    alt_result[21], alt_result[1], alt_result[11], 4);
  __b64_from_24bit (&cp, &buflen,
		    alt_result[12], alt_result[22], alt_result[2], 4);
  __b64_from_24bit (&cp, &buflen,
		    alt_result[3], alt_result[13], alt_result[23], 4);
  __b64_from_24bit (&cp, &buflen,
		    alt_result[24], alt_result[4], alt_result[14], 4);
  __b64_from_24bit (&cp, &buflen,
		    alt_result[15], alt_result[25], alt_result[5], 4);
  __b64_from_24bit (&cp, &buflen,
		    alt_result[6], alt_result[16], alt_result[26], 4);
  __b64_from_24bit (&cp, &buflen,
		    alt_result[27], alt_result[7], alt_result[17], 4);
  __b64_from_24bit (&cp, &buflen,
		    alt_result[18], alt_result[28], alt_result[8], 4);
  __b64_from_24bit (&cp, &buflen,
		    alt_result[9], alt_result[19], alt_result[29], 4);
  __b64_from_24bit (&cp, &buflen,
		    0, alt_result[31], alt_result[30], 3);
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
     inside the SHA256 implementation as well.  */
#ifndef USE_NSS
  __sha256_init_ctx (&ctx);
  __sha256_finish_ctx (&ctx, alt_result);
  explicit_bzero (&ctx, sizeof (ctx));
  explicit_bzero (&alt_ctx, sizeof (alt_ctx));
#endif
  explicit_bzero (temp_result, sizeof (temp_result));
  explicit_bzero (p_bytes, key_len);
  explicit_bzero (s_bytes, salt_len);
  if (copied_key != NULL)
    explicit_bzero (copied_key, key_len);
  if (copied_salt != NULL)
    explicit_bzero (copied_salt, salt_len);

  free (free_key);
  free (free_pbytes);
  return buffer;
}

#ifndef _LIBC
# define libc_freeres_ptr(decl) decl
#endif
libc_freeres_ptr (static char *buffer);

/* This entry point is equivalent to the `crypt' function in Unix
   libcs.  */
char *
__sha256_crypt (const char *key, const char *salt)
{
  /* We don't want to have an arbitrary limit in the size of the
     password.  We can compute an upper bound for the size of the
     result in advance and so we can prepare the buffer we pass to
     `sha256_crypt_r'.  */
  static int buflen;
  int needed = (sizeof (sha256_salt_prefix) - 1
		+ sizeof (sha256_rounds_prefix) + 9 + 1
		+ strlen (salt) + 1 + 43 + 1);

  if (buflen < needed)
    {
      char *new_buffer = (char *) realloc (buffer, needed);
      if (new_buffer == NULL)
	return NULL;

      buffer = new_buffer;
      buflen = needed;
    }

  return __sha256_crypt_r (key, salt, buffer, buflen);
}

#ifndef _LIBC
static void
__attribute__ ((__destructor__))
free_mem (void)
{
  free (buffer);
}
#endif
