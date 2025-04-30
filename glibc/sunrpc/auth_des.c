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
/*
 * auth_des.c, client-side implementation of DES authentication
 */

#include <string.h>
#include <stdint.h>
#include <rpc/des_crypt.h>
#include <rpc/types.h>
#include <rpc/auth.h>
#include <rpc/auth_des.h>
#include <rpc/xdr.h>
#include <netinet/in.h>		/* XXX: just to get htonl() and ntohl() */
#include <sys/socket.h>
#include <time.h>
#include <shlib-compat.h>

#define MILLION		1000000L
#define RTIME_TIMEOUT 5		/* seconds to wait for sync */

#define AUTH_PRIVATE(auth)	(struct ad_private *) auth->ah_private
#define ALLOC(object_type)	(object_type *) mem_alloc(sizeof(object_type))
#define FREE(ptr, size)		mem_free((char *)(ptr), (int) size)
#define ATTEMPT(xdr_op)		if (!(xdr_op)) return (FALSE)

#define debug(msg)		/* printf("%s\n", msg) */


/*
 * DES authenticator operations vector
 */
static void authdes_nextverf (AUTH *);
static bool_t authdes_marshal (AUTH *, XDR *);
static bool_t authdes_validate (AUTH *, struct opaque_auth *);
static bool_t authdes_refresh (AUTH *);
static void authdes_destroy (AUTH *);
static bool_t synchronize (struct sockaddr *, struct rpc_timeval *);

static const struct auth_ops authdes_ops = {
  authdes_nextverf,
  authdes_marshal,
  authdes_validate,
  authdes_refresh,
  authdes_destroy
};


/*
 * This struct is pointed to by the ah_private field of an "AUTH *"
 */
struct ad_private {
  char *ad_fullname;		/* client's full name */
  u_int ad_fullnamelen;		/* length of name, rounded up */
  char *ad_servername;		/* server's full name */
  u_int ad_servernamelen;	/* length of name, rounded up */
  uint32_t ad_window;		/* client specified window */
  bool_t ad_dosync;		/* synchronize? */
  struct sockaddr ad_syncaddr;	/* remote host to synch with */
  struct rpc_timeval ad_timediff;	/* server's time - client's time */
  uint32_t ad_nickname;		/* server's nickname for client */
  struct authdes_cred ad_cred;	/* storage for credential */
  struct authdes_verf ad_verf;	/* storage for verifier */
  struct rpc_timeval ad_timestamp;	/* timestamp sent */
  des_block ad_xkey;		/* encrypted conversation key */
  u_char ad_pkey[1024];		/* Servers actual public key */
};


/*
 * Create the client des authentication object
 */
AUTH *
authdes_create (const char *servername, u_int window,
		struct sockaddr *syncaddr, des_block *ckey)
  /* servername - network name of server */
  /* window     - time to live */
  /* syncaddr   - optional addr of host to sync with */
  /* ckey       - optional conversation key to use */
{
  char pkey_data[1024];
  netobj pkey;

  if (!getpublickey (servername, pkey_data))
    return NULL;

  pkey.n_bytes = pkey_data;
  pkey.n_len = strlen (pkey_data) + 1;
  return authdes_pk_create (servername, &pkey, window, syncaddr, ckey);
}
#ifdef EXPORT_RPC_SYMBOLS
libc_hidden_def (authdes_create)
#else
libc_hidden_nolink_sunrpc (authdes_create, GLIBC_2_1)
#endif

AUTH *
authdes_pk_create (const char *servername, netobj *pkey, u_int window,
		   struct sockaddr *syncaddr, des_block *ckey)
{
  AUTH *auth;
  struct ad_private *ad;
  char namebuf[MAXNETNAMELEN + 1];

  /*
   * Allocate everything now
   */
  auth = ALLOC (AUTH);
  ad = ALLOC (struct ad_private);

  if (auth == NULL || ad == NULL)
    {
      debug ("authdes_create: out of memory");
      goto failed;
    }

  memset (ad, 0, sizeof (struct ad_private));
  memcpy (ad->ad_pkey, pkey->n_bytes, pkey->n_len);
  if (!getnetname (namebuf))
    goto failed;
  ad->ad_fullnamelen = RNDUP (strlen (namebuf));
  ad->ad_fullname = mem_alloc (ad->ad_fullnamelen + 1);

  ad->ad_servernamelen = strlen (servername);
  ad->ad_servername = mem_alloc (ad->ad_servernamelen + 1);

  if (ad->ad_fullname == NULL || ad->ad_servername == NULL)
    {
      debug ("authdes_create: out of memory");
      goto failed;
    }

  /*
   * Set up private data
   */
  memcpy (ad->ad_fullname, namebuf, ad->ad_fullnamelen + 1);
  memcpy (ad->ad_servername, servername, ad->ad_servernamelen + 1);
  ad->ad_timediff.tv_sec = ad->ad_timediff.tv_usec = 0;
  if (syncaddr != NULL)
    {
      ad->ad_syncaddr = *syncaddr;
      ad->ad_dosync = TRUE;
    }
  else
    ad->ad_dosync = FALSE;

  ad->ad_window = window;
  if (ckey == NULL)
    {
      if (key_gendes (&auth->ah_key) < 0)
	{
	  debug ("authdes_create: unable to gen conversation key");
	  goto failed;
	}
    }
  else
    auth->ah_key = *ckey;

  /*
   * Set up auth handle
   */
  auth->ah_cred.oa_flavor = AUTH_DES;
  auth->ah_verf.oa_flavor = AUTH_DES;
  auth->ah_ops = (struct auth_ops *) &authdes_ops;
  auth->ah_private = (caddr_t) ad;

  if (!authdes_refresh (auth))
    goto failed;

  return auth;

failed:
  if (auth != NULL)
    FREE (auth, sizeof (AUTH));
  if (ad != NULL)
    {
      if (ad->ad_fullname != NULL)
	FREE (ad->ad_fullname, ad->ad_fullnamelen + 1);
      if (ad->ad_servername != NULL)
	FREE (ad->ad_servername, ad->ad_servernamelen + 1);
      FREE (ad, sizeof (struct ad_private));
    }
  return NULL;
}
#ifdef EXPORT_RPC_SYMBOLS
libc_hidden_def (authdes_pk_create)
#else
libc_hidden_nolink_sunrpc (authdes_pk_create, GLIBC_2_1)
#endif

/*
 * Implement the five authentication operations
 */


/*
 * 1. Next Verifier
 */
/*ARGSUSED */
static void
authdes_nextverf (AUTH *auth)
{
  /* what the heck am I supposed to do??? */
}



/*
 * 2. Marshal
 */
static bool_t
authdes_marshal (AUTH *auth, XDR *xdrs)
{
  struct ad_private *ad = AUTH_PRIVATE (auth);
  struct authdes_cred *cred = &ad->ad_cred;
  struct authdes_verf *verf = &ad->ad_verf;
  des_block cryptbuf[2];
  des_block ivec;
  int status;
  int len;
  register int32_t *ixdr;
  struct timespec now;

  /*
   * Figure out the "time", accounting for any time difference
   * with the server if necessary.
   */
  __clock_gettime (CLOCK_REALTIME, &now);
  ad->ad_timestamp.tv_sec = now.tv_sec + ad->ad_timediff.tv_sec;
  ad->ad_timestamp.tv_usec = (now.tv_nsec / 1000) + ad->ad_timediff.tv_usec;
  if (ad->ad_timestamp.tv_usec >= MILLION)
    {
      ad->ad_timestamp.tv_usec -= MILLION;
      ad->ad_timestamp.tv_sec += 1;
    }

  /*
   * XDR the timestamp and possibly some other things, then
   * encrypt them.
   * XXX We have a real Year 2038 problem here.
   */
  ixdr = (int32_t *) cryptbuf;
  IXDR_PUT_INT32 (ixdr, ad->ad_timestamp.tv_sec);
  IXDR_PUT_INT32 (ixdr, ad->ad_timestamp.tv_usec);
  if (ad->ad_cred.adc_namekind == ADN_FULLNAME)
    {
      IXDR_PUT_U_INT32 (ixdr, ad->ad_window);
      IXDR_PUT_U_INT32 (ixdr, ad->ad_window - 1);
      ivec.key.high = ivec.key.low = 0;
      status = cbc_crypt ((char *) &auth->ah_key, (char *) cryptbuf,
	      2 * sizeof (des_block), DES_ENCRYPT | DES_HW, (char *) &ivec);
    }
  else
    status = ecb_crypt ((char *) &auth->ah_key, (char *) cryptbuf,
			sizeof (des_block), DES_ENCRYPT | DES_HW);

  if (DES_FAILED (status))
    {
      debug ("authdes_marshal: DES encryption failure");
      return FALSE;
    }
  ad->ad_verf.adv_xtimestamp = cryptbuf[0];
  if (ad->ad_cred.adc_namekind == ADN_FULLNAME)
    {
      ad->ad_cred.adc_fullname.window = cryptbuf[1].key.high;
      ad->ad_verf.adv_winverf = cryptbuf[1].key.low;
    }
  else
    {
      ad->ad_cred.adc_nickname = ad->ad_nickname;
      ad->ad_verf.adv_winverf = 0;
    }

  /*
   * Serialize the credential and verifier into opaque
   * authentication data.
   */
  if (ad->ad_cred.adc_namekind == ADN_FULLNAME)
    len = ((1 + 1 + 2 + 1) * BYTES_PER_XDR_UNIT + ad->ad_fullnamelen);
  else
    len = (1 + 1) * BYTES_PER_XDR_UNIT;

  if ((ixdr = xdr_inline (xdrs, 2 * BYTES_PER_XDR_UNIT)) != NULL)
    {
      IXDR_PUT_INT32 (ixdr, AUTH_DES);
      IXDR_PUT_U_INT32 (ixdr, len);
    }
  else
    {
      ATTEMPT (xdr_putint32 (xdrs, &auth->ah_cred.oa_flavor));
      ATTEMPT (xdr_putint32 (xdrs, &len));
    }
  ATTEMPT (xdr_authdes_cred (xdrs, cred));

  len = (2 + 1) * BYTES_PER_XDR_UNIT;
  if ((ixdr = xdr_inline (xdrs, 2 * BYTES_PER_XDR_UNIT)) != NULL)
    {
      IXDR_PUT_INT32 (ixdr, AUTH_DES);
      IXDR_PUT_U_INT32 (ixdr, len);
    }
  else
    {
      ATTEMPT (xdr_putint32 (xdrs, &auth->ah_verf.oa_flavor));
      ATTEMPT (xdr_putint32 (xdrs, &len));
    }
  ATTEMPT (xdr_authdes_verf (xdrs, verf));

  return TRUE;
}


/*
 * 3. Validate
 */
static bool_t
authdes_validate (AUTH *auth, struct opaque_auth *rverf)
{
  struct ad_private *ad = AUTH_PRIVATE (auth);
  struct authdes_verf verf;
  int status;
  register uint32_t *ixdr;

  if (rverf->oa_length != (2 + 1) * BYTES_PER_XDR_UNIT)
    return FALSE;

  ixdr = (uint32_t *) rverf->oa_base;
  verf.adv_xtimestamp.key.high = *ixdr++;
  verf.adv_xtimestamp.key.low = *ixdr++;
  verf.adv_int_u = *ixdr++;	/* nickname not XDR'd ! */

  /*
   * Decrypt the timestamp
   */
  status = ecb_crypt ((char *) &auth->ah_key, (char *) &verf.adv_xtimestamp,
		      sizeof (des_block), DES_DECRYPT | DES_HW);

  if (DES_FAILED (status))
    {
      debug ("authdes_validate: DES decryption failure");
      return FALSE;
    }

  /*
   * xdr the decrypted timestamp
   */
  ixdr = (uint32_t *) verf.adv_xtimestamp.c;
  verf.adv_timestamp.tv_sec = IXDR_GET_U_INT32 (ixdr) + 1;
  verf.adv_timestamp.tv_usec = IXDR_GET_U_INT32 (ixdr);

  /*
   * validate
   */
  if (memcmp ((char *) &ad->ad_timestamp, (char *) &verf.adv_timestamp,
	      sizeof (struct rpc_timeval)) != 0)
    {
      debug ("authdes_validate: verifier mismatch\n");
      return FALSE;
    }

  /*
   * We have a nickname now, let's use it
   */
  ad->ad_nickname = verf.adv_nickname;
  ad->ad_cred.adc_namekind = ADN_NICKNAME;
  return TRUE;
}

/*
 * 4. Refresh
 */
static bool_t
authdes_refresh (AUTH *auth)
{
  netobj pkey;
  struct ad_private *ad = AUTH_PRIVATE (auth);
  struct authdes_cred *cred = &ad->ad_cred;

  if (ad->ad_dosync && !synchronize (&ad->ad_syncaddr, &ad->ad_timediff))
    {
      /*
       * Hope the clocks are synced!
       */
      ad->ad_timediff.tv_sec = ad->ad_timediff.tv_usec = 0;
      debug ("authdes_refresh: unable to synchronize with server");
    }
  ad->ad_xkey = auth->ah_key;
  pkey.n_bytes = (char *) (ad->ad_pkey);
  pkey.n_len = strlen ((char *) ad->ad_pkey) + 1;
  if (key_encryptsession_pk (ad->ad_servername, &pkey, &ad->ad_xkey) < 0)
    {
      debug ("authdes_create: unable to encrypt conversation key");
      return FALSE;
    }
  cred->adc_fullname.key = ad->ad_xkey;
  cred->adc_namekind = ADN_FULLNAME;
  cred->adc_fullname.name = ad->ad_fullname;
  return TRUE;
}

/*
 * 5. Destroy
 */
static void
authdes_destroy (AUTH *auth)
{
  struct ad_private *ad = AUTH_PRIVATE (auth);

  FREE (ad->ad_fullname, ad->ad_fullnamelen + 1);
  FREE (ad->ad_servername, ad->ad_servernamelen + 1);
  FREE (ad, sizeof (struct ad_private));
  FREE (auth, sizeof (AUTH));
}

/*
 * Synchronize with the server at the given address, that is,
 * adjust timep to reflect the delta between our clocks
 */
static bool_t
synchronize (struct sockaddr *syncaddr, struct rpc_timeval *timep)
{
  struct timespec mytime;
  struct rpc_timeval timeout;
  long int myusec;

  timeout.tv_sec = RTIME_TIMEOUT;
  timeout.tv_usec = 0;
  if (rtime ((struct sockaddr_in *) syncaddr, timep, &timeout) < 0)
    return FALSE;

  __clock_gettime (CLOCK_REALTIME, &mytime);
  timep->tv_sec -= mytime.tv_sec;
  myusec = mytime.tv_nsec / 1000;
  if (myusec > timep->tv_usec)
    {
      timep->tv_sec -= 1;
      timep->tv_usec += MILLION;
    }
  timep->tv_usec -= myusec;
  return TRUE;
}
