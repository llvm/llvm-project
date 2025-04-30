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
 *
 * svcauth_des.c, server-side des authentication
 *
 * We insure for the service the following:
 * (1) The timestamp microseconds do not exceed 1 million.
 * (2) The timestamp plus the window is less than the current time.
 * (3) The timestamp is not less than the one previously
 *     seen in the current session.
 *
 * It is up to the server to determine if the window size is
 * too small .
 *
 */

#include <limits.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/param.h>
#include <netinet/in.h>
#include <rpc/rpc.h>
#include <rpc/xdr.h>
#include <rpc/auth.h>
#include <rpc/auth_des.h>
#include <rpc/svc_auth.h>
#include <rpc/svc.h>
#include <rpc/des_crypt.h>
#include <shlib-compat.h>

#define debug(msg)		/*printf("svcauth_des: %s\n", msg) */

#define BEFORE(t1, t2) timercmp(t1, t2, <)

/*
 * LRU cache of conversation keys and some other useful items.
 */
#define AUTHDES_CACHESZ 64
struct cache_entry
  {
    des_block key;		/* conversation key */
    char *rname;		/* client's name */
    u_int window;		/* credential lifetime window */
    struct rpc_timeval laststamp;	/* detect replays of creds */
    char *localcred;		/* generic local credential */
  };
#define authdes_cache RPC_THREAD_VARIABLE(authdes_cache_s)
#define authdes_lru RPC_THREAD_VARIABLE(authdes_lru_s)

static void cache_init (void); /* initialize the cache */
static short cache_spot (des_block *, char *, struct rpc_timeval *);
  /* find an entry in the cache */
static void cache_ref (uint32_t sid); /* note that sid was ref'd */

static void invalidate (char *cred); /* invalidate entry in cache */

/* Cache statistics.  Accidental historic export without a matching
   declaration in any header file.  */
#ifndef SHARED
static
#endif
struct
  {
    u_long ncachehits;		/* times cache hit, and is not replay */
    u_long ncachereplays;	/* times cache hit, and is replay */
    u_long ncachemisses;	/* times cache missed */
  }
svcauthdes_stats;
#ifdef SHARED
compat_symbol (libc, svcauthdes_stats, svcauthdes_stats, GLIBC_2_0);
#endif

/*
 * Service side authenticator for AUTH_DES
 */
enum auth_stat
_svcauth_des (register struct svc_req *rqst, register struct rpc_msg *msg)
{
  register uint32_t *ixdr;
  des_block cryptbuf[2];
  register struct authdes_cred *cred;
  struct authdes_verf verf;
  int status;
  register struct cache_entry *entry;
  uint32_t sid = 0;
  des_block *sessionkey;
  des_block ivec;
  u_int window;
  struct rpc_timeval timestamp;
  uint32_t namelen;
  struct area
    {
      struct authdes_cred area_cred;
      char area_netname[MAXNETNAMELEN + 1];
    }
   *area;

  if (authdes_cache == NULL)
    cache_init ();
  if (authdes_cache == NULL) /* No free memory */
    return AUTH_FAILED;

  area = (struct area *) rqst->rq_clntcred;
  cred = (struct authdes_cred *) &area->area_cred;

  /*
   * Get the credential
   */
  if (msg->rm_call.cb_cred.oa_length <= 0 ||
      msg->rm_call.cb_cred.oa_length > MAX_AUTH_BYTES)
    return AUTH_BADCRED;

  ixdr = (uint32_t *) msg->rm_call.cb_cred.oa_base;
  cred->adc_namekind = IXDR_GET_ENUM (ixdr, enum authdes_namekind);
  switch (cred->adc_namekind)
    {
    case ADN_FULLNAME:
      namelen = IXDR_GET_U_INT32 (ixdr);
      if (namelen > MAXNETNAMELEN)
	{
	  return AUTH_BADCRED;
	}
      cred->adc_fullname.name = area->area_netname;
      memcpy (cred->adc_fullname.name, (char *) ixdr, namelen);
      cred->adc_fullname.name[namelen] = 0;
      ixdr += (RNDUP (namelen) / BYTES_PER_XDR_UNIT);
      cred->adc_fullname.key.key.high = *ixdr++;
      cred->adc_fullname.key.key.low = *ixdr++;
      cred->adc_fullname.window = *ixdr++;
      break;
    case ADN_NICKNAME:
      cred->adc_nickname = *ixdr++;
      break;
    default:
      return AUTH_BADCRED;
    }

  /*
   * Get the verifier
   */
  if (msg->rm_call.cb_verf.oa_length <= 0 ||
      msg->rm_call.cb_verf.oa_length > MAX_AUTH_BYTES)
    return AUTH_BADCRED;

  ixdr = (uint32_t *) msg->rm_call.cb_verf.oa_base;
  verf.adv_xtimestamp.key.high = *ixdr++;
  verf.adv_xtimestamp.key.low = *ixdr++;
  verf.adv_int_u = *ixdr++;

  /*
   * Get the conversation key
   */
  if (cred->adc_namekind == ADN_FULLNAME)
    {
      netobj pkey;
      char pkey_data[1024];

      sessionkey = &cred->adc_fullname.key;
      if (!getpublickey (cred->adc_fullname.name, pkey_data))
	{
	  debug("getpublickey");
	  return AUTH_BADCRED;
	}
      pkey.n_bytes = pkey_data;
      pkey.n_len = strlen (pkey_data) + 1;
      if (key_decryptsession_pk (cred->adc_fullname.name, &pkey,
				 sessionkey) < 0)
	{
	  debug ("decryptsessionkey");
	  return AUTH_BADCRED;	/* key not found */
	}
    }
  else
    {				/* ADN_NICKNAME */
      if (cred->adc_nickname >= AUTHDES_CACHESZ)
	{
	  debug ("bad nickname");
	  return AUTH_BADCRED;	/* garbled credential */
	}
      else
	sid = cred->adc_nickname;

      /* XXX This could be wrong, but else we have a
	 security problem */
      if (authdes_cache[sid].rname == NULL)
	return AUTH_BADCRED;
      sessionkey = &authdes_cache[sid].key;
    }


  /*
   * Decrypt the timestamp
   */
  cryptbuf[0] = verf.adv_xtimestamp;
  if (cred->adc_namekind == ADN_FULLNAME)
    {
      cryptbuf[1].key.high = cred->adc_fullname.window;
      cryptbuf[1].key.low = verf.adv_winverf;
      ivec.key.high = ivec.key.low = 0;
      status = cbc_crypt ((char *) sessionkey, (char *) cryptbuf,
			  2 * sizeof (des_block), DES_DECRYPT | DES_HW,
			  (char *) &ivec);
    }
  else
    status = ecb_crypt ((char *) sessionkey, (char *) cryptbuf,
			sizeof (des_block), DES_DECRYPT | DES_HW);

  if (DES_FAILED (status))
    {
      debug ("decryption failure");
      return AUTH_FAILED;	/* system error */
    }

  /*
   * XDR the decrypted timestamp
   */
  ixdr = (uint32_t *) cryptbuf;
  timestamp.tv_sec = IXDR_GET_INT32 (ixdr);
  timestamp.tv_usec = IXDR_GET_INT32 (ixdr);

  /*
   * Check for valid credentials and verifiers.
   * They could be invalid because the key was flushed
   * out of the cache, and so a new session should begin.
   * Be sure and send AUTH_REJECTED{CRED, VERF} if this is the case.
   */
  {
    struct timeval current;
    int nick;
    u_int winverf;

    if (cred->adc_namekind == ADN_FULLNAME)
      {
	short tmp_spot;

	window = IXDR_GET_U_INT32 (ixdr);
	winverf = IXDR_GET_U_INT32 (ixdr);
	if (winverf != window - 1)
	  {
	    debug ("window verifier mismatch");
	    return AUTH_BADCRED;	/* garbled credential */
	  }
	tmp_spot = cache_spot (sessionkey, cred->adc_fullname.name,
			       &timestamp);
	if (tmp_spot < 0 || tmp_spot > AUTHDES_CACHESZ)
	  {
	    debug ("replayed credential");
	    return AUTH_REJECTEDCRED;		/* replay */
	  }
	sid = tmp_spot;
	nick = 0;
      }
    else
      {				/* ADN_NICKNAME */
	window = authdes_cache[sid].window;
	nick = 1;
      }

    if (timestamp.tv_usec >= USEC_PER_SEC)
      {
	debug ("invalid usecs");
	/* cached out (bad key), or garbled verifier */
	return nick ? AUTH_REJECTEDVERF : AUTH_BADVERF;
      }
    if (nick && BEFORE (&timestamp, &authdes_cache[sid].laststamp))
      {
	debug ("timestamp before last seen");
	return AUTH_REJECTEDVERF;	/* replay */
      }
    {
      struct timespec now;
      __clock_gettime (CLOCK_REALTIME, &now);
      TIMESPEC_TO_TIMEVAL (&current, &now);
    }
    current.tv_sec -= window;	/* allow for expiration */
    if (!BEFORE (&current, &timestamp))
      {
	debug ("timestamp expired");
	/* replay, or garbled credential */
	return nick ? AUTH_REJECTEDVERF : AUTH_BADCRED;
      }
  }

  /*
   * Set up the reply verifier
   */
  verf.adv_nickname = sid;

  /*
   * xdr the timestamp before encrypting
   */
  ixdr = (uint32_t *) cryptbuf;
  IXDR_PUT_INT32 (ixdr, timestamp.tv_sec - 1);
  IXDR_PUT_INT32 (ixdr, timestamp.tv_usec);

  /*
   * encrypt the timestamp
   */
  status = ecb_crypt ((char *) sessionkey, (char *) cryptbuf,
		      sizeof (des_block), DES_ENCRYPT | DES_HW);
  if (DES_FAILED (status))
    {
      debug ("encryption failure");
      return AUTH_FAILED;	/* system error */
    }
  verf.adv_xtimestamp = cryptbuf[0];

  /*
   * Serialize the reply verifier, and update rqst
   */
  ixdr = (uint32_t *) msg->rm_call.cb_verf.oa_base;
  *ixdr++ = verf.adv_xtimestamp.key.high;
  *ixdr++ = verf.adv_xtimestamp.key.low;
  *ixdr++ = verf.adv_int_u;

  rqst->rq_xprt->xp_verf.oa_flavor = AUTH_DES;
  rqst->rq_xprt->xp_verf.oa_base = msg->rm_call.cb_verf.oa_base;
  rqst->rq_xprt->xp_verf.oa_length =
    (char *) ixdr - msg->rm_call.cb_verf.oa_base;

  /*
   * We succeeded, commit the data to the cache now and
   * finish cooking the credential.
   */
  entry = &authdes_cache[sid];
  entry->laststamp = timestamp;
  cache_ref (sid);
  if (cred->adc_namekind == ADN_FULLNAME)
    {
      size_t full_len;

      cred->adc_fullname.window = window;
      cred->adc_nickname = sid;	/* save nickname */
      if (entry->rname != NULL)
	mem_free (entry->rname, strlen (entry->rname) + 1);
      full_len = strlen (cred->adc_fullname.name) + 1;
      entry->rname = mem_alloc ((u_int) full_len);
      if (entry->rname != NULL)
	memcpy (entry->rname, cred->adc_fullname.name, full_len);
      else
	{
	  debug ("out of memory");
	  return AUTH_FAILED; /* out of memory is bad */
	}
      entry->key = *sessionkey;
      entry->window = window;
      invalidate (entry->localcred);	/* mark any cached cred invalid */
    }
  else
    {				/* ADN_NICKNAME */
      /*
       * nicknames are cooked into fullnames
       */
      cred->adc_namekind = ADN_FULLNAME;
      cred->adc_fullname.name = entry->rname;
      cred->adc_fullname.key = entry->key;
      cred->adc_fullname.window = entry->window;
    }
  return AUTH_OK;		/* we made it! */
}


/*
 * Initialize the cache
 */
static void
cache_init (void)
{
  register int i;

  authdes_cache = (struct cache_entry *)
    calloc (sizeof (struct cache_entry) * AUTHDES_CACHESZ, 1);
  if (authdes_cache == NULL)
    return;

  authdes_lru = (int *) mem_alloc (sizeof (int) * AUTHDES_CACHESZ);
  /*
   * Initialize the lru list
   */
  for (i = 0; i < AUTHDES_CACHESZ; ++i)
    authdes_lru[i] = i;
}


/*
 * Find the lru victim
 */
static short
cache_victim (void)
{
  return authdes_lru[AUTHDES_CACHESZ - 1];
}

/*
 * Note that sid was referenced
 */
static void
cache_ref (register uint32_t sid)
{
  register int i;
  register int curr;
  register int prev;

  prev = authdes_lru[0];
  authdes_lru[0] = sid;
  for (i = 1; prev != sid; ++i)
    {
      curr = authdes_lru[i];
      authdes_lru[i] = prev;
      prev = curr;
    }
}

/*
 * Find a spot in the cache for a credential containing
 * the items given.  Return -1 if a replay is detected, otherwise
 * return the spot in the cache.
 */
static short
cache_spot (register des_block *key, char *name,
	    struct rpc_timeval *timestamp)
{
  register struct cache_entry *cp;
  register int i;
  register uint32_t hi;

  hi = key->key.high;
  for (cp = authdes_cache, i = 0; i < AUTHDES_CACHESZ; ++i, ++cp)
    {
      if (cp->key.key.high == hi &&
	  cp->key.key.low == key->key.low &&
	  cp->rname != NULL &&
	  memcmp (cp->rname, name, strlen (name) + 1) == 0)
	{
	  if (BEFORE (timestamp, &cp->laststamp))
	    {
	      ++svcauthdes_stats.ncachereplays;
	      return -1;	/* replay */
	    }
	  ++svcauthdes_stats.ncachehits;
	  return i;		/* refresh */
	}
    }
  ++svcauthdes_stats.ncachemisses;
  return cache_victim ();	/* new credential */
}

/*
 * Local credential handling stuff.
 * NOTE: bsd unix dependent.
 * Other operating systems should put something else here.
 */
#define UNKNOWN 	-2	/* grouplen, if cached cred is unknown user */
#define INVALID		-1	/* grouplen, if cache entry is invalid */

struct bsdcred
{
  uid_t uid;			/* cached uid */
  gid_t gid;			/* cached gid */
  int grouplen;			/* length of cached groups */
  int grouplen_max;		/* length of allocated cached groups */
  gid_t groups[0];		/* cached groups */
};

/*
 * Map a des credential into a unix cred.
 * We cache the credential here so the application does
 * not have to make an rpc call every time to interpret
 * the credential.
 */
int
authdes_getucred (const struct authdes_cred *adc, uid_t * uid, gid_t * gid,
		  short *grouplen, gid_t * groups)
{
  unsigned sid;
  register int i;
  uid_t i_uid;
  gid_t i_gid;
  int i_grouplen;
  struct bsdcred *cred;

  sid = adc->adc_nickname;
  if (sid >= AUTHDES_CACHESZ)
    {
      debug ("invalid nickname");
      return 0;
    }
  cred = (struct bsdcred *) authdes_cache[sid].localcred;
  if (cred == NULL || cred->grouplen == INVALID)
    {
      /*
       * not in cache: lookup
       */
      if (!netname2user (adc->adc_fullname.name, &i_uid, &i_gid,
			 &i_grouplen, groups))
	{
	  debug ("unknown netname");
	  if (cred != NULL)
	    cred->grouplen = UNKNOWN;	/* mark as lookup up, but not found */
	  return 0;
	}

      if (cred != NULL && cred->grouplen_max < i_grouplen)
	{
	  /* We already have an allocated data structure.  But it is
	     too small.  */
	  free (cred);
	  authdes_cache[sid].localcred = NULL;
	  cred = NULL;
	}

      if (cred == NULL)
	{
	  /* We should allocate room for at least NGROUPS groups.  */
	  int ngroups_max = MAX (i_grouplen, NGROUPS);

	  cred = (struct bsdcred *) mem_alloc (sizeof (struct bsdcred)
					       + ngroups_max * sizeof (gid_t));
	  if (cred == NULL)
	    return 0;

	  authdes_cache[sid].localcred = (char *) cred;
	  cred->grouplen = INVALID;
	  cred->grouplen_max = ngroups_max;
	}

      debug ("missed ucred cache");
      *uid = cred->uid = i_uid;
      *gid = cred->gid = i_gid;
      cred->grouplen = i_grouplen;
      for (i = i_grouplen - 1; i >= 0; --i)
	cred->groups[i] = groups[i];
      /* Make sure no too large values are reported.  */
      *grouplen = MIN (SHRT_MAX, i_grouplen);
      return 1;
    }
  else if (cred->grouplen == UNKNOWN)
    {
      /*
       * Already lookup up, but no match found
       */
      return 0;
    }

  /*
   * cached credentials
   */
  *uid = cred->uid;
  *gid = cred->gid;

  /* Another stupidity in the interface: *grouplen is of type short.
     So we might have to cut the information passed up short.  */
  int grouplen_copy = MIN (SHRT_MAX, cred->grouplen);
  *grouplen = grouplen_copy;
  for (i = grouplen_copy - 1; i >= 0; --i)
    groups[i] = cred->groups[i];
  return 1;
}
libc_hidden_nolink_sunrpc (authdes_getucred, GLIBC_2_1)

static void
invalidate (char *cred)
{
  if (cred == NULL)
    return;
  ((struct bsdcred *) cred)->grouplen = INVALID;
}
