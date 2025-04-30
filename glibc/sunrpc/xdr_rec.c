/*
 * xdr_rec.c, Implements TCP/IP based XDR streams with a "record marking"
 * layer above tcp (for rpc's use).
 *
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
 * These routines interface XDRSTREAMS to a tcp/ip connection.
 * There is a record marking layer between the xdr stream
 * and the tcp transport level.  A record is composed on one or more
 * record fragments.  A record fragment is a thirty-two bit header followed
 * by n bytes of data, where n is contained in the header.  The header
 * is represented as a htonl(u_long).  The high order bit encodes
 * whether or not the fragment is the last fragment of the record
 * (1 => fragment is last, 0 => more fragments to follow.
 * The other 31 bits encode the byte length of the fragment.
 */

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <rpc/rpc.h>
#include <libintl.h>
#include <wchar.h>
#include <libio/iolibio.h>
#include <shlib-compat.h>

static bool_t xdrrec_getlong (XDR *, long *);
static bool_t xdrrec_putlong (XDR *, const long *);
static bool_t xdrrec_getbytes (XDR *, caddr_t, u_int);
static bool_t xdrrec_putbytes (XDR *, const char *, u_int);
static u_int xdrrec_getpos (const XDR *);
static bool_t xdrrec_setpos (XDR *, u_int);
static int32_t *xdrrec_inline (XDR *, u_int);
static void xdrrec_destroy (XDR *);
static bool_t xdrrec_getint32 (XDR *, int32_t *);
static bool_t xdrrec_putint32 (XDR *, const int32_t *);

static const struct xdr_ops xdrrec_ops = {
  xdrrec_getlong,
  xdrrec_putlong,
  xdrrec_getbytes,
  xdrrec_putbytes,
  xdrrec_getpos,
  xdrrec_setpos,
  xdrrec_inline,
  xdrrec_destroy,
  xdrrec_getint32,
  xdrrec_putint32
};

/*
 * A record is composed of one or more record fragments.
 * A record fragment is a two-byte header followed by zero to
 * 2**32-1 bytes.  The header is treated as a long unsigned and is
 * encode/decoded to the network via htonl/ntohl.  The low order 31 bits
 * are a byte count of the fragment.  The highest order bit is a boolean:
 * 1 => this fragment is the last fragment of the record,
 * 0 => this fragment is followed by more fragment(s).
 *
 * The fragment/record machinery is not general;  it is constructed to
 * meet the needs of xdr and rpc based on tcp.
 */

#define LAST_FRAG (1UL << 31)

typedef struct rec_strm
  {
    caddr_t tcp_handle;
    caddr_t the_buffer;
    /*
     * out-going bits
     */
    int (*writeit) (char *, char *, int);
    caddr_t out_base;		/* output buffer (points to frag header) */
    caddr_t out_finger;		/* next output position */
    caddr_t out_boundry;	/* data cannot up to this address */
    uint32_t *frag_header;	/* beginning of curren fragment */
    bool_t frag_sent;		/* true if buffer sent in middle of record */
    /*
     * in-coming bits
     */
    int (*readit) (char *, char *, int);
    u_long in_size;		/* fixed size of the input buffer */
    caddr_t in_base;
    caddr_t in_finger;		/* location of next byte to be had */
    caddr_t in_boundry;		/* can read up to this location */
    long fbtbc;			/* fragment bytes to be consumed */
    bool_t last_frag;
    u_int sendsize;
    u_int recvsize;
  }
RECSTREAM;

static u_int fix_buf_size (u_int);
static bool_t skip_input_bytes (RECSTREAM *, long);
static bool_t flush_out (RECSTREAM *, bool_t);
static bool_t set_input_fragment (RECSTREAM *);
static bool_t get_input_bytes (RECSTREAM *, caddr_t, int);

/*
 * Create an xdr handle for xdrrec
 * xdrrec_create fills in xdrs.  Sendsize and recvsize are
 * send and recv buffer sizes (0 => use default).
 * tcp_handle is an opaque handle that is passed as the first parameter to
 * the procedures readit and writeit.  Readit and writeit are read and
 * write respectively.   They are like the system
 * calls expect that they take an opaque handle rather than an fd.
 */
void
xdrrec_create (XDR *xdrs, u_int sendsize,
	       u_int recvsize, caddr_t tcp_handle,
	       int (*readit) (char *, char *, int),
	       int (*writeit) (char *, char *, int))
{
  RECSTREAM *rstrm = (RECSTREAM *) mem_alloc (sizeof (RECSTREAM));
  caddr_t tmp;
  char *buf;

  sendsize = fix_buf_size (sendsize);
  recvsize = fix_buf_size (recvsize);
  buf = mem_alloc (sendsize + recvsize + BYTES_PER_XDR_UNIT);

  if (rstrm == NULL || buf == NULL)
    {
      (void) __fxprintf (NULL, "%s: %s", __func__, _("out of memory\n"));
      mem_free (rstrm, sizeof (RECSTREAM));
      mem_free (buf, sendsize + recvsize + BYTES_PER_XDR_UNIT);
      /*
       *  This is bad.  Should rework xdrrec_create to
       *  return a handle, and in this case return NULL
       */
      return;
    }
  /*
   * adjust sizes and allocate buffer quad byte aligned
   */
  rstrm->sendsize = sendsize;
  rstrm->recvsize = recvsize;
  rstrm->the_buffer = buf;
  tmp = rstrm->the_buffer;
  if ((size_t)tmp % BYTES_PER_XDR_UNIT)
    tmp += BYTES_PER_XDR_UNIT - (size_t)tmp % BYTES_PER_XDR_UNIT;
  rstrm->out_base = tmp;
  rstrm->in_base = tmp + sendsize;
  /*
   * now the rest ...
   */
  /* We have to add the cast since the `struct xdr_ops' in `struct XDR'
     is not `const'.  */
  xdrs->x_ops = (struct xdr_ops *) &xdrrec_ops;
  xdrs->x_private = (caddr_t) rstrm;
  rstrm->tcp_handle = tcp_handle;
  rstrm->readit = readit;
  rstrm->writeit = writeit;
  rstrm->out_finger = rstrm->out_boundry = rstrm->out_base;
  rstrm->frag_header = (uint32_t *) rstrm->out_base;
  rstrm->out_finger += 4;
  rstrm->out_boundry += sendsize;
  rstrm->frag_sent = FALSE;
  rstrm->in_size = recvsize;
  rstrm->in_boundry = rstrm->in_base;
  rstrm->in_finger = (rstrm->in_boundry += recvsize);
  rstrm->fbtbc = 0;
  rstrm->last_frag = TRUE;
}
libc_hidden_nolink_sunrpc (xdrrec_create, GLIBC_2_0)


/*
 * The routines defined below are the xdr ops which will go into the
 * xdr handle filled in by xdrrec_create.
 */

static bool_t
xdrrec_getlong (XDR *xdrs, long *lp)
{
  RECSTREAM *rstrm = (RECSTREAM *) xdrs->x_private;
  int32_t *buflp = (int32_t *) rstrm->in_finger;
  int32_t mylong;

  /* first try the inline, fast case */
  if (rstrm->fbtbc >= BYTES_PER_XDR_UNIT &&
      rstrm->in_boundry - (char *) buflp >= BYTES_PER_XDR_UNIT)
    {
      *lp = (int32_t) ntohl (*buflp);
      rstrm->fbtbc -= BYTES_PER_XDR_UNIT;
      rstrm->in_finger += BYTES_PER_XDR_UNIT;
    }
  else
    {
      if (!xdrrec_getbytes (xdrs, (caddr_t) & mylong,
			    BYTES_PER_XDR_UNIT))
	return FALSE;
      *lp = (int32_t) ntohl (mylong);
    }
  return TRUE;
}

static bool_t
xdrrec_putlong (XDR *xdrs, const long *lp)
{
  RECSTREAM *rstrm = (RECSTREAM *) xdrs->x_private;
  int32_t *dest_lp = (int32_t *) rstrm->out_finger;

  if ((rstrm->out_finger += BYTES_PER_XDR_UNIT) > rstrm->out_boundry)
    {
      /*
       * this case should almost never happen so the code is
       * inefficient
       */
      rstrm->out_finger -= BYTES_PER_XDR_UNIT;
      rstrm->frag_sent = TRUE;
      if (!flush_out (rstrm, FALSE))
	return FALSE;
      dest_lp = (int32_t *) rstrm->out_finger;
      rstrm->out_finger += BYTES_PER_XDR_UNIT;
    }
  *dest_lp = htonl (*lp);
  return TRUE;
}

static bool_t	   /* must manage buffers, fragments, and records */
xdrrec_getbytes (XDR *xdrs, caddr_t addr, u_int len)
{
  RECSTREAM *rstrm = (RECSTREAM *) xdrs->x_private;
  u_int current;

  while (len > 0)
    {
      current = rstrm->fbtbc;
      if (current == 0)
	{
	  if (rstrm->last_frag)
	    return FALSE;
	  if (!set_input_fragment (rstrm))
	    return FALSE;
	  continue;
	}
      current = (len < current) ? len : current;
      if (!get_input_bytes (rstrm, addr, current))
	return FALSE;
      addr += current;
      rstrm->fbtbc -= current;
      len -= current;
    }
  return TRUE;
}

static bool_t
xdrrec_putbytes (XDR *xdrs, const char *addr, u_int len)
{
  RECSTREAM *rstrm = (RECSTREAM *) xdrs->x_private;
  u_int current;

  while (len > 0)
    {
      current = rstrm->out_boundry - rstrm->out_finger;
      current = (len < current) ? len : current;
      memcpy (rstrm->out_finger, addr, current);
      rstrm->out_finger += current;
      addr += current;
      len -= current;
      if (rstrm->out_finger == rstrm->out_boundry && len > 0)
	{
	  rstrm->frag_sent = TRUE;
	  if (!flush_out (rstrm, FALSE))
	    return FALSE;
	}
    }
  return TRUE;
}

static u_int
xdrrec_getpos (const XDR *xdrs)
{
  RECSTREAM *rstrm = (RECSTREAM *) xdrs->x_private;
  long pos;

  pos = __lseek ((int) (long) rstrm->tcp_handle, (long) 0, 1);
  if (pos != -1)
    switch (xdrs->x_op)
      {

      case XDR_ENCODE:
	pos += rstrm->out_finger - rstrm->out_base;
	break;

      case XDR_DECODE:
	pos -= rstrm->in_boundry - rstrm->in_finger;
	break;

      default:
	pos = (u_int) - 1;
	break;
      }
  return (u_int) pos;
}

static bool_t
xdrrec_setpos (XDR *xdrs, u_int pos)
{
  RECSTREAM *rstrm = (RECSTREAM *) xdrs->x_private;
  u_int currpos = xdrrec_getpos (xdrs);
  int delta = currpos - pos;
  caddr_t newpos;

  if ((int) currpos != -1)
    switch (xdrs->x_op)
      {

      case XDR_ENCODE:
	newpos = rstrm->out_finger - delta;
	if (newpos > (caddr_t) rstrm->frag_header &&
	    newpos < rstrm->out_boundry)
	  {
	    rstrm->out_finger = newpos;
	    return TRUE;
	  }
	break;

      case XDR_DECODE:
	newpos = rstrm->in_finger - delta;
	if ((delta < (int) (rstrm->fbtbc)) &&
	    (newpos <= rstrm->in_boundry) &&
	    (newpos >= rstrm->in_base))
	  {
	    rstrm->in_finger = newpos;
	    rstrm->fbtbc -= delta;
	    return TRUE;
	  }
	break;

      default:
	break;
      }
  return FALSE;
}

static int32_t *
xdrrec_inline (XDR *xdrs, u_int len)
{
  RECSTREAM *rstrm = (RECSTREAM *) xdrs->x_private;
  int32_t *buf = NULL;

  switch (xdrs->x_op)
    {

    case XDR_ENCODE:
      if ((rstrm->out_finger + len) <= rstrm->out_boundry)
	{
	  buf = (int32_t *) rstrm->out_finger;
	  rstrm->out_finger += len;
	}
      break;

    case XDR_DECODE:
      if ((len <= rstrm->fbtbc) &&
	  ((rstrm->in_finger + len) <= rstrm->in_boundry))
	{
	  buf = (int32_t *) rstrm->in_finger;
	  rstrm->fbtbc -= len;
	  rstrm->in_finger += len;
	}
      break;

    default:
      break;
    }
  return buf;
}

static void
xdrrec_destroy (XDR *xdrs)
{
  RECSTREAM *rstrm = (RECSTREAM *) xdrs->x_private;

  mem_free (rstrm->the_buffer,
	    rstrm->sendsize + rstrm->recvsize + BYTES_PER_XDR_UNIT);
  mem_free ((caddr_t) rstrm, sizeof (RECSTREAM));
}

static bool_t
xdrrec_getint32 (XDR *xdrs, int32_t *ip)
{
  RECSTREAM *rstrm = (RECSTREAM *) xdrs->x_private;
  int32_t *bufip = (int32_t *) rstrm->in_finger;
  int32_t mylong;

  /* first try the inline, fast case */
  if (rstrm->fbtbc >= BYTES_PER_XDR_UNIT &&
      rstrm->in_boundry - (char *) bufip >= BYTES_PER_XDR_UNIT)
    {
      *ip = ntohl (*bufip);
      rstrm->fbtbc -= BYTES_PER_XDR_UNIT;
      rstrm->in_finger += BYTES_PER_XDR_UNIT;
    }
  else
    {
      if (!xdrrec_getbytes (xdrs, (caddr_t) &mylong,
			    BYTES_PER_XDR_UNIT))
	return FALSE;
      *ip = ntohl (mylong);
    }
  return TRUE;
}

static bool_t
xdrrec_putint32 (XDR *xdrs, const int32_t *ip)
{
  RECSTREAM *rstrm = (RECSTREAM *) xdrs->x_private;
  int32_t *dest_ip = (int32_t *) rstrm->out_finger;

  if ((rstrm->out_finger += BYTES_PER_XDR_UNIT) > rstrm->out_boundry)
    {
      /*
       * this case should almost never happen so the code is
       * inefficient
       */
      rstrm->out_finger -= BYTES_PER_XDR_UNIT;
      rstrm->frag_sent = TRUE;
      if (!flush_out (rstrm, FALSE))
	return FALSE;
      dest_ip = (int32_t *) rstrm->out_finger;
      rstrm->out_finger += BYTES_PER_XDR_UNIT;
    }
  *dest_ip = htonl (*ip);
  return TRUE;
}

/*
 * Exported routines to manage xdr records
 */

/*
 * Before reading (deserializing from the stream, one should always call
 * this procedure to guarantee proper record alignment.
 */
bool_t
xdrrec_skiprecord (XDR *xdrs)
{
  RECSTREAM *rstrm = (RECSTREAM *) xdrs->x_private;

  while (rstrm->fbtbc > 0 || (!rstrm->last_frag))
    {
      if (!skip_input_bytes (rstrm, rstrm->fbtbc))
	return FALSE;
      rstrm->fbtbc = 0;
      if ((!rstrm->last_frag) && (!set_input_fragment (rstrm)))
	return FALSE;
    }
  rstrm->last_frag = FALSE;
  return TRUE;
}
libc_hidden_nolink_sunrpc (xdrrec_skiprecord, GLIBC_2_0)

/*
 * Lookahead function.
 * Returns TRUE iff there is no more input in the buffer
 * after consuming the rest of the current record.
 */
bool_t
xdrrec_eof (XDR *xdrs)
{
  RECSTREAM *rstrm = (RECSTREAM *) xdrs->x_private;

  while (rstrm->fbtbc > 0 || (!rstrm->last_frag))
    {
      if (!skip_input_bytes (rstrm, rstrm->fbtbc))
	return TRUE;
      rstrm->fbtbc = 0;
      if ((!rstrm->last_frag) && (!set_input_fragment (rstrm)))
	return TRUE;
    }
  if (rstrm->in_finger == rstrm->in_boundry)
    return TRUE;
  return FALSE;
}
libc_hidden_nolink_sunrpc (xdrrec_eof, GLIBC_2_0)

/*
 * The client must tell the package when an end-of-record has occurred.
 * The second parameter tells whether the record should be flushed to the
 * (output) tcp stream.  (This lets the package support batched or
 * pipelined procedure calls.)  TRUE => immediate flush to tcp connection.
 */
bool_t
xdrrec_endofrecord (XDR *xdrs, bool_t sendnow)
{
  RECSTREAM *rstrm = (RECSTREAM *) xdrs->x_private;
  u_long len;		/* fragment length */

  if (sendnow || rstrm->frag_sent
      || rstrm->out_finger + BYTES_PER_XDR_UNIT >= rstrm->out_boundry)
    {
      rstrm->frag_sent = FALSE;
      return flush_out (rstrm, TRUE);
    }
  len = (rstrm->out_finger - (char *) rstrm->frag_header
	 - BYTES_PER_XDR_UNIT);
  *rstrm->frag_header = htonl ((u_long) len | LAST_FRAG);
  rstrm->frag_header = (uint32_t *) rstrm->out_finger;
  rstrm->out_finger += BYTES_PER_XDR_UNIT;
  return TRUE;
}
libc_hidden_nolink_sunrpc (xdrrec_endofrecord, GLIBC_2_0)


/*
 * Internal useful routines
 */
static bool_t
flush_out (RECSTREAM *rstrm, bool_t eor)
{
  u_long eormask = (eor == TRUE) ? LAST_FRAG : 0;
  u_long len = (rstrm->out_finger - (char *) rstrm->frag_header
		- BYTES_PER_XDR_UNIT);

  *rstrm->frag_header = htonl (len | eormask);
  len = rstrm->out_finger - rstrm->out_base;
  if ((*(rstrm->writeit)) (rstrm->tcp_handle, rstrm->out_base, (int) len)
      != (int) len)
    return FALSE;
  rstrm->frag_header = (uint32_t *) rstrm->out_base;
  rstrm->out_finger = (caddr_t) rstrm->out_base + BYTES_PER_XDR_UNIT;
  return TRUE;
}

static bool_t	/* knows nothing about records!  Only about input buffers */
fill_input_buf (RECSTREAM *rstrm)
{
  caddr_t where;
  size_t i;
  int len;

  where = rstrm->in_base;
  i = (size_t) rstrm->in_boundry % BYTES_PER_XDR_UNIT;
  where += i;
  len = rstrm->in_size - i;
  if ((len = (*(rstrm->readit)) (rstrm->tcp_handle, where, len)) == -1)
    return FALSE;
  rstrm->in_finger = where;
  where += len;
  rstrm->in_boundry = where;
  return TRUE;
}

static bool_t	/* knows nothing about records!  Only about input buffers */
get_input_bytes (RECSTREAM *rstrm, caddr_t addr, int len)
{
  int current;

  while (len > 0)
    {
      current = rstrm->in_boundry - rstrm->in_finger;
      if (current == 0)
	{
	  if (!fill_input_buf (rstrm))
	    return FALSE;
	  continue;
	}
      current = (len < current) ? len : current;
      memcpy (addr, rstrm->in_finger, current);
      rstrm->in_finger += current;
      addr += current;
      len -= current;
    }
  return TRUE;
}

static bool_t /* next two bytes of the input stream are treated as a header */
set_input_fragment (RECSTREAM *rstrm)
{
  uint32_t header;

  if (! get_input_bytes (rstrm, (caddr_t)&header, BYTES_PER_XDR_UNIT))
    return FALSE;
  header = ntohl (header);
  rstrm->last_frag = ((header & LAST_FRAG) == 0) ? FALSE : TRUE;
  /*
   * Sanity check. Try not to accept wildly incorrect fragment
   * sizes. Unfortunately, only a size of zero can be identified as
   * 'wildely incorrect', and this only, if it is not the last
   * fragment of a message. Ridiculously large fragment sizes may look
   * wrong, but we don't have any way to be certain that they aren't
   * what the client actually intended to send us. Many existing RPC
   * implementations may sent a fragment of size zero as the last
   * fragment of a message.
   */
  if (header == 0)
    return FALSE;
  rstrm->fbtbc = header & ~LAST_FRAG;
  return TRUE;
}

static bool_t	/* consumes input bytes; knows nothing about records! */
skip_input_bytes (RECSTREAM *rstrm, long cnt)
{
  int current;

  while (cnt > 0)
    {
      current = rstrm->in_boundry - rstrm->in_finger;
      if (current == 0)
	{
	  if (!fill_input_buf (rstrm))
	    return FALSE;
	  continue;
	}
      current = (cnt < current) ? cnt : current;
      rstrm->in_finger += current;
      cnt -= current;
    }
  return TRUE;
}

static u_int
fix_buf_size (u_int s)
{
  if (s < 100)
    s = 4000;
  return RNDUP (s);
}
