/*
 * xdr_mem.c, XDR implementation using memory buffers.
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
 * If you have some data to be interpreted as external data representation
 * or to be converted to external data representation in a memory buffer,
 * then this is the package for you.
 */

#include <string.h>
#include <limits.h>
#include <rpc/rpc.h>
#include <shlib-compat.h>

static bool_t xdrmem_getlong (XDR *, long *);
static bool_t xdrmem_putlong (XDR *, const long *);
static bool_t xdrmem_getbytes (XDR *, caddr_t, u_int);
static bool_t xdrmem_putbytes (XDR *, const char *, u_int);
static u_int xdrmem_getpos (const XDR *);
static bool_t xdrmem_setpos (XDR *, u_int);
static int32_t *xdrmem_inline (XDR *, u_int);
static void xdrmem_destroy (XDR *);
static bool_t xdrmem_getint32 (XDR *, int32_t *);
static bool_t xdrmem_putint32 (XDR *, const int32_t *);

static const struct xdr_ops xdrmem_ops =
{
  xdrmem_getlong,
  xdrmem_putlong,
  xdrmem_getbytes,
  xdrmem_putbytes,
  xdrmem_getpos,
  xdrmem_setpos,
  xdrmem_inline,
  xdrmem_destroy,
  xdrmem_getint32,
  xdrmem_putint32
};

/*
 * The procedure xdrmem_create initializes a stream descriptor for a
 * memory buffer.
 */
void
xdrmem_create (XDR *xdrs, const caddr_t addr, u_int size, enum xdr_op op)
{
  xdrs->x_op = op;
  /* We have to add the const since the `struct xdr_ops' in `struct XDR'
     is not `const'.  */
  xdrs->x_ops = (struct xdr_ops *) &xdrmem_ops;
  xdrs->x_private = xdrs->x_base = addr;
  xdrs->x_handy = size;
}
#ifdef EXPORT_RPC_SYMBOLS
libc_hidden_def (xdrmem_create)
#else
libc_hidden_nolink_sunrpc (xdrmem_create, GLIBC_2_0)
#endif

/*
 * Nothing needs to be done for the memory case.  The argument is clearly
 * const.
 */

static void
xdrmem_destroy (XDR *xdrs)
{
}

/*
 * Gets the next word from the memory referenced by xdrs and places it
 * in the long pointed to by lp.  It then increments the private word to
 * point at the next element.  Neither object pointed to is const
 */
static bool_t
xdrmem_getlong (XDR *xdrs, long *lp)
{
  if (xdrs->x_handy < 4)
    return FALSE;
  xdrs->x_handy -= 4;
  *lp = (int32_t) ntohl ((*((int32_t *) (xdrs->x_private))));
  xdrs->x_private += 4;
  return TRUE;
}

/*
 * Puts the long pointed to by lp in the memory referenced by xdrs.  It
 * then increments the private word to point at the next element.  The
 * long pointed at is const
 */
static bool_t
xdrmem_putlong (XDR *xdrs, const long *lp)
{
  if (xdrs->x_handy < 4)
    return FALSE;
  xdrs->x_handy -= 4;
  *(int32_t *) xdrs->x_private = htonl (*lp);
  xdrs->x_private += 4;
  return TRUE;
}

/*
 * Gets an unaligned number of bytes from the xdrs structure and writes them
 * to the address passed in addr.  Be very careful when calling this routine
 * as it could leave the xdrs pointing to an unaligned structure which is not
 * a good idea.  None of the things pointed to are const.
 */
static bool_t
xdrmem_getbytes (XDR *xdrs, caddr_t addr, u_int len)
{
  if (xdrs->x_handy < len)
    return FALSE;
  xdrs->x_handy -= len;
  memcpy (addr, xdrs->x_private, len);
  xdrs->x_private += len;
  return TRUE;
}

/*
 * The complementary function to the above.  The same warnings apply about
 * unaligned data.  The source address is const.
 */
static bool_t
xdrmem_putbytes (XDR *xdrs, const char *addr, u_int len)
{
  if (xdrs->x_handy < len)
    return FALSE;
  xdrs->x_handy -= len;
  memcpy (xdrs->x_private, addr, len);
  xdrs->x_private += len;
  return TRUE;
}

/*
 * Not sure what this one does.  But it clearly doesn't modify the contents
 * of xdrs.  **FIXME** does this not assume u_int == u_long?
 */
static u_int
xdrmem_getpos (const XDR *xdrs)
{
  return (u_long) xdrs->x_private - (u_long) xdrs->x_base;
}

/*
 * xdrs modified
 */
static bool_t
xdrmem_setpos (XDR *xdrs, u_int pos)
{
  caddr_t newaddr = xdrs->x_base + pos;
  caddr_t lastaddr = xdrs->x_private + xdrs->x_handy;
  size_t handy = lastaddr - newaddr;

  if (newaddr > lastaddr
      || newaddr < xdrs->x_base
      || handy != (u_int) handy)
    return FALSE;

  xdrs->x_private = newaddr;
  xdrs->x_handy = (u_int) handy;
  return TRUE;
}

/*
 * xdrs modified
 */
static int32_t *
xdrmem_inline (XDR *xdrs, u_int len)
{
  int32_t *buf = 0;

  if (xdrs->x_handy >= len)
    {
      xdrs->x_handy -= len;
      buf = (int32_t *) xdrs->x_private;
      xdrs->x_private += len;
    }
  return buf;
}

/*
 * Gets the next word from the memory referenced by xdrs and places it
 * in the int pointed to by ip.  It then increments the private word to
 * point at the next element.  Neither object pointed to is const
 */
static bool_t
xdrmem_getint32 (XDR *xdrs, int32_t *ip)
{
  if (xdrs->x_handy < 4)
    return FALSE;
  xdrs->x_handy -= 4;
  *ip = ntohl ((*((int32_t *) (xdrs->x_private))));
  xdrs->x_private += 4;
  return TRUE;
}

/*
 * Puts the long pointed to by lp in the memory referenced by xdrs.  It
 * then increments the private word to point at the next element.  The
 * long pointed at is const
 */
static bool_t
xdrmem_putint32 (XDR *xdrs, const int32_t *ip)
{
  if (xdrs->x_handy < 4)
    return FALSE;
  xdrs->x_handy -= 4;
  *(int32_t *) xdrs->x_private = htonl (*ip);
  xdrs->x_private += 4;
  return TRUE;
}
