/*
 * xdr_reference.c, Generic XDR routines implementation.
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
 * These are the "non-trivial" xdr primitives used to serialize and
 * de-serialize "pointers".  See xdr.h for more info on the interface to xdr.
 */

#include <stdio.h>
#include <string.h>
#include <rpc/types.h>
#include <rpc/xdr.h>
#include <libintl.h>
#include <wchar.h>
#include <libio/iolibio.h>
#include <shlib-compat.h>

#define LASTUNSIGNED	((u_int)0-1)

/*
 * XDR an indirect pointer
 * xdr_reference is for recursively translating a structure that is
 * referenced by a pointer inside the structure that is currently being
 * translated.  pp references a pointer to storage. If *pp is null
 * the  necessary storage is allocated.
 * size is the size of the referneced structure.
 * proc is the routine to handle the referenced structure.
 */
bool_t
xdr_reference (XDR *xdrs,
	       /* the pointer to work on */
	       caddr_t *pp,
	       /* size of the object pointed to */
	       u_int size,
	       /* xdr routine to handle the object */
	       xdrproc_t proc)
{
  caddr_t loc = *pp;
  bool_t stat;

  if (loc == NULL)
    switch (xdrs->x_op)
      {
      case XDR_FREE:
	return TRUE;

      case XDR_DECODE:
	*pp = loc = (caddr_t) calloc (1, size);
	if (loc == NULL)
	  {
	    (void) __fxprintf (NULL, "%s: %s", __func__, _("out of memory\n"));
	    return FALSE;
	  }
	break;
      default:
	break;
      }

  stat = (*proc) (xdrs, loc, LASTUNSIGNED);

  if (xdrs->x_op == XDR_FREE)
    {
      mem_free (loc, size);
      *pp = NULL;
    }
  return stat;
}
libc_hidden_nolink_sunrpc (xdr_reference, GLIBC_2_0)


/*
 * xdr_pointer():
 *
 * XDR a pointer to a possibly recursive data structure. This
 * differs with xdr_reference in that it can serialize/deserialize
 * trees correctly.
 *
 *  What's sent is actually a union:
 *
 *  union object_pointer switch (boolean b) {
 *  case TRUE: object_data data;
 *  case FALSE: void nothing;
 *  }
 *
 * > objpp: Pointer to the pointer to the object.
 * > obj_size: size of the object.
 * > xdr_obj: routine to XDR an object.
 *
 */
bool_t
xdr_pointer (XDR *xdrs, char **objpp, u_int obj_size, xdrproc_t xdr_obj)
{

  bool_t more_data;

  more_data = (*objpp != NULL);
  if (!xdr_bool (xdrs, &more_data))
    {
      return FALSE;
    }
  if (!more_data)
    {
      *objpp = NULL;
      return TRUE;
    }
  return xdr_reference (xdrs, objpp, obj_size, xdr_obj);
}
#ifdef EXPORT_RPC_SYMBOLS
libc_hidden_def (xdr_pointer)
#else
libc_hidden_nolink_sunrpc (xdr_pointer, GLIBC_2_0)
#endif
