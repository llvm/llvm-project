/* Conversion module for UTF-16.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1999.

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

#include <byteswap.h>
#include <dlfcn.h>
#include <gconv.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* This is the Byte Order Mark character (BOM).  */
#define BOM	0xfeff
/* And in the other byte order.  */
#define BOM_OE	0xfffe


/* Definitions used in the body of the `gconv' function.  */
#define FROM_LOOP		from_utf16_loop
#define TO_LOOP			to_utf16_loop
#define DEFINE_INIT		0
#define DEFINE_FINI		0
#define MIN_NEEDED_FROM		2
#define MAX_NEEDED_FROM		4
#define MIN_NEEDED_TO		4
#define ONE_DIRECTION		0
#define FROM_DIRECTION		(dir == from_utf16)
#define PREPARE_LOOP \
  enum direction dir = ((struct utf16_data *) step->__data)->dir;	      \
  enum variant var = ((struct utf16_data *) step->__data)->var;		      \
  if (__glibc_unlikely (data->__invocation_counter == 0))		      \
    {									      \
      if (var == UTF_16)						      \
	{								      \
	  if (FROM_DIRECTION)						      \
	    {								      \
	      /* We have to find out which byte order the file is	      \
		 encoded in.  */					      \
	      if (inptr + 2 > inend)					      \
		return (inptr == inend					      \
			? __GCONV_EMPTY_INPUT : __GCONV_INCOMPLETE_INPUT);    \
									      \
	      if (get16u (inptr) == BOM)				      \
		/* Simply ignore the BOM character.  */			      \
		*inptrp = inptr += 2;					      \
	      else if (get16u (inptr) == BOM_OE)			      \
		{							      \
		  data->__flags |= __GCONV_SWAP;			      \
		  *inptrp = inptr += 2;					      \
		}							      \
	    }								      \
	  else if (!FROM_DIRECTION && !data->__internal_use)		      \
	    {								      \
	      /* Emit the Byte Order Mark.  */				      \
	      if (__glibc_unlikely (outbuf + 2 > outend))		      \
		return __GCONV_FULL_OUTPUT;				      \
									      \
	      put16u (outbuf, BOM);					      \
	      outbuf += 2;						      \
	    }								      \
	}								      \
      else if ((var == UTF_16LE && BYTE_ORDER == BIG_ENDIAN)		      \
	       || (var == UTF_16BE && BYTE_ORDER == LITTLE_ENDIAN))	      \
	data->__flags |= __GCONV_SWAP;					      \
    }									      \
  const int swap = data->__flags & __GCONV_SWAP;
#define EXTRA_LOOP_ARGS		, swap


/* Direction of the transformation.  */
enum direction
{
  illegal_dir,
  to_utf16,
  from_utf16
};

enum variant
{
  illegal_var,
  UTF_16,
  UTF_16LE,
  UTF_16BE
};

struct utf16_data
{
  enum direction dir;
  enum variant var;
};


extern int gconv_init (struct __gconv_step *step);
int
gconv_init (struct __gconv_step *step)
{
  /* Determine which direction.  */
  struct utf16_data *new_data;
  enum direction dir = illegal_dir;
  enum variant var = illegal_var;
  int result;

  if (__strcasecmp (step->__from_name, "UTF-16//") == 0)
    {
      dir = from_utf16;
      var = UTF_16;
    }
  else if (__strcasecmp (step->__to_name, "UTF-16//") == 0)
    {
      dir = to_utf16;
      var = UTF_16;
    }
  else if (__strcasecmp (step->__from_name, "UTF-16BE//") == 0)
    {
      dir = from_utf16;
      var = UTF_16BE;
    }
  else if (__strcasecmp (step->__to_name, "UTF-16BE//") == 0)
    {
      dir = to_utf16;
      var = UTF_16BE;
    }
  else if (__strcasecmp (step->__from_name, "UTF-16LE//") == 0)
    {
      dir = from_utf16;
      var = UTF_16LE;
    }
  else if (__strcasecmp (step->__to_name, "UTF-16LE//") == 0)
    {
      dir = to_utf16;
      var = UTF_16LE;
    }

  result = __GCONV_NOCONV;
  if (__builtin_expect (dir, to_utf16) != illegal_dir)
    {
      new_data = (struct utf16_data *) malloc (sizeof (struct utf16_data));

      result = __GCONV_NOMEM;
      if (new_data != NULL)
	{
	  new_data->dir = dir;
	  new_data->var = var;
	  step->__data = new_data;

	  if (dir == from_utf16)
	    {
	      step->__min_needed_from = MIN_NEEDED_FROM;
	      step->__max_needed_from = MAX_NEEDED_FROM;
	      step->__min_needed_to = MIN_NEEDED_TO;
	      step->__max_needed_to = MIN_NEEDED_TO;
	    }
	  else
	    {
	      step->__min_needed_from = MIN_NEEDED_TO;
	      step->__max_needed_from = MIN_NEEDED_TO;
	      step->__min_needed_to = MIN_NEEDED_FROM;
	      step->__max_needed_to = MAX_NEEDED_FROM;
	    }

	  step->__stateful = 0;

	  result = __GCONV_OK;
	}
    }

  return result;
}


extern void gconv_end (struct __gconv_step *data);
void
gconv_end (struct __gconv_step *data)
{
  free (data->__data);
}


/* Convert from the internal (UCS4-like) format to UTF-16.  */
#define MIN_NEEDED_INPUT	MIN_NEEDED_TO
#define MIN_NEEDED_OUTPUT	MIN_NEEDED_FROM
#define MAX_NEEDED_OUTPUT	MAX_NEEDED_FROM
#define LOOPFCT			TO_LOOP
#define BODY \
  {									      \
    uint32_t c = get32 (inptr);						      \
									      \
    if (__glibc_unlikely (c >= 0xd800 && c < 0xe000))			      \
      {									      \
	/* Surrogate characters in UCS-4 input are not valid.		      \
	   We must catch this.  If we let surrogates pass through,	      \
	   attackers could make a security hole exploit by		      \
	   synthesizing any desired plane 1-16 character.  */		      \
	result = __GCONV_ILLEGAL_INPUT;					      \
	if (! ignore_errors_p ())					      \
	  break;							      \
	inptr += 4;							      \
	++*irreversible;						      \
	continue;							      \
      }									      \
									      \
    if (swap)								      \
      {									      \
	if (__glibc_unlikely (c >= 0x10000))				      \
	  {								      \
	    if (__glibc_unlikely (c >= 0x110000))			      \
	      {								      \
		STANDARD_TO_LOOP_ERR_HANDLER (4);			      \
	      }								      \
									      \
	    /* Generate a surrogate character.  */			      \
	    if (__glibc_unlikely (outptr + 4 > outend))			      \
	      {								      \
		/* Overflow in the output buffer.  */			      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
									      \
	    put16 (outptr, bswap_16 (0xd7c0 + (c >> 10)));		      \
	    outptr += 2;						      \
	    put16 (outptr, bswap_16 (0xdc00 + (c & 0x3ff)));		      \
	  }								      \
	else								      \
	  put16 (outptr, bswap_16 (c));					      \
      }									      \
    else								      \
      {									      \
	if (__glibc_unlikely (c >= 0x10000))				      \
	  {								      \
	    if (__glibc_unlikely (c >= 0x110000))			      \
	      {								      \
		STANDARD_TO_LOOP_ERR_HANDLER (4);			      \
	      }								      \
									      \
	    /* Generate a surrogate character.  */			      \
	    if (__glibc_unlikely (outptr + 4 > outend))			      \
	      {								      \
		/* Overflow in the output buffer.  */			      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
									      \
	    put16 (outptr, 0xd7c0 + (c >> 10));				      \
	    outptr += 2;						      \
	    put16 (outptr, 0xdc00 + (c & 0x3ff));			      \
	  }								      \
	else								      \
	  put16 (outptr, c);						      \
      }									      \
    outptr += 2;							      \
    inptr += 4;								      \
  }
#define LOOP_NEED_FLAGS
#define EXTRA_LOOP_DECLS \
	, int swap
#include <iconv/loop.c>


/* Convert from UTF-16 to the internal (UCS4-like) format.  */
#define MIN_NEEDED_INPUT	MIN_NEEDED_FROM
#define MAX_NEEDED_INPUT	MAX_NEEDED_FROM
#define MIN_NEEDED_OUTPUT	MIN_NEEDED_TO
#define LOOPFCT			FROM_LOOP
#define BODY \
  {									      \
    uint16_t u1 = get16 (inptr);					      \
									      \
    if (swap)								      \
      {									      \
	u1 = bswap_16 (u1);						      \
									      \
	if (__builtin_expect (u1 < 0xd800, 1) || u1 > 0xdfff)		      \
	  {								      \
	    /* No surrogate.  */					      \
	    put32 (outptr, u1);						      \
	    inptr += 2;							      \
	  }								      \
	else								      \
	  {								      \
	    uint16_t u2;						      \
									      \
	    if (__glibc_unlikely (u1 >= 0xdc00))			      \
	      {								      \
		/* This is no valid first word for a surrogate.  */	      \
		STANDARD_FROM_LOOP_ERR_HANDLER (2);			      \
	      }								      \
									      \
	    /* It's a surrogate character.  At least the first word says      \
	       it is.  */						      \
	    if (__glibc_unlikely (inptr + 4 > inend))			      \
	      {								      \
		/* We don't have enough input for another complete input      \
		   character.  */					      \
		result = __GCONV_INCOMPLETE_INPUT;			      \
		break;							      \
	      }								      \
									      \
	    inptr += 2;							      \
	    u2 = bswap_16 (get16 (inptr));				      \
	    if (__builtin_expect (u2 < 0xdc00, 0)			      \
		|| __builtin_expect (u2 > 0xdfff, 0))			      \
	      {								      \
		/* This is no valid second word for a surrogate.  */	      \
		inptr -= 2;						      \
		STANDARD_FROM_LOOP_ERR_HANDLER (2);			      \
	      }								      \
									      \
	    put32 (outptr, ((u1 - 0xd7c0) << 10) + (u2 - 0xdc00));	      \
	    inptr += 2;							      \
	  }								      \
      }									      \
    else								      \
      {									      \
	if (__builtin_expect (u1 < 0xd800, 1) || u1 > 0xdfff)		      \
	  {								      \
	    /* No surrogate.  */					      \
	    put32 (outptr, u1);						      \
	    inptr += 2;							      \
	  }								      \
	else								      \
	  {								      \
	    if (__glibc_unlikely (u1 >= 0xdc00))			      \
	      {								      \
		/* This is no valid first word for a surrogate.  */	      \
		STANDARD_FROM_LOOP_ERR_HANDLER (2);			      \
	      }								      \
									      \
	    /* It's a surrogate character.  At least the first word says      \
	       it is.  */						      \
	    if (__glibc_unlikely (inptr + 4 > inend))			      \
	      {								      \
		/* We don't have enough input for another complete input      \
		   character.  */					      \
		result = __GCONV_INCOMPLETE_INPUT;			      \
		break;							      \
	      }								      \
									      \
	    inptr += 2;							      \
	    uint16_t u2 = get16 (inptr);				      \
	    if (__builtin_expect (u2 < 0xdc00, 0)			      \
		|| __builtin_expect (u2 > 0xdfff, 0))			      \
	      {								      \
		/* This is no valid second word for a surrogate.  */	      \
		inptr -= 2;						      \
		STANDARD_FROM_LOOP_ERR_HANDLER (2);			      \
	      }								      \
									      \
	    put32 (outptr, ((u1 - 0xd7c0) << 10) + (u2 - 0xdc00));	      \
	    inptr += 2;							      \
	  }								      \
      }									      \
    outptr += 4;							      \
  }
#define LOOP_NEED_FLAGS
#define EXTRA_LOOP_DECLS \
	, int swap
#include <iconv/loop.c>


/* Now define the toplevel functions.  */
#include <iconv/skeleton.c>
