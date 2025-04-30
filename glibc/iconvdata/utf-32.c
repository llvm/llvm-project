/* Conversion module for UTF-32.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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
#define BOM	0x0000feffu
/* And in the other byte order.  */
#define BOM_OE	0xfffe0000u


/* Definitions used in the body of the `gconv' function.  */
#define FROM_LOOP		from_utf32_loop
#define TO_LOOP			to_utf32_loop
#define DEFINE_INIT		0
#define DEFINE_FINI		0
#define MIN_NEEDED_FROM		4
#define MIN_NEEDED_TO		4
#define ONE_DIRECTION		0
#define FROM_DIRECTION		(dir == from_utf32)
#define PREPARE_LOOP \
  enum direction dir = ((struct utf32_data *) step->__data)->dir;	      \
  enum variant var = ((struct utf32_data *) step->__data)->var;		      \
  int swap;								      \
  if (FROM_DIRECTION && var == UTF_32)					      \
    {									      \
      if (__glibc_unlikely (data->__invocation_counter == 0))		      \
	{								      \
	  /* We have to find out which byte order the file is encoded in.  */ \
	  if (inptr + 4 > inend)					      \
	    return (inptr == inend					      \
		    ? __GCONV_EMPTY_INPUT : __GCONV_INCOMPLETE_INPUT);	      \
									      \
	  if (get32u (inptr) == BOM)					      \
	    /* Simply ignore the BOM character.  */			      \
	    *inptrp = inptr += 4;					      \
	  else if (get32u (inptr) == BOM_OE)				      \
	    {								      \
	      data->__flags |= __GCONV_SWAP;				      \
	      *inptrp = inptr += 4;					      \
	    }								      \
	}								      \
    }									      \
  else if (!FROM_DIRECTION && var == UTF_32 && !data->__internal_use	      \
	   && data->__invocation_counter == 0)				      \
    {									      \
      /* Emit the Byte Order Mark.  */					      \
      if (__glibc_unlikely (outbuf + 4 > outend))			      \
	return __GCONV_FULL_OUTPUT;					      \
									      \
      put32u (outbuf, BOM);						      \
      outbuf += 4;							      \
    }									      \
  else if (__builtin_expect (data->__invocation_counter == 0, 0)	      \
	   && ((var == UTF_32LE && BYTE_ORDER == BIG_ENDIAN)		      \
	       || (var == UTF_32BE && BYTE_ORDER == LITTLE_ENDIAN)))	      \
    data->__flags |= __GCONV_SWAP;					      \
  swap = data->__flags & __GCONV_SWAP;
#define EXTRA_LOOP_ARGS		, var, swap


/* Direction of the transformation.  */
enum direction
{
  illegal_dir,
  to_utf32,
  from_utf32
};

enum variant
{
  illegal_var,
  UTF_32,
  UTF_32LE,
  UTF_32BE
};

struct utf32_data
{
  enum direction dir;
  enum variant var;
};


extern int gconv_init (struct __gconv_step *step);
int
gconv_init (struct __gconv_step *step)
{
  /* Determine which direction.  */
  struct utf32_data *new_data;
  enum direction dir = illegal_dir;
  enum variant var = illegal_var;
  int result;

  if (__strcasecmp (step->__from_name, "UTF-32//") == 0)
    {
      dir = from_utf32;
      var = UTF_32;
    }
  else if (__strcasecmp (step->__to_name, "UTF-32//") == 0)
    {
      dir = to_utf32;
      var = UTF_32;
    }
  else if (__strcasecmp (step->__from_name, "UTF-32BE//") == 0)
    {
      dir = from_utf32;
      var = UTF_32BE;
    }
  else if (__strcasecmp (step->__to_name, "UTF-32BE//") == 0)
    {
      dir = to_utf32;
      var = UTF_32BE;
    }
  else if (__strcasecmp (step->__from_name, "UTF-32LE//") == 0)
    {
      dir = from_utf32;
      var = UTF_32LE;
    }
  else if (__strcasecmp (step->__to_name, "UTF-32LE//") == 0)
    {
      dir = to_utf32;
      var = UTF_32LE;
    }

  result = __GCONV_NOCONV;
  if (__builtin_expect (dir, to_utf32) != illegal_dir)
    {
      new_data = (struct utf32_data *) malloc (sizeof (struct utf32_data));

      result = __GCONV_NOMEM;
      if (new_data != NULL)
	{
	  new_data->dir = dir;
	  new_data->var = var;
	  step->__data = new_data;

	  if (dir == from_utf32)
	    {
	      step->__min_needed_from = MIN_NEEDED_FROM;
	      step->__max_needed_from = MIN_NEEDED_FROM;
	      step->__min_needed_to = MIN_NEEDED_TO;
	      step->__max_needed_to = MIN_NEEDED_TO;
	    }
	  else
	    {
	      step->__min_needed_from = MIN_NEEDED_TO;
	      step->__max_needed_from = MIN_NEEDED_TO;
	      step->__min_needed_to = MIN_NEEDED_FROM;
	      step->__max_needed_to = MIN_NEEDED_FROM;
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


/* Convert from the internal (UCS4-like) format to UTF-32.  */
#define MIN_NEEDED_INPUT	MIN_NEEDED_TO
#define MIN_NEEDED_OUTPUT	MIN_NEEDED_FROM
#define LOOPFCT			TO_LOOP
#define BODY \
  {									      \
    uint32_t c = get32 (inptr);						      \
									      \
    if (__glibc_unlikely (c >= 0x110000))				      \
      {									      \
	STANDARD_TO_LOOP_ERR_HANDLER (4);				      \
      }									      \
    else if (__glibc_unlikely (c >= 0xd800 && c < 0xe000))		      \
      {									      \
	/* Surrogate characters in UCS-4 input are not valid.		      \
	   We must catch this.  If we let surrogates pass through,	      \
	   attackers could make a security hole exploit by		      \
	   generating "irregular UTF-32" sequences.  */			      \
	result = __GCONV_ILLEGAL_INPUT;					      \
	if (! ignore_errors_p ())					      \
	  break;							      \
	inptr += 4;							      \
	++*irreversible;						      \
	continue;							      \
      }									      \
									      \
    if (swap)								      \
      c = bswap_32 (c);							      \
    put32 (outptr, c);							      \
									      \
    outptr += 4;							      \
    inptr += 4;								      \
  }
#define LOOP_NEED_FLAGS
#define EXTRA_LOOP_DECLS \
	, enum variant var, int swap
#include <iconv/loop.c>


/* Convert from UTF-32 to the internal (UCS4-like) format.  */
#define MIN_NEEDED_INPUT	MIN_NEEDED_FROM
#define MIN_NEEDED_OUTPUT	MIN_NEEDED_TO
#define LOOPFCT			FROM_LOOP
#define BODY \
  {									      \
    uint32_t u1 = get32 (inptr);					      \
									      \
    if (swap)								      \
      u1 = bswap_32 (u1);						      \
									      \
    if (__glibc_unlikely (u1 >= 0x110000 || (u1 >= 0xd800 && u1 < 0xe000)))   \
      {									      \
	/* This is illegal.  */						      \
	STANDARD_FROM_LOOP_ERR_HANDLER (4);				      \
      }									      \
									      \
    put32 (outptr, u1);							      \
    inptr += 4;								      \
    outptr += 4;							      \
  }
#define LOOP_NEED_FLAGS
#define EXTRA_LOOP_DECLS \
	, enum variant var, int swap
#include <iconv/loop.c>


/* Now define the toplevel functions.  */
#include <iconv/skeleton.c>
