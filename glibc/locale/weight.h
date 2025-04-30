/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written by Ulrich Drepper, <drepper@cygnus.com>.

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

#ifndef _WEIGHT_H_
#define _WEIGHT_H_	1

#include <libc-diag.h>

/* Find index of weight.  */
static inline int32_t __attribute__ ((always_inline))
findidx (const int32_t *table,
	 const int32_t *indirect,
	 const unsigned char *extra,
	 const unsigned char **cpp, size_t len)
{
  int_fast32_t i = table[*(*cpp)++];
  const unsigned char *cp;
  const unsigned char *usrc;

  if (i >= 0)
    /* This is an index into the weight table.  Cool.  */
    return i;

  /* Oh well, more than one sequence starting with this byte.
     Search for the correct one.  */
  cp = &extra[-i];
  usrc = *cpp;
  --len;
  while (1)
    {
      size_t nhere;

      /* The first thing is the index.  */
      i = *((const int32_t *) cp);
      cp += sizeof (int32_t);

      /* Next is the length of the byte sequence.  These are always
	 short byte sequences so there is no reason to call any
	 function (even if they are inlined).  */
      nhere = *cp++;

      if (i >= 0)
	{
	  /* It is a single character.  If it matches we found our
	     index.  Note that at the end of each list there is an
	     entry of length zero which represents the single byte
	     sequence.  The first (and here only) byte was tested
	     already.  */
	  size_t cnt;

	  /* With GCC 5.3 when compiling with -Os the compiler warns
	     that seq2.back_us, which becomes usrc, might be used
	     uninitialized.  This can't be true because we pass a length
	     of -1 for len at the same time which means that this loop
	     never executes.  */
	  DIAG_PUSH_NEEDS_COMMENT;
	  DIAG_IGNORE_Os_NEEDS_COMMENT (5, "-Wmaybe-uninitialized");
	  for (cnt = 0; cnt < nhere && cnt < len; ++cnt)
	    if (cp[cnt] != usrc[cnt])
	      break;
	  DIAG_POP_NEEDS_COMMENT;

	  if (cnt == nhere)
	    {
	      /* Found it.  */
	      *cpp += nhere;
	      return i;
	    }

	  /* Up to the next entry.  */
	  cp += nhere;
	  if (!LOCFILE_ALIGNED_P (1 + nhere))
	    cp += LOCFILE_ALIGN - (1 + nhere) % LOCFILE_ALIGN;
	}
      else
	{
	  /* This is a range of characters.  First decide whether the
	     current byte sequence lies in the range.  */
	  size_t cnt;
	  size_t offset = 0;

	  for (cnt = 0; cnt < nhere && cnt < len; ++cnt)
	    if (cp[cnt] != usrc[cnt])
	      break;

	  if (cnt != nhere)
	    {
	      if (cnt == len || cp[cnt] > usrc[cnt])
		{
		  /* Cannot be in this range.  */
		  cp += 2 * nhere;
		  if (!LOCFILE_ALIGNED_P (1 + 2 * nhere))
		    cp += (LOCFILE_ALIGN
			   - (1 + 2 * nhere) % LOCFILE_ALIGN);
		  continue;
		}

	      /* Test against the end of the range.  */
	      for (cnt = 0; cnt < nhere; ++cnt)
		if (cp[nhere + cnt] != usrc[cnt])
		  break;

	      if (cnt != nhere && cp[nhere + cnt] < usrc[cnt])
		{
		  /* Cannot be in this range.  */
		  cp += 2 * nhere;
		  if (!LOCFILE_ALIGNED_P (1 + 2 * nhere))
		    cp += (LOCFILE_ALIGN
			   - (1 + 2 * nhere) % LOCFILE_ALIGN);
		  continue;
		}

	      /* This range matches the next characters.  Now find
		 the offset in the indirect table.  */
	      for (cnt = 0; cp[cnt] == usrc[cnt]; ++cnt);

	      do
		{
		  offset <<= 8;
		  /* With GCC 7 when compiling with -Os the compiler
		     warns that seq1.back_us and seq2.back_us, which
		     become usrc, might be used uninitialized.  This
		     is impossible for the same reason as described
		     above.  */
		  DIAG_PUSH_NEEDS_COMMENT;
		  DIAG_IGNORE_Os_NEEDS_COMMENT (7, "-Wmaybe-uninitialized");
		  offset += usrc[cnt] - cp[cnt];
		  DIAG_POP_NEEDS_COMMENT;
		}
	      while (++cnt < nhere);
	    }

	  *cpp += nhere;
	  return indirect[-i + offset];
	}
    }

  /* NOTREACHED */
  return 0x43219876;
}

#endif	/* weight.h */
