/* memrchr -- find the last occurrence of a byte in a memory block
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Based on strlen implementation by Torbjorn Granlund (tege@sics.se),
   with help from Dan Sahlin (dan@sics.se) and
   commentary by Jim Blandy (jimb@ai.mit.edu);
   adaptation to memchr suggested by Dick Karpinski (dick@cca.ucsf.edu),
   and implemented by Roland McGrath (roland@ai.mit.edu).

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

#include <stdlib.h>

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#if defined _LIBC
# include <string.h>
# include <memcopy.h>
#endif

#if defined HAVE_LIMITS_H || defined _LIBC
# include <limits.h>
#endif

#define LONG_MAX_32_BITS 2147483647

#ifndef LONG_MAX
# define LONG_MAX LONG_MAX_32_BITS
#endif

#include <sys/types.h>

#undef __memrchr
#undef memrchr

#ifndef weak_alias
# define __memrchr memrchr
#endif

/* Search no more than N bytes of S for C.  */
void *
#ifndef MEMRCHR
__memrchr
#else
MEMRCHR
#endif
     (const void *s, int c_in, size_t n)
{
  const unsigned char *char_ptr;
  const unsigned long int *longword_ptr;
  unsigned long int longword, magic_bits, charmask;
  unsigned char c;

  c = (unsigned char) c_in;

  /* Handle the last few characters by reading one character at a time.
     Do this until CHAR_PTR is aligned on a longword boundary.  */
  for (char_ptr = (const unsigned char *) s + n;
       n > 0 && ((unsigned long int) char_ptr
		 & (sizeof (longword) - 1)) != 0;
       --n)
    if (*--char_ptr == c)
      return (void *) char_ptr;

  /* All these elucidatory comments refer to 4-byte longwords,
     but the theory applies equally well to 8-byte longwords.  */

  longword_ptr = (const unsigned long int *) char_ptr;

  /* Bits 31, 24, 16, and 8 of this number are zero.  Call these bits
     the "holes."  Note that there is a hole just to the left of
     each byte, with an extra at the end:

     bits:  01111110 11111110 11111110 11111111
     bytes: AAAAAAAA BBBBBBBB CCCCCCCC DDDDDDDD

     The 1-bits make sure that carries propagate to the next 0-bit.
     The 0-bits provide holes for carries to fall into.  */
  magic_bits = -1;
  magic_bits = magic_bits / 0xff * 0xfe << 1 >> 1 | 1;

  /* Set up a longword, each of whose bytes is C.  */
  charmask = c | (c << 8);
  charmask |= charmask << 16;
#if LONG_MAX > LONG_MAX_32_BITS
  charmask |= charmask << 32;
#endif

  /* Instead of the traditional loop which tests each character,
     we will test a longword at a time.  The tricky part is testing
     if *any of the four* bytes in the longword in question are zero.  */
  while (n >= sizeof (longword))
    {
      /* We tentatively exit the loop if adding MAGIC_BITS to
	 LONGWORD fails to change any of the hole bits of LONGWORD.

	 1) Is this safe?  Will it catch all the zero bytes?
	 Suppose there is a byte with all zeros.  Any carry bits
	 propagating from its left will fall into the hole at its
	 least significant bit and stop.  Since there will be no
	 carry from its most significant bit, the LSB of the
	 byte to the left will be unchanged, and the zero will be
	 detected.

	 2) Is this worthwhile?  Will it ignore everything except
	 zero bytes?  Suppose every byte of LONGWORD has a bit set
	 somewhere.  There will be a carry into bit 8.  If bit 8
	 is set, this will carry into bit 16.  If bit 8 is clear,
	 one of bits 9-15 must be set, so there will be a carry
	 into bit 16.  Similarly, there will be a carry into bit
	 24.  If one of bits 24-30 is set, there will be a carry
	 into bit 31, so all of the hole bits will be changed.

	 The one misfire occurs when bits 24-30 are clear and bit
	 31 is set; in this case, the hole at bit 31 is not
	 changed.  If we had access to the processor carry flag,
	 we could close this loophole by putting the fourth hole
	 at bit 32!

	 So it ignores everything except 128's, when they're aligned
	 properly.

	 3) But wait!  Aren't we looking for C, not zero?
	 Good point.  So what we do is XOR LONGWORD with a longword,
	 each of whose bytes is C.  This turns each byte that is C
	 into a zero.  */

      longword = *--longword_ptr ^ charmask;

      /* Add MAGIC_BITS to LONGWORD.  */
      if ((((longword + magic_bits)

	    /* Set those bits that were unchanged by the addition.  */
	    ^ ~longword)

	   /* Look at only the hole bits.  If any of the hole bits
	      are unchanged, most likely one of the bytes was a
	      zero.  */
	   & ~magic_bits) != 0)
	{
	  /* Which of the bytes was C?  If none of them were, it was
	     a misfire; continue the search.  */

	  const unsigned char *cp = (const unsigned char *) longword_ptr;

#if LONG_MAX > 2147483647
	  if (cp[7] == c)
	    return (void *) &cp[7];
	  if (cp[6] == c)
	    return (void *) &cp[6];
	  if (cp[5] == c)
	    return (void *) &cp[5];
	  if (cp[4] == c)
	    return (void *) &cp[4];
#endif
	  if (cp[3] == c)
	    return (void *) &cp[3];
	  if (cp[2] == c)
	    return (void *) &cp[2];
	  if (cp[1] == c)
	    return (void *) &cp[1];
	  if (cp[0] == c)
	    return (void *) cp;
	}

      n -= sizeof (longword);
    }

  char_ptr = (const unsigned char *) longword_ptr;

  while (n-- > 0)
    {
      if (*--char_ptr == c)
	return (void *) char_ptr;
    }

  return 0;
}
#ifndef MEMRCHR
# ifdef weak_alias
weak_alias (__memrchr, memrchr)
# endif
#endif
