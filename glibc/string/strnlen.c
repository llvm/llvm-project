/* Find the length of STRING, but scan at most MAXLEN characters.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
   Contributed by Jakub Jelinek <jakub@redhat.com>.

   Based on strlen written by Torbjorn Granlund (tege@sics.se),
   with help from Dan Sahlin (dan@sics.se);
   commentary by Jim Blandy (jimb@ai.mit.edu).

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#include <string.h>
#include <stdlib.h>

/* Find the length of S, but scan at most MAXLEN characters.  If no
   '\0' terminator is found in that many characters, return MAXLEN.  */

#ifdef STRNLEN
# define __strnlen STRNLEN
#endif

size_t
__strnlen (const char *str, size_t maxlen)
{
  const char *char_ptr, *end_ptr = str + maxlen;
  const unsigned long int *longword_ptr;
  unsigned long int longword, himagic, lomagic;

  if (maxlen == 0)
    return 0;

  if (__glibc_unlikely (end_ptr < str))
    end_ptr = (const char *) ~0UL;

  /* Handle the first few characters by reading one character at a time.
     Do this until CHAR_PTR is aligned on a longword boundary.  */
  for (char_ptr = str; ((unsigned long int) char_ptr
			& (sizeof (longword) - 1)) != 0;
       ++char_ptr)
    if (*char_ptr == '\0')
      {
	if (char_ptr > end_ptr)
	  char_ptr = end_ptr;
	return char_ptr - str;
      }

  /* All these elucidatory comments refer to 4-byte longwords,
     but the theory applies equally well to 8-byte longwords.  */

  longword_ptr = (unsigned long int *) char_ptr;

  /* Bits 31, 24, 16, and 8 of this number are zero.  Call these bits
     the "holes."  Note that there is a hole just to the left of
     each byte, with an extra at the end:

     bits:  01111110 11111110 11111110 11111111
     bytes: AAAAAAAA BBBBBBBB CCCCCCCC DDDDDDDD

     The 1-bits make sure that carries propagate to the next 0-bit.
     The 0-bits provide holes for carries to fall into.  */
  himagic = 0x80808080L;
  lomagic = 0x01010101L;
  if (sizeof (longword) > 4)
    {
      /* 64-bit version of the magic.  */
      /* Do the shift in two steps to avoid a warning if long has 32 bits.  */
      himagic = ((himagic << 16) << 16) | himagic;
      lomagic = ((lomagic << 16) << 16) | lomagic;
    }
  if (sizeof (longword) > 8)
    abort ();

  /* Instead of the traditional loop which tests each character,
     we will test a longword at a time.  The tricky part is testing
     if *any of the four* bytes in the longword in question are zero.  */
  while (longword_ptr < (unsigned long int *) end_ptr)
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
	 properly.  */

      longword = *longword_ptr++;

      if ((longword - lomagic) & himagic)
	{
	  /* Which of the bytes was the zero?  If none of them were, it was
	     a misfire; continue the search.  */

	  const char *cp = (const char *) (longword_ptr - 1);

	  char_ptr = cp;
	  if (cp[0] == 0)
	    break;
	  char_ptr = cp + 1;
	  if (cp[1] == 0)
	    break;
	  char_ptr = cp + 2;
	  if (cp[2] == 0)
	    break;
	  char_ptr = cp + 3;
	  if (cp[3] == 0)
	    break;
	  if (sizeof (longword) > 4)
	    {
	      char_ptr = cp + 4;
	      if (cp[4] == 0)
		break;
	      char_ptr = cp + 5;
	      if (cp[5] == 0)
		break;
	      char_ptr = cp + 6;
	      if (cp[6] == 0)
		break;
	      char_ptr = cp + 7;
	      if (cp[7] == 0)
		break;
	    }
	}
      char_ptr = end_ptr;
    }

  if (char_ptr > end_ptr)
    char_ptr = end_ptr;
  return char_ptr - str;
}
#ifndef STRNLEN
libc_hidden_def (__strnlen)
weak_alias (__strnlen, strnlen)
#endif
libc_hidden_def (strnlen)
