/* Copyright (C) 2006-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2006.

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

#include <string.h>
#include <netinet/in.h>
#include <netinet/ip6.h>


/* RFC 3542, 10.1

   This function returns the number of bytes needed for the empty
   extension header i.e., without any options.  If EXTBUF is not NULL it
   also initializes the extension header to have the correct length
   field.  In that case if the EXTLEN value is not a positive (i.e.,
   non-zero) multiple of 8 the function fails and returns -1.  */
int
inet6_opt_init (void *extbuf, socklen_t extlen)
{
  if (extbuf != NULL)
    {
      if (extlen <= 0 || (extlen % 8) != 0 || extlen > 256 * 8)
	return -1;

      /* Fill in the length in units of 8 octets.  */
      struct ip6_hbh *extp = (struct ip6_hbh *) extbuf;

      /* RFC 2460 requires that the header extension length is the
	 length of the option header in 8-byte units, not including
	 the first 8 bytes.  Hence we have to subtract one.  */
      extp->ip6h_len = extlen / 8 - 1;
    }

  return sizeof (struct ip6_hbh);
}


static void
add_padding (uint8_t *extbuf, int offset, int npad)
{
  if (npad == 1)
    extbuf[offset] = IP6OPT_PAD1;
  else if (npad > 0)
    {
      struct ip6_opt *pad_opt = (struct ip6_opt *) (extbuf + offset);

      pad_opt->ip6o_type = IP6OPT_PADN;
      pad_opt->ip6o_len = npad - sizeof (struct ip6_opt);
      /* Clear the memory used by the padding.  */
      memset (pad_opt + 1, '\0', pad_opt->ip6o_len);
    }
}



/* RFC 3542, 10.2

   This function returns the updated total length taking into account
   adding an option with length 'len' and alignment 'align'.  If
   EXTBUF is not NULL then, in addition to returning the length, the
   function inserts any needed pad option, initializes the option
   (setting the type and length fields) and returns a pointer to the
   location for the option content in databufp.  If the option does
   not fit in the extension header buffer the function returns -1.  */
int
inet6_opt_append (void *extbuf, socklen_t extlen, int offset, uint8_t type,
		  socklen_t len, uint8_t align, void **databufp)
{
  /* Check minimum offset.  */
  if (offset < sizeof (struct ip6_hbh))
    return -1;

  /* One cannot add padding options.  */
  if (type == IP6OPT_PAD1 || type == IP6OPT_PADN)
    return -1;

  /* The option length must fit in one octet.  */
  if (len > 255)
    return -1;

  /* The alignment can only by 1, 2, 4, or 8 and must not exceed the
     option length.  */
  if (align == 0 || align > 8 || (align & (align - 1)) != 0 || align > len)
    return -1;

  /* Determine the needed padding for alignment.  Following the
     current content of the buffer we have the is the IPv6 option type
     and length, followed immediately by the data.  The data has the
     alignment constraints.  Therefore padding must be inserted in the
     form of padding options before the new option. */
  int data_offset = offset + sizeof (struct ip6_opt);
  int npad = (align - data_offset % align) & (align - 1);

  if (extbuf != NULL)
    {
      /* Now we can check whether the buffer is large enough.  */
      if (data_offset + npad + len > extlen)
	return -1;

      add_padding (extbuf, offset, npad);

      offset += npad;

      /* Now prepare the option itself.  */
      struct ip6_opt *opt = (struct ip6_opt *) ((uint8_t *) extbuf + offset);

      opt->ip6o_type = type;
      opt->ip6o_len = len;

      *databufp = opt + 1;
    }
  else
    offset += npad;

  return offset + sizeof (struct ip6_opt) + len;
}


/* RFC 3542, 10.3

   This function returns the updated total length taking into account
   the final padding of the extension header to make it a multiple of
   8 bytes.  If EXTBUF is not NULL the function also initializes the
   option by inserting a Pad1 or PadN option of the proper length.  */
int
inet6_opt_finish (void *extbuf, socklen_t extlen, int offset)
{
  /* Check minimum offset.  */
  if (offset < sizeof (struct ip6_hbh))
    return -1;

  /* Required padding at the end.  */
  int npad = (8 - (offset & 7)) & 7;

  if (extbuf != NULL)
    {
      /* Make sure the buffer is large enough.  */
      if (offset + npad > extlen)
	return -1;

      add_padding (extbuf, offset, npad);
    }

  return offset + npad;
}


/* RFC 3542, 10.4

   This function inserts data items of various sizes in the data
   portion of the option.  VAL should point to the data to be
   inserted.  OFFSET specifies where in the data portion of the option
   the value should be inserted; the first byte after the option type
   and length is accessed by specifying an offset of zero.  */
int
inet6_opt_set_val (void *databuf, int offset, void *val, socklen_t vallen)
{
  memcpy ((uint8_t *) databuf + offset, val, vallen);

  return offset + vallen;
}


/* RFC 3542, 10.5

   This function parses received option extension headers returning
   the next option.  EXTBUF and EXTLEN specifies the extension header.
   OFFSET should either be zero (for the first option) or the length
   returned by a previous call to 'inet6_opt_next' or
   'inet6_opt_find'.  It specifies the position where to continue
   scanning the extension buffer.  */
int
inet6_opt_next (void *extbuf, socklen_t extlen, int offset, uint8_t *typep,
		socklen_t *lenp, void **databufp)
{
  if (offset == 0)
    offset = sizeof (struct ip6_hbh);
  else if (offset < sizeof (struct ip6_hbh))
    return -1;

  while (offset < extlen)
    {
      struct ip6_opt *opt = (struct ip6_opt *) ((uint8_t *) extbuf + offset);

      if (opt->ip6o_type == IP6OPT_PAD1)
	/* Single byte padding.  */
	++offset;
      else if (opt->ip6o_type == IP6OPT_PADN)
	offset += sizeof (struct ip6_opt) + opt->ip6o_len;
      else
	{
	  /* Check whether the option is valid.  */
	  offset += sizeof (struct ip6_opt) + opt->ip6o_len;
	  if (offset > extlen)
	    return -1;

	  *typep = opt->ip6o_type;
	  *lenp = opt->ip6o_len;
	  *databufp = opt + 1;
	  return offset;
	}
    }

  return -1;
}


/* RFC 3542, 10.6

   This function is similar to the previously described
   'inet6_opt_next' function, except this function lets the caller
   specify the option type to be searched for, instead of always
   returning the next option in the extension header.  */
int
inet6_opt_find (void *extbuf, socklen_t extlen, int offset, uint8_t type,
		socklen_t *lenp, void **databufp)
{
  if (offset == 0)
    offset = sizeof (struct ip6_hbh);
  else if (offset < sizeof (struct ip6_hbh))
    return -1;

  while (offset < extlen)
    {
      struct ip6_opt *opt = (struct ip6_opt *) ((uint8_t *) extbuf + offset);

      if (opt->ip6o_type == IP6OPT_PAD1)
	{
	  /* Single byte padding.  */
	  ++offset;
	  if (type == IP6OPT_PAD1)
	    {
	      *lenp = 0;
	      *databufp = (uint8_t *) extbuf + offset;
	      return offset;
	    }
	}
      else if (opt->ip6o_type != type)
	offset += sizeof (struct ip6_opt) + opt->ip6o_len;
      else
	{
	  /* Check whether the option is valid.  */
	  offset += sizeof (struct ip6_opt) + opt->ip6o_len;
	  if (offset > extlen)
	    return -1;

	  *lenp = opt->ip6o_len;
	  *databufp = opt + 1;
	  return offset;
	}
    }

  return -1;
}


/* RFC 3542, 10.7

   This function extracts data items of various sizes in the data
   portion of the option.  */
int
inet6_opt_get_val (void *databuf, int offset, void *val, socklen_t vallen)
{
  memcpy (val, (uint8_t *) databuf + offset, vallen);

  return offset + vallen;
}
