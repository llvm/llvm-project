/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2003.

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

#include <assert.h>
#include <string.h>
#include <netinet/in.h>
#include <netinet/ip6.h>
#include <sys/param.h>


static void
add_pad (struct cmsghdr *cmsg, int len)
{
  unsigned char *p = CMSG_DATA (cmsg) + cmsg->cmsg_len - CMSG_LEN (0);

  if (len == 1)
    /* Special handling for 1, a one-byte solution.  */
    *p++ = IP6OPT_PAD1;
  else if (len != 0)
    {
      /* Multibyte padding.  */
      *p++ = IP6OPT_PADN;
      *p++ = len - 2;	/* Discount the two header bytes.  */
      /* The rest is filled with zero.  */
      memset (p, '\0', len - 2);
      p += len - 2;
    }

  /* Account for the bytes.  */
  cmsg->cmsg_len += len;
}


static int
get_opt_end (const uint8_t **result, const uint8_t *startp,
	     const uint8_t *endp)
{
  if (startp >= endp)
    /* Out of bounds.  */
    return -1;

  if (*startp == IP6OPT_PAD1)
    {
      /* Just this one byte.  */
      *result = startp + 1;
      return 0;
    }

  /* Now we know there must be at least two bytes.  */
  if (startp + 2 > endp
      /* Now we can get the length byte.  */
      || startp + startp[1] + 2 > endp)
    return -1;

  *result = startp + startp[1] + 2;

  return 0;
}


static uint8_t *option_alloc (struct cmsghdr *cmsg, int datalen, int multx,
			      int plusy);


/* RFC 2292, 6.3.1

   This function returns the number of bytes required to hold an option
   when it is stored as ancillary data, including the cmsghdr structure
   at the beginning, and any padding at the end (to make its size a
   multiple of 8 bytes).  The argument is the size of the structure
   defining the option, which must include any pad bytes at the
   beginning (the value y in the alignment term "xn + y"), the type
   byte, the length byte, and the option data.  */
int
inet6_option_space (int nbytes)
{
  /* Add room for the extension header.  */
  nbytes += sizeof (struct ip6_ext);

  return CMSG_SPACE (roundup (nbytes, 8));
}
link_warning (inet6_option_space,
	      "inet6_option_space is obsolete, use the RFC 3542 interfaces")


/* RFC 2292, 6.3.2

   This function is called once per ancillary data object that will
   contain either Hop-by-Hop or Destination options.  It returns 0 on
   success or -1 on an error.  */
int
inet6_option_init (void *bp, struct cmsghdr **cmsgp, int type)
{
  /* Only Hop-by-Hop or Destination options allowed.  */
  if (type != IPV6_HOPOPTS && type != IPV6_DSTOPTS)
    return -1;

  /* BP is a pointer to the previously allocated space.  */
  struct cmsghdr *newp = (struct cmsghdr *) bp;

  /* Initialize the message header.

     Length: No data yet, only the cmsghdr struct.  */
  newp->cmsg_len = CMSG_LEN (0);
  /* Originating protocol: obviously IPv6.  */
  newp->cmsg_level = IPPROTO_IPV6;
  /* Message type.  */
  newp->cmsg_type = type;

  /* Pass up the result.  */
  *cmsgp = newp;

  return 0;
}
link_warning (inet6_option_init,
	      "inet6_option_init is obsolete, use the RFC 3542 interfaces")


/* RFC 2292, 6.3.3

   This function appends a Hop-by-Hop option or a Destination option
   into an ancillary data object that has been initialized by
   inet6_option_init().  This function returns 0 if it succeeds or -1 on
   an error.  */
int
inet6_option_append (struct cmsghdr *cmsg, const uint8_t *typep, int multx,
		     int plusy)
{
  /* typep is a pointer to the 8-bit option type.  It is assumed that this
     field is immediately followed by the 8-bit option data length field,
     which is then followed immediately by the option data.

     The option types IP6OPT_PAD1 and IP6OPT_PADN also must be handled.  */
  int len = typep[0] == IP6OPT_PAD1 ? 1 : typep[1] + 2;

  /* Get the pointer to the space in the message.  */
  uint8_t *ptr = option_alloc (cmsg, len, multx, plusy);
  if (ptr == NULL)
    /* Some problem with the parameters.  */
    return -1;

  /* Copy the content.  */
  memcpy (ptr, typep, len);

  return 0;
}
link_warning (inet6_option_append,
	      "inet6_option_append is obsolete, use the RFC 3542 interfaces")


/* RFC 2292, 6.3.4

   This function appends a Hop-by-Hop option or a Destination option
   into an ancillary data object that has been initialized by
   inet6_option_init().  This function returns a pointer to the 8-bit
   option type field that starts the option on success, or NULL on an
   error.  */
static uint8_t *
option_alloc (struct cmsghdr *cmsg, int datalen, int multx, int plusy)
{
  /* The RFC limits the value of the alignment values.  */
  if ((multx != 1 && multx != 2 && multx != 4 && multx != 8)
      || ! (plusy >= 0 && plusy <= 7))
    return NULL;

  /* Current data size.  */
  int dsize = cmsg->cmsg_len - CMSG_LEN (0);

  /* The first two bytes of the option are for the extended header.  */
  if (__glibc_unlikely (dsize == 0))
    {
      cmsg->cmsg_len += sizeof (struct ip6_ext);
      dsize = sizeof (struct ip6_ext);
    }

  /* First add padding.  */
  add_pad (cmsg, ((multx - (dsize & (multx - 1))) & (multx - 1)) + plusy);

  /* Return the pointer to the start of the option space.  */
  uint8_t *result = CMSG_DATA (cmsg) + cmsg->cmsg_len - CMSG_LEN (0);
  cmsg->cmsg_len += datalen;

  /* The extended option header length is measured in 8-byte groups.
     To represent the current length we might have to add padding.  */
  dsize = cmsg->cmsg_len - CMSG_LEN (0);
  add_pad (cmsg, (8 - (dsize & (8 - 1))) & (8 - 1));

  /* Record the new length of the option.  */
  assert (((cmsg->cmsg_len - CMSG_LEN (0)) % 8) == 0);
  int len8b = (cmsg->cmsg_len - CMSG_LEN (0)) / 8 - 1;
  if (len8b >= 256)
    /* Too long.  */
    return NULL;

  struct ip6_ext *ie = (void *) CMSG_DATA (cmsg);
  ie->ip6e_len = len8b;

  return result;
}


uint8_t *
inet6_option_alloc (struct cmsghdr *cmsg, int datalen, int multx, int plusy)
{
  return option_alloc (cmsg, datalen, multx, plusy);
}
link_warning (inet6_option_alloc,
	      "inet6_option_alloc is obsolete, use the RFC 3542 interfaces")


/* RFC 2292, 6.3.5

   This function processes the next Hop-by-Hop option or Destination
   option in an ancillary data object.  If another option remains to be
   processed, the return value of the function is 0 and *tptrp points to
   the 8-bit option type field (which is followed by the 8-bit option
   data length, followed by the option data).  If no more options remain
   to be processed, the return value is -1 and *tptrp is NULL.  If an
   error occurs, the return value is -1 and *tptrp is not NULL.  */
int
inet6_option_next (const struct cmsghdr *cmsg, uint8_t **tptrp)
{
  /* Make sure it is an option of the right type.  */
  if (cmsg->cmsg_level != IPPROTO_IPV6
      || (cmsg->cmsg_type != IPV6_HOPOPTS && cmsg->cmsg_type != IPV6_DSTOPTS))
    return -1;

  /* Pointer to the extension header.  We only compute the address, we
     don't access anything yet.  */
  const struct ip6_ext *ip6e = (const struct ip6_ext *) CMSG_DATA (cmsg);

  /* Make sure the message is long enough.  */
  if (cmsg->cmsg_len < CMSG_LEN (sizeof (struct ip6_ext))
      /* Now we can access the extension header.  */
      || cmsg->cmsg_len < CMSG_LEN ((ip6e->ip6e_len + 1) * 8))
    /* Too small.  */
    return -1;

  /* Determine the address of the byte past the message.  */
  const uint8_t *endp = CMSG_DATA (cmsg) + (ip6e->ip6e_len + 1) * 8;

  const uint8_t *result;
  if (*tptrp == NULL)
    /* This is the first call, return the first option if there is one.  */
    result = (const uint8_t *) (ip6e + 1);
  else
    {
      /* Make sure *TPTRP points to a beginning of a new option in
	 the message.  The upper limit is checked in get_opt_end.  */
      if (*tptrp < (const uint8_t *) (ip6e + 1))
	return -1;

      /* Get the beginning of the next option.  */
      if (get_opt_end (&result, *tptrp, endp) != 0)
	return -1;
    }

  /* We know where the next option starts.  */
  *tptrp = (uint8_t *) result;

  /* Check the option is fully represented in the message.  */
  return get_opt_end (&result, result, endp);
}
link_warning (inet6_option_next,
	      "inet6_option_next is obsolete, use the RFC 3542 interfaces")


/* RFC 2292, 6.3.6

   This function is similar to the previously described
   inet6_option_next() function, except this function lets the caller
   specify the option type to be searched for, instead of always
   returning the next option in the ancillary data object.  cmsg is a
   pointer to cmsghdr structure of which cmsg_level equals IPPROTO_IPV6
   and cmsg_type equals either IPV6_HOPOPTS or IPV6_DSTOPTS.  */
int
inet6_option_find (const struct cmsghdr *cmsg, uint8_t **tptrp, int type)
{
  /* Make sure it is an option of the right type.  */
  if (cmsg->cmsg_level != IPPROTO_IPV6
      || (cmsg->cmsg_type != IPV6_HOPOPTS && cmsg->cmsg_type != IPV6_DSTOPTS))
    return -1;

  /* Pointer to the extension header.  We only compute the address, we
     don't access anything yet.  */
  const struct ip6_ext *ip6e = (const struct ip6_ext *) CMSG_DATA (cmsg);

  /* Make sure the message is long enough.  */
  if (cmsg->cmsg_len < CMSG_LEN (sizeof (struct ip6_ext))
      /* Now we can access the extension header.  */
      || cmsg->cmsg_len < CMSG_LEN ((ip6e->ip6e_len + 1) * 8))
    /* Too small.  */
    return -1;

  /* Determine the address of the byte past the message.  */
  const uint8_t *endp = CMSG_DATA (cmsg) + (ip6e->ip6e_len + 1) * 8;

  const uint8_t *next;
  if (*tptrp == NULL)
    /* This is the first call, return the first option if there is one.  */
    next = (const uint8_t *) (ip6e + 1);
  else
    {
      /* Make sure *TPTRP points to a beginning of a new option in
	 the message.  The upper limit is checked in get_opt_end.  */
      if (*tptrp < (const uint8_t *) (ip6e + 1))
	return -1;

      /* Get the beginning of the next option.  */
      if (get_opt_end (&next, *tptrp, endp) != 0)
	return -1;
    }

  /* Now search for the appropriate typed entry.  */
  const uint8_t *result;
  do
    {
      result = next;

      /* Get the end of this entry.  */
      if (get_opt_end (&next, result, endp) != 0)
	return -1;
    }
  while (*result != type);

  /* We know where the next option starts.  */
  *tptrp = (uint8_t *) result;

  /* Success.  */
  return 0;
}
link_warning (inet6_option_find,
	      "inet6_option_find is obsolete, use the RFC 3542 interfaces")
