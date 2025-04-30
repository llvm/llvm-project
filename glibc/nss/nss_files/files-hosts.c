/* Hosts file parser in nss_files module.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

#include <assert.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <arpa/nameser.h>
#include <netdb.h>
#include <resolv/resolv-internal.h>
#include <scratch_buffer.h>
#include <alloc_buffer.h>
#include <nss.h>

/* Get implementation for some internal functions.  */
#include "../resolv/res_hconf.h"


#define ENTNAME		hostent
#define DATABASE	"hosts"
#define NEED_H_ERRNO

#define EXTRA_ARGS	 , af
#define EXTRA_ARGS_DECL	 , int af

#define ENTDATA hostent_data
struct hostent_data
  {
    unsigned char host_addr[16]; /* IPv4 or IPv6 address.  */
    char *h_addr_ptrs[2];	/* Points to that and null terminator.  */
  };

#define TRAILING_LIST_MEMBER		h_aliases
#define TRAILING_LIST_SEPARATOR_P	isspace
#include "files-parse.c"
LINE_PARSER
("#",
 {
   char *addr;

   STRING_FIELD (addr, isspace, 1);

   /* Parse address.  */
   if (__inet_pton (af == AF_UNSPEC ? AF_INET : af, addr, entdata->host_addr)
       > 0)
     af = af == AF_UNSPEC ? AF_INET : af;
   else
     {
       if (af == AF_INET
	   && __inet_pton (AF_INET6, addr, entdata->host_addr) > 0)
	 {
	   if (IN6_IS_ADDR_V4MAPPED (entdata->host_addr))
	     memcpy (entdata->host_addr, entdata->host_addr + 12, INADDRSZ);
	   else if (IN6_IS_ADDR_LOOPBACK (entdata->host_addr))
	     {
	       in_addr_t localhost = htonl (INADDR_LOOPBACK);
	       memcpy (entdata->host_addr, &localhost, sizeof (localhost));
	     }
	   else
	     /* Illegal address: ignore line.  */
	     return 0;
	 }
       else if (af == AF_UNSPEC
		&& __inet_pton (AF_INET6, addr, entdata->host_addr) > 0)
	 af = AF_INET6;
       else
	 /* Illegal address: ignore line.  */
	 return 0;
     }

   /* We always return entries of the requested form.  */
   result->h_addrtype = af;
   result->h_length = af == AF_INET ? INADDRSZ : IN6ADDRSZ;

   /* Store a pointer to the address in the expected form.  */
   entdata->h_addr_ptrs[0] = (char *) entdata->host_addr;
   entdata->h_addr_ptrs[1] = NULL;
   result->h_addr_list = entdata->h_addr_ptrs;

   STRING_FIELD (result->h_name, isspace, 1);
 })

#define EXTRA_ARGS_VALUE , AF_INET
#include "files-XXX.c"
#undef EXTRA_ARGS_VALUE

/* We only need to consider IPv4 mapped addresses if the input to the
   gethostbyaddr() function is an IPv6 address.  */
#define EXTRA_ARGS_VALUE , af
DB_LOOKUP (hostbyaddr, ,,,
	   {
	     if (result->h_length == (int) len
		 && ! memcmp (addr, result->h_addr_list[0], len))
	       break;
	   }, const void *addr, socklen_t len, int af)
#undef EXTRA_ARGS_VALUE

/* Type of the address and alias arrays.  */
#define DYNARRAY_STRUCT array
#define DYNARRAY_ELEMENT char *
#define DYNARRAY_PREFIX array_
#include <malloc/dynarray-skeleton.c>

static enum nss_status
gethostbyname3_multi (FILE * stream, const char *name, int af,
		      struct hostent *result, char *buffer, size_t buflen,
		      int *errnop, int *herrnop)
{
  assert (af == AF_INET || af == AF_INET6);

  /* We have to get all host entries from the file.  */
  struct scratch_buffer tmp_buffer;
  scratch_buffer_init (&tmp_buffer);
  struct hostent tmp_result_buf;
  struct array addresses;
  array_init (&addresses);
  struct array aliases;
  array_init (&aliases);
  enum nss_status status;

  /* Preserve the addresses and aliases encountered so far.  */
  for (size_t i = 0; result->h_addr_list[i] != NULL; ++i)
    array_add (&addresses, result->h_addr_list[i]);
  for (size_t i = 0; result->h_aliases[i] != NULL; ++i)
    array_add (&aliases, result->h_aliases[i]);

  /* The output buffer re-uses now-unused space at the end of the
     buffer, starting with the aliases array.  It comes last in the
     data produced by internal_getent.  (The alias names themselves
     are still located in the line read in internal_getent, which is
     stored at the beginning of the buffer.)  */
  struct alloc_buffer outbuf;
  {
    char *bufferend = (char *) result->h_aliases;
    outbuf = alloc_buffer_create (bufferend, buffer + buflen - bufferend);
  }

  while (true)
    {
      status = internal_getent (stream, &tmp_result_buf, tmp_buffer.data,
				tmp_buffer.length, errnop, herrnop, af);
      /* Enlarge the buffer if necessary.  */
      if (status == NSS_STATUS_TRYAGAIN && *herrnop == NETDB_INTERNAL
	  && *errnop == ERANGE)
	{
	  if (!scratch_buffer_grow (&tmp_buffer))
	    {
	      *errnop = ENOMEM;
	      /* *herrnop and status already have the right value.  */
	      break;
	    }
	  /* Loop around and retry with a larger buffer.  */
	}
      else if (status == NSS_STATUS_SUCCESS)
	{
	  /* A line was read.  Check that it matches the search
	     criteria.  */

	  int matches = 1;
	  struct hostent *old_result = result;
	  result = &tmp_result_buf;
	  /* The following piece is a bit clumsy but we want to use
	     the `LOOKUP_NAME_CASE' value.  The optimizer should do
	     its job.  */
	  do
	    {
	      LOOKUP_NAME_CASE (h_name, h_aliases)
		result = old_result;
	    }
	  while ((matches = 0));

	  /* If the line matches, we need to copy the addresses and
	     aliases, so that we can reuse tmp_buffer for the next
	     line.  */
	  if (matches)
	    {
	      /* Record the addresses.  */
	      for (size_t i = 0; tmp_result_buf.h_addr_list[i] != NULL; ++i)
		{
		  /* Allocate the target space in the output buffer,
		     depending on the address family.  */
		  void *target;
		  if (af == AF_INET)
		    {
		      assert (tmp_result_buf.h_length == 4);
		      target = alloc_buffer_alloc (&outbuf, struct in_addr);
		    }
		  else if (af == AF_INET6)
		    {
		      assert (tmp_result_buf.h_length == 16);
		      target = alloc_buffer_alloc (&outbuf, struct in6_addr);
		    }
		  else
		    __builtin_unreachable ();

		  if (target == NULL)
		    {
		      /* Request a larger output buffer.  */
		      *errnop = ERANGE;
		      *herrnop = NETDB_INTERNAL;
		      status = NSS_STATUS_TRYAGAIN;
		      break;
		    }
		  memcpy (target, tmp_result_buf.h_addr_list[i],
			  tmp_result_buf.h_length);
		  array_add (&addresses, target);
		}

	      /* Record the aliases.  */
	      for (size_t i = 0; tmp_result_buf.h_aliases[i] != NULL; ++i)
		{
		  char *alias = tmp_result_buf.h_aliases[i];
		  array_add (&aliases,
			     alloc_buffer_copy_string (&outbuf, alias));
		}

	      /* If the real name is different add, it also to the
		 aliases.  This means that there is a duplication in
		 the alias list but this is really the user's
		 problem.  */
	      {
		char *new_name = tmp_result_buf.h_name;
		if (strcmp (old_result->h_name, new_name) != 0)
		  array_add (&aliases,
			     alloc_buffer_copy_string (&outbuf, new_name));
	      }

	      /* Report memory allocation failures during the
		 expansion of the temporary arrays.  */
	      if (array_has_failed (&addresses) || array_has_failed (&aliases))
		{
		  *errnop = ENOMEM;
		  *herrnop = NETDB_INTERNAL;
		  status = NSS_STATUS_UNAVAIL;
		  break;
		}

	      /* Request a larger output buffer if we ran out of room.  */
	      if (alloc_buffer_has_failed (&outbuf))
		{
		  *errnop = ERANGE;
		  *herrnop = NETDB_INTERNAL;
		  status = NSS_STATUS_TRYAGAIN;
		  break;
		}

	      result = old_result;
	    } /* If match was found.  */

	  /* If no match is found, loop around and fetch another
	     line.  */

	} /* status == NSS_STATUS_SUCCESS.  */
      else
	/* internal_getent returned an error.  */
	break;
    } /* while (true) */

  /* Propagate the NSS_STATUS_TRYAGAIN error to the caller.  It means
     that we may not have loaded the complete result.
     NSS_STATUS_NOTFOUND, however, means that we reached the end of
     the file successfully.  */
  if (status != NSS_STATUS_TRYAGAIN)
    status = NSS_STATUS_SUCCESS;

  if (status == NSS_STATUS_SUCCESS)
    {
      /* Copy the address and alias arrays into the output buffer and
	 add NULL terminators.  The pointed-to elements were directly
	 written into the output buffer above and do not need to be
	 copied again.  */
      size_t addresses_count = array_size (&addresses);
      size_t aliases_count = array_size (&aliases);
      char **out_addresses = alloc_buffer_alloc_array
	(&outbuf, char *, addresses_count + 1);
      char **out_aliases = alloc_buffer_alloc_array
	(&outbuf, char *, aliases_count + 1);
      if (out_addresses == NULL || out_aliases == NULL)
	{
	  /* The output buffer is not large enough.  */
	  *errnop = ERANGE;
	  *herrnop = NETDB_INTERNAL;
	  status = NSS_STATUS_TRYAGAIN;
	  /* Fall through to function exit.  */
	}
      else
	{
	  /* Everything is allocated in place.  Make the copies and
	     adjust the array pointers.  */
	  memcpy (out_addresses, array_begin (&addresses),
		  addresses_count * sizeof (char *));
	  out_addresses[addresses_count] = NULL;
	  memcpy (out_aliases, array_begin (&aliases),
		  aliases_count * sizeof (char *));
	  out_aliases[aliases_count] = NULL;

	  result->h_addr_list = out_addresses;
	  result->h_aliases = out_aliases;

	  status = NSS_STATUS_SUCCESS;
	}
    }

  scratch_buffer_free (&tmp_buffer);
  array_free (&addresses);
  array_free (&aliases);
  return status;
}

enum nss_status
_nss_files_gethostbyname3_r (const char *name, int af, struct hostent *result,
			     char *buffer, size_t buflen, int *errnop,
			     int *herrnop, int32_t *ttlp, char **canonp)
{
  FILE *stream = NULL;
  uintptr_t pad = -(uintptr_t) buffer % __alignof__ (struct hostent_data);
  buffer += pad;
  buflen = buflen > pad ? buflen - pad : 0;

  /* Open file.  */
  enum nss_status status = internal_setent (&stream);

  if (status == NSS_STATUS_SUCCESS)
    {
      while ((status = internal_getent (stream, result, buffer, buflen, errnop,
					herrnop, af))
	     == NSS_STATUS_SUCCESS)
	{
	  LOOKUP_NAME_CASE (h_name, h_aliases)
	}

      if (status == NSS_STATUS_SUCCESS
	  && _res_hconf.flags & HCONF_FLAG_MULTI)
	status = gethostbyname3_multi
	  (stream, name, af, result, buffer, buflen, errnop, herrnop);

      fclose (stream);
    }

  if (canonp && status == NSS_STATUS_SUCCESS)
    *canonp = result->h_name;

  return status;
}
libc_hidden_def (_nss_files_gethostbyname3_r)

enum nss_status
_nss_files_gethostbyname_r (const char *name, struct hostent *result,
			    char *buffer, size_t buflen, int *errnop,
			    int *herrnop)
{
  return _nss_files_gethostbyname3_r (name, AF_INET, result, buffer, buflen,
				      errnop, herrnop, NULL, NULL);
}
libc_hidden_def (_nss_files_gethostbyname_r)

enum nss_status
_nss_files_gethostbyname2_r (const char *name, int af, struct hostent *result,
			     char *buffer, size_t buflen, int *errnop,
			     int *herrnop)
{
  return _nss_files_gethostbyname3_r (name, af, result, buffer, buflen,
				      errnop, herrnop, NULL, NULL);
}
libc_hidden_def (_nss_files_gethostbyname2_r)

enum nss_status
_nss_files_gethostbyname4_r (const char *name, struct gaih_addrtuple **pat,
			     char *buffer, size_t buflen, int *errnop,
			     int *herrnop, int32_t *ttlp)
{
  FILE *stream = NULL;

  /* Open file.  */
  enum nss_status status = internal_setent (&stream);

  if (status == NSS_STATUS_SUCCESS)
    {
      bool any = false;
      bool got_canon = false;
      while (1)
	{
	  /* Align the buffer for the next record.  */
	  uintptr_t pad = (-(uintptr_t) buffer
			   % __alignof__ (struct hostent_data));
	  buffer += pad;
	  buflen = buflen > pad ? buflen - pad : 0;

	  struct hostent result;
	  status = internal_getent (stream, &result, buffer, buflen, errnop,
				    herrnop, AF_UNSPEC);
	  if (status != NSS_STATUS_SUCCESS)
	    break;

	  int naliases = 0;
	  if (__strcasecmp (name, result.h_name) != 0)
	    {
	      for (; result.h_aliases[naliases] != NULL; ++naliases)
		if (! __strcasecmp (name, result.h_aliases[naliases]))
		  break;
	      if (result.h_aliases[naliases] == NULL)
		continue;

	      /* We know this alias exist.  Count it.  */
	      ++naliases;
	    }

	  /* Determine how much memory has been used so far.  */
	  // XXX It is not necessary to preserve the aliases array
	  while (result.h_aliases[naliases] != NULL)
	    ++naliases;
	  char *bufferend = (char *) &result.h_aliases[naliases + 1];
	  assert (buflen >= bufferend - buffer);
	  buflen -= bufferend - buffer;
	  buffer = bufferend;

	  /* We found something.  */
	  any = true;

	  /* Create the record the caller expects.  There is only one
	     address.  */
	  assert (result.h_addr_list[1] == NULL);
	  if (*pat == NULL)
	    {
	      uintptr_t pad = (-(uintptr_t) buffer
			       % __alignof__ (struct gaih_addrtuple));
	      buffer += pad;
	      buflen = buflen > pad ? buflen - pad : 0;

	      if (__builtin_expect (buflen < sizeof (struct gaih_addrtuple),
				    0))
		{
		  *errnop = ERANGE;
		  *herrnop = NETDB_INTERNAL;
		  status = NSS_STATUS_TRYAGAIN;
		  break;
		}

	      *pat = (struct gaih_addrtuple *) buffer;
	      buffer += sizeof (struct gaih_addrtuple);
	      buflen -= sizeof (struct gaih_addrtuple);
	    }

	  (*pat)->next = NULL;
	  (*pat)->name = got_canon ? NULL : result.h_name;
	  got_canon = true;
	  (*pat)->family = result.h_addrtype;
	  memcpy ((*pat)->addr, result.h_addr_list[0], result.h_length);
	  (*pat)->scopeid = 0;

	  pat = &((*pat)->next);

	  /* If we only look for the first matching entry we are done.  */
	  if ((_res_hconf.flags & HCONF_FLAG_MULTI) == 0)
	    break;
	}

      /* If we have to look for multiple records and found one, this
	 is a success.  */
      if (status == NSS_STATUS_NOTFOUND && any)
	{
	  assert ((_res_hconf.flags & HCONF_FLAG_MULTI) != 0);
	  status = NSS_STATUS_SUCCESS;
	}

      fclose (stream);
    }
  else if (status == NSS_STATUS_TRYAGAIN)
    {
      *errnop = errno;
      *herrnop = TRY_AGAIN;
    }
  else
    {
      *errnop = errno;
      *herrnop = NO_DATA;
    }

  return status;
}
libc_hidden_def (_nss_files_gethostbyname4_r)
