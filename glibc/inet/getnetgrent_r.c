/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
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
#include <atomic.h>
#include <libc-lock.h>
#include <errno.h>
#include <netdb.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "netgroup.h"
#include "nsswitch.h"
#include <sysdep.h>
#include <nscd/nscd_proto.h>


/* Protect above variable against multiple uses at the same time.  */
__libc_lock_define_initialized (static, lock)

/* The whole information for the set/get/endnetgrent functions are
   kept in this structure.  */
static struct __netgrent dataset;

/* Set up NIP to run through the services.  Return nonzero if there are no
   services (left).  */
static int
setup (void **fctp, nss_action_list *nipp)
{
  int no_more;

  no_more = __nss_netgroup_lookup2 (nipp, "setnetgrent", NULL, fctp);

  return no_more;
}

/* Free used memory.  */
static void
free_memory (struct __netgrent *data)
{
  while (data->known_groups != NULL)
    {
      struct name_list *tmp = data->known_groups;
      data->known_groups = data->known_groups->next;
      free (tmp);
    }

  while (data->needed_groups != NULL)
    {
      struct name_list *tmp = data->needed_groups;
      data->needed_groups = data->needed_groups->next;
      free (tmp);
    }
}

static void
endnetgrent_hook (struct __netgrent *datap)
{
  enum nss_status (*endfct) (struct __netgrent *);

  if (datap->nip == NULL || datap->nip == (nss_action_list) -1l)
    return;

  endfct = __nss_lookup_function (datap->nip, "endnetgrent");
  if (endfct != NULL)
    (void) (*endfct) (datap);
  datap->nip = NULL;
}

static int
__internal_setnetgrent_reuse (const char *group, struct __netgrent *datap,
			      int *errnop)
{
  union
  {
    enum nss_status (*f) (const char *, struct __netgrent *);
    void *ptr;
  } fct;
  enum nss_status status = NSS_STATUS_UNAVAIL;
  struct name_list *new_elem;

  /* Free data from previous service.  */
  endnetgrent_hook (datap);

  /* Cycle through all the services and run their setnetgrent functions.  */
  int no_more = setup (&fct.ptr, &datap->nip);
  while (! no_more)
    {
      assert (datap->data == NULL);

      /* Ignore status, we force check in `__nss_next2'.  */
      status = DL_CALL_FCT (*fct.f, (group, datap));

      nss_action_list old_nip = datap->nip;
      no_more = __nss_next2 (&datap->nip, "setnetgrent", NULL, &fct.ptr,
			     status, 0);

      if (status == NSS_STATUS_SUCCESS && ! no_more)
	{
	  enum nss_status (*endfct) (struct __netgrent *);

	  endfct = __nss_lookup_function (old_nip, "endnetgrent");
	  if (endfct != NULL)
	    (void) DL_CALL_FCT (*endfct, (datap));
	}
    }

  /* Add the current group to the list of known groups.  */
  size_t group_len = strlen (group) + 1;
  new_elem = (struct name_list *) malloc (sizeof (struct name_list)
					  + group_len);
  if (new_elem == NULL)
    {
      *errnop = errno;
      status = NSS_STATUS_TRYAGAIN;
    }
  else
    {
      new_elem->next = datap->known_groups;
      memcpy (new_elem->name, group, group_len);
      datap->known_groups = new_elem;
    }

  return status == NSS_STATUS_SUCCESS;
}

int
__internal_setnetgrent (const char *group, struct __netgrent *datap)
{
  /* Free list of all netgroup names from last run.  */
  free_memory (datap);

  return __internal_setnetgrent_reuse (group, datap, &errno);
}
libc_hidden_def (__internal_setnetgrent)

static int
nscd_setnetgrent (const char *group)
{
#ifdef USE_NSCD
  if (__nss_not_use_nscd_netgroup > 0
      && ++__nss_not_use_nscd_netgroup > NSS_NSCD_RETRY)
    __nss_not_use_nscd_netgroup = 0;

  if (!__nss_not_use_nscd_netgroup
      && !__nss_database_custom[NSS_DBSIDX_netgroup])
    return __nscd_setnetgrent (group, &dataset);
#endif
  return -1;
}

int
setnetgrent (const char *group)
{
  int result;

  __libc_lock_lock (lock);

  result = nscd_setnetgrent (group);
  if (result < 0)
    result = __internal_setnetgrent (group, &dataset);

  __libc_lock_unlock (lock);

  return result;
}

void
__internal_endnetgrent (struct __netgrent *datap)
{
  endnetgrent_hook (datap);
  /* Now free list of all netgroup names from last run.  */
  free_memory (datap);
}
libc_hidden_def (__internal_endnetgrent)


void
endnetgrent (void)
{
  __libc_lock_lock (lock);

  __internal_endnetgrent (&dataset);

  __libc_lock_unlock (lock);
}

#ifdef USE_NSCD
static const char *
get_nonempty_val (const char *in)
{
  if (*in == '\0')
    return NULL;
  return in;
}

static enum nss_status
nscd_getnetgrent (struct __netgrent *datap, char *buffer, size_t buflen,
		  int *errnop)
{
  if (datap->cursor >= datap->data + datap->data_size)
    return NSS_STATUS_UNAVAIL;

  datap->type = triple_val;
  datap->val.triple.host = get_nonempty_val (datap->cursor);
  datap->cursor = (char *) __rawmemchr (datap->cursor, '\0') + 1;
  datap->val.triple.user = get_nonempty_val (datap->cursor);
  datap->cursor = (char *) __rawmemchr (datap->cursor, '\0') + 1;
  datap->val.triple.domain = get_nonempty_val (datap->cursor);
  datap->cursor = (char *) __rawmemchr (datap->cursor, '\0') + 1;

  return NSS_STATUS_SUCCESS;
}
#endif

int
__internal_getnetgrent_r (char **hostp, char **userp, char **domainp,
			  struct __netgrent *datap,
			  char *buffer, size_t buflen, int *errnop)
{
  enum nss_status (*fct) (struct __netgrent *, char *, size_t, int *);

  /* Initialize status to return if no more functions are found.  */
  enum nss_status status = NSS_STATUS_NOTFOUND;

  /* Run through available functions, starting with the same function last
     run.  We will repeat each function as long as it succeeds, and then go
     on to the next service action.  */
  int no_more = datap->nip == NULL;
  if (! no_more)
    {
#ifdef USE_NSCD
      /* This bogus function pointer is a special marker left by
	 __nscd_setnetgrent to tell us to use the data it left
	 before considering any modules.  */
      if (datap->nip == (nss_action_list) -1l)
	fct = nscd_getnetgrent;
      else
#endif
	{
	  fct = __nss_lookup_function (datap->nip, "getnetgrent_r");
	  no_more = fct == NULL;
	}

      while (! no_more)
	{
	  status = DL_CALL_FCT (*fct, (datap, buffer, buflen, &errno));

	  if (status == NSS_STATUS_RETURN
	      /* The service returned a NOTFOUND, but there are more groups that
		 we need to resolve before we give up.  */
	      || (status == NSS_STATUS_NOTFOUND && datap->needed_groups != NULL))
	    {
	      /* This was the last one for this group.  Look at next group
		 if available.  */
	      int found = 0;
	      while (datap->needed_groups != NULL && ! found)
		{
		  struct name_list *tmp = datap->needed_groups;
		  datap->needed_groups = datap->needed_groups->next;
		  tmp->next = datap->known_groups;
		  datap->known_groups = tmp;

		  found = __internal_setnetgrent_reuse (datap->known_groups->name,
							datap, errnop);
		}

	      if (found && datap->nip != NULL)
		{
		  fct = __nss_lookup_function (datap->nip, "getnetgrent_r");
		  if (fct != NULL)
		    continue;
		}
	    }
	  else if (status == NSS_STATUS_SUCCESS && datap->type == group_val)
	    {
	      /* The last entry was a name of another netgroup.  */
	      struct name_list *namep;

	      /* Ignore if we've seen the name before.  */
	      for (namep = datap->known_groups; namep != NULL;
		   namep = namep->next)
		if (strcmp (datap->val.group, namep->name) == 0)
		  break;
	      if (namep == NULL)
		for (namep = datap->needed_groups; namep != NULL;
		     namep = namep->next)
		  if (strcmp (datap->val.group, namep->name) == 0)
		    break;
	      if (namep != NULL)
		/* Really ignore.  */
		continue;

	      size_t group_len = strlen (datap->val.group) + 1;
	      namep = (struct name_list *) malloc (sizeof (struct name_list)
						  + group_len);
	      if (namep == NULL)
		/* We are out of memory.  */
		status = NSS_STATUS_RETURN;
	      else
		{
		  namep->next = datap->needed_groups;
		  memcpy (namep->name, datap->val.group, group_len);
		  datap->needed_groups = namep;
		  /* And get the next entry.  */
		  continue;
		}
	    }
	  break;
	}
    }

  if (status == NSS_STATUS_SUCCESS)
    {
      *hostp = (char *) datap->val.triple.host;
      *userp = (char *) datap->val.triple.user;
      *domainp = (char *) datap->val.triple.domain;
    }

  return status == NSS_STATUS_SUCCESS ? 1 : 0;
}
libc_hidden_def (__internal_getnetgrent_r)

/* The real entry point.  */
int
__getnetgrent_r (char **hostp, char **userp, char **domainp,
		 char *buffer, size_t buflen)
{
  enum nss_status status;

  __libc_lock_lock (lock);

  status = __internal_getnetgrent_r (hostp, userp, domainp, &dataset,
				     buffer, buflen, &errno);

  __libc_lock_unlock (lock);

  return status;
}
weak_alias (__getnetgrent_r, getnetgrent_r)

/* Test whether given (host,user,domain) triple is in NETGROUP.  */
int
innetgr (const char *netgroup, const char *host, const char *user,
	 const char *domain)
{
#ifdef USE_NSCD
  if (__nss_not_use_nscd_netgroup > 0
      && ++__nss_not_use_nscd_netgroup > NSS_NSCD_RETRY)
    __nss_not_use_nscd_netgroup = 0;

  if (!__nss_not_use_nscd_netgroup
      && !__nss_database_custom[NSS_DBSIDX_netgroup])
    {
      int result = __nscd_innetgr (netgroup, host, user, domain);
      if (result >= 0)
	return result;
    }
#endif

  union
  {
    enum nss_status (*f) (const char *, struct __netgrent *);
    void *ptr;
  } setfct;
  void (*endfct) (struct __netgrent *);
  int (*getfct) (struct __netgrent *, char *, size_t, int *);
  struct __netgrent entry;
  int result = 0;
  const char *current_group = netgroup;

  memset (&entry, '\0', sizeof (entry));

  /* Walk through the services until we found an answer or we shall
     not work further.  We can do some optimization here.  Since all
     services must provide the `setnetgrent' function we can do all
     the work during one walk through the service list.  */
  while (1)
    {
      int no_more = setup (&setfct.ptr, &entry.nip);
      while (! no_more)
	{
	  assert (entry.data == NULL);

	  /* Open netgroup.  */
	  enum nss_status status = DL_CALL_FCT (*setfct.f,
						(current_group, &entry));

	  if (status == NSS_STATUS_SUCCESS
	      && (getfct = __nss_lookup_function (entry.nip, "getnetgrent_r"))
		 != NULL)
	    {
	      char buffer[1024];

	      while (DL_CALL_FCT (*getfct,
				  (&entry, buffer, sizeof buffer, &errno))
		     == NSS_STATUS_SUCCESS)
		{
		  if (entry.type == group_val)
		    {
		      /* Make sure we haven't seen the name before.  */
		      struct name_list *namep;

		      for (namep = entry.known_groups; namep != NULL;
			   namep = namep->next)
			if (strcmp (entry.val.group, namep->name) == 0)
			  break;
		      if (namep == NULL)
			for (namep = entry.needed_groups; namep != NULL;
			     namep = namep->next)
			  if (strcmp (entry.val.group, namep->name) == 0)
			    break;
		      if (namep == NULL
			  && strcmp (netgroup, entry.val.group) != 0)
			{
			  size_t group_len = strlen (entry.val.group) + 1;
			  namep =
			    (struct name_list *) malloc (sizeof (*namep)
							 + group_len);
			  if (namep == NULL)
			    {
			      /* Out of memory, simply return.  */
			      result = -1;
			      break;
			    }

			  namep->next = entry.needed_groups;
			  memcpy (namep->name, entry.val.group, group_len);
			  entry.needed_groups = namep;
			}
		    }
		  else
		    {
		      if ((entry.val.triple.host == NULL || host == NULL
			   || __strcasecmp (entry.val.triple.host, host) == 0)
			  && (entry.val.triple.user == NULL || user == NULL
			      || strcmp (entry.val.triple.user, user) == 0)
			  && (entry.val.triple.domain == NULL || domain == NULL
			      || __strcasecmp (entry.val.triple.domain,
					       domain) == 0))
			{
			  result = 1;
			  break;
			}
		    }
		}

	      /* If we found one service which does know the given
		 netgroup we don't try further.  */
	      status = NSS_STATUS_RETURN;
	    }

	  /* Free all resources of the service.  */
	  endfct = __nss_lookup_function (entry.nip, "endnetgrent");
	  if (endfct != NULL)
	    DL_CALL_FCT (*endfct, (&entry));

	  if (result != 0)
	    break;

	  /* Look for the next service.  */
	  no_more = __nss_next2 (&entry.nip, "setnetgrent", NULL,
				 &setfct.ptr, status, 0);
	}

      if (result == 0 && entry.needed_groups != NULL)
	{
	  struct name_list *tmp = entry.needed_groups;
	  entry.needed_groups = tmp->next;
	  tmp->next = entry.known_groups;
	  entry.known_groups = tmp;
	  current_group = tmp->name;
	  continue;
	}

      /* No way out.  */
      break;
    }

  /* Free the memory.  */
  free_memory (&entry);

  return result == 1;
}
libc_hidden_def (innetgr)
