/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Thorsten Kukuk <kukuk@suse.de>, 1996.

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

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <grp.h>
#include <nss.h>
#include <nsswitch.h>
#include <stdio_ext.h>
#include <string.h>
#include <libc-lock.h>
#include <kernel-features.h>
#include <nss_files.h>

NSS_DECLARE_MODULE_FUNCTIONS (compat)

static nss_action_list ni;
static enum nss_status (*setgrent_impl) (int stayopen);
static enum nss_status (*getgrnam_r_impl) (const char *name,
					   struct group * grp, char *buffer,
					   size_t buflen, int *errnop);
static enum nss_status (*getgrgid_r_impl) (gid_t gid, struct group * grp,
					   char *buffer, size_t buflen,
					   int *errnop);
static enum nss_status (*getgrent_r_impl) (struct group * grp, char *buffer,
					   size_t buflen, int *errnop);
static enum nss_status (*endgrent_impl) (void);

/* Get the declaration of the parser function.  */
#define ENTNAME grent
#define STRUCTURE group
#define EXTERN_PARSER
#include <nss/nss_files/files-parse.c>

/* Structure for remembering -group members ... */
#define BLACKLIST_INITIAL_SIZE 512
#define BLACKLIST_INCREMENT 256
struct blacklist_t
{
  char *data;
  int current;
  int size;
};

struct ent_t
{
  bool files;
  enum nss_status setent_status;
  FILE *stream;
  struct blacklist_t blacklist;
};
typedef struct ent_t ent_t;

static ent_t ext_ent = { true, NSS_STATUS_SUCCESS, NULL, { NULL, 0, 0 }};

/* Protect global state against multiple changers.  */
__libc_lock_define_initialized (static, lock)

/* Prototypes for local functions.  */
static void blacklist_store_name (const char *, ent_t *);
static bool in_blacklist (const char *, int, ent_t *);

/* Initialize the NSS interface/functions. The calling function must
   hold the lock.  */
static void
init_nss_interface (void)
{
  if (__nss_database_get (nss_database_group_compat, &ni))
    {
      setgrent_impl = __nss_lookup_function (ni, "setgrent");
      getgrnam_r_impl = __nss_lookup_function (ni, "getgrnam_r");
      getgrgid_r_impl = __nss_lookup_function (ni, "getgrgid_r");
      getgrent_r_impl = __nss_lookup_function (ni, "getgrent_r");
      endgrent_impl = __nss_lookup_function (ni, "endgrent");
    }
}

static enum nss_status
internal_setgrent (ent_t *ent, int stayopen, int needent)
{
  enum nss_status status = NSS_STATUS_SUCCESS;

  ent->files = true;

  if (ent->blacklist.data != NULL)
    {
      ent->blacklist.current = 1;
      ent->blacklist.data[0] = '|';
      ent->blacklist.data[1] = '\0';
    }
  else
    ent->blacklist.current = 0;

  if (ent->stream == NULL)
    {
      ent->stream = __nss_files_fopen ("/etc/group");

      if (ent->stream == NULL)
	status = errno == EAGAIN ? NSS_STATUS_TRYAGAIN : NSS_STATUS_UNAVAIL;
    }
  else
    rewind (ent->stream);

  if (needent && status == NSS_STATUS_SUCCESS && setgrent_impl)
    ent->setent_status = setgrent_impl (stayopen);

  return status;
}


enum nss_status
_nss_compat_setgrent (int stayopen)
{
  enum nss_status result;

  __libc_lock_lock (lock);

  if (ni == NULL)
    init_nss_interface ();

  result = internal_setgrent (&ext_ent, stayopen, 1);

  __libc_lock_unlock (lock);

  return result;
}


static enum nss_status __attribute_warn_unused_result__
internal_endgrent (ent_t *ent)
{
  if (ent->stream != NULL)
    {
      fclose (ent->stream);
      ent->stream = NULL;
    }

  if (ent->blacklist.data != NULL)
    {
      ent->blacklist.current = 1;
      ent->blacklist.data[0] = '|';
      ent->blacklist.data[1] = '\0';
    }
  else
    ent->blacklist.current = 0;

  return NSS_STATUS_SUCCESS;
}

/* Like internal_endgrent, but preserve errno in all cases.  */
static void
internal_endgrent_noerror (ent_t *ent)
{
  int saved_errno = errno;
  enum nss_status unused __attribute__ ((unused)) = internal_endgrent (ent);
  __set_errno (saved_errno);
}

enum nss_status
_nss_compat_endgrent (void)
{
  enum nss_status result;

  __libc_lock_lock (lock);

  if (endgrent_impl)
    endgrent_impl ();

  result = internal_endgrent (&ext_ent);

  __libc_lock_unlock (lock);

  return result;
}

/* get the next group from NSS  (+ entry) */
static enum nss_status
getgrent_next_nss (struct group *result, ent_t *ent, char *buffer,
		   size_t buflen, int *errnop)
{
  if (!getgrent_r_impl)
    return NSS_STATUS_UNAVAIL;

  /* If the setgrent call failed, say so.  */
  if (ent->setent_status != NSS_STATUS_SUCCESS)
    return ent->setent_status;

  do
    {
      enum nss_status status;

      if ((status = getgrent_r_impl (result, buffer, buflen, errnop))
	  != NSS_STATUS_SUCCESS)
	return status;
    }
  while (in_blacklist (result->gr_name, strlen (result->gr_name), ent));

  return NSS_STATUS_SUCCESS;
}

/* This function handle the +group entrys in /etc/group */
static enum nss_status
getgrnam_plusgroup (const char *name, struct group *result, ent_t *ent,
		    char *buffer, size_t buflen, int *errnop)
{
  if (!getgrnam_r_impl)
    return NSS_STATUS_UNAVAIL;

  enum nss_status status = getgrnam_r_impl (name, result, buffer, buflen,
					    errnop);
  if (status != NSS_STATUS_SUCCESS)
    return status;

  if (in_blacklist (result->gr_name, strlen (result->gr_name), ent))
    return NSS_STATUS_NOTFOUND;

  /* We found the entry.  */
  return NSS_STATUS_SUCCESS;
}

static enum nss_status
getgrent_next_file (struct group *result, ent_t *ent,
		    char *buffer, size_t buflen, int *errnop)
{
  struct parser_data *data = (void *) buffer;
  while (1)
    {
      fpos_t pos;
      int parse_res = 0;
      char *p;

      do
	{
	  /* We need at least 3 characters for one line.  */
	  if (__glibc_unlikely (buflen < 3))
	    {
	    erange:
	      *errnop = ERANGE;
	      return NSS_STATUS_TRYAGAIN;
	    }

	  fgetpos (ent->stream, &pos);
	  buffer[buflen - 1] = '\xff';
	  p = fgets_unlocked (buffer, buflen, ent->stream);
	  if (p == NULL && feof_unlocked (ent->stream))
	    return NSS_STATUS_NOTFOUND;

	  if (p == NULL || __builtin_expect (buffer[buflen - 1] != '\xff', 0))
	    {
	    erange_reset:
	      fsetpos (ent->stream, &pos);
	      goto erange;
	    }

	  /* Terminate the line for any case.  */
	  buffer[buflen - 1] = '\0';

	  /* Skip leading blanks.  */
	  while (isspace (*p))
	    ++p;
	}
      while (*p == '\0' || *p == '#' /* Ignore empty and comment lines. */
	     /* Parse the line.  If it is invalid, loop to
	        get the next line of the file to parse.  */
	     || !(parse_res = _nss_files_parse_grent (p, result, data, buflen,
						      errnop)));

      if (__glibc_unlikely (parse_res == -1))
	/* The parser ran out of space.  */
	goto erange_reset;

      if (result->gr_name[0] != '+' && result->gr_name[0] != '-')
	/* This is a real entry.  */
	break;

      /* -group */
      if (result->gr_name[0] == '-' && result->gr_name[1] != '\0'
	  && result->gr_name[1] != '@')
	{
	  blacklist_store_name (&result->gr_name[1], ent);
	  continue;
	}

      /* +group */
      if (result->gr_name[0] == '+' && result->gr_name[1] != '\0'
	  && result->gr_name[1] != '@')
	{
	  size_t len = strlen (result->gr_name);
	  char buf[len];
	  enum nss_status status;

	  /* Store the group in the blacklist for the "+" at the end of
	     /etc/group */
	  memcpy (buf, &result->gr_name[1], len);
	  status = getgrnam_plusgroup (&result->gr_name[1], result, ent,
				       buffer, buflen, errnop);
	  blacklist_store_name (buf, ent);
	  if (status == NSS_STATUS_SUCCESS)	/* We found the entry. */
	    break;
	  else if (status == NSS_STATUS_RETURN /* We couldn't parse the entry*/
		   || status == NSS_STATUS_NOTFOUND)	/* No group in NIS */
	    continue;
	  else
	    {
	      if (status == NSS_STATUS_TRYAGAIN)
		/* The parser ran out of space.  */
		goto erange_reset;

	      return status;
	    }
	}

      /* +:... */
      if (result->gr_name[0] == '+' && result->gr_name[1] == '\0')
	{
	  ent->files = false;

	  return getgrent_next_nss (result, ent, buffer, buflen, errnop);
	}
    }

  return NSS_STATUS_SUCCESS;
}


enum nss_status
_nss_compat_getgrent_r (struct group *grp, char *buffer, size_t buflen,
			int *errnop)
{
  enum nss_status result = NSS_STATUS_SUCCESS;

  __libc_lock_lock (lock);

  /* Be prepared that the setgrent function was not called before.  */
  if (ni == NULL)
    init_nss_interface ();

  if (ext_ent.stream == NULL)
    result = internal_setgrent (&ext_ent, 1, 1);

  if (result == NSS_STATUS_SUCCESS)
    {
      if (ext_ent.files)
	result = getgrent_next_file (grp, &ext_ent, buffer, buflen, errnop);
      else
	result = getgrent_next_nss (grp, &ext_ent, buffer, buflen, errnop);
    }
  __libc_lock_unlock (lock);

  return result;
}

/* Searches in /etc/group and the NIS/NIS+ map for a special group */
static enum nss_status
internal_getgrnam_r (const char *name, struct group *result, ent_t *ent,
		     char *buffer, size_t buflen, int *errnop)
{
  struct parser_data *data = (void *) buffer;
  while (1)
    {
      fpos_t pos;
      int parse_res = 0;
      char *p;

      do
	{
	  /* We need at least 3 characters for one line.  */
	  if (__glibc_unlikely (buflen < 3))
	    {
	    erange:
	      *errnop = ERANGE;
	      return NSS_STATUS_TRYAGAIN;
	    }

	  fgetpos (ent->stream, &pos);
	  buffer[buflen - 1] = '\xff';
	  p = fgets_unlocked (buffer, buflen, ent->stream);
	  if (p == NULL && feof_unlocked (ent->stream))
	    return NSS_STATUS_NOTFOUND;

	  if (p == NULL || __builtin_expect (buffer[buflen - 1] != '\xff', 0))
	    {
	    erange_reset:
	      fsetpos (ent->stream, &pos);
	      goto erange;
	    }

	  /* Terminate the line for any case.  */
	  buffer[buflen - 1] = '\0';

	  /* Skip leading blanks.  */
	  while (isspace (*p))
	    ++p;
	}
      while (*p == '\0' || *p == '#' /* Ignore empty and comment lines. */
	     /* Parse the line.  If it is invalid, loop to
	        get the next line of the file to parse.  */
	     || !(parse_res = _nss_files_parse_grent (p, result, data, buflen,
						      errnop)));

      if (__glibc_unlikely (parse_res == -1))
	/* The parser ran out of space.  */
	goto erange_reset;

      /* This is a real entry.  */
      if (result->gr_name[0] != '+' && result->gr_name[0] != '-')
	{
	  if (strcmp (result->gr_name, name) == 0)
	    return NSS_STATUS_SUCCESS;
	  else
	    continue;
	}

      /* -group */
      if (result->gr_name[0] == '-' && result->gr_name[1] != '\0')
	{
	  if (strcmp (&result->gr_name[1], name) == 0)
	    return NSS_STATUS_NOTFOUND;
	  else
	    continue;
	}

      /* +group */
      if (result->gr_name[0] == '+' && result->gr_name[1] != '\0')
	{
	  if (strcmp (name, &result->gr_name[1]) == 0)
	    {
	      enum nss_status status;

	      status = getgrnam_plusgroup (name, result, ent,
					   buffer, buflen, errnop);
	      if (status == NSS_STATUS_RETURN)
		/* We couldn't parse the entry */
		continue;
	      else
		return status;
	    }
	}
      /* +:... */
      if (result->gr_name[0] == '+' && result->gr_name[1] == '\0')
	{
	  enum nss_status status;

	  status = getgrnam_plusgroup (name, result, ent,
				       buffer, buflen, errnop);
	  if (status == NSS_STATUS_RETURN)
	    /* We couldn't parse the entry */
	    continue;
	  else
	    return status;
	}
    }

  return NSS_STATUS_SUCCESS;
}

enum nss_status
_nss_compat_getgrnam_r (const char *name, struct group *grp,
			char *buffer, size_t buflen, int *errnop)
{
  ent_t ent = { true, NSS_STATUS_SUCCESS, NULL, { NULL, 0, 0 }};
  enum nss_status result;

  if (name[0] == '-' || name[0] == '+')
    return NSS_STATUS_NOTFOUND;

  __libc_lock_lock (lock);

  if (ni == NULL)
    init_nss_interface ();

  __libc_lock_unlock (lock);

  result = internal_setgrent (&ent, 0, 0);

  if (result == NSS_STATUS_SUCCESS)
    result = internal_getgrnam_r (name, grp, &ent, buffer, buflen, errnop);

  internal_endgrent_noerror (&ent);

  return result;
}

/* Searches in /etc/group and the NIS/NIS+ map for a special group id */
static enum nss_status
internal_getgrgid_r (gid_t gid, struct group *result, ent_t *ent,
		     char *buffer, size_t buflen, int *errnop)
{
  struct parser_data *data = (void *) buffer;
  while (1)
    {
      fpos_t pos;
      int parse_res = 0;
      char *p;

      do
	{
	  /* We need at least 3 characters for one line.  */
	  if (__glibc_unlikely (buflen < 3))
	    {
	    erange:
	      *errnop = ERANGE;
	      return NSS_STATUS_TRYAGAIN;
	    }

	  fgetpos (ent->stream, &pos);
	  buffer[buflen - 1] = '\xff';
	  p = fgets_unlocked (buffer, buflen, ent->stream);
	  if (p == NULL && feof_unlocked (ent->stream))
	    return NSS_STATUS_NOTFOUND;

	  if (p == NULL || __builtin_expect (buffer[buflen - 1] != '\xff', 0))
	    {
	    erange_reset:
	      fsetpos (ent->stream, &pos);
	      goto erange;
	    }

	  /* Terminate the line for any case.  */
	  buffer[buflen - 1] = '\0';

	  /* Skip leading blanks.  */
	  while (isspace (*p))
	    ++p;
	}
      while (*p == '\0' || *p == '#' /* Ignore empty and comment lines. */
	     /* Parse the line.  If it is invalid, loop to
	        get the next line of the file to parse.  */
	     || !(parse_res = _nss_files_parse_grent (p, result, data, buflen,
						      errnop)));

      if (__glibc_unlikely (parse_res == -1))
	/* The parser ran out of space.  */
	goto erange_reset;

      /* This is a real entry.  */
      if (result->gr_name[0] != '+' && result->gr_name[0] != '-')
	{
	  if (result->gr_gid == gid)
	    return NSS_STATUS_SUCCESS;
	  else
	    continue;
	}

      /* -group */
      if (result->gr_name[0] == '-' && result->gr_name[1] != '\0')
	{
	  blacklist_store_name (&result->gr_name[1], ent);
	  continue;
	}

      /* +group */
      if (result->gr_name[0] == '+' && result->gr_name[1] != '\0')
	{
	  /* Yes, no +1, see the memcpy call below.  */
	  size_t len = strlen (result->gr_name);
	  char buf[len];
	  enum nss_status status;

	  /* Store the group in the blacklist for the "+" at the end of
	     /etc/group */
	  memcpy (buf, &result->gr_name[1], len);
	  status = getgrnam_plusgroup (&result->gr_name[1], result, ent,
				       buffer, buflen, errnop);
	  blacklist_store_name (buf, ent);
	  if (status == NSS_STATUS_SUCCESS && result->gr_gid == gid)
	    break;
	  else
	    continue;
	}
      /* +:... */
      if (result->gr_name[0] == '+' && result->gr_name[1] == '\0')
	{
	  if (!getgrgid_r_impl)
	    return NSS_STATUS_UNAVAIL;

	  enum nss_status status = getgrgid_r_impl (gid, result,
						    buffer, buflen, errnop);
	  if (status == NSS_STATUS_RETURN) /* We couldn't parse the entry */
	    return NSS_STATUS_NOTFOUND;
	  else
	    return status;
	}
    }

  return NSS_STATUS_SUCCESS;
}

enum nss_status
_nss_compat_getgrgid_r (gid_t gid, struct group *grp,
			char *buffer, size_t buflen, int *errnop)
{
  ent_t ent = { true, NSS_STATUS_SUCCESS, NULL, { NULL, 0, 0 }};
  enum nss_status result;

  __libc_lock_lock (lock);

  if (ni == NULL)
    init_nss_interface ();

  __libc_lock_unlock (lock);

  result = internal_setgrent (&ent, 0, 0);

  if (result == NSS_STATUS_SUCCESS)
    result = internal_getgrgid_r (gid, grp, &ent, buffer, buflen, errnop);

  internal_endgrent_noerror (&ent);

  return result;
}


/* Support routines for remembering -@netgroup and -user entries.
   The names are stored in a single string with `|' as separator. */
static void
blacklist_store_name (const char *name, ent_t *ent)
{
  int namelen = strlen (name);
  char *tmp;

  /* first call, setup cache */
  if (ent->blacklist.size == 0)
    {
      ent->blacklist.size = MAX (BLACKLIST_INITIAL_SIZE, 2 * namelen);
      ent->blacklist.data = malloc (ent->blacklist.size);
      if (ent->blacklist.data == NULL)
	return;
      ent->blacklist.data[0] = '|';
      ent->blacklist.data[1] = '\0';
      ent->blacklist.current = 1;
    }
  else
    {
      if (in_blacklist (name, namelen, ent))
	return;			/* no duplicates */

      if (ent->blacklist.current + namelen + 1 >= ent->blacklist.size)
	{
	  ent->blacklist.size += MAX (BLACKLIST_INCREMENT, 2 * namelen);
	  tmp = realloc (ent->blacklist.data, ent->blacklist.size);
	  if (tmp == NULL)
	    {
	      free (ent->blacklist.data);
	      ent->blacklist.size = 0;
	      return;
	    }
	  ent->blacklist.data = tmp;
	}
    }

  tmp = stpcpy (ent->blacklist.data + ent->blacklist.current, name);
  *tmp++ = '|';
  *tmp = '\0';
  ent->blacklist.current += namelen + 1;

  return;
}

/* Return whether ent->blacklist contains name.  */
static bool
in_blacklist (const char *name, int namelen, ent_t *ent)
{
  char buf[namelen + 3];
  char *cp;

  if (ent->blacklist.data == NULL)
    return false;

  buf[0] = '|';
  cp = stpcpy (&buf[1], name);
  *cp++ = '|';
  *cp = '\0';
  return strstr (ent->blacklist.data, buf) != NULL;
}
