/* Group merging implementation.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <grp.h>
#include <grp-merge.h>

#define BUFCHECK(size)			\
  ({					\
    do					\
      {					\
	if (c + (size) > buflen)	\
          {				\
	    free (members);		\
	    return ERANGE;		\
	  }				\
      }					\
    while (0);				\
  })

int
__copy_grp (const struct group srcgrp, const size_t buflen,
	    struct group *destgrp, char *destbuf, char **endptr)
{
  size_t i;
  size_t c = 0;
  size_t len;
  size_t memcount;
  char **members = NULL;

  /* Copy the GID.  */
  destgrp->gr_gid = srcgrp.gr_gid;

  /* Copy the name.  */
  len = strlen (srcgrp.gr_name) + 1;
  BUFCHECK (len);
  memcpy (&destbuf[c], srcgrp.gr_name, len);
  destgrp->gr_name = &destbuf[c];
  c += len;

  /* Copy the password.  */
  len = strlen (srcgrp.gr_passwd) + 1;
  BUFCHECK (len);
  memcpy (&destbuf[c], srcgrp.gr_passwd, len);
  destgrp->gr_passwd = &destbuf[c];
  c += len;

  /* Count all of the members.  */
  for (memcount = 0; srcgrp.gr_mem[memcount]; memcount++)
    ;

  /* Allocate a temporary holding area for the pointers to the member
     contents, including space for a NULL-terminator.  */
  members = malloc (sizeof (char *) * (memcount + 1));
  if (members == NULL)
    return ENOMEM;

  /* Copy all of the group members to destbuf and add a pointer to each of
     them into the 'members' array.  */
  for (i = 0; srcgrp.gr_mem[i]; i++)
    {
      len = strlen (srcgrp.gr_mem[i]) + 1;
      BUFCHECK (len);
      memcpy (&destbuf[c], srcgrp.gr_mem[i], len);
      members[i] = &destbuf[c];
      c += len;
    }
  members[i] = NULL;

  /* Align for pointers.  We can't simply align C because we need to
     align destbuf[c].  */
  if ((((uintptr_t)destbuf + c) & (__alignof__(char **) - 1)) != 0)
    {
      uintptr_t mis_align = ((uintptr_t)destbuf + c) & (__alignof__(char **) - 1);
      c += __alignof__(char **) - mis_align;
    }

  /* Copy the pointers from the members array into the buffer and assign them
     to the gr_mem member of destgrp.  */
  destgrp->gr_mem = (char **) &destbuf[c];
  len = sizeof (char *) * (memcount + 1);
  BUFCHECK (len);
  memcpy (&destbuf[c], members, len);
  c += len;
  free (members);
  members = NULL;

  /* Save the count of members at the end.  */
  BUFCHECK (sizeof (size_t));
  memcpy (&destbuf[c], &memcount, sizeof (size_t));
  c += sizeof (size_t);

  if (endptr)
    *endptr = destbuf + c;
  return 0;
}
libc_hidden_def (__copy_grp)

/* Check that the name, GID and passwd fields match, then
   copy in the gr_mem array.  */
int
__merge_grp (struct group *savedgrp, char *savedbuf, char *savedend,
	     size_t buflen, struct group *mergegrp, char *mergebuf)
{
  size_t c, i, len;
  size_t savedmemcount;
  size_t memcount;
  size_t membersize;
  char **members = NULL;

  /* We only support merging members of groups with identical names and
     GID values. If we hit this case, we need to overwrite the current
     buffer with the saved one (which is functionally equivalent to
     treating the new lookup as NSS_STATUS_NOTFOUND).  */
  if (mergegrp->gr_gid != savedgrp->gr_gid
      || strcmp (mergegrp->gr_name, savedgrp->gr_name))
    return __copy_grp (*savedgrp, buflen, mergegrp, mergebuf, NULL);

  /* Get the count of group members from the last sizeof (size_t) bytes in the
     mergegrp buffer.  */
  savedmemcount = *(size_t *) (savedend - sizeof (size_t));

  /* Get the count of new members to add.  */
  for (memcount = 0; mergegrp->gr_mem[memcount]; memcount++)
    ;

  /* Create a temporary array to hold the pointers to the member values from
     both the saved and merge groups.  */
  membersize = savedmemcount + memcount + 1;
  members = malloc (sizeof (char *) * membersize);
  if (members == NULL)
    return ENOMEM;

  /* Copy in the existing member pointers from the saved group
     Note: this is not NULL-terminated yet.  */
  memcpy (members, savedgrp->gr_mem, sizeof (char *) * savedmemcount);

  /* Back up into the savedbuf until we get back to the NULL-terminator of the
     group member list. (This means walking back savedmemcount + 1 (char *) pointers
     and the member count value.
     The value of c is going to be the used length of the buffer backed up by
     the member count and further backed up by the size of the pointers.  */
  c = savedend - savedbuf
      - sizeof (size_t)
      - sizeof (char *) * (savedmemcount + 1);

  /* Add all the new group members, overwriting the old NULL-terminator while
     adding the new pointers to the temporary array.  */
  for (i = 0; mergegrp->gr_mem[i]; i++)
    {
      len = strlen (mergegrp->gr_mem[i]) + 1;
      BUFCHECK (len);
      memcpy (&savedbuf[c], mergegrp->gr_mem[i], len);
      members[savedmemcount + i] = &savedbuf[c];
      c += len;
    }
  /* Add the NULL-terminator.  */
  members[savedmemcount + memcount] = NULL;

  /* Align for pointers.  We can't simply align C because we need to
     align savedbuf[c].  */
  if ((((uintptr_t)savedbuf + c) & (__alignof__(char **) - 1)) != 0)
    {
      uintptr_t mis_align = ((uintptr_t)savedbuf + c) & (__alignof__(char **) - 1);
      c += __alignof__(char **) - mis_align;
    }

  /* Copy the member array back into the buffer after the member list and free
     the member array.  */
  savedgrp->gr_mem = (char **) &savedbuf[c];
  len = sizeof (char *) * membersize;
  BUFCHECK (len);
  memcpy (&savedbuf[c], members, len);
  c += len;

  free (members);
  members = NULL;

  /* Finally, copy the results back into mergebuf, since that's the buffer
     that we were provided by the caller.  */
  return __copy_grp (*savedgrp, buflen, mergegrp, mergebuf, NULL);
}
libc_hidden_def (__merge_grp)
