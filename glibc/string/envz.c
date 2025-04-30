/* Routines for dealing with '\0' separated environment vectors
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written by Miles Bader <miles@gnu.org>

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

#include <malloc.h>
#include <string.h>

#include <envz.h>

/* The character separating names from values in an envz.  */
#define SEP '='

/* Returns a pointer to the entry in ENVZ for NAME, or 0 if there is none.
   If NAME contains the separator character, only the portion before it is
   used in the comparison.  */
char *
envz_entry (const char *envz, size_t envz_len, const char *name)
{
  while (envz_len)
    {
      const char *p = name;
      const char *entry = envz;	/* Start of this entry. */

      /* See how far NAME and ENTRY match.  */
      while (envz_len && *p == *envz && *p && *p != SEP)
	p++, envz++, envz_len--;

      if ((*envz == '\0' || *envz == SEP) && (*p == '\0' || *p == SEP))
	/* Bingo! */
	return (char *) entry;

      /* No match, skip to the next entry.  */
      while (envz_len && *envz)
	envz++, envz_len--;
      if (envz_len)
	envz++, envz_len--;	/* skip '\0' */
    }

  return 0;
}
libc_hidden_def (envz_entry)

/* Returns a pointer to the value portion of the entry in ENVZ for NAME, or 0
   if there is none.  */
char *
envz_get (const char *envz, size_t envz_len, const char *name)
{
  char *entry = envz_entry (envz, envz_len, name);
  if (entry)
    {
      while (*entry && *entry != SEP)
	entry++;
      if (*entry)
	entry++;
      else
	entry = 0;		/* A null entry.  */
    }
  return entry;
}

/* Remove the entry for NAME from ENVZ & ENVZ_LEN, if any.  */
void
envz_remove (char **envz, size_t *envz_len, const char *name)
{
  char *entry = envz_entry (*envz, *envz_len, name);
  if (entry)
    argz_delete (envz, envz_len, entry);
}
libc_hidden_def (envz_remove)

/* Adds an entry for NAME with value VALUE to ENVZ & ENVZ_LEN.  If an entry
   with the same name already exists in ENVZ, it is removed.  If VALUE is
   NULL, then the new entry will a special null one, for which envz_get will
   return NULL, although envz_entry will still return an entry; this is handy
   because when merging with another envz, the null entry can override an
   entry in the other one.  Null entries can be removed with envz_strip ().  */
error_t
envz_add (char **envz, size_t *envz_len, const char *name, const char *value)
{
  envz_remove (envz, envz_len, name);

  if (value)
    /* Add the new value, if there is one.  */
    {
      size_t name_len = strlen (name);
      size_t value_len = strlen (value);
      size_t old_envz_len = *envz_len;
      size_t new_envz_len = old_envz_len + name_len + 1 + value_len + 1;
      char *new_envz = realloc (*envz, new_envz_len);

      if (new_envz)
	{
	  memcpy (new_envz + old_envz_len, name, name_len);
	  new_envz[old_envz_len + name_len] = SEP;
	  memcpy (new_envz + old_envz_len + name_len + 1, value, value_len);
	  new_envz[new_envz_len - 1] = 0;

	  *envz = new_envz;
	  *envz_len = new_envz_len;

	  return 0;
	}
      else
	return ENOMEM;
    }
  else
    /* Add a null entry.  */
    return __argz_add (envz, envz_len, name);
}

/* Adds each entry in ENVZ2 to ENVZ & ENVZ_LEN, as if with envz_add().  If
   OVERRIDE is true, then values in ENVZ2 will supersede those with the same
   name in ENV, otherwise not.  */
error_t
envz_merge (char **envz, size_t *envz_len, const char *envz2,
	    size_t envz2_len, int override)
{
  error_t err = 0;

  while (envz2_len && ! err)
    {
      char *old = envz_entry (*envz, *envz_len, envz2);
      size_t new_len = strlen (envz2) + 1;

      if (! old)
	err = __argz_append (envz, envz_len, envz2, new_len);
      else if (override)
	{
	  argz_delete (envz, envz_len, old);
	  err = __argz_append (envz, envz_len, envz2, new_len);
	}

      envz2 += new_len;
      envz2_len -= new_len;
    }

  return err;
}

/* Remove null entries.  */
void
envz_strip (char **envz, size_t *envz_len)
{
  char *entry = *envz;
  size_t left = *envz_len;
  while (left)
    {
      size_t entry_len = strlen (entry) + 1;
      left -= entry_len;
      if (! strchr (entry, SEP))
	/* Null entry. */
	memmove (entry, entry + entry_len, left);
      else
	entry += entry_len;
    }
  *envz_len = entry - *envz;
}
