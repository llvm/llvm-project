/* Utilities for reading/writing fstab, mtab, etc.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#include <alloca.h>
#include <mntent.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdio_ext.h>
#include <string.h>
#include <sys/types.h>

#define flockfile(s) _IO_flockfile (s)
#define funlockfile(s) _IO_funlockfile (s)

#undef __setmntent
#undef __endmntent
#undef __getmntent_r

/* Prepare to begin reading and/or writing mount table entries from the
   beginning of FILE.  MODE is as for `fopen'.  */
FILE *
__setmntent (const char *file, const char *mode)
{
  /* Extend the mode parameter with "c" to disable cancellation in the
     I/O functions and "e" to set FD_CLOEXEC.  */
  size_t modelen = strlen (mode);
  char newmode[modelen + 3];
  memcpy (mempcpy (newmode, mode, modelen), "ce", 3);
  FILE *result = fopen (file, newmode);

  if (result != NULL)
    /* We do the locking ourselves.  */
    __fsetlocking (result, FSETLOCKING_BYCALLER);

  return result;
}
libc_hidden_def (__setmntent)
weak_alias (__setmntent, setmntent)


/* Close a stream opened with `setmntent'.  */
int
__endmntent (FILE *stream)
{
  if (stream)		/* SunOS 4.x allows for NULL stream */
    fclose (stream);
  return 1;		/* SunOS 4.x says to always return 1 */
}
libc_hidden_def (__endmntent)
weak_alias (__endmntent, endmntent)


/* Since the values in a line are separated by spaces, a name cannot
   contain a space.  Therefore some programs encode spaces in names
   by the strings "\040".  We undo the encoding when reading an entry.
   The decoding happens in place.  */
static char *
decode_name (char *buf)
{
  char *rp = buf;
  char *wp = buf;

  do
    if (rp[0] == '\\' && rp[1] == '0' && rp[2] == '4' && rp[3] == '0')
      {
	/* \040 is a SPACE.  */
	*wp++ = ' ';
	rp += 3;
      }
    else if (rp[0] == '\\' && rp[1] == '0' && rp[2] == '1' && rp[3] == '1')
      {
	/* \011 is a TAB.  */
	*wp++ = '\t';
	rp += 3;
      }
    else if (rp[0] == '\\' && rp[1] == '0' && rp[2] == '1' && rp[3] == '2')
      {
	/* \012 is a NEWLINE.  */
	*wp++ = '\n';
	rp += 3;
      }
    else if (rp[0] == '\\' && rp[1] == '\\')
      {
	/* We have to escape \\ to be able to represent all characters.  */
	*wp++ = '\\';
	rp += 1;
      }
    else if (rp[0] == '\\' && rp[1] == '1' && rp[2] == '3' && rp[3] == '4')
      {
	/* \134 is also \\.  */
	*wp++ = '\\';
	rp += 3;
      }
    else
      *wp++ = *rp;
  while (*rp++ != '\0');

  return buf;
}

static bool
get_mnt_entry (FILE *stream, struct mntent *mp, char *buffer, int bufsiz)
{
  char *cp;
  char *head;

  do
    {
      char *end_ptr;

      if (__fgets_unlocked (buffer, bufsiz, stream) == NULL)
	  return false;

      end_ptr = strchr (buffer, '\n');
      if (end_ptr != NULL)	/* chop newline */
	{
	  /* Do not walk past the start of buffer if it's all whitespace.  */
	  while (end_ptr != buffer
		 && (end_ptr[-1] == ' ' || end_ptr[-1] == '\t'))
            end_ptr--;
	  *end_ptr = '\0';
	}
      else
	{
	  /* Not the whole line was read.  Do it now but forget it.  */
	  char tmp[1024];
	  while (__fgets_unlocked (tmp, sizeof tmp, stream) != NULL)
	    if (strchr (tmp, '\n') != NULL)
	      break;
	}

      head = buffer + strspn (buffer, " \t");
      /* skip empty lines and comment lines:  */
    }
  while (head[0] == '\0' || head[0] == '#');

  cp = __strsep (&head, " \t");
  mp->mnt_fsname = cp != NULL ? decode_name (cp) : (char *) "";
  if (head)
    head += strspn (head, " \t");
  cp = __strsep (&head, " \t");
  mp->mnt_dir = cp != NULL ? decode_name (cp) : (char *) "";
  if (head)
    head += strspn (head, " \t");
  cp = __strsep (&head, " \t");
  mp->mnt_type = cp != NULL ? decode_name (cp) : (char *) "";
  if (head)
    head += strspn (head, " \t");
  cp = __strsep (&head, " \t");
  mp->mnt_opts = cp != NULL ? decode_name (cp) : (char *) "";
  switch (head ? sscanf (head, " %d %d ", &mp->mnt_freq, &mp->mnt_passno) : 0)
    {
    case 0:
      mp->mnt_freq = 0;
      /* Fall through.  */
    case 1:
      mp->mnt_passno = 0;
      /* Fall through.  */
    case 2:
      break;
    }

  return true;
}

/* Read one mount table entry from STREAM.  Returns a pointer to storage
   reused on the next call, or null for EOF or error (use feof/ferror to
   check).  */
struct mntent *
__getmntent_r (FILE *stream, struct mntent *mp, char *buffer, int bufsiz)
{
  struct mntent *result;

  flockfile (stream);
  while (true)
    if (get_mnt_entry (stream, mp, buffer, bufsiz))
      {
	/* If the file system is autofs look for a mount option hint
	   ("ignore") to skip the entry.  */
	if (strcmp (mp->mnt_type, "autofs") == 0 && __hasmntopt (mp, "ignore"))
	  memset (mp, 0, sizeof (*mp));
	else
	  {
	    result = mp;
	    break;
	  }
      }
    else
      {
	result = NULL;
	break;
      }
  funlockfile (stream);

  return result;
}
libc_hidden_def (__getmntent_r)
weak_alias (__getmntent_r, getmntent_r)

/* Write STR into STREAM, escaping whitespaces as we go.  Do not check for
   errors here; we check the stream status in __ADDMNTENT.  */
static void
write_string (FILE *stream, const char *str)
{
  char c;
  const char *encode_chars = " \t\n\\";

  while ((c = *str++) != '\0')
    {
      if (strchr (encode_chars, c) == NULL)
	__putc_unlocked (c, stream);
      else
	{
	  __putc_unlocked ('\\', stream);
	  __putc_unlocked (((c & 0xc0) >> 6) + '0', stream);
	  __putc_unlocked (((c & 0x38) >> 3) + '0', stream);
	  __putc_unlocked (((c & 0x07) >> 0) + '0', stream);
	}
    }
  __putc_unlocked (' ', stream);
}

/* Write the mount table entry described by MNT to STREAM.
   Return zero on success, nonzero on failure.  */
int
__addmntent (FILE *stream, const struct mntent *mnt)
{
  int ret = 1;

  if (fseek (stream, 0, SEEK_END))
    return ret;

  flockfile (stream);

  write_string (stream, mnt->mnt_fsname);
  write_string (stream, mnt->mnt_dir);
  write_string (stream, mnt->mnt_type);
  write_string (stream, mnt->mnt_opts);
  fprintf (stream, "%d %d\n", mnt->mnt_freq, mnt->mnt_passno);

  ret = __ferror_unlocked (stream) != 0 || fflush (stream) != 0;

  funlockfile (stream);

  return ret;
}
weak_alias (__addmntent, addmntent)


/* Search MNT->mnt_opts for an option matching OPT.
   Returns the address of the substring, or null if none found.  */
char *
__hasmntopt (const struct mntent *mnt, const char *opt)
{
  const size_t optlen = strlen (opt);
  char *rest = mnt->mnt_opts, *p;

  while ((p = strstr (rest, opt)) != NULL)
    {
      if ((p == rest || p[-1] == ',')
	  && (p[optlen] == '\0' || p[optlen] == '=' || p[optlen] == ','))
	return p;

      rest = strchr (p, ',');
      if (rest == NULL)
	break;
      ++rest;
    }

  return NULL;
}
libc_hidden_def (__hasmntopt)
weak_alias (__hasmntopt, hasmntopt)
