/* Handle configuration data.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#include <dirent.h>
#include <libc-symbols.h>
#include <locale.h>
#include <sys/types.h>

#if IS_IN (libc)
# include <libio/libioP.h>
# define __getdelim(line, len, c, fp) __getdelim (line, len, c, fp)

# undef isspace
# define isspace(__c) __isspace_l ((__c), _nl_C_locobj_ptr)
# define asprintf __asprintf
# define opendir __opendir
# define readdir __readdir
# define closedir __closedir
# define mempcpy __mempcpy
# define lstat64 __lstat64
# define feof_unlocked __feof_unlocked
#endif

/* Name of the file containing the module information in the directories
   along the path.  */
static const char gconv_conf_filename[] = "gconv-modules";
static const char gconv_conf_dirname[] = "gconv-modules.d";

static void add_alias (char *);
static void add_module (char *, const char *, size_t, int);

/* Read the next configuration file.  */
static bool
read_conf_file (const char *filename, const char *directory, size_t dir_len)
{
  /* Note the file is opened with cancellation in the I/O functions
     disabled.  */
  FILE *fp = fopen (filename, "rce");
  char *line = NULL;
  size_t line_len = 0;
  static int modcounter;

  /* Don't complain if a file is not present or readable, simply silently
     ignore it.  */
  if (fp == NULL)
    return false;

  /* No threads reading from this stream.  */
  __fsetlocking (fp, FSETLOCKING_BYCALLER);

  /* Process the known entries of the file.  Comments start with `#' and
     end with the end of the line.  Empty lines are ignored.  */
  while (!feof_unlocked (fp))
    {
      char *rp, *endp, *word;
      ssize_t n = __getdelim (&line, &line_len, '\n', fp);
      if (n < 0)
	/* An error occurred.  */
	break;

      rp = line;
      /* Terminate the line (excluding comments or newline) by an NUL byte
	 to simplify the following code.  */
      endp = strchr (rp, '#');
      if (endp != NULL)
	*endp = '\0';
      else
	if (rp[n - 1] == '\n')
	  rp[n - 1] = '\0';

      while (isspace (*rp))
	++rp;

      /* If this is an empty line go on with the next one.  */
      if (rp == endp)
	continue;

      word = rp;
      while (*rp != '\0' && !isspace (*rp))
	++rp;

      if (rp - word == sizeof ("alias") - 1
	  && memcmp (word, "alias", sizeof ("alias") - 1) == 0)
	add_alias (rp);
      else if (rp - word == sizeof ("module") - 1
	       && memcmp (word, "module", sizeof ("module") - 1) == 0)
	add_module (rp, directory, dir_len, modcounter++);
      /* else */
	/* Otherwise ignore the line.  */
    }

  free (line);

  fclose (fp);
  return true;
}

static __always_inline bool
gconv_parseconfdir (const char *dir, size_t dir_len)
{
  /* No slash needs to be inserted between dir and gconv_conf_filename;
     dir already ends in a slash.  */
  char *buf = malloc (dir_len + sizeof (gconv_conf_dirname));
  bool found = false;

  if (buf == NULL)
    return false;

  char *cp = mempcpy (mempcpy (buf, dir, dir_len), gconv_conf_filename,
		      sizeof (gconv_conf_filename));

  /* Read the gconv-modules configuration file first.  */
  found = read_conf_file (buf, dir, dir_len);

  /* Next, see if there is a gconv-modules.d directory containing
     configuration files and if it is non-empty.  */
  cp--;
  cp[0] = '.';
  cp[1] = 'd';
  cp[2] = '\0';

  DIR *confdir = opendir (buf);
  if (confdir != NULL)
    {
      struct dirent *ent;
      while ((ent = readdir (confdir)) != NULL)
	{
	  if (ent->d_type != DT_REG && ent->d_type != DT_UNKNOWN)
	    continue;

	  size_t len = strlen (ent->d_name);
	  const char *suffix = ".conf";

	  if (len > strlen (suffix)
	      && strcmp (ent->d_name + len - strlen (suffix), suffix) == 0)
	    {
	      char *conf;
	      struct stat64 st;
	      if (asprintf (&conf, "%s/%s", buf, ent->d_name) < 0)
		continue;
	      if (ent->d_type == DT_UNKNOWN
		  && (lstat64 (conf, &st) == -1
		      || !S_ISREG (st.st_mode)))
		continue;

	      found |= read_conf_file (conf, dir, dir_len);
	      free (conf);
	    }
	}
      closedir (confdir);
    }
  free (buf);
  return found;
}
