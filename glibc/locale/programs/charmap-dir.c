/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published
   by the Free Software Foundation; version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <https://www.gnu.org/licenses/>.  */

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <libintl.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

#include "localedef.h"
#include "charmap-dir.h"

/* The data type of a charmap directory being traversed.  */
struct charmap_dir
{
  DIR *dir;
  /* The directory pathname, ending in a slash.  */
  char *directory;
  size_t directory_len;
  /* Scratch area used for returning pathnames.  */
  char *pathname;
  size_t pathname_size;
};

/* Starts a charmap directory traversal.
   Returns a CHARMAP_DIR, or NULL if the directory doesn't exist.  */
CHARMAP_DIR *
charmap_opendir (const char *directory)
{
  struct charmap_dir *cdir;
  DIR *dir;
  size_t len;
  int add_slash;

  dir = opendir (directory);
  if (dir == NULL)
    {
      record_error (1, errno, gettext ("\
cannot read character map directory `%s'"),
		    directory);
      return NULL;
    }

  cdir = (struct charmap_dir *) xmalloc (sizeof (struct charmap_dir));
  cdir->dir = dir;

  len = strlen (directory);
  add_slash = (len == 0 || directory[len - 1] != '/');
  cdir->directory = (char *) xmalloc (len + add_slash + 1);
  memcpy (cdir->directory, directory, len);
  if (add_slash)
    cdir->directory[len] = '/';
  cdir->directory[len + add_slash] = '\0';
  cdir->directory_len = len + add_slash;

  cdir->pathname = NULL;
  cdir->pathname_size = 0;

  return cdir;
}

/* Reads the next directory entry.
   Returns its charmap name, or NULL if past the last entry or upon error.
   The storage returned may be overwritten by a later charmap_readdir
   call on the same CHARMAP_DIR.  */
const char *
charmap_readdir (CHARMAP_DIR *cdir)
{
  for (;;)
    {
      struct dirent64 *dirent;
      size_t len;
      size_t size;
      char *filename;
      mode_t mode;

      dirent = readdir64 (cdir->dir);
      if (dirent == NULL)
        return NULL;
      if (strcmp (dirent->d_name, ".") == 0)
        continue;
      if (strcmp (dirent->d_name, "..") == 0)
        continue;

      len = strlen (dirent->d_name);

      size = cdir->directory_len + len + 1;
      if (size > cdir->pathname_size)
        {
          free (cdir->pathname);
          if (size < 2 * cdir->pathname_size)
            size = 2 * cdir->pathname_size;
          cdir->pathname = (char *) xmalloc (size);
          cdir->pathname_size = size;
        }

      stpcpy (stpcpy (cdir->pathname, cdir->directory), dirent->d_name);
      filename = cdir->pathname + cdir->directory_len;

      if (dirent->d_type != DT_UNKNOWN && dirent->d_type != DT_LNK)
        mode = DTTOIF (dirent->d_type);
      else
        {
          struct stat64 statbuf;

          if (stat64 (cdir->pathname, &statbuf) < 0)
            continue;

          mode = statbuf.st_mode;
        }

      if (!S_ISREG (mode))
        continue;

      /* For compressed charmaps, the canonical charmap name does not
         include the extension.  */
      if (len > 3 && memcmp (&filename[len - 3], ".gz", 3) == 0)
        filename[len - 3] = '\0';
      else if (len > 4 && memcmp (&filename[len - 4], ".bz2", 4) == 0)
        filename[len - 4] = '\0';

      return filename;
    }
}

/* Finishes a charmap directory traversal, and frees the resources
   attached to the CHARMAP_DIR.  */
int
charmap_closedir (CHARMAP_DIR *cdir)
{
  DIR *dir = cdir->dir;

  free (cdir->directory);
  free (cdir->pathname);
  free (cdir);
  return closedir (dir);
}

/* Creates a subprocess decompressing the given pathname, and returns
   a stream reading its output (the decompressed data).  */
static
FILE *
fopen_uncompressed (const char *pathname, const char *compressor)
{
  int pfd;

  pfd = open (pathname, O_RDONLY);
  if (pfd >= 0)
    {
      struct stat64 statbuf;
      int fd[2];

      if (fstat64 (pfd, &statbuf) >= 0
          && S_ISREG (statbuf.st_mode)
          && pipe (fd) >= 0)
        {
          char *argv[4]
	    = { (char *) compressor, (char *) "-d", (char *) "-c", NULL };
          posix_spawn_file_actions_t actions;

          if (posix_spawn_file_actions_init (&actions) == 0)
            {
              if (posix_spawn_file_actions_adddup2 (&actions,
                                                    fd[1], STDOUT_FILENO) == 0
                  && posix_spawn_file_actions_addclose (&actions, fd[1]) == 0
                  && posix_spawn_file_actions_addclose (&actions, fd[0]) == 0
                  && posix_spawn_file_actions_adddup2 (&actions,
                                                       pfd, STDIN_FILENO) == 0
                  && posix_spawn_file_actions_addclose (&actions, pfd) == 0
                  && posix_spawnp (NULL, compressor, &actions, NULL,
                                   argv, environ) == 0)
                {
                  posix_spawn_file_actions_destroy (&actions);
                  close (fd[1]);
                  close (pfd);
                  return fdopen (fd[0], "r");
                }
              posix_spawn_file_actions_destroy (&actions);
            }
          close (fd[1]);
          close (fd[0]);
        }
      close (pfd);
    }
  return NULL;
}

/* Opens a charmap for reading, given its name (not an alias name).  */
FILE *
charmap_open (const char *directory, const char *name)
{
  size_t dlen = strlen (directory);
  int add_slash = (dlen == 0 || directory[dlen - 1] != '/');
  size_t nlen = strlen (name);
  char *pathname;
  char *p;
  FILE *stream;

  pathname = alloca (dlen + add_slash + nlen + 5);
  p = stpcpy (pathname, directory);
  if (add_slash)
    *p++ = '/';
  p = stpcpy (p, name);

  stream = fopen (pathname, "rm");
  if (stream != NULL)
    return stream;

  memcpy (p, ".gz", 4);
  stream = fopen_uncompressed (pathname, "gzip");
  if (stream != NULL)
    return stream;

  memcpy (p, ".bz2", 5);
  stream = fopen_uncompressed (pathname, "bzip2");
  if (stream != NULL)
    return stream;

  return NULL;
}

/* An empty alias list.  Avoids the need to return NULL from
   charmap_aliases.  */
static char *empty[1];

/* Returns a NULL terminated list of alias names of a charmap.  */
char **
charmap_aliases (const char *directory, const char *name)
{
  FILE *stream;
  char **aliases;
  size_t naliases;

  stream = charmap_open (directory, name);
  if (stream == NULL)
    return empty;

  aliases = NULL;
  naliases = 0;

  while (!feof (stream))
    {
      char *alias = NULL;
      char junk[BUFSIZ];

      if (fscanf (stream, " <code_set_name> %ms", &alias) == 1
          || fscanf (stream, "%% alias %ms", &alias) == 1)
        {
          aliases = (char **) xrealloc (aliases,
                                        (naliases + 2) * sizeof (char *));
          aliases[naliases++] = alias;
        }

      /* Read the rest of the line.  */
      if (fgets (junk, sizeof junk, stream) != NULL)
        {
          if (strstr (junk, "CHARMAP") != NULL)
            /* We cannot expect more aliases from now on.  */
            break;

          while (strchr (junk, '\n') == NULL
                 && fgets (junk, sizeof junk, stream) != NULL)
            continue;
        }
    }

  fclose (stream);

  if (naliases == 0)
    return empty;

  aliases[naliases] = NULL;
  return aliases;
}

/* Frees an alias list returned by charmap_aliases.  */
void
charmap_free_aliases (char **aliases)
{
  if (aliases != empty)
    {
      char **p;

      for (p = aliases; *p; p++)
        free (*p);

      free (aliases);
    }
}
