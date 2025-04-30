/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <unistd.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <paths.h>
#include <confstr.h>
#include <sys/param.h>

#ifndef PATH_MAX
# ifdef MAXPATHLEN
#  define PATH_MAX MAXPATHLEN
# else
#  define PATH_MAX 1024
# endif
#endif

/* The file is accessible but it is not an executable file.  Invoke
   the shell to interpret it as a script.  */
static void
maybe_script_execute (const char *file, char *const argv[], char *const envp[])
{
  ptrdiff_t argc;
  for (argc = 0; argv[argc] != NULL; argc++)
    {
      if (argc == INT_MAX - 1)
	{
	  errno = E2BIG;
	  return;
	}
    }

  /* Construct an argument list for the shell based on original arguments:
     1. Empty list (argv = { NULL }, argc = 1 }: new argv will contain 3
	arguments - default shell, script to execute, and ending NULL.
     2. Non empty argument list (argc = { ..., NULL }, argc > 1}: new argv
	will contain also the default shell and the script to execute.  It
	will also skip the script name in arguments and only copy script
	arguments.  */
  char *new_argv[argc > 1 ? 2 + argc : 3];
  new_argv[0] = (char *) _PATH_BSHELL;
  new_argv[1] = (char *) file;
  if (argc > 1)
    memcpy (new_argv + 2, argv + 1, argc * sizeof (char *));
  else
    new_argv[2] = NULL;

  /* Execute the shell.  */
  __execve (new_argv[0], new_argv, envp);
}

static int
__execvpe_common (const char *file, char *const argv[], char *const envp[],
	          bool exec_script)
{
  /* We check the simple case first. */
  if (*file == '\0')
    {
      __set_errno (ENOENT);
      return -1;
    }

  /* Don't search when it contains a slash.  */
  if (strchr (file, '/') != NULL)
    {
      __execve (file, argv, envp);

      if (errno == ENOEXEC && exec_script)
        maybe_script_execute (file, argv, envp);

      return -1;
    }

  const char *path = getenv ("PATH");
  if (!path)
    path = CS_PATH;
  /* Although GLIBC does not enforce NAME_MAX, we set it as the maximum
     size to avoid unbounded stack allocation.  Same applies for
     PATH_MAX.  */
  size_t file_len = __strnlen (file, NAME_MAX) + 1;
  size_t path_len = __strnlen (path, PATH_MAX - 1) + 1;

  /* NAME_MAX does not include the terminating null character.  */
  if ((file_len - 1 > NAME_MAX)
      || !__libc_alloca_cutoff (path_len + file_len + 1))
    {
      errno = ENAMETOOLONG;
      return -1;
    }

  const char *subp;
  bool got_eacces = false;
  /* The resulting string maximum size would be potentially a entry
     in PATH plus '/' (path_len + 1) and then the the resulting file name
     plus '\0' (file_len since it already accounts for the '\0').  */
  char buffer[path_len + file_len + 1];
  for (const char *p = path; ; p = subp)
    {
      subp = __strchrnul (p, ':');

      /* PATH is larger than PATH_MAX and thus potentially larger than
	 the stack allocation.  */
      if (subp - p >= path_len)
	{
          /* If there is only one path, bail out.  */
	  if (*subp == '\0')
	    break;
	  /* Otherwise skip to next one.  */
	  continue;
	}

      /* Use the current path entry, plus a '/' if nonempty, plus the file to
         execute.  */
      char *pend = mempcpy (buffer, p, subp - p);
      *pend = '/';
      memcpy (pend + (p < subp), file, file_len);

      __execve (buffer, argv, envp);

      if (errno == ENOEXEC && exec_script)
        /* This has O(P*C) behavior, where P is the length of the path and C
           is the argument count.  A better strategy would be allocate the
           substitute argv and reuse it each time through the loop (so it
           behaves as O(P+C) instead.  */
        maybe_script_execute (buffer, argv, envp);

      switch (errno)
	{
	  case EACCES:
	  /* Record that we got a 'Permission denied' error.  If we end
	     up finding no executable we can use, we want to diagnose
	     that we did find one but were denied access.  */
	    got_eacces = true;
	  case ENOENT:
	  case ESTALE:
	  case ENOTDIR:
	  /* Those errors indicate the file is missing or not executable
	     by us, in which case we want to just try the next path
	     directory.  */
	  case ENODEV:
	  case ETIMEDOUT:
	  /* Some strange filesystems like AFS return even
	     stranger error numbers.  They cannot reasonably mean
	     anything else so ignore those, too.  */
	    break;

          default:
	  /* Some other error means we found an executable file, but
	     something went wrong executing it; return the error to our
	     caller.  */
	    return -1;
	}

      if (*subp++ == '\0')
	break;
    }

  /* We tried every element and none of them worked.  */
  if (got_eacces)
    /* At least one failure was due to permissions, so report that
       error.  */
    __set_errno (EACCES);

  return -1;
}

/* Execute FILE, searching in the `PATH' environment variable if it contains
   no slashes, with arguments ARGV and environment from ENVP.  */
int
__execvpe (const char *file, char *const argv[], char *const envp[])
{
  return __execvpe_common (file, argv, envp, true);
}
weak_alias (__execvpe, execvpe)

/* Same as __EXECVPE, but does not try to execute NOEXEC files.  */
int
__execvpex (const char *file, char *const argv[], char *const envp[])
{
  return __execvpe_common (file, argv, envp, false);
}
