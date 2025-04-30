/* Return the canonical absolute name of a given file.
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

#ifndef _LIBC
/* Don't use __attribute__ __nonnull__ in this compilation unit.  Otherwise gcc
   optimizes away the name == NULL test below.  */
# define _GL_ARG_NONNULL(params)

# define _GL_USE_STDLIB_ALLOC 1
# include <libc-config.h>
#endif

/* Specification.  */
#include <stdlib.h>

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdbool.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <eloop-threshold.h>
#include <filename.h>
#include <idx.h>
#include <intprops.h>
#include <scratch_buffer.h>

#ifdef _LIBC
# include <shlib-compat.h>
# define GCC_LINT 1
# define _GL_ATTRIBUTE_PURE __attribute__ ((__pure__))
#else
# define __canonicalize_file_name canonicalize_file_name
# define __realpath realpath
# include "pathmax.h"
# define __faccessat faccessat
# if defined _WIN32 && !defined __CYGWIN__
#  define __getcwd _getcwd
# elif HAVE_GETCWD
#  if IN_RELOCWRAPPER
    /* When building the relocatable program wrapper, use the system's getcwd
       function, not the gnulib override, otherwise we would get a link error.
     */
#   undef getcwd
#  endif
#  if defined VMS && !defined getcwd
    /* We want the directory in Unix syntax, not in VMS syntax.
       The gnulib override of 'getcwd' takes 2 arguments; the original VMS
       'getcwd' takes 3 arguments.  */
#   define __getcwd(buf, max) getcwd (buf, max, 0)
#  else
#   define __getcwd getcwd
#  endif
# else
#  define __getcwd(buf, max) getwd (buf)
# endif
# define __mempcpy mempcpy
# define __pathconf pathconf
# define __rawmemchr rawmemchr
# define __readlink readlink
# define __stat stat
#endif

/* Suppress bogus GCC -Wmaybe-uninitialized warnings.  */
#if defined GCC_LINT || defined lint
# define IF_LINT(Code) Code
#else
# define IF_LINT(Code) /* empty */
#endif

#ifndef DOUBLE_SLASH_IS_DISTINCT_ROOT
# define DOUBLE_SLASH_IS_DISTINCT_ROOT false
#endif

#if defined _LIBC || !FUNC_REALPATH_WORKS

/* Return true if FILE's existence can be shown, false (setting errno)
   otherwise.  Follow symbolic links.  */
static bool
file_accessible (char const *file)
{
# if defined _LIBC || HAVE_FACCESSAT
  return __faccessat (AT_FDCWD, file, F_OK, AT_EACCESS) == 0;
# else
  struct stat st;
  return __stat (file, &st) == 0 || errno == EOVERFLOW;
# endif
}

/* True if concatenating END as a suffix to a file name means that the
   code needs to check that the file name is that of a searchable
   directory, since the canonicalize_filename_mode_stk code won't
   check this later anyway when it checks an ordinary file name
   component within END.  END must either be empty, or start with a
   slash.  */

static bool _GL_ATTRIBUTE_PURE
suffix_requires_dir_check (char const *end)
{
  /* If END does not start with a slash, the suffix is OK.  */
  while (ISSLASH (*end))
    {
      /* Two or more slashes act like a single slash.  */
      do
        end++;
      while (ISSLASH (*end));

      switch (*end++)
        {
        default: return false;  /* An ordinary file name component is OK.  */
        case '\0': return true; /* Trailing "/" is trouble.  */
        case '.': break;        /* Possibly "." or "..".  */
        }
      /* Trailing "/.", or "/.." even if not trailing, is trouble.  */
      if (!*end || (*end == '.' && (!end[1] || ISSLASH (end[1]))))
        return true;
    }

  return false;
}

/* Append this to a file name to test whether it is a searchable directory.
   On POSIX platforms "/" suffices, but "/./" is sometimes needed on
   macOS 10.13 <https://bugs.gnu.org/30350>, and should also work on
   platforms like AIX 7.2 that need at least "/.".  */

#if defined _LIBC || defined LSTAT_FOLLOWS_SLASHED_SYMLINK
static char const dir_suffix[] = "/";
#else
static char const dir_suffix[] = "/./";
#endif

/* Return true if DIR is a searchable dir, false (setting errno) otherwise.
   DIREND points to the NUL byte at the end of the DIR string.
   Store garbage into DIREND[0 .. strlen (dir_suffix)].  */

static bool
dir_check (char *dir, char *dirend)
{
  strcpy (dirend, dir_suffix);
  return file_accessible (dir);
}

static idx_t
get_path_max (void)
{
# ifdef PATH_MAX
  long int path_max = PATH_MAX;
# else
  /* The caller invoked realpath with a null RESOLVED, even though
     PATH_MAX is not defined as a constant.  The glibc manual says
     programs should not do this, and POSIX says the behavior is undefined.
     Historically, glibc here used the result of pathconf, or 1024 if that
     failed; stay consistent with this (dubious) historical practice.  */
  int err = errno;
  long int path_max = __pathconf ("/", _PC_PATH_MAX);
  __set_errno (err);
# endif
  return path_max < 0 ? 1024 : path_max <= IDX_MAX ? path_max : IDX_MAX;
}

/* Act like __realpath (see below), with an additional argument
   rname_buf that can be used as temporary storage.

   If GCC_LINT is defined, do not inline this function with GCC 10.1
   and later, to avoid creating a pointer to the stack that GCC
   -Wreturn-local-addr incorrectly complains about.  See:
   https://gcc.gnu.org/bugzilla/show_bug.cgi?id=93644
   Although the noinline attribute can hurt performance a bit, no better way
   to pacify GCC is known; even an explicit #pragma does not pacify GCC.
   When the GCC bug is fixed this workaround should be limited to the
   broken GCC versions.  */
#if __GNUC_PREREQ (10, 1)
# if defined GCC_LINT || defined lint
__attribute__ ((__noinline__))
# elif __OPTIMIZE__ && !__NO_INLINE__
#  define GCC_BOGUS_WRETURN_LOCAL_ADDR
# endif
#endif
static char *
realpath_stk (const char *name, char *resolved,
              struct scratch_buffer *rname_buf)
{
  char *dest;
  char const *start;
  char const *end;
  int num_links = 0;

  if (name == NULL)
    {
      /* As per Single Unix Specification V2 we must return an error if
         either parameter is a null pointer.  We extend this to allow
         the RESOLVED parameter to be NULL in case the we are expected to
         allocate the room for the return value.  */
      __set_errno (EINVAL);
      return NULL;
    }

  if (name[0] == '\0')
    {
      /* As per Single Unix Specification V2 we must return an error if
         the name argument points to an empty string.  */
      __set_errno (ENOENT);
      return NULL;
    }

  struct scratch_buffer extra_buffer, link_buffer;
  scratch_buffer_init (&extra_buffer);
  scratch_buffer_init (&link_buffer);
  scratch_buffer_init (rname_buf);
  char *rname_on_stack = rname_buf->data;
  char *rname = rname_on_stack;
  bool end_in_extra_buffer = false;
  bool failed = true;

  /* This is always zero for Posix hosts, but can be 2 for MS-Windows
     and MS-DOS X:/foo/bar file names.  */
  idx_t prefix_len = FILE_SYSTEM_PREFIX_LEN (name);

  if (!IS_ABSOLUTE_FILE_NAME (name))
    {
      while (!__getcwd (rname, rname_buf->length))
        {
          if (errno != ERANGE)
            {
              dest = rname;
              goto error;
            }
          if (!scratch_buffer_grow (rname_buf))
            goto error_nomem;
          rname = rname_buf->data;
        }
      dest = __rawmemchr (rname, '\0');
      start = name;
      prefix_len = FILE_SYSTEM_PREFIX_LEN (rname);
    }
  else
    {
      dest = __mempcpy (rname, name, prefix_len);
      *dest++ = '/';
      if (DOUBLE_SLASH_IS_DISTINCT_ROOT)
        {
          if (prefix_len == 0 /* implies ISSLASH (name[0]) */
              && ISSLASH (name[1]) && !ISSLASH (name[2]))
            *dest++ = '/';
          *dest = '\0';
        }
      start = name + prefix_len;
    }

  for ( ; *start; start = end)
    {
      /* Skip sequence of multiple file name separators.  */
      while (ISSLASH (*start))
        ++start;

      /* Find end of component.  */
      for (end = start; *end && !ISSLASH (*end); ++end)
        /* Nothing.  */;

      /* Length of this file name component; it can be zero if a file
         name ends in '/'.  */
      idx_t startlen = end - start;

      if (startlen == 0)
        break;
      else if (startlen == 1 && start[0] == '.')
        /* nothing */;
      else if (startlen == 2 && start[0] == '.' && start[1] == '.')
        {
          /* Back up to previous component, ignore if at root already.  */
          if (dest > rname + prefix_len + 1)
            for (--dest; dest > rname && !ISSLASH (dest[-1]); --dest)
              continue;
          if (DOUBLE_SLASH_IS_DISTINCT_ROOT
              && dest == rname + 1 && !prefix_len
              && ISSLASH (*dest) && !ISSLASH (dest[1]))
            dest++;
        }
      else
        {
          if (!ISSLASH (dest[-1]))
            *dest++ = '/';

          while (rname + rname_buf->length - dest
                 < startlen + sizeof dir_suffix)
            {
              idx_t dest_offset = dest - rname;
              if (!scratch_buffer_grow_preserve (rname_buf))
                goto error_nomem;
              rname = rname_buf->data;
              dest = rname + dest_offset;
            }

          dest = __mempcpy (dest, start, startlen);
          *dest = '\0';

          char *buf;
          ssize_t n;
          while (true)
            {
              buf = link_buffer.data;
              idx_t bufsize = link_buffer.length;
              n = __readlink (rname, buf, bufsize - 1);
              if (n < bufsize - 1)
                break;
              if (!scratch_buffer_grow (&link_buffer))
                goto error_nomem;
            }
          if (0 <= n)
            {
              if (++num_links > __eloop_threshold ())
                {
                  __set_errno (ELOOP);
                  goto error;
                }

              buf[n] = '\0';

              char *extra_buf = extra_buffer.data;
              idx_t end_idx IF_LINT (= 0);
              if (end_in_extra_buffer)
                end_idx = end - extra_buf;
              size_t len = strlen (end);
              if (INT_ADD_OVERFLOW (len, n))
                {
                  __set_errno (ENOMEM);
                  goto error_nomem;
                }
              while (extra_buffer.length <= len + n)
                {
                  if (!scratch_buffer_grow_preserve (&extra_buffer))
                    goto error_nomem;
                  extra_buf = extra_buffer.data;
                }
              if (end_in_extra_buffer)
                end = extra_buf + end_idx;

              /* Careful here, end may be a pointer into extra_buf... */
              memmove (&extra_buf[n], end, len + 1);
              name = end = memcpy (extra_buf, buf, n);
              end_in_extra_buffer = true;

              if (IS_ABSOLUTE_FILE_NAME (buf))
                {
                  idx_t pfxlen = FILE_SYSTEM_PREFIX_LEN (buf);

                  dest = __mempcpy (rname, buf, pfxlen);
                  *dest++ = '/'; /* It's an absolute symlink */
                  if (DOUBLE_SLASH_IS_DISTINCT_ROOT)
                    {
                      if (ISSLASH (buf[1]) && !ISSLASH (buf[2]) && !pfxlen)
                        *dest++ = '/';
                      *dest = '\0';
                    }
                  /* Install the new prefix to be in effect hereafter.  */
                  prefix_len = pfxlen;
                }
              else
                {
                  /* Back up to previous component, ignore if at root
                     already: */
                  if (dest > rname + prefix_len + 1)
                    for (--dest; dest > rname && !ISSLASH (dest[-1]); --dest)
                      continue;
                  if (DOUBLE_SLASH_IS_DISTINCT_ROOT && dest == rname + 1
                      && ISSLASH (*dest) && !ISSLASH (dest[1]) && !prefix_len)
                    dest++;
                }
            }
          else if (! (suffix_requires_dir_check (end)
                      ? dir_check (rname, dest)
                      : errno == EINVAL))
            goto error;
        }
    }
  if (dest > rname + prefix_len + 1 && ISSLASH (dest[-1]))
    --dest;
  if (DOUBLE_SLASH_IS_DISTINCT_ROOT && dest == rname + 1 && !prefix_len
      && ISSLASH (*dest) && !ISSLASH (dest[1]))
    dest++;
  failed = false;

error:
  *dest++ = '\0';
  if (resolved != NULL && dest - rname <= get_path_max ())
    rname = strcpy (resolved, rname);

error_nomem:
  scratch_buffer_free (&extra_buffer);
  scratch_buffer_free (&link_buffer);

  if (failed || rname == resolved)
    {
      scratch_buffer_free (rname_buf);
      return failed ? NULL : resolved;
    }

  return scratch_buffer_dupfree (rname_buf, dest - rname);
}

/* Return the canonical absolute name of file NAME.  A canonical name
   does not contain any ".", ".." components nor any repeated file name
   separators ('/') or symlinks.  All file name components must exist.  If
   RESOLVED is null, the result is malloc'd; otherwise, if the
   canonical name is PATH_MAX chars or more, returns null with 'errno'
   set to ENAMETOOLONG; if the name fits in fewer than PATH_MAX chars,
   returns the name in RESOLVED.  If the name cannot be resolved and
   RESOLVED is non-NULL, it contains the name of the first component
   that cannot be resolved.  If the name can be resolved, RESOLVED
   holds the same value as the value returned.  */

char *
__realpath (const char *name, char *resolved)
{
  #ifdef GCC_BOGUS_WRETURN_LOCAL_ADDR
   #warning "GCC might issue a bogus -Wreturn-local-addr warning here."
   #warning "See <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=93644>."
  #endif
  struct scratch_buffer rname_buffer;
  return realpath_stk (name, resolved, &rname_buffer);
}
libc_hidden_def (__realpath)
versioned_symbol (libc, __realpath, realpath, GLIBC_2_3);
#endif /* !FUNC_REALPATH_WORKS || defined _LIBC */


#if SHLIB_COMPAT(libc, GLIBC_2_0, GLIBC_2_3)
char *
attribute_compat_text_section
__old_realpath (const char *name, char *resolved)
{
  if (resolved == NULL)
    {
      __set_errno (EINVAL);
      return NULL;
    }

  return __realpath (name, resolved);
}
compat_symbol (libc, __old_realpath, realpath, GLIBC_2_0);
#endif


char *
__canonicalize_file_name (const char *name)
{
  return __realpath (name, NULL);
}
weak_alias (__canonicalize_file_name, canonicalize_file_name)
