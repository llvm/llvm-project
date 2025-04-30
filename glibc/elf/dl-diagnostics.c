/* Print diagnostics data in ld.so.
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

#include <gnu/lib-names.h>
#include <stdbool.h>
#include <stddef.h>
#include <unistd.h>

#include <dl-diagnostics.h>
#include <dl-hwcaps.h>
#include <dl-main.h>
#include <dl-procinfo.h>
#include <dl-sysdep.h>
#include <ldsodefs.h>
#include "trusted-dirs.h"
#include "version.h"

/* Write CH to standard output.  */
static void
_dl_putc (char ch)
{
  _dl_write (STDOUT_FILENO, &ch, 1);
}

/* Print CH to standard output, quoting it if necessary.  */
static void
print_quoted_char (char ch)
{
  if (ch < ' ' || ch > '~')
    {
      char buf[4];
      buf[0] = '\\';
      buf[1] = '0' + ((ch >> 6) & 7);
      buf[2] = '0' + ((ch >> 6) & 7);
      buf[3] = '0' + (ch & 7);
      _dl_write (STDOUT_FILENO, buf, 4);
    }
  else
    {
      if (ch == '\\' || ch == '"')
        _dl_putc ('\\');
      _dl_putc (ch);
    }
}

/* Print S of LEN bytes to standard output, quoting characters as
   needed.  */
static void
print_string_length (const char *s, size_t len)
{
  _dl_putc ('"');
  for (size_t i = 0; i < len; ++i)
    print_quoted_char (s[i]);
  _dl_putc ('"');
}

void
_dl_diagnostics_print_string (const char *s)
{
  if (s == NULL)
    {
      _dl_printf ("0x0");
      return;
    }

  _dl_putc ('"');
  while (*s != '\0')
    {
      print_quoted_char (*s);
      ++s;
    }
  _dl_putc ('"');
}

void
_dl_diagnostics_print_labeled_string (const char *label, const char *s)
{
  _dl_printf ("%s=", label);
  _dl_diagnostics_print_string (s);
  _dl_putc ('\n');
}

void
_dl_diagnostics_print_labeled_value (const char *label, uint64_t value)
{
  if (sizeof (value) == sizeof (unsigned long int))
    /* _dl_printf can print 64-bit values directly.  */
    _dl_printf ("%s=0x%lx\n", label, (unsigned long int) value);
  else
    {
      uint32_t high = value >> 32;
      uint32_t low = value;
      if (high == 0)
        _dl_printf ("%s=0x%x\n", label, low);
      else
        _dl_printf ("%s=0x%x%08x\n", label, high, low);
    }
}

/* Return true if ENV is an unfiltered environment variable.  */
static bool
unfiltered_envvar (const char *env, size_t *name_length)
{
  char *env_equal = strchr (env, '=');
  if (env_equal == NULL)
    {
      /* Always dump malformed entries.  */
      *name_length = strlen (env);
      return true;
    }
  size_t envname_length = env_equal - env;
  *name_length = envname_length;

  /* LC_ and LD_ variables.  */
  if (env[0] == 'L' && (env[1] == 'C' || env[1] == 'D')
      && env[2] == '_')
    return true;

  /* MALLOC_ variables.  */
  if (strncmp (env, "MALLOC_", strlen ("MALLOC_")) == 0)
    return true;

  static const char unfiltered[] =
    "DATEMSK\0"
    "GCONV_PATH\0"
    "GETCONF_DIR\0"
    "GETCONF_DIR\0"
    "GLIBC_TUNABLES\0"
    "GMON_OUTPUT_PREFIX\0"
    "HESIOD_CONFIG\0"
    "HES_DOMAIN\0"
    "HOSTALIASES\0"
    "I18NPATH\0"
    "IFS\0"
    "LANG\0"
    "LOCALDOMAIN\0"
    "LOCPATH\0"
    "MSGVERB\0"
    "NIS_DEFAULTS\0"
    "NIS_GROUP\0"
    "NIS_PATH\0"
    "NLSPATH\0"
    "PATH\0"
    "POSIXLY_CORRECT\0"
    "RESOLV_HOST_CONF\0"
    "RES_OPTIONS\0"
    "SEV_LEVEL\0"
    "TMPDIR\0"
    "TZ\0"
    "TZDIR\0"
    /* Two null bytes at the end to mark the end of the list via an
       empty substring.  */
    ;
  for (const char *candidate = unfiltered; *candidate != '\0'; )
    {
      size_t candidate_length = strlen (candidate);
      if (candidate_length == envname_length
          && memcmp (candidate, env, candidate_length) == 0)
        return true;
      candidate += candidate_length + 1;
    }

  return false;
}

/* Dump the process environment.  */
static void
print_environ (char **environ)
{
  unsigned int index = 0;
  for (char **envp = environ; *envp != NULL; ++envp)
    {
      char *env = *envp;
      size_t name_length;
      bool unfiltered = unfiltered_envvar (env, &name_length);
      _dl_printf ("env%s[0x%x]=",
                  unfiltered ? "" : "_filtered", index);
      if (unfiltered)
        _dl_diagnostics_print_string (env);
      else
        print_string_length (env, name_length);
      _dl_putc ('\n');
      ++index;
    }
}

/* Print configured paths and the built-in search path.  */
static void
print_paths (void)
{
  _dl_diagnostics_print_labeled_string ("path.prefix", PREFIX);
  _dl_diagnostics_print_labeled_string ("path.rtld", RTLD);
  _dl_diagnostics_print_labeled_string ("path.sysconfdir", SYSCONFDIR);

  unsigned int index = 0;
  static const char *system_dirs = SYSTEM_DIRS "\0";
  for (const char *e = system_dirs; *e != '\0'; )
    {
      size_t len = strlen (e);
      _dl_printf ("path.system_dirs[0x%x]=", index);
      print_string_length (e, len);
      _dl_putc ('\n');
      ++index;
      e += len + 1;
    }
}

/* Print information about the glibc version.  */
static void
print_version (void)
{
  _dl_diagnostics_print_labeled_string ("version.release", RELEASE);
  _dl_diagnostics_print_labeled_string ("version.version", VERSION);
}

void
_dl_print_diagnostics (char **environ)
{
#ifdef HAVE_DL_DISCOVER_OSVERSION
  _dl_diagnostics_print_labeled_value
    ("dl_discover_osversion", _dl_discover_osversion ());
#endif
  _dl_diagnostics_print_labeled_string ("dl_dst_lib", DL_DST_LIB);
  _dl_diagnostics_print_labeled_value ("dl_hwcap", GLRO (dl_hwcap));
  _dl_diagnostics_print_labeled_value ("dl_hwcap_important", HWCAP_IMPORTANT);
  _dl_diagnostics_print_labeled_value ("dl_hwcap2", GLRO (dl_hwcap2));
  _dl_diagnostics_print_labeled_string
    ("dl_hwcaps_subdirs", _dl_hwcaps_subdirs);
  _dl_diagnostics_print_labeled_value
    ("dl_hwcaps_subdirs_active", _dl_hwcaps_subdirs_active ());
  _dl_diagnostics_print_labeled_value ("dl_osversion", GLRO (dl_osversion));
  _dl_diagnostics_print_labeled_value ("dl_pagesize", GLRO (dl_pagesize));
  _dl_diagnostics_print_labeled_string ("dl_platform", GLRO (dl_platform));
  _dl_diagnostics_print_labeled_string
    ("dl_profile_output", GLRO (dl_profile_output));
  _dl_diagnostics_print_labeled_value
    ("dl_string_platform", _dl_string_platform ( GLRO (dl_platform)));

  _dl_diagnostics_print_labeled_string ("dso.ld", LD_SO);
  _dl_diagnostics_print_labeled_string ("dso.libc", LIBC_SO);

  print_environ (environ);
  print_paths ();
  print_version ();

  _dl_diagnostics_kernel ();
  _dl_diagnostics_cpu ();

  _exit (EXIT_SUCCESS);
}
