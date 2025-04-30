/* Print usage information and help for ld.so.
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

#include <dl-cache.h>
#include <dl-main.h>
#include <ldsodefs.h>
#include <unistd.h>
#include "version.h"

#include <dl-procinfo.h>
#include <dl-hwcaps.h>

void
_dl_usage (const char *argv0, const char *wrong_option)
{
  if (wrong_option != NULL)
    _dl_error_printf ("%s: unrecognized option '%s'\n", argv0, wrong_option);
  else
    _dl_error_printf ("%s: missing program name\n", argv0);
  _dl_error_printf ("Try '%s --help' for more information.\n", argv0);
  _exit (EXIT_FAILURE);
}

void
_dl_version (void)
{
  _dl_printf ("\
ld.so " PKGVERSION RELEASE " release version " VERSION ".\n\
Copyright (C) 2021 Free Software Foundation, Inc.\n\
This is free software; see the source for copying conditions.\n\
There is NO warranty; not even for MERCHANTABILITY or FITNESS FOR A\n\
PARTICULAR PURPOSE.\n\
");
  _exit (EXIT_SUCCESS);
}

/* Print part of the library search path (from a single source).  */
static void
print_search_path_for_help_1 (struct r_search_path_elem **list)
{
  if (list == NULL || list == (void *) -1)
    /* Path is missing or marked as inactive.  */
    return;

  for (; *list != NULL; ++list)
    {
      _dl_write (STDOUT_FILENO, "  ", 2);
      const char *name = (*list)->dirname;
      size_t namelen = (*list)->dirnamelen;
      if (namelen == 0)
        {
          /* The empty string denotes the current directory.  */
          name = ".";
          namelen = 1;
        }
      else if (namelen > 1)
        /* Remove the trailing slash.  */
        --namelen;
      _dl_write (STDOUT_FILENO, name, namelen);
      _dl_printf (" (%s)\n", (*list)->what);
    }
}

/* Prints the library search path.  See _dl_init_paths in dl-load.c
   how this information is populated.  */
static void
print_search_path_for_help (struct dl_main_state *state)
{
  if (__rtld_search_dirs.dirs == NULL)
    /* The run-time search paths have not yet been initialized.  */
    call_init_paths (state);

  _dl_printf ("\nShared library search path:\n");

  /* The print order should reflect the processing in
     _dl_map_object.  */

  struct link_map *map = GL(dl_ns)[LM_ID_BASE]._ns_loaded;
  if (map != NULL)
    print_search_path_for_help_1 (map->l_rpath_dirs.dirs);

  print_search_path_for_help_1 (__rtld_env_path_list.dirs);

  if (map != NULL)
    print_search_path_for_help_1 (map->l_runpath_dirs.dirs);

  _dl_printf ("  (libraries located via %s)\n", LD_SO_CACHE);

  print_search_path_for_help_1 (__rtld_search_dirs.dirs);
}

/* Helper function for printing flags associated with a HWCAP name.  */
static void
print_hwcap_1 (bool *first, bool active, const char *label)
{
  if (active)
    {
      if (*first)
        {
          _dl_printf (" (");
          *first = false;
        }
      else
        _dl_printf (", ");
      _dl_printf ("%s", label);
    }
}

/* Called after a series of print_hwcap_1 calls to emit the line
   terminator.  */
static void
print_hwcap_1_finish (bool *first)
{
  if (*first)
    _dl_printf ("\n");
  else
    _dl_printf (")\n");
}

/* Print the header for print_hwcaps_subdirectories.  */
static void
print_hwcaps_subdirectories_header (bool *nothing_printed)
{
  if (*nothing_printed)
    {
      _dl_printf ("\n\
Subdirectories of glibc-hwcaps directories, in priority order:\n");
      *nothing_printed = false;
    }
}

/* Print the HWCAP name itself, indented.  */
static void
print_hwcaps_subdirectories_name (const struct dl_hwcaps_split *split)
{
  _dl_write (STDOUT_FILENO, "  ", 2);
  _dl_write (STDOUT_FILENO, split->segment, split->length);
}

/* Print the list of recognized glibc-hwcaps subdirectories.  */
static void
print_hwcaps_subdirectories (const struct dl_main_state *state)
{
  bool nothing_printed = true;
  struct dl_hwcaps_split split;

  /* The prepended glibc-hwcaps subdirectories.  */
  _dl_hwcaps_split_init (&split, state->glibc_hwcaps_prepend);
  while (_dl_hwcaps_split (&split))
    {
      print_hwcaps_subdirectories_header (&nothing_printed);
      print_hwcaps_subdirectories_name (&split);
      bool first = true;
      print_hwcap_1 (&first, true, "searched");
      print_hwcap_1_finish (&first);
    }

  /* The built-in glibc-hwcaps subdirectories.  Do the filtering
     manually, so that more precise diagnostics are possible.  */
  uint32_t mask = _dl_hwcaps_subdirs_active ();
  _dl_hwcaps_split_init (&split, _dl_hwcaps_subdirs);
  while (_dl_hwcaps_split (&split))
    {
      print_hwcaps_subdirectories_header (&nothing_printed);
      print_hwcaps_subdirectories_name (&split);
      bool first = true;
      print_hwcap_1 (&first, mask & 1, "supported");
      bool listed = _dl_hwcaps_contains (state->glibc_hwcaps_mask,
                                         split.segment, split.length);
      print_hwcap_1 (&first, !listed, "masked");
      print_hwcap_1 (&first, (mask & 1) && listed, "searched");
      print_hwcap_1_finish (&first);
      mask >>= 1;
    }

  if (nothing_printed)
    _dl_printf ("\n\
No subdirectories of glibc-hwcaps directories are searched.\n");
}

/* Write a list of hwcap subdirectories to standard output.  See
 _dl_important_hwcaps in dl-hwcaps.c.  */
static void
print_legacy_hwcap_directories (void)
{
  _dl_printf ("\n\
Legacy HWCAP subdirectories under library search path directories:\n");

  const char *platform = GLRO (dl_platform);
  if (platform != NULL)
    _dl_printf ("  %s (AT_PLATFORM; supported, searched)\n", platform);

  _dl_printf ("  tls (supported, searched)\n");

  uint64_t hwcap_mask = GET_HWCAP_MASK();
  uint64_t searched = GLRO (dl_hwcap) & hwcap_mask;
  for (int n = 63; n >= 0; --n)
    {
      uint64_t bit = 1ULL << n;
      if (HWCAP_IMPORTANT & bit)
        {
          _dl_printf ("  %s", _dl_hwcap_string (n));
          bool first = true;
          print_hwcap_1 (&first, GLRO (dl_hwcap) & bit, "supported");
          print_hwcap_1 (&first, !(hwcap_mask & bit), "masked");
          print_hwcap_1 (&first, searched & bit, "searched");
          print_hwcap_1_finish (&first);
        }
    }
}

void
_dl_help (const char *argv0, struct dl_main_state *state)
{
  _dl_printf ("\
Usage: %s [OPTION]... EXECUTABLE-FILE [ARGS-FOR-PROGRAM...]\n\
You have invoked 'ld.so', the program interpreter for dynamically-linked\n\
ELF programs.  Usually, the program interpreter is invoked automatically\n\
when a dynamically-linked executable is started.\n\
\n\
You may invoke the program interpreter program directly from the command\n\
line to load and run an ELF executable file; this is like executing that\n\
file itself, but always uses the program interpreter you invoked,\n\
instead of the program interpreter specified in the executable file you\n\
run.  Invoking the program interpreter directly provides access to\n\
additional diagnostics, and changing the dynamic linker behavior without\n\
setting environment variables (which would be inherited by subprocesses).\n\
\n\
  --list                list all dependencies and how they are resolved\n\
  --verify              verify that given object really is a dynamically linked\n\
                        object we can handle\n\
  --inhibit-cache       Do not use " LD_SO_CACHE "\n\
  --library-path PATH   use given PATH instead of content of the environment\n\
                        variable LD_LIBRARY_PATH\n\
  --glibc-hwcaps-prepend LIST\n\
                        search glibc-hwcaps subdirectories in LIST\n\
  --glibc-hwcaps-mask LIST\n\
                        only search built-in subdirectories if in LIST\n\
  --inhibit-rpath LIST  ignore RUNPATH and RPATH information in object names\n\
                        in LIST\n\
  --audit LIST          use objects named in LIST as auditors\n\
  --preload LIST        preload objects named in LIST\n\
  --argv0 STRING        set argv[0] to STRING before running\n"
#if HAVE_TUNABLES
"\
  --list-tunables       list all tunables with minimum and maximum values\n"
#endif
"\
  --list-diagnostics    list diagnostics information\n\
  --help                display this help and exit\n\
  --version             output version information and exit\n\
\n\
This program interpreter self-identifies as: " RTLD "\n\
",
              argv0);
  print_search_path_for_help (state);
  print_hwcaps_subdirectories (state);
  print_legacy_hwcap_directories ();
  _exit (EXIT_SUCCESS);
}
