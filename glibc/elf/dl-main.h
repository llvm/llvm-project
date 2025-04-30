/* Information collection during ld.so startup.
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

#ifndef _DL_MAIN
#define _DL_MAIN

#include <ldsodefs.h>
#include <limits.h>
#include <stdlib.h>

/* Length limits for names and paths, to protect the dynamic linker,
   particularly when __libc_enable_secure is active.  */
#ifdef NAME_MAX
# define SECURE_NAME_LIMIT NAME_MAX
#else
# define SECURE_NAME_LIMIT 255
#endif
#ifdef PATH_MAX
# define SECURE_PATH_LIMIT PATH_MAX
#else
# define SECURE_PATH_LIMIT 1024
#endif

/* Strings containing colon-separated lists of audit modules.  */
struct audit_list
{
  /* Array of strings containing colon-separated path lists.  Each
     audit module needs its own namespace, so pre-allocate the largest
     possible list.  */
  const char *audit_strings[DL_NNS];

  /* Number of entries added to audit_strings.  */
  size_t length;

  /* Index into the audit_strings array (for the iteration phase).  */
  size_t current_index;

  /* Tail of audit_strings[current_index] which still needs
     processing.  */
  const char *current_tail;

  /* Scratch buffer for returning a name which is part of the strings
     in audit_strings.  */
  char fname[SECURE_NAME_LIMIT];
};

/* This is a list of all the modes the dynamic loader can be in.  */
enum rtld_mode
  {
    rtld_mode_normal, rtld_mode_list, rtld_mode_verify, rtld_mode_trace,
    rtld_mode_list_tunables, rtld_mode_list_diagnostics, rtld_mode_help,
  };

/* Aggregated state information extracted from environment variables
   and the ld.so command line.  */
struct dl_main_state
{
  struct audit_list audit_list;

  /* The library search path.  */
  const char *library_path;

  /* Where library_path comes from.  LD_LIBRARY_PATH or --library-path.  */
  const char *library_path_source;

  /* The list preloaded objects from LD_PRELOAD.  */
  const char *preloadlist;

  /* The preload list passed as a command argument.  */
  const char *preloadarg;

  /* Additional glibc-hwcaps subdirectories to search first.
     Colon-separated list.  */
  const char *glibc_hwcaps_prepend;

  /* Mask for the internal glibc-hwcaps subdirectories.
     Colon-separated list.  */
  const char *glibc_hwcaps_mask;

  enum rtld_mode mode;

  /* True if any of the debugging options is enabled.  */
  bool any_debug;

  /* True if information about versions has to be printed.  */
  bool version_info;
};

/* Helper function to invoke _dl_init_paths with the right arguments
   from *STATE.  */
static inline void
call_init_paths (const struct dl_main_state *state)
{
  _dl_init_paths (state->library_path, state->library_path_source,
                  state->glibc_hwcaps_prepend, state->glibc_hwcaps_mask);
}

/* Print ld.so usage information and exit.  */
_Noreturn void _dl_usage (const char *argv0, const char *wrong_option)
  attribute_hidden;

/* Print ld.so version information and exit.  */
_Noreturn void _dl_version (void) attribute_hidden;

/* Print ld.so --help output and exit.  */
_Noreturn void _dl_help (const char *argv0, struct dl_main_state *state)
  attribute_hidden;

/* Print a diagnostics dump.  */
_Noreturn void _dl_print_diagnostics (char **environ) attribute_hidden;

#endif /* _DL_MAIN */
