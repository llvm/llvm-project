/* Bug 11941: Improper assert map->l_init_called in dlclose.
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

/* This is the primary DSO that is loaded by the appliation.  This DSO
   then loads a plugin with RTLD_NODELETE.  This plugin depends on this
   DSO.  This dependency chain means that at application shutdown the
   plugin will be destructed first.  Thus by the time this DSO is
   destructed we will be calling dlclose on an object that has already
   been destructed.  It is allowed to call dlclose in this way and
   should not assert.  */
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

/* Plugin to load.  */
static void *plugin_lib = NULL;
/* Plugin function.  */
static void (*plugin_func) (void);
#define LIB_PLUGIN "tst-nodelete-dlclose-plugin.so"

/* This function is never called but the plugin references it.
   We do this to avoid any future --as-needed from removing the
   plugin's DT_NEEDED on this DSO (required for the test).  */
void
primary_reference (void)
{
  printf ("INFO: Called primary_reference function.\n");
}

void
primary (void)
{
  char *error;

  plugin_lib = dlopen (LIB_PLUGIN, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
  if (plugin_lib == NULL)
    {
      printf ("ERROR: Unable to load plugin library.\n");
      exit (EXIT_FAILURE);
    }
  dlerror ();

  plugin_func = (void (*) (void)) dlsym (plugin_lib, "plugin_func");
  error = dlerror ();
  if (error != NULL)
    {
      printf ("ERROR: Unable to find symbol with error \"%s\".",
	      error);
      exit (EXIT_FAILURE);
    }

  return;
}

__attribute__ ((destructor))
static void
primary_dtor (void)
{
  int ret;

  printf ("INFO: Calling primary destructor.\n");

  /* The destructor runs in the test driver also, which
     hasn't called primary, in that case do nothing.  */
  if (plugin_lib == NULL)
    return;

  ret = dlclose (plugin_lib);
  if (ret != 0)
    {
      printf ("ERROR: Calling dlclose failed with \"%s\"\n",
	      dlerror ());
      exit (EXIT_FAILURE);
    }
}
