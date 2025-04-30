/* Test dtv setup if entries don't have monotone increasing generation.
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
   <http://www.gnu.org/licenses/>.  */

#include <array_length.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <support/check.h>
#include <support/support.h>
#include <support/test-driver.h>
#include <support/xdlfcn.h>
#include <support/xthread.h>

#define NMOD 100
static void *mod[NMOD];

static void
load_fail (void)
{
  /* Expected to fail because of a missing symbol.  */
  void *m = dlopen ("tst-tls20mod-bad.so", RTLD_NOW);
  if (m != NULL)
    FAIL_EXIT1 ("dlopen of tst-tls20mod-bad.so succeeded\n");
}

static void
load_mod (int i)
{
  char *buf = xasprintf ("tst-tls-manydynamic%02dmod.so", i);
  mod[i] = xdlopen (buf, RTLD_LAZY);
  free (buf);
}

static void
unload_mod (int i)
{
  if (mod[i] != NULL)
    xdlclose (mod[i]);
  mod[i] = NULL;
}

static void
access (int i)
{
  char *buf = xasprintf ("tls_global_%02d", i);
  dlerror ();
  int *p = dlsym (mod[i], buf);
  if (test_verbose)
    printf ("mod[%d]: &tls = %p\n", i, p);
  if (p == NULL)
    FAIL_EXIT1 ("dlsym failed: %s\n", dlerror ());
  TEST_COMPARE (*p, 0);
  ++*p;
  free (buf);
}

static void
access_mod (const char *modname, void *mod, int i)
{
  char *modsym = xasprintf ("tls_global_%d", i);
  dlerror ();
  int *p = dlsym (mod, modsym);
  if (test_verbose)
    printf ("%s: &tls = %p\n", modname, p);
  if (p == NULL)
    FAIL_EXIT1 ("dlsym failed: %s\n", dlerror ());
  TEST_COMPARE (*p, 0);
  ++*p;
  free (modsym);
}

static void
access_dep (int i)
{
  char *modname = xasprintf ("tst-tls-manydynamic%dmod-dep.so", i);
  void *moddep = xdlopen (modname, RTLD_LAZY);
  access_mod (modname, moddep, i);
  free (modname);
  xdlclose (moddep);
}

struct start_args
{
  const char *modname;
  void *mod;
  int modi;
  int ndeps;
  const int *deps;
};

static void *
start (void *a)
{
  struct start_args *args = a;

  for (int i = 0; i < NMOD; i++)
    if (mod[i] != NULL)
      access (i);

  if (args != NULL)
    {
      access_mod (args->modname, args->mod, args->modi);
      for (int n = 0; n < args->ndeps; n++)
	access_dep (args->deps[n]);
    }

  return 0;
}

/* This test gaps with shared libraries with dynamic TLS that has no
   dependencies.  The DTV gap is set with by trying to load an invalid
   module, the entry should be used on the dlopen.  */
static void
do_test_no_depedency (void)
{
  for (int i = 0; i < NMOD; i++)
    {
      load_mod (i);
      /* Bump the generation of mod[0] without using new dtv slot.  */
      unload_mod (0);
      load_fail (); /* Ensure GL(dl_tls_dtv_gaps) is true: see bug 27135.  */
      load_mod (0);
      /* Access TLS in all loaded modules.  */
      pthread_t t = xpthread_create (0, start, 0);
      xpthread_join (t);
    }
  for (int i = 0; i < NMOD; i++)
    unload_mod (i);
}

/* The following test check DTV gaps handling with shared libraries that has
   dependencies.  It defines 5 different sets:

   1. Single dependency:
      mod0 -> mod1
   2. Double dependency:
      mod2 -> [mod3,mod4]
   3. Double dependency with each dependency depent of another module:
      mod5 -> [mod6,mod7] -> mod8
   4. Long chain with one double dependency in the middle:
      mod9 -> [mod10, mod11] -> mod12 -> mod13
   5. Long chain with two double depedencies in the middle:
      mod14 -> mod15 -> [mod16, mod17]
      mod15 -> [mod18, mod19]

   This does not cover all the possible gaps and configuration, but it
   should check if different dynamic shared sets are placed correctly in
   different gaps configurations.  */

static int
nmodules (uint32_t v)
{
  unsigned int r = 0;
  while (v >>= 1)
    r++;
  return r + 1;
}

static inline bool
is_mod_set (uint32_t g, uint32_t n)
{
  return (1U << (n - 1)) & g;
}

static void
print_gap (uint32_t g)
{
  if (!test_verbose)
    return;
  printf ("gap: ");
  int nmods = nmodules (g);
  for (int n = 1; n <= nmods; n++)
    printf ("%c", ((1 << (n - 1)) & g) == 0 ? 'G' : 'M');
  printf ("\n");
}

static void
do_test_dependency (void)
{
  /* Maps the module and its dependencies, use thread to access the TLS on
     each loaded module.  */
  static const int tlsmanydeps0[] = { 1 };
  static const int tlsmanydeps1[] = { 3, 4 };
  static const int tlsmanydeps2[] = { 6, 7, 8 };
  static const int tlsmanydeps3[] = { 10, 11, 12 };
  static const int tlsmanydeps4[] = { 15, 16, 17, 18, 19 };
  static const struct tlsmanydeps_t
  {
    int modi;
    int ndeps;
    const int *deps;
  } tlsmanydeps[] =
  {
    {  0, array_length (tlsmanydeps0), tlsmanydeps0 },
    {  2, array_length (tlsmanydeps1), tlsmanydeps1 },
    {  5, array_length (tlsmanydeps2), tlsmanydeps2 },
    {  9, array_length (tlsmanydeps3), tlsmanydeps3 },
    { 14, array_length (tlsmanydeps4), tlsmanydeps4 },
  };

  /* The gap configuration is defined as a bitmap: the bit set represents a
     loaded module prior the tests execution, while a bit unsed is a module
     unloaded.  Not all permtation will show gaps, but it is simpler than
     define each one independently.  */
  for (uint32_t g = 0; g < 64; g++)
    {
      print_gap (g);
      int nmods = nmodules (g);

      int mods[nmods];
      /* We use '0' as indication for a gap, to avoid the dlclose on iteration
	 cleanup.  */
      for (int n = 1; n < nmods; n++)
	{
	  load_mod (n);
	   mods[n] = n;
	}
      for (int n = 1; n < nmods; n++)
	{
	  if (!is_mod_set (g, n))
	    {
	      unload_mod (n);
	      mods[n] = 0;
	    }
	}

      for (int t = 0; t < array_length (tlsmanydeps); t++)
	{
	  char *moddepname = xasprintf ("tst-tls-manydynamic%dmod-dep.so",
					tlsmanydeps[t].modi);
	  void *moddep = xdlopen (moddepname, RTLD_LAZY);

	  /* Access TLS in all loaded modules.  */
	  struct start_args args =
	    {
	      moddepname,
	      moddep,
	      tlsmanydeps[t].modi,
	      tlsmanydeps[t].ndeps,
	      tlsmanydeps[t].deps
	    };
	  pthread_t t = xpthread_create (0, start, &args);
	  xpthread_join (t);

	  free (moddepname);
	  xdlclose (moddep);
	}

      for (int n = 1; n <= nmods; n++)
	if (mods[n] != 0)
	  unload_mod (n);
    }
}

/* The following test check DTV gaps handling with shared libraries that has
   invalid dependencies.  It defines 5 different sets:

   1. Single dependency:
      mod0 -> invalid
   2. Double dependency:
      mod1 -> [mod2,invalid]
   3. Double dependency with each dependency depent of another module:
      mod3 -> [mod4,mod5] -> invalid
   4. Long chain with one double dependency in the middle:
      mod6 -> [mod7, mod8] -> mod12 -> invalid
   5. Long chain with two double depedencies in the middle:
      mod10 -> mod11 -> [mod12, mod13]
      mod12 -> [mod14, invalid]

   This does not cover all the possible gaps and configuration, but it
   should check if different dynamic shared sets are placed correctly in
   different gaps configurations.  */

static void
do_test_invalid_dependency (bool bind_now)
{
  static const int tlsmanydeps[] = { 0, 1, 3, 6, 10 };

  /* The gap configuration is defined as a bitmap: the bit set represents a
     loaded module prior the tests execution, while a bit unsed is a module
     unloaded.  Not all permtation will show gaps, but it is simpler than
     define each one independently.  */
  for (uint32_t g = 0; g < 64; g++)
    {
      print_gap (g);
      int nmods = nmodules (g);

      int mods[nmods];
      /* We use '0' as indication for a gap, to avoid the dlclose on iteration
	 cleanup.  */
      for (int n = 1; n < nmods; n++)
	{
	  load_mod (n);
	   mods[n] = n;
	}
      for (int n = 1; n < nmods; n++)
	{
	  if (!is_mod_set (g, n))
	    {
	      unload_mod (n);
	      mods[n] = 0;
	    }
	}

      for (int t = 0; t < array_length (tlsmanydeps); t++)
	{
	  char *moddepname = xasprintf ("tst-tls-manydynamic%dmod-dep-bad.so",
					tlsmanydeps[t]);
	  void *moddep;
	  if (bind_now)
	    {
	      moddep = dlopen (moddepname, RTLD_NOW);
	      TEST_VERIFY (moddep == 0);
	    }
	  else
	    moddep = dlopen (moddepname, RTLD_LAZY);

	  /* Access TLS in all loaded modules.  */
	  pthread_t t = xpthread_create (0, start, NULL);
	  xpthread_join (t);

	  free (moddepname);
	  if (!bind_now)
	    xdlclose (moddep);
	}

      for (int n = 1; n <= nmods; n++)
	if (mods[n] != 0)
	  unload_mod (n);
    }
}

static int
do_test (void)
{
  do_test_no_depedency ();
  do_test_dependency ();
  do_test_invalid_dependency (true);
  do_test_invalid_dependency (false);

  return 0;
}

#include <support/test-driver.c>
