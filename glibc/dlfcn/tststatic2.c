#include <dlfcn.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gnu/lib-names.h>
#include <first-versions.h>

static int
do_test (void)
{
  void *handle = dlopen ("modstatic2-nonexistent.so", RTLD_LAZY);
  if (handle == NULL)
    printf ("nonexistent: %s\n", dlerror ());
  else
    exit (1);

  handle = dlopen ("modstatic2.so", RTLD_LAZY);
  if (handle == NULL)
    {
      printf ("%s\n", dlerror ());
      exit (1);
    }

  int (*test) (FILE *, int);
  test = dlsym (handle, "test");
  if (test == NULL)
    {
      printf ("%s\n", dlerror ());
      exit (1);
    }

  Dl_info info;
  int res = dladdr (test, &info);
  if (res == 0)
    {
      puts ("dladdr returned 0");
      exit (1);
    }
  else
    {
      if (strstr (info.dli_fname, "modstatic2.so") == NULL
	  || strcmp (info.dli_sname, "test") != 0)
	{
	  printf ("fname %s sname %s\n", info.dli_fname, info.dli_sname);
	  exit (1);
	}
      if (info.dli_saddr != (void *) test)
	{
	  printf ("saddr %p != test %p\n", info.dli_saddr, test);
	  exit (1);
	}
    }

  ElfW(Sym) *sym;
  void *symp;
  res = dladdr1 (test, &info, &symp, RTLD_DL_SYMENT);
  if (res == 0)
    {
      puts ("dladdr1 returned 0");
      exit (1);
    }
  else
    {
      if (strstr (info.dli_fname, "modstatic2.so") == NULL
	  || strcmp (info.dli_sname, "test") != 0)
	{
	  printf ("fname %s sname %s\n", info.dli_fname, info.dli_sname);
	  exit (1);
	}
      if (info.dli_saddr != (void *) test)
	{
	  printf ("saddr %p != test %p\n", info.dli_saddr, test);
	  exit (1);
	}
      sym = symp;
      if (sym == NULL)
	{
	  puts ("sym == NULL\n");
	  exit (1);
	}
      if (ELF32_ST_BIND (sym->st_info) != STB_GLOBAL
	  || ELF32_ST_VISIBILITY (sym->st_other) != STV_DEFAULT)
	{
	  printf ("bind %d visibility %d\n",
		  (int) ELF32_ST_BIND (sym->st_info),
		  (int) ELF32_ST_VISIBILITY (sym->st_other));
	  exit (1);
	}
    }

  Lmid_t lmid;
  res = dlinfo (handle, RTLD_DI_LMID, &lmid);
  if (res != 0)
    {
      printf ("dlinfo returned %d %s\n", res, dlerror ());
      exit (1);
    }
  else if (lmid != LM_ID_BASE)
    {
      printf ("lmid %d != %d\n", (int) lmid, (int) LM_ID_BASE);
      exit (1);
    }

  res = test (stdout, 2);
  if (res != 4)
    {
      printf ("Got %i, expected 4\n", res);
      exit (1);
    }

  void *handle2 = dlopen (LIBDL_SO, RTLD_LAZY);
  if (handle2 == NULL)
    {
      printf ("libdl.so: %s\n", dlerror ());
      exit (1);
    }

  /* _exit is very unlikely to receive a second symbol version.  */
  void *exit_ptr = dlvsym (handle2, "_exit", FIRST_VERSION_libc__exit_STRING);
  if (exit_ptr == NULL)
    {
      printf ("dlvsym: %s\n", dlerror ());
      exit (1);
    }
  if (exit_ptr != dlsym (handle2, "_exit"))
    {
      printf ("dlvsym for _exit does not match dlsym\n");
      exit (1);
    }

  void *(*dlsymfn) (void *, const char *);
  dlsymfn = dlsym (handle2, "dlsym");
  if (dlsymfn == NULL)
    {
      printf ("dlsym \"dlsym\": %s\n", dlerror ());
      exit (1);
    }
  void *test2 = dlsymfn (handle, "test");
  if (test2 == NULL)
    {
      printf ("%s\n", dlerror ());
      exit (1);
    }
  else if (test2 != (void *) test)
    {
      printf ("test %p != test2 %p\n", test, test2);
      exit (1);
    }

  dlclose (handle2);
  dlclose (handle);

  handle = dlmopen (LM_ID_BASE, "modstatic2.so", RTLD_LAZY);
  if (handle == NULL)
    {
      printf ("%s\n", dlerror ());
      exit (1);
    }
  dlclose (handle);

  handle = dlmopen (LM_ID_NEWLM, "modstatic2.so", RTLD_LAZY);
  if (handle == NULL)
    printf ("LM_ID_NEWLM: %s\n", dlerror ());
  else
    {
      puts ("LM_ID_NEWLM unexpectedly succeeded");
      exit (1);
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
