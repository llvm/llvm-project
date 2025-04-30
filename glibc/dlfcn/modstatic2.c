#include <dlfcn.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gnu/lib-names.h>
#include <first-versions.h>

int test (FILE *out, int a);

int
test (FILE *out, int a)
{
  fputs ("in modstatic2.c (test)\n", out);

  void *handle = dlopen ("modstatic2-nonexistent.so", RTLD_LAZY);
  if (handle == NULL)
    fprintf (out, "nonexistent: %s\n", dlerror ());
  else
    exit (1);

  handle = dlopen ("modstatic2.so", RTLD_LAZY);
  if (handle == NULL)
    {
      fprintf (out, "%s\n", dlerror ());
      exit (1);
    }

  int (*test2) (FILE *, int);
  test2 = dlsym (handle, "test");
  if (test2 == NULL)
    {
      fprintf (out, "%s\n", dlerror ());
      exit (1);
    }
  if (test2 != test)
    {
      fprintf (out, "test %p != test2 %p\n", test, test2);
      exit (1);
    }

  Dl_info info;
  int res = dladdr (test2, &info);
  if (res == 0)
    {
      fputs ("dladdr returned 0\n", out);
      exit (1);
    }
  else
    {
      if (strstr (info.dli_fname, "modstatic2.so") == NULL
	  || strcmp (info.dli_sname, "test") != 0)
	{
	  fprintf (out, "fname %s sname %s\n", info.dli_fname, info.dli_sname);
	  exit (1);
	}
      if (info.dli_saddr != (void *) test2)
	{
	  fprintf (out, "saddr %p != test %p\n", info.dli_saddr, test2);
	  exit (1);
	}
    }

  ElfW(Sym) *sym;
  void *symp;
  res = dladdr1 (test2, &info, &symp, RTLD_DL_SYMENT);
  if (res == 0)
    {
      fputs ("dladdr1 returned 0\n", out);
      exit (1);
    }
  else
    {
      if (strstr (info.dli_fname, "modstatic2.so") == NULL
	  || strcmp (info.dli_sname, "test") != 0)
	{
	  fprintf (out, "fname %s sname %s\n", info.dli_fname, info.dli_sname);
	  exit (1);
	}
      if (info.dli_saddr != (void *) test2)
	{
	  fprintf (out, "saddr %p != test %p\n", info.dli_saddr, test2);
	  exit (1);
	}
      sym = symp;
      if (sym == NULL)
	{
	  fputs ("sym == NULL\n", out);
	  exit (1);
	}
      if (ELF32_ST_BIND (sym->st_info) != STB_GLOBAL
	  || ELF32_ST_VISIBILITY (sym->st_other) != STV_DEFAULT)
	{
	  fprintf (out, "bind %d visibility %d\n",
		   (int) ELF32_ST_BIND (sym->st_info),
		   (int) ELF32_ST_VISIBILITY (sym->st_other));
	  exit (1);
	}
    }

  Lmid_t lmid;
  res = dlinfo (handle, RTLD_DI_LMID, &lmid);
  if (res != 0)
    {
      fprintf (out, "dlinfo returned %d %s\n", res, dlerror ());
      exit (1);
    }
  else if (lmid != LM_ID_BASE)
    {
      fprintf (out, "lmid %d != %d\n", (int) lmid, (int) LM_ID_BASE);
      exit (1);
    }

  void *handle2 = dlopen (LIBDL_SO, RTLD_LAZY);
  if (handle2 == NULL)
    {
      fprintf (out, "libdl.so: %s\n", dlerror ());
      exit (1);
    }

  /* _exit is very unlikely to receive a second symbol version.  */
  void *exit_ptr = dlvsym (handle2, "_exit", FIRST_VERSION_libc__exit_STRING);
  if (exit_ptr == NULL)
    {
      fprintf (out, "dlvsym: %s\n", dlerror ());
      exit (1);
    }
  if (exit_ptr != dlsym (handle2, "_exit"))
    {
      fprintf (out, "dlvsym for _exit does not match dlsym\n");
      exit (1);
    }

  void *(*dlsymfn) (void *, const char *);
  dlsymfn = dlsym (handle2, "dlsym");
  if (dlsymfn == NULL)
    {
      fprintf (out, "dlsym \"dlsym\": %s\n", dlerror ());
      exit (1);
    }
  void *test3 = dlsymfn (handle, "test");
  if (test3 == NULL)
    {
      fprintf (out, "%s\n", dlerror ());
      exit (1);
    }
  else if (test3 != (void *) test2)
    {
      fprintf (out, "test2 %p != test3 %p\n", test2, test3);
      exit (1);
    }

  dlclose (handle2);
  dlclose (handle);

  handle = dlmopen (LM_ID_BASE, "modstatic2.so", RTLD_LAZY);
  if (handle == NULL)
    {
      fprintf (out, "%s\n", dlerror ());
      exit (1);
    }
  dlclose (handle);

  handle = dlmopen (LM_ID_NEWLM, "modstatic2.so", RTLD_LAZY);
  if (handle == NULL)
    fprintf (out, "LM_ID_NEWLM: %s\n", dlerror ());
  else
    {
      fputs ("LM_ID_NEWLM unexpectedly succeeded\n", out);
      exit (1);
    }

  handle = dlopen ("modstatic.so", RTLD_LAZY);
  if (handle == NULL)
    {
      fprintf (out, "%s\n", dlerror ());
      exit (1);
    }

  int (*test4) (int);
  test4 = dlsym (handle, "test");
  if (test4 == NULL)
    {
      fprintf (out, "%s\n", dlerror ());
      exit (1);
    }

  res = test4 (16);
  if (res != 16 + 16)
    {
      fprintf (out, "modstatic.so (test) returned %d\n", res);
      exit (1);
    }

  res = dladdr1 (test4, &info, &symp, RTLD_DL_SYMENT);
  if (res == 0)
    {
      fputs ("dladdr1 returned 0\n", out);
      exit (1);
    }
  else
    {
      if (strstr (info.dli_fname, "modstatic.so") == NULL
	  || strcmp (info.dli_sname, "test") != 0)
	{
	  fprintf (out, "fname %s sname %s\n", info.dli_fname, info.dli_sname);
	  exit (1);
	}
      if (info.dli_saddr != (void *) test4)
	{
	  fprintf (out, "saddr %p != test %p\n", info.dli_saddr, test4);
	  exit (1);
	}
      sym = symp;
      if (sym == NULL)
	{
	  fputs ("sym == NULL\n", out);
	  exit (1);
	}
      if (ELF32_ST_BIND (sym->st_info) != STB_GLOBAL
	  || ELF32_ST_VISIBILITY (sym->st_other) != STV_DEFAULT)
	{
	  fprintf (out, "bind %d visibility %d\n",
		   (int) ELF32_ST_BIND (sym->st_info),
		   (int) ELF32_ST_VISIBILITY (sym->st_other));
	  exit (1);
	}
    }

  dlclose (handle);

  fputs ("leaving modstatic2.c (test)\n", out);
  return a + a;
}
