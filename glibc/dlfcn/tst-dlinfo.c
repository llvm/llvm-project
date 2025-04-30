/* Test for dlinfo.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
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

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <error.h>

#define TEST_FUNCTION do_test ()

static int
do_test (void)
{
  int status = 0;

  void *handle = dlopen ("glreflib3.so", RTLD_NOW);
  if (handle == NULL)
    error (EXIT_FAILURE, 0, "cannot load: glreflib1.so: %s", dlerror ());

#define TRY(req, arg)							      \
  if (dlinfo (handle, req, arg) != 0)					      \
    {									      \
      printf ("dlinfo failed for %s: %s\n", #req, dlerror ());		      \
      status = 1;							      \
    }									      \
  else

  struct link_map *l;
  TRY (RTLD_DI_LINKMAP, &l)
    {
      if (l != handle)
	{
	  printf ("bogus link_map? %p != %p\n", l, handle);
	  status = 1;
	}
    }

  char origin[8192];		/* >= PATH_MAX, in theory */
  TRY (RTLD_DI_ORIGIN, origin)
    {
      printf ("origin: %s\n", origin);
    }

  Dl_serinfo counts;
  TRY (RTLD_DI_SERINFOSIZE, &counts)
    {
      Dl_serinfo *buf = alloca (counts.dls_size);
      buf->dls_cnt = counts.dls_cnt;
      buf->dls_size = counts.dls_size;
      printf ("%u library directories\n", buf->dls_cnt);
      TRY (RTLD_DI_SERINFO, buf)
	{
	  if (counts.dls_cnt != buf->dls_cnt)
	    {
	      printf ("??? became %u library directories\n", buf->dls_cnt);
	      status = 1;
	    }
	  for (unsigned int i = 0; i < buf->dls_cnt; ++i)
	    printf ("\t%#02x\t%s\n",
		    buf->dls_serpath[i].dls_flags,
		    buf->dls_serpath[i].dls_name);
	}
    }

  unsigned long int lmid = 0xdeadbeefUL;
  if (dlinfo (handle, RTLD_DI_LMID, &lmid) != 0)
    printf ("dlinfo refuses RTLD_DI_LMID: %s\n", dlerror ());
  else
    {
      printf ("dlinfo RTLD_DI_LMID worked? %#lx\n", lmid);
      status = lmid == 0xdeadbeefUL;
    }

#undef TRY
  dlclose (handle);

  return status;
}

#include "../test-skeleton.c"
