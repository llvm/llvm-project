/* Copyright (C) 2011-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gmail.com>, 2011.

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

#define E(name) E_(name, CLASS)
#define E_(name, cl) E__(name, cl)
#define E__(name, cl) name##cl
#define EW(type) EW_(Elf, CLASS, type)
#define EW_(e, w, t) EW__(e, w, _##t)
#define EW__(e, w, t) e##w##t

struct E(link_map)
{
  EW(Addr) l_addr;
  EW(Addr) l_name;
  EW(Addr) l_ld;
  EW(Addr) l_next;
  EW(Addr) l_prev;
  EW(Addr) l_real;
  Lmid_t l_ns;
  EW(Addr) l_libname;
};
#if CLASS == __ELF_NATIVE_CLASS
_Static_assert (offsetof (struct link_map, l_addr)
		== offsetof (struct E(link_map), l_addr), "l_addr");
_Static_assert (offsetof (struct link_map, l_name)
		== offsetof (struct E(link_map), l_name), "l_name");
_Static_assert (offsetof (struct link_map, l_next)
		== offsetof (struct E(link_map), l_next), "l_next");
#endif


struct E(libname_list)
{
  EW(Addr) name;
  EW(Addr) next;
};
#if CLASS == __ELF_NATIVE_CLASS
_Static_assert (offsetof (struct libname_list, name)
		== offsetof (struct E(libname_list), name), "name");
_Static_assert (offsetof (struct libname_list, next)
		== offsetof (struct E(libname_list), next), "next");
#endif

struct E(r_debug)
{
  int r_version;
#if CLASS == 64
  int pad;
#endif
  EW(Addr) r_map;
};
#if CLASS == __ELF_NATIVE_CLASS
_Static_assert (offsetof (struct r_debug, r_version)
		== offsetof (struct E(r_debug), r_version), "r_version");
_Static_assert (offsetof (struct r_debug, r_map)
		== offsetof (struct E(r_debug), r_map), "r_map");
#endif


static int

E(find_maps) (const char *exe, int memfd, pid_t pid, void *auxv,
	      size_t auxv_size)
{
  EW(Addr) phdr = 0;
  unsigned int phnum = 0;
  unsigned int phent = 0;

  EW(auxv_t) *auxvXX = (EW(auxv_t) *) auxv;
  for (int i = 0; i < auxv_size / sizeof (EW(auxv_t)); ++i)
    switch (auxvXX[i].a_type)
      {
      case AT_PHDR:
	phdr = auxvXX[i].a_un.a_val;
	break;
      case AT_PHNUM:
	phnum = auxvXX[i].a_un.a_val;
	break;
      case AT_PHENT:
	phent = auxvXX[i].a_un.a_val;
	break;
      default:
	break;
      }

  if (phdr == 0 || phnum == 0 || phent == 0)
    error (EXIT_FAILURE, 0, gettext ("cannot find program header of process"));

  EW(Phdr) *p = xmalloc (phnum * phent);
  if (pread (memfd, p, phnum * phent, phdr) != phnum * phent)
    error (EXIT_FAILURE, 0, gettext ("cannot read program header"));

  /* Determine the load offset.  We need this for interpreting the
     other program header entries so we do this in a separate loop.
     Fortunately it is the first time unless someone does something
     stupid when linking the application.  */
  EW(Addr) offset = 0;
  for (unsigned int i = 0; i < phnum; ++i)
    if (p[i].p_type == PT_PHDR)
      {
	offset = phdr - p[i].p_vaddr;
	break;
      }

  EW(Addr) list = 0;
  char *interp = NULL;
  for (unsigned int i = 0; i < phnum; ++i)
    if (p[i].p_type == PT_DYNAMIC)
      {
	EW(Dyn) *dyn = xmalloc (p[i].p_filesz);
	if (pread (memfd, dyn, p[i].p_filesz, offset + p[i].p_vaddr)
	    != p[i].p_filesz)
	  error (EXIT_FAILURE, 0, gettext ("cannot read dynamic section"));

	/* Search for the DT_DEBUG entry.  */
	for (unsigned int j = 0; j < p[i].p_filesz / sizeof (EW(Dyn)); ++j)
	  if (dyn[j].d_tag == DT_DEBUG && dyn[j].d_un.d_ptr != 0)
	    {
	      struct E(r_debug) r;
	      if (pread (memfd, &r, sizeof (r), dyn[j].d_un.d_ptr)
		  != sizeof (r))
		error (EXIT_FAILURE, 0, gettext ("cannot read r_debug"));

	      if (r.r_map != 0)
		{
		  list = r.r_map;
		  break;
		}
	    }

	free (dyn);
	break;
      }
    else if (p[i].p_type == PT_INTERP)
      {
	interp = xmalloc (p[i].p_filesz);
	if (pread (memfd, interp, p[i].p_filesz, offset + p[i].p_vaddr)
	    != p[i].p_filesz)
	  error (EXIT_FAILURE, 0, gettext ("cannot read program interpreter"));
      }

  if (list == 0)
    {
      if (interp == NULL)
	{
	  // XXX check whether the executable itself is the loader
	  exit (EXIT_FAILURE);
	}

      // XXX perhaps try finding ld.so and _r_debug in it
      exit (EXIT_FAILURE);
    }

  free (p);
  free (interp);

  /* Print the PID and program name first.  */
  printf ("%lu:\t%s\n", (unsigned long int) pid, exe);

  /* Iterate over the list of objects and print the information.  */
  struct scratch_buffer tmpbuf;
  scratch_buffer_init (&tmpbuf);
  int status = 0;
  do
    {
      struct E(link_map) m;
      if (pread (memfd, &m, sizeof (m), list) != sizeof (m))
	error (EXIT_FAILURE, 0, gettext ("cannot read link map"));

      EW(Addr) name_offset = m.l_name;
      while (1)
	{
	  ssize_t n = pread (memfd, tmpbuf.data, tmpbuf.length, name_offset);
	  if (n == -1)
	    error (EXIT_FAILURE, 0, gettext ("cannot read object name"));

	  if (memchr (tmpbuf.data, '\0', n) != NULL)
	    break;

	  if (!scratch_buffer_grow (&tmpbuf))
	    error (EXIT_FAILURE, 0,
		   gettext ("cannot allocate buffer for object name"));
	}

      /* The m.l_name and m.l_libname.name for loader linkmap points to same
	 values (since BZ#387 fix).  Trying to use l_libname name as the
	 shared object name might lead to an infinite loop (BZ#18035).  */

      /* Skip over the executable.  */
      if (((char *)tmpbuf.data)[0] != '\0')
	printf ("%s\n", (char *)tmpbuf.data);

      list = m.l_next;
    }
  while (list != 0);

  scratch_buffer_free (&tmpbuf);
  return status;
}


#undef CLASS
