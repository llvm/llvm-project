/* AArch64 BTI functions.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.

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

#include <unistd.h>
#include <errno.h>
#include <libintl.h>
#include <ldsodefs.h>
#include <sys/mman.h>

/* See elf/dl-load.h.  */
#ifndef MAP_COPY
# define MAP_COPY (MAP_PRIVATE | MAP_DENYWRITE)
#endif

/* Enable BTI protection for MAP.  */

void
_dl_bti_protect (struct link_map *map, int fd)
{
  const size_t pagesz = GLRO(dl_pagesize);
  const ElfW(Phdr) *phdr;

  for (phdr = map->l_phdr; phdr < &map->l_phdr[map->l_phnum]; ++phdr)
    if (phdr->p_type == PT_LOAD && (phdr->p_flags & PF_X))
      {
	size_t vstart = ALIGN_DOWN (phdr->p_vaddr, pagesz);
	size_t vend = ALIGN_UP (phdr->p_vaddr + phdr->p_filesz, pagesz);
	off_t off = ALIGN_DOWN (phdr->p_offset, pagesz);
	void *start = (void *) (vstart + map->l_addr);
	size_t len = vend - vstart;

	unsigned prot = PROT_EXEC | PROT_BTI;
	if (phdr->p_flags & PF_R)
	  prot |= PROT_READ;
	if (phdr->p_flags & PF_W)
	  prot |= PROT_WRITE;

	if (fd == -1)
	  /* Ignore failures for kernel mapped binaries.  */
	  __mprotect (start, len, prot);
	else
	  map->l_mach.bti_fail = __mmap (start, len, prot,
					 MAP_FIXED|MAP_COPY|MAP_FILE,
					 fd, off) == MAP_FAILED;
      }
}


static void
bti_failed (struct link_map *l, const char *program)
{
  if (program)
    _dl_fatal_printf ("%s: %s: failed to turn on BTI protection\n",
		      program, l->l_name);
  else
    /* Note: the errno value is not available any more.  */
    _dl_signal_error (0, l->l_name, "dlopen",
		      N_("failed to turn on BTI protection"));
}


/* Enable BTI for L and its dependencies.  */

void
_dl_bti_check (struct link_map *l, const char *program)
{
  if (!GLRO(dl_aarch64_cpu_features).bti)
    return;

  if (l->l_mach.bti_fail)
    bti_failed (l, program);

  unsigned int i = l->l_searchlist.r_nlist;
  while (i-- > 0)
    {
      struct link_map *dep = l->l_initfini[i];
      if (dep->l_mach.bti_fail)
	bti_failed (dep, program);
    }
}
