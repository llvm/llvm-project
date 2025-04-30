/* Map in a shared object's segments from the file.
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

#ifndef _DL_LOAD_H
#define _DL_LOAD_H	1

#include <link.h>
#include <sys/mman.h>


/* On some systems, no flag bits are given to specify file mapping.  */
#ifndef MAP_FILE
# define MAP_FILE       0
#endif

/* The right way to map in the shared library files is MAP_COPY, which
   makes a virtual copy of the data at the time of the mmap call; this
   guarantees the mapped pages will be consistent even if the file is
   overwritten.  Some losing VM systems like Linux's lack MAP_COPY.  All we
   get is MAP_PRIVATE, which copies each page when it is modified; this
   means if the file is overwritten, we may at some point get some pages
   from the new version after starting with pages from the old version.

   To make up for the lack and avoid the overwriting problem,
   what Linux does have is MAP_DENYWRITE.  This prevents anyone
   from modifying the file while we have it mapped.  */
#ifndef MAP_COPY
# ifdef MAP_DENYWRITE
#  define MAP_COPY      (MAP_PRIVATE | MAP_DENYWRITE)
# else
#  define MAP_COPY      MAP_PRIVATE
# endif
#endif

/* Some systems link their relocatable objects for another base address
   than 0.  We want to know the base address for these such that we can
   subtract this address from the segment addresses during mapping.
   This results in a more efficient address space usage.  Defaults to
   zero for almost all systems.  */
#ifndef MAP_BASE_ADDR
# define MAP_BASE_ADDR(l)       0
#endif


/* Handle situations where we have a preferred location in memory for
   the shared objects.  */
#ifdef ELF_PREFERRED_ADDRESS_DATA
ELF_PREFERRED_ADDRESS_DATA;
#endif
#ifndef ELF_PREFERRED_ADDRESS
# define ELF_PREFERRED_ADDRESS(loader, maplength, mapstartpref) (mapstartpref)
#endif
#ifndef ELF_FIXED_ADDRESS
# define ELF_FIXED_ADDRESS(loader, mapstart) ((void) 0)
#endif


/* This structure describes one PT_LOAD command.
   Its details have been expanded out and converted.  */
struct loadcmd
{
  ElfW(Addr) mapstart, mapend, dataend, allocend;
  ElfW(Off) mapoff;
  int prot;                             /* PROT_* bits.  */
};


/* This is a subroutine of _dl_map_segments.  It should be called for each
   load command, some time after L->l_addr has been set correctly.  It is
   responsible for setting up the l_text_end and l_phdr fields.  */
static __always_inline void
_dl_postprocess_loadcmd (struct link_map *l, const ElfW(Ehdr) *header,
                         const struct loadcmd *c)
{
  if (c->prot & PROT_EXEC)
    l->l_text_end = l->l_addr + c->mapend;

  if (l->l_phdr == 0
      && c->mapoff <= header->e_phoff
      && ((size_t) (c->mapend - c->mapstart + c->mapoff)
          >= header->e_phoff + header->e_phnum * sizeof (ElfW(Phdr))))
    /* Found the program header in this segment.  */
    l->l_phdr = (void *) (uintptr_t) (c->mapstart + header->e_phoff
                                      - c->mapoff);
}


/* This is a subroutine of _dl_map_object_from_fd.  It is responsible
   for filling in several fields in *L: l_map_start, l_map_end, l_addr,
   l_contiguous, l_text_end, l_phdr.  On successful return, all the
   segments are mapped (or copied, or whatever) from the file into their
   final places in the address space, with the correct page permissions,
   and any bss-like regions already zeroed.  It returns a null pointer
   on success, or an error message string (to be translated) on error
   (having also set errno).

   The file <dl-map-segments.h> defines this function.  The canonical
   implementation in elf/dl-map-segments.h might be replaced by a sysdeps
   version.  */
static const char *_dl_map_segments (struct link_map *l, int fd,
                                     const ElfW(Ehdr) *header, int type,
                                     const struct loadcmd loadcmds[],
                                     size_t nloadcmds,
                                     const size_t maplength,
                                     bool has_holes,
                                     struct link_map *loader);

/* All the error message strings _dl_map_segments might return are
   listed here so that different implementations in different sysdeps
   dl-map-segments.h files all use consistent strings that are
   guaranteed to have translations.  */
#define DL_MAP_SEGMENTS_ERROR_MAP_SEGMENT \
  N_("failed to map segment from shared object")
#define DL_MAP_SEGMENTS_ERROR_MPROTECT \
  N_("cannot change memory protections")
#define DL_MAP_SEGMENTS_ERROR_MAP_ZERO_FILL \
  N_("cannot map zero-fill pages")

extern typeof (&__mmap) volatile __mmap_hook;
extern typeof (&__mprotect) volatile __mprotect_hook;

#endif	/* dl-load.h */
