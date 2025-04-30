/* Manage function descriptors.  Generic version.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library; if not, write to the Free
   Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
   02111-1307 USA.  */

#include <libintl.h>
#include <unistd.h>
#include <string.h>
#include <sys/param.h>
#include <sys/mman.h>
#include <link.h>
#include <ldsodefs.h>
#include <elf/dynamic-link.h>
#include <dl-fptr.h>
#include <dl-unmap-segments.h>
#include <atomic.h>
#include <libc-pointer-arith.h>

#ifndef ELF_MACHINE_BOOT_FPTR_TABLE_LEN
/* ELF_MACHINE_BOOT_FPTR_TABLE_LEN should be greater than the number of
   dynamic symbols in ld.so.  */
# define ELF_MACHINE_BOOT_FPTR_TABLE_LEN 256
#endif

#ifndef ELF_MACHINE_LOAD_ADDRESS
# error "ELF_MACHINE_LOAD_ADDRESS is not defined."
#endif

#ifndef COMPARE_AND_SWAP
# define COMPARE_AND_SWAP(ptr, old, new) \
  (catomic_compare_and_exchange_bool_acq (ptr, new, old) == 0)
#endif

ElfW(Addr) _dl_boot_fptr_table [ELF_MACHINE_BOOT_FPTR_TABLE_LEN];

static struct local
  {
    struct fdesc_table *root;
    struct fdesc *free_list;
    unsigned int npages;		/* # of pages to allocate */
    /* the next to members MUST be consecutive! */
    struct fdesc_table boot_table;
    struct fdesc boot_fdescs[1024];
  }
local =
  {
#ifdef SHARED
    /* Address of .boot_table is not known until runtime.  */
    .root = 0,
#else
    .root = &local.boot_table,
#endif
    .npages = 2,
    .boot_table =
      {
	.len = sizeof (local.boot_fdescs) / sizeof (local.boot_fdescs[0]),
	.first_unused = 0
      }
  };

/* Create a new fdesc table and return a pointer to the first fdesc
   entry.  The fdesc lock must have been acquired already.  */

static struct fdesc_table *
new_fdesc_table (struct local *l, size_t *size)
{
  size_t old_npages = l->npages;
  size_t new_npages = old_npages + old_npages;
  struct fdesc_table *new_table;

  /* If someone has just created a new table, we return NULL to tell
     the caller to use the new table.  */
  if (! COMPARE_AND_SWAP (&l->npages, old_npages, new_npages))
    return (struct fdesc_table *) NULL;

  *size = old_npages * GLRO(dl_pagesize);
  new_table = __mmap (NULL, *size,
		      PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
  if (new_table == MAP_FAILED)
    _dl_signal_error (errno, NULL, NULL,
		      N_("cannot map pages for fdesc table"));

  new_table->len
    = (*size - sizeof (*new_table)) / sizeof (struct fdesc);
  new_table->first_unused = 1;
  return new_table;
}

/* Must call _dl_fptr_init before using any other function.  */
void
_dl_fptr_init (void)
{
  struct local *l;

  ELF_MACHINE_LOAD_ADDRESS (l, local);
  l->root = &l->boot_table;
}

static ElfW(Addr)
make_fdesc (ElfW(Addr) ip, ElfW(Addr) gp)
{
  struct fdesc *fdesc = NULL;
  struct fdesc_table *root;
  unsigned int old;
  struct local *l;

  ELF_MACHINE_LOAD_ADDRESS (l, local);

 retry:
  root = l->root;
  while (1)
    {
      old = root->first_unused;
      if (old >= root->len)
	break;
      else if (COMPARE_AND_SWAP (&root->first_unused, old, old + 1))
	{
	  fdesc = &root->fdesc[old];
	  goto install;
	}
    }

  if (l->free_list)
    {
      /* Get it from free-list.  */
      do
	{
	  fdesc = l->free_list;
	  if (fdesc == NULL)
	    goto retry;
	}
      while (! COMPARE_AND_SWAP ((ElfW(Addr) *) &l->free_list,
				 (ElfW(Addr)) fdesc, fdesc->ip));
    }
  else
    {
      /* Create a new fdesc table.  */
      size_t size;
      struct fdesc_table *new_table = new_fdesc_table (l, &size);

      if (new_table == NULL)
	goto retry;

      new_table->next = root;
      if (! COMPARE_AND_SWAP ((ElfW(Addr) *) &l->root,
			      (ElfW(Addr)) root,
			      (ElfW(Addr)) new_table))
	{
	  /* Someone has just installed a new table. Return NULL to
	     tell the caller to use the new table.  */
	  __munmap (new_table, size);
	  goto retry;
	}

      /* Note that the first entry was reserved while allocating the
	 memory for the new page.  */
      fdesc = &new_table->fdesc[0];
    }

 install:
  fdesc->gp = gp;
  fdesc->ip = ip;

  return (ElfW(Addr)) fdesc;
}


static inline ElfW(Addr) * __attribute__ ((always_inline))
make_fptr_table (struct link_map *map)
{
  const ElfW(Sym) *symtab = (const void *) D_PTR (map, l_info[DT_SYMTAB]);
  const char *strtab = (const void *) D_PTR (map, l_info[DT_STRTAB]);
  ElfW(Addr) *fptr_table;
  size_t size;
  size_t len;
  const ElfW(Sym) *symtabend;

  /* Determine the end of the dynamic symbol table using the hash.  */
  if (map->l_info[DT_HASH] != NULL)
    symtabend = (symtab + ((Elf_Symndx *) D_PTR (map, l_info[DT_HASH]))[1]);
  else
  /* There is no direct way to determine the number of symbols in the
     dynamic symbol table and no hash table is present.  The ELF
     binary is ill-formed but what shall we do?  Use the beginning of
     the string table which generally follows the symbol table.  */
    symtabend = (const ElfW(Sym) *) strtab;

  len = (((char *) symtabend - (char *) symtab)
	 / map->l_info[DT_SYMENT]->d_un.d_val);
  size = ALIGN_UP (len * sizeof (fptr_table[0]), GLRO(dl_pagesize));

  /* We don't support systems without MAP_ANON.  We avoid using malloc
     because this might get called before malloc is setup.  */
  fptr_table = __mmap (NULL, size,
		       PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE,
		       -1, 0);
  if (fptr_table == MAP_FAILED)
    _dl_signal_error (errno, NULL, NULL,
		      N_("cannot map pages for fptr table"));

  if (COMPARE_AND_SWAP ((ElfW(Addr) *) &map->l_mach.fptr_table,
			(ElfW(Addr)) NULL, (ElfW(Addr)) fptr_table))
    map->l_mach.fptr_table_len = len;
  else
    __munmap (fptr_table, len * sizeof (fptr_table[0]));

  return map->l_mach.fptr_table;
}


ElfW(Addr)
_dl_make_fptr (struct link_map *map, const ElfW(Sym) *sym,
	       ElfW(Addr) ip)
{
  ElfW(Addr) *ftab = map->l_mach.fptr_table;
  const ElfW(Sym) *symtab;
  Elf_Symndx symidx;
  struct local *l;

  if (__builtin_expect (ftab == NULL, 0))
    ftab = make_fptr_table (map);

  symtab = (const void *) D_PTR (map, l_info[DT_SYMTAB]);
  symidx = sym - symtab;

  if (symidx >= map->l_mach.fptr_table_len)
    _dl_signal_error (0, NULL, NULL,
		      N_("internal error: symidx out of range of fptr table"));

  while (ftab[symidx] == 0)
    {
      /* GOT has already been relocated in elf_get_dynamic_info -
	 don't try to relocate it again.  */
      ElfW(Addr) fdesc
	= make_fdesc (ip, map->l_info[DT_PLTGOT]->d_un.d_ptr);

      if (__builtin_expect (COMPARE_AND_SWAP (&ftab[symidx], (ElfW(Addr)) NULL,
					      fdesc), 1))
	{
	  /* Noone has updated the entry and the new function
	     descriptor has been installed.  */
#if 0
	  const char *strtab
	    = (const void *) D_PTR (map, l_info[DT_STRTAB]);

	  ELF_MACHINE_LOAD_ADDRESS (l, local);
	  if (l->root != &l->boot_table
	      || l->boot_table.first_unused > 20)
	    _dl_debug_printf ("created fdesc symbol `%s' at %lx\n",
			      strtab + sym->st_name, ftab[symidx]);
#endif
	  break;
	}
      else
	{
	  /* We created a duplicated function descriptor. We put it on
	     free-list.  */
	  struct fdesc *f = (struct fdesc *) fdesc;

	  ELF_MACHINE_LOAD_ADDRESS (l, local);

	  do
	    f->ip = (ElfW(Addr)) l->free_list;
	  while (! COMPARE_AND_SWAP ((ElfW(Addr) *) &l->free_list,
				     f->ip, fdesc));
	}
    }

  return ftab[symidx];
}


void
_dl_unmap (struct link_map *map)
{
  ElfW(Addr) *ftab = map->l_mach.fptr_table;
  struct fdesc *head = NULL, *tail = NULL;
  size_t i;

  _dl_unmap_segments (map);

  if (ftab == NULL)
    return;

  /* String together the fdesc structures that are being freed.  */
  for (i = 0; i < map->l_mach.fptr_table_len; ++i)
    {
      if (ftab[i])
	{
	  *(struct fdesc **) ftab[i] = head;
	  head = (struct fdesc *) ftab[i];
	  if (tail == NULL)
	    tail = head;
	}
    }

  /* Prepend the new list to the free_list: */
  if (tail)
    do
      tail->ip = (ElfW(Addr)) local.free_list;
    while (! COMPARE_AND_SWAP ((ElfW(Addr) *) &local.free_list,
			       tail->ip, (ElfW(Addr)) head));

  __munmap (ftab, (map->l_mach.fptr_table_len
		   * sizeof (map->l_mach.fptr_table[0])));

  map->l_mach.fptr_table = NULL;
}

extern ElfW(Addr) _dl_fixup (struct link_map *, ElfW(Word)) attribute_hidden;

static inline Elf32_Addr
elf_machine_resolve (void)
{
  Elf32_Addr addr;

  asm ("b,l     1f,%0\n"
"	addil	L'_dl_runtime_resolve - ($PIC_pcrel$0 - 1),%0\n"
"1:	ldo	R'_dl_runtime_resolve - ($PIC_pcrel$0 - 5)(%%r1),%0\n"
       : "=r" (addr) : : "r1");

  return addr;
}

static inline int
_dl_read_access_allowed (unsigned int *addr)
{
  int result;

  asm ("proberi	(%1),3,%0" : "=r" (result) : "r" (addr) : );

  return result;
}

ElfW(Addr)
_dl_lookup_address (const void *address)
{
  ElfW(Addr) addr = (ElfW(Addr)) address;
  ElfW(Word) reloc_arg;
  volatile unsigned int *desc;
  unsigned int *gptr;

  /* Return ADDR if the least-significant two bits of ADDR are not consistent
     with ADDR being a linker defined function pointer.  The normal value for
     a code address in a backtrace is 3.  */
  if (((unsigned int) addr & 3) != 2)
    return addr;

  /* Handle special case where ADDR points to page 0.  */
  if ((unsigned int) addr < 4096)
    return addr;

  /* Clear least-significant two bits from descriptor address.  */
  desc = (unsigned int *) ((unsigned int) addr & ~3);
  if (!_dl_read_access_allowed (desc))
    return addr;

  /* First load the relocation offset.  */
  reloc_arg = (ElfW(Word)) desc[1];
  atomic_full_barrier();

  /* Then load first word of candidate descriptor.  It should be a pointer
     with word alignment and point to memory that can be read.  */
  gptr = (unsigned int *) desc[0];
  if (((unsigned int) gptr & 3) != 0
      || !_dl_read_access_allowed (gptr))
    return addr;

  /* See if descriptor requires resolution.  The following trampoline is
     used in each global offset table for function resolution:

		ldw 0(r20),r21
		bv r0(r21)
		ldw 4(r20),r21
     tramp:	b,l .-12,r20
		depwi 0,31,2,r20
		.word _dl_runtime_resolve
		.word "_dl_runtime_resolve ltp"
     got:	.word _DYNAMIC
		.word "struct link map address" */
  if (gptr[0] == 0xea9f1fdd			/* b,l .-12,r20     */
      && gptr[1] == 0xd6801c1e			/* depwi 0,31,2,r20 */
      && (ElfW(Addr)) gptr[2] == elf_machine_resolve ())
    {
      struct link_map *l = (struct link_map *) gptr[5];

      /* If gp has been resolved, we need to hunt for relocation offset.  */
      if (!(reloc_arg & PA_GP_RELOC))
	reloc_arg = _dl_fix_reloc_arg (addr, l);

      _dl_fixup (l, reloc_arg);
    }

  return (ElfW(Addr)) desc[0];
}
