/* Machine-dependent ELF dynamic relocation functions.  PowerPC version.
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

#include <unistd.h>
#include <string.h>
#include <sys/param.h>
#include <link.h>
#include <ldsodefs.h>
#include <elf/dynamic-link.h>
#include <dl-machine.h>
#include <_itoa.h>

/* Stuff for the PLT.  */
#define PLT_INITIAL_ENTRY_WORDS 18
#define PLT_LONGBRANCH_ENTRY_WORDS 0
#define PLT_TRAMPOLINE_ENTRY_WORDS 6
#define PLT_DOUBLE_SIZE (1<<13)
#define PLT_ENTRY_START_WORDS(entry_number) \
  (PLT_INITIAL_ENTRY_WORDS + (entry_number)*2				\
   + ((entry_number) > PLT_DOUBLE_SIZE					\
      ? ((entry_number) - PLT_DOUBLE_SIZE)*2				\
      : 0))
#define PLT_DATA_START_WORDS(num_entries) PLT_ENTRY_START_WORDS(num_entries)

/* Macros to build PowerPC opcode words.  */
#define OPCODE_ADDI(rd,ra,simm) \
  (0x38000000 | (rd) << 21 | (ra) << 16 | ((simm) & 0xffff))
#define OPCODE_ADDIS(rd,ra,simm) \
  (0x3c000000 | (rd) << 21 | (ra) << 16 | ((simm) & 0xffff))
#define OPCODE_ADD(rd,ra,rb) \
  (0x7c000214 | (rd) << 21 | (ra) << 16 | (rb) << 11)
#define OPCODE_B(target) (0x48000000 | ((target) & 0x03fffffc))
#define OPCODE_BA(target) (0x48000002 | ((target) & 0x03fffffc))
#define OPCODE_BCTR() 0x4e800420
#define OPCODE_LWZ(rd,d,ra) \
  (0x80000000 | (rd) << 21 | (ra) << 16 | ((d) & 0xffff))
#define OPCODE_LWZU(rd,d,ra) \
  (0x84000000 | (rd) << 21 | (ra) << 16 | ((d) & 0xffff))
#define OPCODE_MTCTR(rd) (0x7C0903A6 | (rd) << 21)
#define OPCODE_RLWINM(ra,rs,sh,mb,me) \
  (0x54000000 | (rs) << 21 | (ra) << 16 | (sh) << 11 | (mb) << 6 | (me) << 1)

#define OPCODE_LI(rd,simm)    OPCODE_ADDI(rd,0,simm)
#define OPCODE_ADDIS_HI(rd,ra,value) \
  OPCODE_ADDIS(rd,ra,((value) + 0x8000) >> 16)
#define OPCODE_LIS_HI(rd,value) OPCODE_ADDIS_HI(rd,0,value)
#define OPCODE_SLWI(ra,rs,sh) OPCODE_RLWINM(ra,rs,sh,0,31-sh)


#define PPC_DCBST(where) asm volatile ("dcbst 0,%0" : : "r"(where) : "memory")
#define PPC_SYNC asm volatile ("sync" : : : "memory")
#define PPC_ISYNC asm volatile ("sync; isync" : : : "memory")
#define PPC_ICBI(where) asm volatile ("icbi 0,%0" : : "r"(where) : "memory")
#define PPC_DIE asm volatile ("tweq 0,0")

/* Use this when you've modified some code, but it won't be in the
   instruction fetch queue (or when it doesn't matter if it is). */
#define MODIFIED_CODE_NOQUEUE(where) \
     do { PPC_DCBST(where); PPC_SYNC; PPC_ICBI(where); } while (0)
/* Use this when it might be in the instruction queue. */
#define MODIFIED_CODE(where) \
     do { PPC_DCBST(where); PPC_SYNC; PPC_ICBI(where); PPC_ISYNC; } while (0)


/* The idea here is that to conform to the ABI, we are supposed to try
   to load dynamic objects between 0x10000 (we actually use 0x40000 as
   the lower bound, to increase the chance of a memory reference from
   a null pointer giving a segfault) and the program's load address;
   this may allow us to use a branch instruction in the PLT rather
   than a computed jump.  The address is only used as a preference for
   mmap, so if we get it wrong the worst that happens is that it gets
   mapped somewhere else.  */

ElfW(Addr)
__elf_preferred_address (struct link_map *loader, size_t maplength,
			 ElfW(Addr) mapstartpref)
{
  ElfW(Addr) low, high;
  struct link_map *l;
  Lmid_t nsid;

  /* If the object has a preference, load it there!  */
  if (mapstartpref != 0)
    return mapstartpref;

  /* Otherwise, quickly look for a suitable gap between 0x3FFFF and
     0x70000000.  0x3FFFF is so that references off NULL pointers will
     cause a segfault, 0x70000000 is just paranoia (it should always
     be superseded by the program's load address).  */
  low =  0x0003FFFF;
  high = 0x70000000;
  for (nsid = 0; nsid < DL_NNS; ++nsid)
    for (l = GL(dl_ns)[nsid]._ns_loaded; l; l = l->l_next)
      {
	ElfW(Addr) mapstart, mapend;
	mapstart = l->l_map_start & ~(GLRO(dl_pagesize) - 1);
	mapend = l->l_map_end | (GLRO(dl_pagesize) - 1);
	assert (mapend > mapstart);

	/* Prefer gaps below the main executable, note that l ==
	   _dl_loaded does not work for static binaries loading
	   e.g. libnss_*.so.  */
	if ((mapend >= high || l->l_type == lt_executable)
	    && high >= mapstart)
	  high = mapstart;
	else if (mapend >= low && low >= mapstart)
	  low = mapend;
	else if (high >= mapend && mapstart >= low)
	  {
	    if (high - mapend >= mapstart - low)
	      low = mapend;
	    else
	      high = mapstart;
	  }
      }

  high -= 0x10000; /* Allow some room between objects.  */
  maplength = (maplength | (GLRO(dl_pagesize) - 1)) + 1;
  if (high <= low || high - low < maplength )
    return 0;
  return high - maplength;  /* Both high and maplength are page-aligned.  */
}

/* Set up the loaded object described by L so its unrelocated PLT
   entries will jump to the on-demand fixup code in dl-runtime.c.
   Also install a small trampoline to be used by entries that have
   been relocated to an address too far away for a single branch.  */

/* There are many kinds of PLT entries:

   (1)	A direct jump to the actual routine, either a relative or
	absolute branch.  These are set up in __elf_machine_fixup_plt.

   (2)	Short lazy entries.  These cover the first 8192 slots in
        the PLT, and look like (where 'index' goes from 0 to 8191):

	li %r11, index*4
	b  &plt[PLT_TRAMPOLINE_ENTRY_WORDS+1]

   (3)	Short indirect jumps.  These replace (2) when a direct jump
	wouldn't reach.  They look the same except that the branch
	is 'b &plt[PLT_LONGBRANCH_ENTRY_WORDS]'.

   (4)  Long lazy entries.  These cover the slots when a short entry
	won't fit ('index*4' overflows its field), and look like:

	lis %r11, %hi(index*4 + &plt[PLT_DATA_START_WORDS])
	lwzu %r12, %r11, %lo(index*4 + &plt[PLT_DATA_START_WORDS])
	b  &plt[PLT_TRAMPOLINE_ENTRY_WORDS]
	bctr

   (5)	Long indirect jumps.  These replace (4) when a direct jump
	wouldn't reach.  They look like:

	lis %r11, %hi(index*4 + &plt[PLT_DATA_START_WORDS])
	lwz %r12, %r11, %lo(index*4 + &plt[PLT_DATA_START_WORDS])
	mtctr %r12
	bctr

   (6) Long direct jumps.  These are used when thread-safety is not
       required.  They look like:

       lis %r12, %hi(finaladdr)
       addi %r12, %r12, %lo(finaladdr)
       mtctr %r12
       bctr


   The lazy entries, (2) and (4), are set up here in
   __elf_machine_runtime_setup.  (1), (3), and (5) are set up in
   __elf_machine_fixup_plt.  (1), (3), and (6) can also be constructed
   in __process_machine_rela.

   The reason for the somewhat strange construction of the long
   entries, (4) and (5), is that we need to ensure thread-safety.  For
   (1) and (3), this is obvious because only one instruction is
   changed and the PPC architecture guarantees that aligned stores are
   atomic.  For (5), this is more tricky.  When changing (4) to (5),
   the `b' instruction is first changed to `mtctr'; this is safe
   and is why the `lwzu' instruction is not just a simple `addi'.
   Once this is done, and is visible to all processors, the `lwzu' can
   safely be changed to a `lwz'.  */
int
__elf_machine_runtime_setup (struct link_map *map, int lazy, int profile)
{
  if (map->l_info[DT_JMPREL])
    {
      Elf32_Word i;
      Elf32_Word *plt = (Elf32_Word *) D_PTR (map, l_info[DT_PLTGOT]);
      Elf32_Word num_plt_entries = (map->l_info[DT_PLTRELSZ]->d_un.d_val
				    / sizeof (Elf32_Rela));
      Elf32_Word rel_offset_words = PLT_DATA_START_WORDS (num_plt_entries);
      Elf32_Word data_words = (Elf32_Word) (plt + rel_offset_words);
      Elf32_Word size_modified;

      extern void _dl_runtime_resolve (void);
      extern void _dl_prof_resolve (void);

      /* Convert the index in r11 into an actual address, and get the
	 word at that address.  */
      plt[PLT_LONGBRANCH_ENTRY_WORDS] = OPCODE_ADDIS_HI (11, 11, data_words);
      plt[PLT_LONGBRANCH_ENTRY_WORDS + 1] = OPCODE_LWZ (11, data_words, 11);

      /* Call the procedure at that address.  */
      plt[PLT_LONGBRANCH_ENTRY_WORDS + 2] = OPCODE_MTCTR (11);
      plt[PLT_LONGBRANCH_ENTRY_WORDS + 3] = OPCODE_BCTR ();

      if (lazy)
	{
	  Elf32_Word *tramp = plt + PLT_TRAMPOLINE_ENTRY_WORDS;
	  Elf32_Word dlrr;
	  Elf32_Word offset;

#ifndef PROF
	  dlrr = (Elf32_Word) (profile
			       ? _dl_prof_resolve
			       : _dl_runtime_resolve);
	  if (profile && GLRO(dl_profile) != NULL
	      && _dl_name_match_p (GLRO(dl_profile), map))
	    /* This is the object we are looking for.  Say that we really
	       want profiling and the timers are started.  */
	    GL(dl_profile_map) = map;
#else
	  dlrr = (Elf32_Word) _dl_runtime_resolve;
#endif

	  /* For the long entries, subtract off data_words.  */
	  tramp[0] = OPCODE_ADDIS_HI (11, 11, -data_words);
	  tramp[1] = OPCODE_ADDI (11, 11, -data_words);

	  /* Multiply index of entry by 3 (in r11).  */
	  tramp[2] = OPCODE_SLWI (12, 11, 1);
	  tramp[3] = OPCODE_ADD (11, 12, 11);
	  if (dlrr <= 0x01fffffc || dlrr >= 0xfe000000)
	    {
	      /* Load address of link map in r12.  */
	      tramp[4] = OPCODE_LI (12, (Elf32_Word) map);
	      tramp[5] = OPCODE_ADDIS_HI (12, 12, (Elf32_Word) map);

	      /* Call _dl_runtime_resolve.  */
	      tramp[6] = OPCODE_BA (dlrr);
	    }
	  else
	    {
	      /* Get address of _dl_runtime_resolve in CTR.  */
	      tramp[4] = OPCODE_LI (12, dlrr);
	      tramp[5] = OPCODE_ADDIS_HI (12, 12, dlrr);
	      tramp[6] = OPCODE_MTCTR (12);

	      /* Load address of link map in r12.  */
	      tramp[7] = OPCODE_LI (12, (Elf32_Word) map);
	      tramp[8] = OPCODE_ADDIS_HI (12, 12, (Elf32_Word) map);

	      /* Call _dl_runtime_resolve.  */
	      tramp[9] = OPCODE_BCTR ();
	    }

	  /* Set up the lazy PLT entries.  */
	  offset = PLT_INITIAL_ENTRY_WORDS;
	  i = 0;
	  while (i < num_plt_entries && i < PLT_DOUBLE_SIZE)
	    {
	      plt[offset  ] = OPCODE_LI (11, i * 4);
	      plt[offset+1] = OPCODE_B ((PLT_TRAMPOLINE_ENTRY_WORDS + 2
					 - (offset+1))
					* 4);
	      i++;
	      offset += 2;
	    }
	  while (i < num_plt_entries)
	    {
	      plt[offset  ] = OPCODE_LIS_HI (11, i * 4 + data_words);
	      plt[offset+1] = OPCODE_LWZU (12, i * 4 + data_words, 11);
	      plt[offset+2] = OPCODE_B ((PLT_TRAMPOLINE_ENTRY_WORDS
					 - (offset+2))
					* 4);
	      plt[offset+3] = OPCODE_BCTR ();
	      i++;
	      offset += 4;
	    }
	}

      /* Now, we've modified code.  We need to write the changes from
	 the data cache to a second-level unified cache, then make
	 sure that stale data in the instruction cache is removed.
	 (In a multiprocessor system, the effect is more complex.)
	 Most of the PLT shouldn't be in the instruction cache, but
	 there may be a little overlap at the start and the end.

	 Assumes that dcbst and icbi apply to lines of 16 bytes or
	 more.  Current known line sizes are 16, 32, and 128 bytes.
	 The following gets the cache line size, when available.  */

      /* Default minimum 4 words per cache line.  */
      int line_size_words = 4;

      if (lazy && GLRO(dl_cache_line_size) != 0)
	/* Convert bytes to words.  */
	line_size_words = GLRO(dl_cache_line_size) / 4;

      size_modified = lazy ? rel_offset_words : 6;
      for (i = 0; i < size_modified; i += line_size_words)
        PPC_DCBST (plt + i);
      PPC_DCBST (plt + size_modified - 1);
      PPC_SYNC;

      for (i = 0; i < size_modified; i += line_size_words)
        PPC_ICBI (plt + i);
      PPC_ICBI (plt + size_modified - 1);
      PPC_ISYNC;
    }

  return lazy;
}

Elf32_Addr
__elf_machine_fixup_plt (struct link_map *map,
			 Elf32_Addr *reloc_addr, Elf32_Addr finaladdr)
{
  Elf32_Sword delta = finaladdr - (Elf32_Word) reloc_addr;
  if (delta << 6 >> 6 == delta)
    *reloc_addr = OPCODE_B (delta);
  else if (finaladdr <= 0x01fffffc || finaladdr >= 0xfe000000)
    *reloc_addr = OPCODE_BA (finaladdr);
  else
    {
      Elf32_Word *plt, *data_words;
      Elf32_Word index, offset, num_plt_entries;

      num_plt_entries = (map->l_info[DT_PLTRELSZ]->d_un.d_val
			 / sizeof (Elf32_Rela));
      plt = (Elf32_Word *) D_PTR (map, l_info[DT_PLTGOT]);
      offset = reloc_addr - plt;
      index = (offset - PLT_INITIAL_ENTRY_WORDS)/2;
      data_words = plt + PLT_DATA_START_WORDS (num_plt_entries);

      reloc_addr += 1;

      if (index < PLT_DOUBLE_SIZE)
	{
	  data_words[index] = finaladdr;
	  PPC_SYNC;
	  *reloc_addr = OPCODE_B ((PLT_LONGBRANCH_ENTRY_WORDS - (offset+1))
				  * 4);
	}
      else
	{
	  index -= (index - PLT_DOUBLE_SIZE)/2;

	  data_words[index] = finaladdr;
	  PPC_SYNC;

	  reloc_addr[1] = OPCODE_MTCTR (12);
	  MODIFIED_CODE_NOQUEUE (reloc_addr + 1);
	  PPC_SYNC;

	  reloc_addr[0] = OPCODE_LWZ (12,
				      (Elf32_Word) (data_words + index), 11);
	}
    }
  MODIFIED_CODE (reloc_addr);
  return finaladdr;
}

void
_dl_reloc_overflow (struct link_map *map,
		    const char *name,
		    Elf32_Addr *const reloc_addr,
		    const Elf32_Sym *refsym)
{
  char buffer[128];
  char *t;
  t = stpcpy (buffer, name);
  t = stpcpy (t, " relocation at 0x00000000");
  _itoa_word ((unsigned) reloc_addr, t, 16, 0);
  if (refsym)
    {
      const char *strtab;

      strtab = (const void *) D_PTR (map, l_info[DT_STRTAB]);
      t = stpcpy (t, " for symbol `");
      t = stpcpy (t, strtab + refsym->st_name);
      t = stpcpy (t, "'");
    }
  t = stpcpy (t, " out of range");
  _dl_signal_error (0, map->l_name, NULL, buffer);
}

void
__process_machine_rela (struct link_map *map,
			const Elf32_Rela *reloc,
			struct link_map *sym_map,
			const Elf32_Sym *sym,
			const Elf32_Sym *refsym,
			Elf32_Addr *const reloc_addr,
			Elf32_Addr const finaladdr,
			int rinfo, bool skip_ifunc)
{
  union unaligned
    {
      uint16_t u2;
      uint32_t u4;
    } __attribute__((__packed__));

  switch (rinfo)
    {
    case R_PPC_NONE:
      return;

    case R_PPC_ADDR32:
    case R_PPC_GLOB_DAT:
    case R_PPC_RELATIVE:
      *reloc_addr = finaladdr;
      return;

    case R_PPC_IRELATIVE:
      if (__glibc_likely (!skip_ifunc))
	*reloc_addr = ((Elf32_Addr (*) (void)) finaladdr) ();
      return;

    case R_PPC_UADDR32:
      ((union unaligned *) reloc_addr)->u4 = finaladdr;
      break;

    case R_PPC_ADDR24:
      if (__glibc_unlikely (finaladdr > 0x01fffffc && finaladdr < 0xfe000000))
	_dl_reloc_overflow (map,  "R_PPC_ADDR24", reloc_addr, refsym);
      *reloc_addr = (*reloc_addr & 0xfc000003) | (finaladdr & 0x3fffffc);
      break;

    case R_PPC_ADDR16:
      if (__glibc_unlikely (finaladdr > 0x7fff && finaladdr < 0xffff8000))
	_dl_reloc_overflow (map,  "R_PPC_ADDR16", reloc_addr, refsym);
      *(Elf32_Half*) reloc_addr = finaladdr;
      break;

    case R_PPC_UADDR16:
      if (__glibc_unlikely (finaladdr > 0x7fff && finaladdr < 0xffff8000))
	_dl_reloc_overflow (map,  "R_PPC_UADDR16", reloc_addr, refsym);
      ((union unaligned *) reloc_addr)->u2 = finaladdr;
      break;

    case R_PPC_ADDR16_LO:
      *(Elf32_Half*) reloc_addr = finaladdr;
      break;

    case R_PPC_ADDR16_HI:
      *(Elf32_Half*) reloc_addr = finaladdr >> 16;
      break;

    case R_PPC_ADDR16_HA:
      *(Elf32_Half*) reloc_addr = (finaladdr + 0x8000) >> 16;
      break;

    case R_PPC_ADDR14:
    case R_PPC_ADDR14_BRTAKEN:
    case R_PPC_ADDR14_BRNTAKEN:
      if (__glibc_unlikely (finaladdr > 0x7fff && finaladdr < 0xffff8000))
	_dl_reloc_overflow (map,  "R_PPC_ADDR14", reloc_addr, refsym);
      *reloc_addr = (*reloc_addr & 0xffff0003) | (finaladdr & 0xfffc);
      if (rinfo != R_PPC_ADDR14)
	*reloc_addr = ((*reloc_addr & 0xffdfffff)
		       | ((rinfo == R_PPC_ADDR14_BRTAKEN)
			  ^ (finaladdr >> 31)) << 21);
      break;

    case R_PPC_REL24:
      {
	Elf32_Sword delta = finaladdr - (Elf32_Word) reloc_addr;
	if (delta << 6 >> 6 != delta)
	  _dl_reloc_overflow (map,  "R_PPC_REL24", reloc_addr, refsym);
	*reloc_addr = (*reloc_addr & 0xfc000003) | (delta & 0x3fffffc);
      }
      break;

    case R_PPC_COPY:
      if (sym == NULL)
	/* This can happen in trace mode when an object could not be
	   found.  */
	return;
      if (sym->st_size > refsym->st_size
	  || (GLRO(dl_verbose) && sym->st_size < refsym->st_size))
	{
	  const char *strtab;

	  strtab = (const void *) D_PTR (map, l_info[DT_STRTAB]);
	  _dl_error_printf ("\
%s: Symbol `%s' has different size in shared object, consider re-linking\n",
			    RTLD_PROGNAME, strtab + refsym->st_name);
	}
      memcpy (reloc_addr, (char *) finaladdr, MIN (sym->st_size,
						   refsym->st_size));
      return;

    case R_PPC_REL32:
      *reloc_addr = finaladdr - (Elf32_Word) reloc_addr;
      return;

    case R_PPC_JMP_SLOT:
      /* It used to be that elf_machine_fixup_plt was used here,
	 but that doesn't work when ld.so relocates itself
	 for the second time.  On the bright side, there's
         no need to worry about thread-safety here.  */
      {
	Elf32_Sword delta = finaladdr - (Elf32_Word) reloc_addr;
	if (delta << 6 >> 6 == delta)
	  *reloc_addr = OPCODE_B (delta);
	else if (finaladdr <= 0x01fffffc || finaladdr >= 0xfe000000)
	  *reloc_addr = OPCODE_BA (finaladdr);
	else
	  {
	    Elf32_Word *plt, *data_words;
	    Elf32_Word index, offset, num_plt_entries;

	    plt = (Elf32_Word *) D_PTR (map, l_info[DT_PLTGOT]);
	    offset = reloc_addr - plt;

	    if (offset < PLT_DOUBLE_SIZE*2 + PLT_INITIAL_ENTRY_WORDS)
	      {
		index = (offset - PLT_INITIAL_ENTRY_WORDS)/2;
		num_plt_entries = (map->l_info[DT_PLTRELSZ]->d_un.d_val
				   / sizeof (Elf32_Rela));
		data_words = plt + PLT_DATA_START_WORDS (num_plt_entries);
		data_words[index] = finaladdr;
		reloc_addr[0] = OPCODE_LI (11, index * 4);
		reloc_addr[1] = OPCODE_B ((PLT_LONGBRANCH_ENTRY_WORDS
					   - (offset+1))
					  * 4);
		MODIFIED_CODE_NOQUEUE (reloc_addr + 1);
	      }
	    else
	      {
		reloc_addr[0] = OPCODE_LIS_HI (12, finaladdr);
		reloc_addr[1] = OPCODE_ADDI (12, 12, finaladdr);
		reloc_addr[2] = OPCODE_MTCTR (12);
		reloc_addr[3] = OPCODE_BCTR ();
		MODIFIED_CODE_NOQUEUE (reloc_addr + 3);
	      }
	  }
      }
      break;

#define DO_TLS_RELOC(suffix)						      \
    case R_PPC_DTPREL##suffix:						      \
      /* During relocation all TLS symbols are defined and used.	      \
	 Therefore the offset is already correct.  */			      \
      if (sym_map != NULL)						      \
	do_reloc##suffix ("R_PPC_DTPREL"#suffix,			      \
			  TLS_DTPREL_VALUE (sym, reloc));		      \
      break;								      \
    case R_PPC_TPREL##suffix:						      \
      if (sym_map != NULL)						      \
	{								      \
	  CHECK_STATIC_TLS (map, sym_map);				      \
	  do_reloc##suffix ("R_PPC_TPREL"#suffix,			      \
			    TLS_TPREL_VALUE (sym_map, sym, reloc));	      \
	}								      \
      break;

    inline void do_reloc16 (const char *r_name, Elf32_Addr value)
      {
	if (__glibc_unlikely (value > 0x7fff && value < 0xffff8000))
	  _dl_reloc_overflow (map, r_name, reloc_addr, refsym);
	*(Elf32_Half *) reloc_addr = value;
      }
    inline void do_reloc16_LO (const char *r_name, Elf32_Addr value)
      {
	*(Elf32_Half *) reloc_addr = value;
      }
    inline void do_reloc16_HI (const char *r_name, Elf32_Addr value)
      {
	*(Elf32_Half *) reloc_addr = value >> 16;
      }
    inline void do_reloc16_HA (const char *r_name, Elf32_Addr value)
      {
	*(Elf32_Half *) reloc_addr = (value + 0x8000) >> 16;
      }
    DO_TLS_RELOC (16)
    DO_TLS_RELOC (16_LO)
    DO_TLS_RELOC (16_HI)
    DO_TLS_RELOC (16_HA)

    default:
      _dl_reloc_bad_type (map, rinfo, 0);
      return;
    }

  MODIFIED_CODE_NOQUEUE (reloc_addr);
}
