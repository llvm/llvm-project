/* Machine-dependent ELF dynamic relocation inline functions.  Stub version.
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

#define ELF_MACHINE_NAME "stub"

#include <string.h>
#include <link.h>


/* Return nonzero iff ELF header is compatible with the running host.  */
static inline int
elf_machine_matches_host (const Elf32_Ehdr *ehdr)
{
  switch (ehdr->e_machine)
    {
    default:
      return 0;
    }
}


/* Return the link-time address of _DYNAMIC.  */
static inline Elf32_Addr
elf_machine_dynamic (void)
{
#error "Damn, no _DYNAMIC"
}


/* Return the run-time load address of the shared object.  */
static inline Elf32_Addr
elf_machine_load_address (void)
{
#error "Where am I?"
}

/* Fixup a PLT entry to bounce directly to the function at VALUE.  */

static inline ElfW(Addr)
elf_machine_fixup_plt (struct link_map *map, lookup_t t,
		       const ElfW(Sym) *refsym, const ElfW(Sym) *sym,
		       const ElfW(Rel) *reloc,
		       ElfW(Addr) *reloc_addr, ElfW(Addr) value)
{
  return *reloc_addr = value;
}

/* Perform the relocation specified by RELOC and SYM (which is fully resolved).
   LOADADDR is the load address of the object; INFO is an array indexed
   by DT_* of the .dynamic section info.  */

auto inline void
__attribute__ ((always_inline))
elf_machine_rel (Elf32_Addr loadaddr, Elf32_Dyn *info[DT_NUM],
		 const Elf32_Rel *reloc, const Elf32_Sym *sym,
		 Elf32_Addr (*resolve) (const Elf32_Sym **ref,
					Elf32_Addr reloc_addr,
					int noplt))
{
  Elf32_Addr *const reloc_addr = (Elf32_Addr *) reloc->r_offset;
  Elf32_Addr loadbase;

  switch (ELF32_R_TYPE (reloc->r_info))
    {
    case R_MACHINE_COPY:
      loadbase = (*resolve) (&sym, (Elf32_Addr) reloc_addr, 0);
      memcpy (reloc_addr, (void *) (loadbase + sym->st_value), sym->st_size);
      break;
    default:
      _dl_reloc_bad_type (map, ELF32_R_TYPE (reloc->r_info), 0);
      break;
    }
}


auto inline Elf32_Addr
__attribute__ ((always_inline))
elf_machine_rela (Elf32_Addr loadaddr, Elf32_Dyn *info[DT_NUM],
		  const Elf32_Rel *reloc, const Elf32_Sym *sym,
		  Elf32_Addr (*resolve) (const Elf32_Sym **ref,
					 Elf32_Addr reloc_addr,
					 int noplt))
{
  _dl_signal_error (0, "Elf32_Rela relocation requested -- unused on "
		    NULL, ELF_MACHINE_NAME);
}


/* Set up the loaded object described by L so its unrelocated PLT
   entries will jump to the on-demand fixup code in dl-runtime.c.  */

static inline int
elf_machine_runtime_setup (struct link_map *l, int lazy)
{
  extern void _dl_runtime_resolve (Elf32_Word);

  if (lazy)
    {
      /* The GOT entries for functions in the PLT have not yet been filled
         in.  Their initial contents will arrange when called to push an
         offset into the .rel.plt section, push _GLOBAL_OFFSET_TABLE_[1],
         and then jump to _GLOBAL_OFFSET_TABLE[2].  */
      Elf32_Addr *got = (Elf32_Addr *) D_PTR (l, l_info[DT_PLTGOT]);
      got[1] = (Elf32_Addr) l;	/* Identify this shared object.  */

      /* This function will get called to fix up the GOT entry indicated by
         the offset on the stack, and then jump to the resolved address.  */
      got[2] = (Elf32_Addr) &_dl_runtime_resolve;
    }

  return lazy;
}


/* Initial entry point code for the dynamic linker.
   The C function `_dl_start' is the real entry point;
   its return value is the user program's entry point.  */

#define RTLD_START #error need some startup code
