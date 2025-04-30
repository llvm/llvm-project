/* Machine-dependent ELF dynamic relocation inline functions.  Nios II version.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef dl_machine_h
#define dl_machine_h

#define ELF_MACHINE_NAME "nios2"

#include <string.h>
#include <link.h>
#include <dl-tls.h>

/* Return nonzero iff ELF header is compatible with the running host.  */
static inline int
elf_machine_matches_host (const Elf32_Ehdr *ehdr)
{
  return ehdr->e_machine == EM_ALTERA_NIOS2;
}


/* Return the link-time address of _DYNAMIC.  Conveniently, this is the
   first element of the GOT.  */
static inline Elf32_Addr
elf_machine_dynamic (void)
{
  Elf32_Addr *dynamic;
  int tmp;
  asm ("nextpc\t%0\n\t"
       "1: movhi\t%1, %%hiadj(_GLOBAL_OFFSET_TABLE_ - 1b)\n\t"
       "addi\t%1, %1, %%lo(_GLOBAL_OFFSET_TABLE_ - 1b)\n\t"
       "add\t%0, %0, %1\n"
       : "=r" (dynamic), "=r" (tmp));
  return *dynamic;
}


/* Return the run-time load address of the shared object.  */
static inline Elf32_Addr
elf_machine_load_address (void)
{
  Elf32_Addr result;
  int tmp;
  asm ("nextpc\t%0\n\t"
       "1: movhi\t%1, %%hiadj(1b)\n\t"
       "addi\t%1, %1, %%lo(1b)\n\t"
       "sub\t%0, %0, %1\n"
       : "=r" (result), "=r" (tmp));
  return result;
}

/* Set up the loaded object described by L so its unrelocated PLT
   entries will jump to the on-demand fixup code in dl-runtime.c.  */

static inline int __attribute__ ((always_inline))
elf_machine_runtime_setup (struct link_map *l, int lazy, int profile)
{
  extern void _dl_runtime_resolve (Elf32_Word);

  if (l->l_info[DT_JMPREL] && lazy)
    {
      /* The GOT entries for functions in the PLT have not yet been filled
         in.  Their initial contents will arrange when called to load r15 with
         an offset into the .got section, load r14 with
	 _GLOBAL_OFFSET_TABLE_[1], and then jump to _GLOBAL_OFFSET_TABLE[2].
      */
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

#define RTLD_START asm ("\
.text\n\
.globl _start\n\
.type _start, %function\n\
_start:\n\
        /* At start time, all the args are on the stack.  */\n\
        mov r4, sp\n\
\n\
        /* Start the calculation of the GOT pointer.  */\n\
        nextpc r22\n\
1:      movhi r8, %hiadj(_gp_got - 1b)\n\
        addi r8, r8, %lo(_gp_got - 1b)\n\
\n\
        /* Figure out where _dl_start will need to return to.  */\n\
        movhi ra, %hiadj(2f - 1b)\n\
        addi ra, ra, %lo(2f - 1b)\n\
        add ra, ra, r22\n\
\n\
        /* Finish the calculation of the GOT pointer.  */\n\
        add r22, r22, r8\n\
\n\
        br _dl_start\n\
\n\
        /* Save the returned user entry point.  */\n\
2:      mov r16, r2\n\
\n\
        /* Initialize gp.  */\n\
        ldw r4, %got(_rtld_local)(r22)\n\
        ldw r4, 0(r4)\n\
        ldw r8, %call(_dl_nios2_get_gp_value)(r22)\n\
        callr r8\n\
        mov gp, r2\n\
\n\
        /* Find the number of arguments to skip.  */\n\
        ldw r8, %got(_dl_skip_args)(r22)\n\
        ldw r8, 0(r8)\n\
\n\
        /* Find the main_map from the GOT.  */\n\
        ldw r4, %got(_rtld_local)(r22)\n\
        ldw r4, 0(r4)\n\
\n\
        /* Find argc.  */\n\
        ldw r5, 0(sp)\n\
        sub r5, r5, r8\n\
        stw r5, 0(sp)\n\
\n\
        /* Find the first unskipped argument.  */\n\
        slli r8, r8, 2\n\
        addi r6, sp, 4\n\
        add r9, r6, r8\n\
        mov r10, r6\n\
\n\
        /* Shuffle argv down.  */\n\
3:      ldw r11, 0(r9)\n\
        stw r11, 0(r10)\n\
        addi r9, r9, 4\n\
        addi r10, r10, 4\n\
        bne r11, zero, 3b\n\
\n\
        /* Shuffle envp down.  */\n\
        mov r7, r10\n\
4:      ldw r11, 0(r9)\n\
        stw r11, 0(r10)\n\
        addi r9, r9, 4\n\
        addi r10, r10, 4\n\
        bne r11, zero, 4b\n\
\n\
        /* Shuffle auxv down.  */\n\
5:      ldw r11, 4(r9)\n\
        stw r11, 4(r10)\n\
        ldw r11, 0(r9)\n\
        stw r11, 0(r10)\n\
        addi r9, r9, 8\n\
        addi r10, r10, 8\n\
        bne r11, zero, 5b\n\
\n\
        /* Update _dl_argv.  */\n\
        ldw r2, %got(_dl_argv)(r22)\n\
        stw r6, 0(r2)\n\
\n\
        /* Call _dl_init through the PLT.  */\n\
        ldw r8, %call(_dl_init)(r22)\n\
        callr r8\n\
\n\
        /* Find the finalization function.  */\n\
        ldw r4, %got(_dl_fini)(r22)\n\
\n\
        /* Jump to the user's entry point.  */\n\
        jmp r16\n\
");

/* ELF_RTYPE_CLASS_PLT iff TYPE describes relocation of a PLT entry, so
   PLT entries should not be allowed to define the value.
   ELF_RTYPE_CLASS_COPY iff TYPE should not be allowed to resolve to one
   of the main executable's symbols, as for a COPY reloc.  */
#define elf_machine_type_class(type)				\
  ((((type) == R_NIOS2_JUMP_SLOT				\
     || (type) == R_NIOS2_TLS_DTPMOD				\
     || (type) == R_NIOS2_TLS_DTPREL				\
     || (type) == R_NIOS2_TLS_TPREL) * ELF_RTYPE_CLASS_PLT)	\
   | (((type) == R_NIOS2_COPY) * ELF_RTYPE_CLASS_COPY)		\
   | (((type) == R_NIOS2_GLOB_DAT) * ELF_RTYPE_CLASS_EXTERN_PROTECTED_DATA))

/* A reloc type used for ld.so cmdline arg lookups to reject PLT entries.  */
#define ELF_MACHINE_JMP_SLOT  R_NIOS2_JUMP_SLOT

/* The Nios II never uses Elf32_Rel relocations.  */
#define ELF_MACHINE_NO_REL 1
#define ELF_MACHINE_NO_RELA 0

/* Fixup a PLT entry to bounce directly to the function at VALUE.  */

static inline Elf32_Addr
elf_machine_fixup_plt (struct link_map *map, lookup_t t,
		       const ElfW(Sym) *refsym, const ElfW(Sym) *sym,
		       const Elf32_Rela *reloc,
		       Elf32_Addr *reloc_addr, Elf32_Addr value)
{
  return *reloc_addr = value;
}

/* Return the final value of a plt relocation.  */
static inline Elf32_Addr
elf_machine_plt_value (struct link_map *map, const Elf32_Rela *reloc,
                       Elf32_Addr value)
{
  return value;
}

/* Names of the architecture-specific auditing callback functions.  */
#define ARCH_LA_PLTENTER nios2_gnu_pltenter
#define ARCH_LA_PLTEXIT nios2_gnu_pltexit

#endif /* dl_machine_h */

#ifdef RESOLVE_MAP

/* Perform the relocation specified by RELOC and SYM (which is fully resolved).
   LOADADDR is the load address of the object; INFO is an array indexed
   by DT_* of the .dynamic section info.  */

auto inline void __attribute__ ((always_inline))
elf_machine_rela (struct link_map *map, const ElfW(Rela) *reloc,
                  const ElfW(Sym) *sym, const struct r_found_version *version,
                  void *const reloc_addr_arg, int skip_ifunc)
{
  Elf32_Addr *const reloc_addr = reloc_addr_arg;
  const unsigned int r_type = ELF32_R_TYPE (reloc->r_info);

  if (__glibc_unlikely (r_type == R_NIOS2_RELATIVE))
    *reloc_addr = map->l_addr + reloc->r_addend;
  else if (__glibc_unlikely (r_type == R_NIOS2_NONE))
    return;
  else
    {
      const Elf32_Sym *const refsym = sym;
      struct link_map *sym_map = RESOLVE_MAP (&sym, version, r_type);
      Elf32_Addr value = SYMBOL_ADDRESS (sym_map, sym, true);

      switch (r_type)
	{
        case R_NIOS2_COPY:
          if (sym == NULL)
            /* This can happen in trace mode if an object could not be
               found.  */
            break;
          if (sym->st_size > refsym->st_size
              || (sym->st_size < refsym->st_size && GLRO(dl_verbose)))
            {
              const char *strtab;

              strtab = (const void *) D_PTR (map, l_info[DT_STRTAB]);
              _dl_error_printf ("\
%s: Symbol `%s' has different size in shared object, consider re-linking\n",
				rtld_progname ?: "<program name unknown>",
				strtab + refsym->st_name);
            }
          memcpy (reloc_addr_arg, (void *) value,
                  MIN (sym->st_size, refsym->st_size));
          break;
	case R_NIOS2_GLOB_DAT:
	case R_NIOS2_JUMP_SLOT:
# ifdef RTLD_BOOTSTRAP
          /* Fix weak undefined references.  */
          if (sym != NULL && sym->st_value == 0)
            *reloc_addr = 0;
          else
# endif
            *reloc_addr = value;
          break;
#ifndef RTLD_BOOTSTRAP
        case R_NIOS2_TLS_DTPMOD:
          /* Get the information from the link map returned by the
             resolv function.  */
          if (sym_map != NULL)
            *reloc_addr = sym_map->l_tls_modid;
          break;

        case R_NIOS2_TLS_DTPREL:
          *reloc_addr = reloc->r_addend + TLS_DTPREL_VALUE(sym);
          break;

        case R_NIOS2_TLS_TPREL:
          if (sym != NULL)
            {
              CHECK_STATIC_TLS (map, sym_map);
              *reloc_addr = reloc->r_addend + TLS_TPREL_VALUE(sym_map, sym);
            }
          break;
#endif
        case R_NIOS2_BFD_RELOC_32:
          *reloc_addr = value + reloc->r_addend;
          break;

	default:
          _dl_reloc_bad_type (map, r_type, 0);
          break;
	}
    }
}

auto inline void __attribute__((always_inline))
elf_machine_rela_relative (ElfW(Addr) l_addr, const ElfW(Rela) *reloc,
			   void *const reloc_addr_arg)
{
  Elf32_Addr *const reloc_addr = reloc_addr_arg;
  *reloc_addr = l_addr + reloc->r_addend;
}

auto inline void __attribute__((always_inline))
elf_machine_lazy_rel (struct link_map *map,
		      ElfW(Addr) l_addr, const ElfW(Rela) *reloc,
		      int skip_ifunc)
{
  Elf32_Addr *const reloc_addr = (void *) (l_addr + reloc->r_offset);
  if (ELF32_R_TYPE (reloc->r_info) == R_NIOS2_JUMP_SLOT)
    *reloc_addr += l_addr;
  else
    _dl_reloc_bad_type (map, ELF32_R_TYPE (reloc->r_info), 1);
}

#endif /* RESOLVE_MAP */
