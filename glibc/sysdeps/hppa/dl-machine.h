/* Machine-dependent ELF dynamic relocation inline functions.  PA-RISC version.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
   Contributed by David Huggins-Daines <dhd@debian.org>
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
#define dl_machine_h 1

#define ELF_MACHINE_NAME "hppa"

#include <sys/param.h>
#include <assert.h>
#include <string.h>
#include <link.h>
#include <errno.h>
#include <dl-fptr.h>
#include <abort-instr.h>
#include <tls.h>

/* These two definitions must match the definition of the stub in
   bfd/elf32-hppa.c (see plt_stub[]).

   a. Define the size of the *entire* stub we place at the end of the PLT
   table (right up against the GOT).

   b. Define the number of bytes back from the GOT to the entry point of
   the PLT stub. You see the PLT stub must be entered in the middle
   so it can depwi to find it's own address (long jump stub)

   c. Define the size of a single PLT entry so we can jump over the
   last entry to get the stub address */

#define SIZEOF_PLT_STUB (7*4)
#define GOT_FROM_PLT_STUB (4*4)
#define PLT_ENTRY_SIZE (2*4)

/* The gp slot in the function descriptor contains the relocation offset
   before resolution.  To distinguish between a resolved gp value and an
   unresolved relocation offset we set an unused bit in the relocation
   offset.  This would allow us to do a synchronzied two word update
   using this bit (interlocked update), but instead of waiting for the
   update we simply recompute the gp value given that we know the ip.  */
#define PA_GP_RELOC 1

/* Initialize the function descriptor table before relocations */
static inline void
__hppa_init_bootstrap_fdesc_table (struct link_map *map)
{
  ElfW(Addr) *boot_table;

  /* Careful: this will be called before got has been relocated... */
  ELF_MACHINE_LOAD_ADDRESS(boot_table,_dl_boot_fptr_table);

  map->l_mach.fptr_table_len = ELF_MACHINE_BOOT_FPTR_TABLE_LEN;
  map->l_mach.fptr_table = boot_table;
}

#define ELF_MACHINE_BEFORE_RTLD_RELOC(dynamic_info)		\
	__hppa_init_bootstrap_fdesc_table (BOOTSTRAP_MAP);	\
	_dl_fptr_init();

/* Return nonzero iff ELF header is compatible with the running host.  */
static inline int
elf_machine_matches_host (const Elf32_Ehdr *ehdr)
{
  return ehdr->e_machine == EM_PARISC;
}

/* Return the link-time address of _DYNAMIC.  */
static inline Elf32_Addr
elf_machine_dynamic (void) __attribute__ ((const));

static inline Elf32_Addr
elf_machine_dynamic (void)
{
  Elf32_Addr dynamic;

  asm ("bl	1f,%0\n"
"	addil	L'_GLOBAL_OFFSET_TABLE_ - ($PIC_pcrel$0 - 1),%0\n"
"1:	ldw	R'_GLOBAL_OFFSET_TABLE_ - ($PIC_pcrel$0 - 5)(%%r1),%0\n"
       : "=r" (dynamic) : : "r1");

  return dynamic;
}

/* Return the run-time load address of the shared object.  */
static inline Elf32_Addr
elf_machine_load_address (void) __attribute__ ((const));

static inline Elf32_Addr
elf_machine_load_address (void)
{
  Elf32_Addr dynamic;

  asm (
"	bl	1f,%0\n"
"	addil	L'_DYNAMIC - ($PIC_pcrel$0 - 1),%0\n"
"1:	ldo	R'_DYNAMIC - ($PIC_pcrel$0 - 5)(%%r1),%0\n"
   : "=r" (dynamic) : : "r1");

  return dynamic - elf_machine_dynamic ();
}

/* Fixup a PLT entry to bounce directly to the function at VALUE. */
static inline struct fdesc __attribute__ ((always_inline))
elf_machine_fixup_plt (struct link_map *map, lookup_t t,
		       const ElfW(Sym) *refsym, const ElfW(Sym) *sym,
		       const Elf32_Rela *reloc,
		       Elf32_Addr *reloc_addr, struct fdesc value)
{
  volatile Elf32_Addr *rfdesc = reloc_addr;
  /* map is the link_map for the caller, t is the link_map for the object
     being called */

  /* We would like the function descriptor to be double word aligned.  This
     helps performance (ip and gp then reside on the same cache line) and
     we can update the pair atomically with a single store.  The linker
     now ensures this alignment but we still have to handle old code.  */
  if ((unsigned int)reloc_addr & 7)
    {
      /* Need to ensure that the gp is visible before the code
         entry point is updated */
      rfdesc[1] = value.gp;
      atomic_full_barrier();
      rfdesc[0] = value.ip;
    }
  else
    {
      /* Update pair atomically with floating point store.  */
      union { ElfW(Word) v[2]; double d; } u;

      u.v[0] = value.ip;
      u.v[1] = value.gp;
      *(volatile double *)rfdesc = u.d;
    }
  return value;
}

/* Return the final value of a plt relocation.  */
static inline struct fdesc
elf_machine_plt_value (struct link_map *map, const Elf32_Rela *reloc,
		       struct fdesc value)
{
  /* We are rela only, return a function descriptor as a plt entry. */
  return (struct fdesc) { value.ip + reloc->r_addend, value.gp };
}

/* Set up the loaded object described by L so its unrelocated PLT
   entries will jump to the on-demand fixup code in dl-runtime.c.  */

static inline int
elf_machine_runtime_setup (struct link_map *l, int lazy, int profile)
{
  Elf32_Addr *got = NULL;
  Elf32_Addr l_addr, iplt, jmprel, end_jmprel, r_type, r_sym;
  const Elf32_Rela *reloc;
  struct fdesc *fptr;
  static union {
    unsigned char c[8];
    Elf32_Addr i[2];
  } sig = {{0x00,0xc0,0xff,0xee, 0xde,0xad,0xbe,0xef}};

  /* If we don't have a PLT we can just skip all this... */
  if (__builtin_expect (l->l_info[DT_JMPREL] == NULL,0))
    return lazy;

  /* All paths use these values */
  l_addr = l->l_addr;
  jmprel = D_PTR(l, l_info[DT_JMPREL]);
  end_jmprel = jmprel + l->l_info[DT_PLTRELSZ]->d_un.d_val;

  extern void _dl_runtime_resolve (void);
  extern void _dl_runtime_profile (void);

  /* Linking lazily */
  if (lazy)
    {
      /* FIXME: Search for the got, but backwards through the relocs, technically we should
	 find it on the first try. However, assuming the relocs got out of order the
	 routine is made a bit more robust by searching them all in case of failure. */
      for (iplt = (end_jmprel - sizeof (Elf32_Rela)); iplt >= jmprel; iplt -= sizeof (Elf32_Rela))
	{

	  reloc = (const Elf32_Rela *) iplt;
	  r_type = ELF32_R_TYPE (reloc->r_info);
	  r_sym = ELF32_R_SYM (reloc->r_info);

	  got = (Elf32_Addr *) (reloc->r_offset + l_addr + PLT_ENTRY_SIZE + SIZEOF_PLT_STUB);

	  /* If we aren't an IPLT, and we aren't NONE then it's a bad reloc */
	  if (__builtin_expect (r_type != R_PARISC_IPLT, 0))
	    {
	      if (__builtin_expect (r_type != R_PARISC_NONE, 0))
		_dl_reloc_bad_type (l, r_type, 1);
	      continue;
	    }

	  /* Check for the plt_stub that binutils placed here for us
	     to use with _dl_runtime_resolve  */
	  if (got[-2] != sig.i[0] || got[-1] != sig.i[1])
	    {
	      got = NULL; /* Not the stub... keep looking */
	    }
	  else
	    {
	      /* Found the GOT! */
	      register Elf32_Addr ltp __asm__ ("%r19");

	      /* Identify this shared object. Second entry in the got. */
	      got[1] = (Elf32_Addr) l;

	      /* This function will be called to perform the relocation. */
	      if (__builtin_expect (!profile, 1))
		{
		  /* If a static application called us, then _dl_runtime_resolve is not
		     a function descriptor, but the *real* address of the function... */
		  if((unsigned long) &_dl_runtime_resolve & 3)
		    {
		      got[-2] = (Elf32_Addr) ((struct fdesc *)
				  ((unsigned long) &_dl_runtime_resolve & ~3))->ip;
		    }
		  else
		    {
		      /* Static executable! */
		      got[-2] = (Elf32_Addr) &_dl_runtime_resolve;
		    }
		}
	      else
		{
		  if (GLRO(dl_profile) != NULL
		      && _dl_name_match_p (GLRO(dl_profile), l))
		    {
		      /* This is the object we are looking for.  Say that
			 we really want profiling and the timers are
			 started.  */
		      GL(dl_profile_map) = l;
		    }

		  if((unsigned long) &_dl_runtime_profile & 3)
		    {
		      got[-2] = (Elf32_Addr) ((struct fdesc *)
				  ((unsigned long) &_dl_runtime_profile & ~3))->ip;
		    }
		  else
		    {
		      /* Static executable */
		      got[-2] = (Elf32_Addr) &_dl_runtime_profile;
		    }
		}
	      /* Plunk in the gp of this function descriptor so we
		 can make the call to _dl_runtime_xxxxxx */
	      got[-1] = ltp;
	      break;
	      /* Done looking for the GOT, and stub is setup */
	    } /* else we found the GOT */
	} /* for, walk the relocs backwards */

      if(!got)
	return 0; /* No lazy linking for you! */

      /* Process all the relocs, now that we know the GOT... */
      for (iplt = jmprel; iplt < end_jmprel; iplt += sizeof (Elf32_Rela))
	{
	  reloc = (const Elf32_Rela *) iplt;
	  r_type = ELF32_R_TYPE (reloc->r_info);
	  r_sym = ELF32_R_SYM (reloc->r_info);

	  if (__builtin_expect (r_type == R_PARISC_IPLT, 1))
	    {
	      fptr = (struct fdesc *) (reloc->r_offset + l_addr);
	      if (r_sym != 0)
		{
		  /* Relocate the pointer to the stub.  */
		  fptr->ip = (Elf32_Addr) got - GOT_FROM_PLT_STUB;

		  /* Instead of the LTP value, we put the reloc offset
		     here.  The trampoline code will load the proper
		     LTP and pass the reloc offset to the fixup
		     function.  */
		  fptr->gp = (iplt - jmprel) | PA_GP_RELOC;
		} /* r_sym != 0 */
	      else
		{
		  /* Relocate this *ABS* entry.  */
		  fptr->ip = reloc->r_addend + l_addr;
		  fptr->gp = D_PTR (l, l_info[DT_PLTGOT]);
		}
	    } /* r_type == R_PARISC_IPLT */
	} /* for all the relocations */
    } /* if lazy */
  else
    {
      for (iplt = jmprel; iplt < end_jmprel; iplt += sizeof (Elf32_Rela))
	{
	  reloc = (const Elf32_Rela *) iplt;
	  r_type = ELF32_R_TYPE (reloc->r_info);
	  r_sym = ELF32_R_SYM (reloc->r_info);

	  if (__builtin_expect ((r_type == R_PARISC_IPLT) && (r_sym == 0), 1))
	    {
	      fptr = (struct fdesc *) (reloc->r_offset + l_addr);
	      /* Relocate this *ABS* entry, set only the gp, the rest is set later
		 when elf_machine_rela_relative is called (WITHOUT the linkmap)  */
	      fptr->gp = D_PTR (l, l_info[DT_PLTGOT]);
	    } /* r_type == R_PARISC_IPLT */
	} /* for all the relocations */
    }
  return lazy;
}


/* Names of the architecture-specific auditing callback functions.  */
#define ARCH_LA_PLTENTER hppa_gnu_pltenter
#define ARCH_LA_PLTEXIT hppa_gnu_pltexit

/* Adjust DL_STACK_END to get value we want in __libc_stack_end.  */
#define DL_STACK_END(cookie) \
  ((void *) (((long) (cookie)) + 0x160))

/* Initial entry point code for the dynamic linker.
   The C function `_dl_start' is the real entry point;
   its return value is the user program's entry point.  */

#define RTLD_START \
/* Set up dp for any non-PIC lib constructors that may be called.  */	\
static struct link_map * __attribute__((used))				\
set_dp (struct link_map *map)						\
{									\
  register Elf32_Addr dp asm ("%r27");					\
  dp = D_PTR (map, l_info[DT_PLTGOT]);					\
  asm volatile ("" : : "r" (dp));					\
  return map;								\
}									\
									\
asm (									\
"	.text\n"							\
"	.globl _start\n"						\
"	.type _start,@function\n"					\
"_start:\n"								\
	/* The kernel does not give us an initial stack frame. */	\
"	ldo	64(%sp),%sp\n"						\
	/* Save the relevant arguments (yes, those are the correct	\
	   registers, the kernel is weird) in their stack slots. */	\
"	stw	%r25,-40(%sp)\n" /* argc */				\
"	stw	%r24,-44(%sp)\n" /* argv */				\
									\
	/* We need the LTP, and we need it now.				\
	   $PIC_pcrel$0 points 8 bytes past the current instruction,	\
	   just like a branch reloc.  This sequence gets us the		\
	   runtime address of _DYNAMIC. */				\
"	bl	0f,%r19\n"						\
"	addil	L'_DYNAMIC - ($PIC_pcrel$0 - 1),%r19\n"			\
"0:	ldo	R'_DYNAMIC - ($PIC_pcrel$0 - 5)(%r1),%r26\n"		\
									\
	/* The link time address is stored in the first entry of the	\
	   GOT.  */							\
"	addil	L'_GLOBAL_OFFSET_TABLE_ - ($PIC_pcrel$0 - 9),%r19\n"	\
"	ldw	R'_GLOBAL_OFFSET_TABLE_ - ($PIC_pcrel$0 - 13)(%r1),%r20\n" \
									\
"	sub	%r26,%r20,%r20\n"	/* Calculate load offset */	\
									\
	/* Rummage through the dynamic entries, looking for		\
	   DT_PLTGOT.  */						\
"	ldw,ma	8(%r26),%r19\n"						\
"1:	cmpib,=,n 3,%r19,2f\n"	/* tag == DT_PLTGOT? */			\
"	cmpib,<>,n 0,%r19,1b\n"						\
"	ldw,ma	8(%r26),%r19\n"						\
									\
	/* Uh oh!  We didn't find one.  Abort. */			\
"	iitlbp	%r0,(%sr0,%r0)\n"					\
									\
"2:	ldw	-4(%r26),%r19\n"	/* Found it, load value. */	\
"	add	%r19,%r20,%r19\n"	/* And add the load offset. */	\
									\
	/* Our initial stack layout is rather different from everyone	\
	   else's due to the unique PA-RISC ABI.  As far as I know it	\
	   looks like this:						\
									\
	   -----------------------------------  (this frame created above) \
	   |         32 bytes of magic       |				\
	   |---------------------------------|				\
	   | 32 bytes argument/sp save area  |				\
	   |---------------------------------|  ((current->mm->env_end)	\
	   |         N bytes of slack        |	 + 63 & ~63)		\
	   |---------------------------------|				\
	   |      envvar and arg strings     |				\
	   |---------------------------------|				\
	   |	    ELF auxiliary info	     |				\
	   |         (up to 28 words)        |				\
	   |---------------------------------|				\
	   |  Environment variable pointers  |				\
	   |         upwards to NULL	     |				\
	   |---------------------------------|				\
	   |        Argument pointers        |				\
	   |         upwards to NULL	     |				\
	   |---------------------------------|				\
	   |          argc (1 word)          |				\
	   -----------------------------------				\
									\
	  So, obviously, we can't just pass %sp to _dl_start.  That's	\
	  okay, argv-4 will do just fine.				\
									\
	  The pleasant part of this is that if we need to skip		\
	  arguments we can just decrement argc and move argv, because	\
	  the stack pointer is utterly unrelated to the location of	\
	  the environment and argument vectors. */			\
									\
	/* This is always within range so we'll be okay. */		\
"	bl	_dl_start,%rp\n"					\
"	ldo	-4(%r24),%r26\n"					\
									\
"	.globl _dl_start_user\n"					\
"	.type _dl_start_user,@function\n"				\
"_dl_start_user:\n"							\
	/* Save the entry point in %r3. */				\
"	copy	%ret0,%r3\n"						\
									\
	/* See if we were called as a command with the executable file	\
	   name as an extra leading argument. */			\
"	addil	LT'_dl_skip_args,%r19\n"				\
"	ldw	RT'_dl_skip_args(%r1),%r20\n"				\
"	ldw	0(%r20),%r20\n"						\
									\
"	ldw	-40(%sp),%r25\n"	/* argc */			\
"	comib,=	0,%r20,.Lnofix\n"	/* FIXME: Mispredicted branch */\
"	ldw	-44(%sp),%r24\n"	/* argv (delay slot) */		\
									\
"	sub	%r25,%r20,%r25\n"					\
"	stw	%r25,-40(%sp)\n"					\
"	sh2add	%r20,%r24,%r24\n"					\
"	stw	%r24,-44(%sp)\n"					\
									\
".Lnofix:\n"								\
"	addil	LT'_rtld_local,%r19\n"					\
"	ldw	RT'_rtld_local(%r1),%r26\n"				\
"	bl	set_dp, %r2\n"						\
"	ldw	0(%r26),%r26\n"						\
									\
	/* Call _dl_init(_dl_loaded, argc, argv, envp). */		\
"	copy	%r28,%r26\n"						\
									\
	/* envp = argv + argc + 1 */					\
"	sh2add	%r25,%r24,%r23\n"					\
"	bl	_dl_init,%r2\n"						\
"	ldo	4(%r23),%r23\n"	/* delay slot */			\
									\
	/* Reload argc, argv to the registers start.S expects.  */	\
"	ldw	-40(%sp),%r25\n"					\
"	ldw	-44(%sp),%r24\n"					\
									\
	/* _dl_fini is a local function in the loader, so we construct	\
	   a false OPD here and pass this to the application.  */	\
	/* FIXME: Should be able to use P%, and LR RR to have the	\
	   the linker construct a proper OPD.  */			\
"	.section .data\n"						\
"__dl_fini_plabel:\n"							\
"	.word	_dl_fini\n"						\
"	.word	0xdeadbeef\n"						\
"	.previous\n"							\
									\
	/* %r3 contains a function pointer, we need to mask out the	\
	   lower bits and load the gp and jump address. */		\
"	depi	0,31,2,%r3\n"						\
"	ldw	0(%r3),%r2\n"						\
"	addil	LT'__dl_fini_plabel,%r19\n"				\
"	ldw	RT'__dl_fini_plabel(%r1),%r23\n"			\
"	stw	%r19,4(%r23)\n"						\
"	ldw	4(%r3),%r19\n"	/* load the object's gp */		\
"	bv	%r0(%r2)\n"						\
"	depi	2,31,2,%r23\n"	/* delay slot */			\
);

/* ELF_RTYPE_CLASS_PLT iff TYPE describes relocation of a PLT entry or
   a TLS variable, so references should not be allowed to define the value.
   ELF_RTYPE_CLASS_COPY iff TYPE should not be allowed to resolve to one
   of the main executable's symbols, as for a COPY reloc.  */
#if !defined RTLD_BOOTSTRAP
# define elf_machine_type_class(type)				\
  ((((type) == R_PARISC_IPLT					\
  || (type) == R_PARISC_EPLT					\
  || (type) == R_PARISC_TLS_DTPMOD32				\
  || (type) == R_PARISC_TLS_DTPOFF32				\
  || (type) == R_PARISC_TLS_TPREL32)				\
  * ELF_RTYPE_CLASS_PLT)					\
  | (((type) == R_PARISC_COPY) * ELF_RTYPE_CLASS_COPY))
#else
#define elf_machine_type_class(type)				\
 ((((type) == R_PARISC_IPLT					\
   || (type) == R_PARISC_EPLT)					\
   * ELF_RTYPE_CLASS_PLT)					\
   | (((type) == R_PARISC_COPY) * ELF_RTYPE_CLASS_COPY))
#endif

/* Used by the runtime in fixup to figure out if reloc is *really* PLT */
#define ELF_MACHINE_JMP_SLOT R_PARISC_IPLT
#define ELF_MACHINE_SIZEOF_JMP_SLOT PLT_ENTRY_SIZE

/* We only use RELA. */
#define ELF_MACHINE_NO_REL 1
#define ELF_MACHINE_NO_RELA 0

/* Return the address of the entry point. */
#define ELF_MACHINE_START_ADDRESS(map, start)			\
({								\
	ElfW(Addr) addr;					\
	DL_DT_FUNCTION_ADDRESS(map, start, static, addr)	\
	addr;							\
})

/* We define an initialization functions.  This is called very early in
 *    _dl_sysdep_start.  */
#define DL_PLATFORM_INIT dl_platform_init ()

static inline void __attribute__ ((unused))
dl_platform_init (void)
{
	if (GLRO(dl_platform) != NULL && *GLRO(dl_platform) == '\0')
	/* Avoid an empty string which would disturb us.  */
		GLRO(dl_platform) = NULL;
}

#endif /* !dl_machine_h */

/* These are only actually used where RESOLVE_MAP is defined, anyway. */
#ifdef RESOLVE_MAP

#define reassemble_21(as21) \
  (  (((as21) & 0x100000) >> 20) \
   | (((as21) & 0x0ffe00) >> 8) \
   | (((as21) & 0x000180) << 7) \
   | (((as21) & 0x00007c) << 14) \
   | (((as21) & 0x000003) << 12))

#define reassemble_14(as14) \
  (  (((as14) & 0x1fff) << 1) \
   | (((as14) & 0x2000) >> 13))

auto void __attribute__((always_inline))
elf_machine_rela (struct link_map *map,
		  const Elf32_Rela *reloc,
		  const Elf32_Sym *sym,
		  const struct r_found_version *version,
		  void *const reloc_addr_arg,
		  int skip_ifunc)
{
  Elf32_Addr *const reloc_addr = reloc_addr_arg;
  const Elf32_Sym *const refsym = sym;
  unsigned long const r_type = ELF32_R_TYPE (reloc->r_info);
  struct link_map *sym_map;
  Elf32_Addr value;

# if !defined RTLD_BOOTSTRAP && !defined HAVE_Z_COMBRELOC && !defined SHARED
  /* This is defined in rtld.c, but nowhere in the static libc.a; make the
     reference weak so static programs can still link.  This declaration
     cannot be done when compiling rtld.c (i.e.  #ifdef RTLD_BOOTSTRAP)
     because rtld.c contains the common defn for _dl_rtld_map, which is
     incompatible with a weak decl in the same file.  */
  weak_extern (GL(dl_rtld_map));
# endif

  /* RESOLVE_MAP will return a null value for undefined syms, and
     non-null for all other syms.  In particular, relocs with no
     symbol (symbol index of zero), also called *ABS* relocs, will be
     resolved to MAP.  (The first entry in a symbol table is all
     zeros, and an all zero Elf32_Sym has a binding of STB_LOCAL.)
     See RESOLVE_MAP definition in elf/dl-reloc.c  */
# ifdef RTLD_BOOTSTRAP
  /* RESOLVE_MAP in rtld.c doesn't have the local sym test.  */
  sym_map = (ELF32_ST_BIND (sym->st_info) != STB_LOCAL
	     ? RESOLVE_MAP (&sym, version, r_type) : map);
# else
  sym_map = RESOLVE_MAP (&sym, version, r_type);
# endif

  if (sym_map)
    {
      value = SYMBOL_ADDRESS (sym_map, sym, true);
      value += reloc->r_addend;
    }
  else
    value = 0;

  switch (r_type)
    {
    case R_PARISC_DIR32:
      /* .eh_frame can have unaligned relocs.  */
      if ((unsigned long) reloc_addr_arg & 3)
	{
	  char *rel_addr = (char *) reloc_addr_arg;
	  rel_addr[0] = value >> 24;
	  rel_addr[1] = value >> 16;
	  rel_addr[2] = value >> 8;
	  rel_addr[3] = value;
	  return;
	}
      break;

    case R_PARISC_DIR21L:
      {
	unsigned int insn = *(unsigned int *)reloc_addr;
	value = (SYMBOL_ADDRESS (sym_map, sym, true)
		 + ((reloc->r_addend + 0x1000) & -0x2000));
	value = value >> 11;
	insn = (insn &~ 0x1fffff) | reassemble_21 (value);
	*(unsigned int *)reloc_addr = insn;
      }
      return;

    case R_PARISC_DIR14R:
      {
	unsigned int insn = *(unsigned int *)reloc_addr;
	value = ((SYMBOL_ADDRESS (sym_map, sym, true) & 0x7ff)
		 + (((reloc->r_addend & 0x1fff) ^ 0x1000) - 0x1000));
	insn = (insn &~ 0x3fff) | reassemble_14 (value);
	*(unsigned int *)reloc_addr = insn;
      }
      return;

    case R_PARISC_PLABEL32:
      /* Easy rule: If there is a symbol and it is global, then we
	 need to make a dynamic function descriptor.  Otherwise we
	 have the address of a PLT slot for a local symbol which we
	 know to be unique. */
      if (sym == NULL
	  || sym_map == NULL
	  || ELF32_ST_BIND (sym->st_info) == STB_LOCAL)
	{
	  break;
	}
      /* Set bit 30 to indicate to $$dyncall that this is a PLABEL.
	 We have to do this outside of the generic function descriptor
	 code, since it doesn't know about our requirement for setting
	 protection bits */
      value = (Elf32_Addr)((unsigned int)_dl_make_fptr (sym_map, sym, value) | 2);
      break;

    case R_PARISC_PLABEL21L:
    case R_PARISC_PLABEL14R:
      {
	unsigned int insn = *(unsigned int *)reloc_addr;

	if (__builtin_expect (sym == NULL, 0))
	  break;

	value = (Elf32_Addr)((unsigned int)_dl_make_fptr (sym_map, sym, value) | 2);

	if (r_type == R_PARISC_PLABEL21L)
	  {
	    value >>= 11;
	    insn = (insn &~ 0x1fffff) | reassemble_21 (value);
	  }
	else
	  {
	    value &= 0x7ff;
	    insn = (insn &~ 0x3fff) | reassemble_14 (value);
	  }

	*(unsigned int *)reloc_addr = insn;
      }
      return;

    case R_PARISC_IPLT:
      if (__builtin_expect (sym_map != NULL, 1))
	{
	  elf_machine_fixup_plt (NULL, sym_map, NULL, NULL, reloc, reloc_addr,
				 DL_FIXUP_MAKE_VALUE(sym_map, value));
	}
      else
	{
	  /* If we get here, it's a (weak) undefined sym.  */
	  elf_machine_fixup_plt (NULL, map, NULL, NULL, reloc, reloc_addr,
				 DL_FIXUP_MAKE_VALUE(map, value));
	}
      return;

    case R_PARISC_COPY:
      if (__builtin_expect (sym == NULL, 0))
	/* This can happen in trace mode if an object could not be
	   found.  */
	break;
      if (__builtin_expect (sym->st_size > refsym->st_size, 0)
	  || (__builtin_expect (sym->st_size < refsym->st_size, 0)
	      && __builtin_expect (GLRO(dl_verbose), 0)))
	{
	  const char *strtab;

	  strtab = (const char *) D_PTR (map, l_info[DT_STRTAB]);
	  _dl_error_printf ("%s: Symbol `%s' has different size in shared object, "
			    "consider re-linking\n",
			    RTLD_PROGNAME, strtab + refsym->st_name);
	}
      memcpy (reloc_addr_arg, (void *) value,
	      MIN (sym->st_size, refsym->st_size));
      return;

#if !defined RTLD_BOOTSTRAP
    case R_PARISC_TLS_DTPMOD32:
      value = sym_map->l_tls_modid;
      break;

    case R_PARISC_TLS_DTPOFF32:
      /* During relocation all TLS symbols are defined and used.
	 Therefore the offset is already correct.  */
      if (sym != NULL)
	*reloc_addr = sym->st_value + reloc->r_addend;
      return;

    case R_PARISC_TLS_TPREL32:
      /* The offset is negative, forward from the thread pointer */
      if (sym != NULL)
	{
	  CHECK_STATIC_TLS (map, sym_map);
	  value = sym_map->l_tls_offset + sym->st_value + reloc->r_addend;
	}
      break;
#endif	/* use TLS */

    case R_PARISC_NONE:	/* Alright, Wilbur. */
      return;

    default:
      _dl_reloc_bad_type (map, r_type, 0);
    }

  *reloc_addr = value;
}

/* hppa doesn't have an R_PARISC_RELATIVE reloc, but uses relocs with
   ELF32_R_SYM (info) == 0 for a similar purpose.  */
auto void __attribute__((always_inline))
elf_machine_rela_relative (Elf32_Addr l_addr,
			   const Elf32_Rela *reloc,
			   void *const reloc_addr_arg)
{
  unsigned long const r_type = ELF32_R_TYPE (reloc->r_info);
  Elf32_Addr *const reloc_addr = reloc_addr_arg;
  static char msgbuf[] = { "Unknown" };
  struct link_map map;
  Elf32_Addr value;

  value = l_addr + reloc->r_addend;

  if (ELF32_R_SYM (reloc->r_info) != 0){
    _dl_error_printf ("%s: In elf_machine_rela_relative "
		      "ELF32_R_SYM (reloc->r_info) != 0. Aborting.",
		      RTLD_PROGNAME);
    ABORT_INSTRUCTION;  /* Crash. */
  }

  switch (r_type)
    {
    case R_PARISC_DIR32:
      /* .eh_frame can have unaligned relocs.  */
      if ((unsigned long) reloc_addr_arg & 3)
	{
	  char *rel_addr = (char *) reloc_addr_arg;
	  rel_addr[0] = value >> 24;
	  rel_addr[1] = value >> 16;
	  rel_addr[2] = value >> 8;
	  rel_addr[3] = value;
	  return;
	}
      break;

    case R_PARISC_PLABEL32:
      break;

    case R_PARISC_IPLT: /* elf_machine_runtime_setup already set gp */
      break;

    case R_PARISC_NONE:
      return;

    default: /* Bad reloc, map unknown (really it's the current map) */
      map.l_name = msgbuf;
      _dl_reloc_bad_type (&map, r_type, 0);
      return;
    }

  *reloc_addr = value;
}

auto void __attribute__((always_inline))
elf_machine_lazy_rel (struct link_map *map,
		      Elf32_Addr l_addr, const Elf32_Rela *reloc,
		      int skip_ifunc)
{
  /* We don't have anything to do here.  elf_machine_runtime_setup has
     done all the relocs already.  */
}

#endif /* RESOLVE_MAP */
