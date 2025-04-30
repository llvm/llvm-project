/* Relocate a shared object and resolve its references to other loaded objects.
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

#include <errno.h>
#include <libintl.h>
#include <stdlib.h>
#include <unistd.h>
#include <ldsodefs.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <sys/types.h>
#include <_itoa.h>
#include <libc-pointer-arith.h>
#include "dynamic-link.h"

/* Statistics function.  */
#ifdef SHARED
# define bump_num_cache_relocations() ++GL(dl_num_cache_relocations)
#else
# define bump_num_cache_relocations() ((void) 0)
#endif


/* We are trying to perform a static TLS relocation in MAP, but it was
   dynamically loaded.  This can only work if there is enough surplus in
   the static TLS area already allocated for each running thread.  If this
   object's TLS segment is too big to fit, we fail with -1.  If it fits,
   we set MAP->l_tls_offset and return 0.
   A portion of the surplus static TLS can be optionally used to optimize
   dynamic TLS access (with TLSDESC or powerpc TLS optimizations).
   If OPTIONAL is true then TLS is allocated for such optimization and
   the caller must have a fallback in case the optional portion of surplus
   TLS runs out.  If OPTIONAL is false then the entire surplus TLS area is
   considered and the allocation only fails if that runs out.  */
int
_dl_try_allocate_static_tls (struct link_map *map, bool optional)
{
  /* If we've already used the variable with dynamic access, or if the
     alignment requirements are too high, fail.  */
  if (map->l_tls_offset == FORCED_DYNAMIC_TLS_OFFSET
      || map->l_tls_align > GLRO (dl_tls_static_align))
    {
    fail:
      return -1;
    }

#if TLS_TCB_AT_TP
  size_t freebytes = GLRO (dl_tls_static_size) - GL(dl_tls_static_used);
  if (freebytes < TLS_TCB_SIZE)
    goto fail;
  freebytes -= TLS_TCB_SIZE;

  size_t blsize = map->l_tls_blocksize + map->l_tls_firstbyte_offset;
  if (freebytes < blsize)
    goto fail;

  size_t n = (freebytes - blsize) / map->l_tls_align;

  /* Account optional static TLS surplus usage.  */
  size_t use = freebytes - n * map->l_tls_align - map->l_tls_firstbyte_offset;
  if (optional && use > GL(dl_tls_static_optional))
    goto fail;
  else if (optional)
    GL(dl_tls_static_optional) -= use;

  size_t offset = GL(dl_tls_static_used) + use;

  map->l_tls_offset = GL(dl_tls_static_used) = offset;
#elif TLS_DTV_AT_TP
  /* dl_tls_static_used includes the TCB at the beginning.  */
  size_t offset = (ALIGN_UP(GL(dl_tls_static_used)
			    - map->l_tls_firstbyte_offset,
			    map->l_tls_align)
		   + map->l_tls_firstbyte_offset);
  size_t used = offset + map->l_tls_blocksize;

  if (used > GLRO (dl_tls_static_size))
    goto fail;

  /* Account optional static TLS surplus usage.  */
  size_t use = used - GL(dl_tls_static_used);
  if (optional && use > GL(dl_tls_static_optional))
    goto fail;
  else if (optional)
    GL(dl_tls_static_optional) -= use;

  map->l_tls_offset = offset;
  map->l_tls_firstbyte_offset = GL(dl_tls_static_used);
  GL(dl_tls_static_used) = used;
#else
# error "Either TLS_TCB_AT_TP or TLS_DTV_AT_TP must be defined"
#endif

  /* If the object is not yet relocated we cannot initialize the
     static TLS region.  Delay it.  */
  if (map->l_real->l_relocated)
    {
#ifdef SHARED
      if (__builtin_expect (THREAD_DTV()[0].counter != GL(dl_tls_generation),
			    0))
	/* Update the slot information data for at least the generation of
	   the DSO we are allocating data for.  */
	(void) _dl_update_slotinfo (map->l_tls_modid);
#endif

      dl_init_static_tls (map);
    }
  else
    map->l_need_tls_init = 1;

  return 0;
}

/* This function intentionally does not return any value but signals error
   directly, as static TLS should be rare and code handling it should
   not be inlined as much as possible.  */
void
__attribute_noinline__
_dl_allocate_static_tls (struct link_map *map)
{
  if (map->l_tls_offset == FORCED_DYNAMIC_TLS_OFFSET
      || _dl_try_allocate_static_tls (map, false))
    {
      _dl_signal_error (0, map->l_name, NULL, N_("\
cannot allocate memory in static TLS block"));
    }
}

#if !THREAD_GSCOPE_IN_TCB
/* Initialize static TLS area and DTV for current (only) thread.
   libpthread implementations should provide their own hook
   to handle all threads.  */
void
_dl_nothread_init_static_tls (struct link_map *map)
{
#if TLS_TCB_AT_TP
  void *dest = (char *) THREAD_SELF - map->l_tls_offset;
#elif TLS_DTV_AT_TP
  void *dest = (char *) THREAD_SELF + map->l_tls_offset + TLS_PRE_TCB_SIZE;
#else
# error "Either TLS_TCB_AT_TP or TLS_DTV_AT_TP must be defined"
#endif

  /* Initialize the memory.  */
  memset (__mempcpy (dest, map->l_tls_initimage, map->l_tls_initimage_size),
	  '\0', map->l_tls_blocksize - map->l_tls_initimage_size);
}
#endif /* !THREAD_GSCOPE_IN_TCB */

#ifndef NESTING

    /* String table object symbols.  */

static struct link_map *glob_l;
static struct r_scope_elem **glob_scope;
static const char *glob_strtab;

/* This macro is used as a callback from the ELF_DYNAMIC_RELOCATE code.  */
#define RESOLVE_MAP(ref, version, r_type) \
    ((ELFW(ST_BIND) ((*ref)->st_info) != STB_LOCAL			      \
      && __glibc_likely (!dl_symbol_visibility_binds_local_p (*ref)))	      \
     ? ((__builtin_expect ((*ref) == glob_l->l_lookup_cache.sym, 0)		      \
	 && elf_machine_type_class (r_type) == glob_l->l_lookup_cache.type_class) \
	? (bump_num_cache_relocations (),				      \
	   (*ref) = glob_l->l_lookup_cache.ret,				      \
	   glob_l->l_lookup_cache.value)					      \
	: ({ lookup_t _lr;						      \
	     int _tc = elf_machine_type_class (r_type);			      \
	     glob_l->l_lookup_cache.type_class = _tc;			      \
	     glob_l->l_lookup_cache.sym = (*ref);				      \
	     const struct r_found_version *v = NULL;			      \
	     if ((version) != NULL && (version)->hash != 0)		      \
	       v = (version);						      \
	     _lr = _dl_lookup_symbol_x (glob_strtab + (*ref)->st_name, glob_l, (ref),   \
					glob_scope, v, _tc,			      \
					DL_LOOKUP_ADD_DEPENDENCY, NULL);      \
	     glob_l->l_lookup_cache.ret = (*ref);				      \
	     glob_l->l_lookup_cache.value = _lr; }))				      \
     : glob_l)

#include "dynamic-link.h"

#endif /* n NESTING */

void
_dl_relocate_object (struct link_map *l, struct r_scope_elem *scope[],
		     int reloc_mode, int consider_profiling)
{
  struct textrels
  {
    caddr_t start;
    size_t len;
    int prot;
    struct textrels *next;
  } *textrels = NULL;
  /* Initialize it to make the compiler happy.  */
  const char *errstring = NULL;
  int lazy = reloc_mode & RTLD_LAZY;
  int skip_ifunc = reloc_mode & __RTLD_NOIFUNC;

#ifdef SHARED
  /* If we are auditing, install the same handlers we need for profiling.  */
  if ((reloc_mode & __RTLD_AUDIT) == 0)
  {
    struct audit_ifaces *afct = GLRO(dl_audit);
    for (unsigned int cnt = 0; cnt < GLRO(dl_naudit); ++cnt)
    {
      /* Profiling is needed only if PLT hooks are provided.  */
      if (afct->ARCH_LA_PLTENTER != NULL || afct->ARCH_LA_PLTEXIT != NULL)
        consider_profiling = 1;
      afct = afct->next;
    }
  }
#elif defined PROF
  /* Never use dynamic linker profiling for gprof profiling code.  */
# define consider_profiling 0
#endif

  if (l->l_relocated)
    return;

  /* If DT_BIND_NOW is set relocate all references in this object.  We
     do not do this if we are profiling, of course.  */
  // XXX Correct for auditing?
  if (!consider_profiling
      && __builtin_expect (l->l_info[DT_BIND_NOW] != NULL, 0))
    lazy = 0;

  if (__glibc_unlikely (GLRO(dl_debug_mask) & DL_DEBUG_RELOC))
    _dl_debug_printf ("\nrelocation processing: %s%s\n",
		      DSO_FILENAME (l->l_name), lazy ? " (lazy)" : "");

  /* DT_TEXTREL is now in level 2 and might phase out at some time.
     But we rewrite the DT_FLAGS entry to a DT_TEXTREL entry to make
     testing easier and therefore it will be available at all time.  */
  if (__glibc_unlikely (l->l_info[DT_TEXTREL] != NULL))
    {
      /* Bletch.  We must make read-only segments writable
	 long enough to relocate them.  */
      const ElfW(Phdr) *ph;
      for (ph = l->l_phdr; ph < &l->l_phdr[l->l_phnum]; ++ph)
	if (ph->p_type == PT_LOAD && (ph->p_flags & PF_W) == 0)
	  {
	    struct textrels *newp;

	    newp = (struct textrels *) alloca (sizeof (*newp));
	    newp->len = ALIGN_UP (ph->p_vaddr + ph->p_memsz, GLRO(dl_pagesize))
			- ALIGN_DOWN (ph->p_vaddr, GLRO(dl_pagesize));
	    newp->start = PTR_ALIGN_DOWN (ph->p_vaddr, GLRO(dl_pagesize))
			  + (caddr_t) l->l_addr;

	    newp->prot = 0;
	    if (ph->p_flags & PF_R)
	      newp->prot |= PROT_READ;
	    if (ph->p_flags & PF_W)
	      newp->prot |= PROT_WRITE;
	    if (ph->p_flags & PF_X)
	      newp->prot |= PROT_EXEC;

	    if (__mprotect (newp->start, newp->len, newp->prot|PROT_WRITE) < 0)
	      {
		errstring = N_("cannot make segment writable for relocation");
	      call_error:
		_dl_signal_error (errno, l->l_name, NULL, errstring);
	      }

	    newp->next = textrels;
	    textrels = newp;
	  }
    }

  {
    /* Do the actual relocation of the object's GOT and other data.  */

#ifdef NESTING

    /* String table object symbols.  */
    const char *strtab = (const void *) D_PTR (l, l_info[DT_STRTAB]);

    /* This macro is used as a callback from the ELF_DYNAMIC_RELOCATE code.  */
#define RESOLVE_MAP(ref, version, r_type) \
    ((ELFW(ST_BIND) ((*ref)->st_info) != STB_LOCAL			      \
      && __glibc_likely (!dl_symbol_visibility_binds_local_p (*ref)))	      \
     ? ((__builtin_expect ((*ref) == l->l_lookup_cache.sym, 0)		      \
	 && elf_machine_type_class (r_type) == l->l_lookup_cache.type_class)  \
	? (bump_num_cache_relocations (),				      \
	   (*ref) = l->l_lookup_cache.ret,				      \
	   l->l_lookup_cache.value)					      \
	: ({ lookup_t _lr;						      \
	     int _tc = elf_machine_type_class (r_type);			      \
	     l->l_lookup_cache.type_class = _tc;			      \
	     l->l_lookup_cache.sym = (*ref);				      \
	     const struct r_found_version *v = NULL;			      \
	     if ((version) != NULL && (version)->hash != 0)		      \
	       v = (version);						      \
	     _lr = _dl_lookup_symbol_x (strtab + (*ref)->st_name, l, (ref),   \
					scope, v, _tc,			      \
					DL_LOOKUP_ADD_DEPENDENCY	      \
					| DL_LOOKUP_FOR_RELOCATE, NULL);      \
	     l->l_lookup_cache.ret = (*ref);				      \
	     l->l_lookup_cache.value = _lr; }))				      \
     : l)

#include "dynamic-link.h"

#else

    glob_l = l;
    glob_scope = scope;
    glob_strtab = (const void *) D_PTR (glob_l, l_info[DT_STRTAB]);

#endif /* NESTING */

    ELF_DYNAMIC_RELOCATE (l, lazy, consider_profiling, skip_ifunc
#ifndef NESTING
			  , NULL
#endif
			  );

#ifndef PROF
    if (__glibc_unlikely (consider_profiling)
	&& l->l_info[DT_PLTRELSZ] != NULL)
      {
	/* Allocate the array which will contain the already found
	   relocations.  If the shared object lacks a PLT (for example
	   if it only contains lead function) the l_info[DT_PLTRELSZ]
	   will be NULL.  */
	size_t sizeofrel = l->l_info[DT_PLTREL]->d_un.d_val == DT_RELA
			   ? sizeof (ElfW(Rela))
			   : sizeof (ElfW(Rel));
	size_t relcount = l->l_info[DT_PLTRELSZ]->d_un.d_val / sizeofrel;
	l->l_reloc_result = calloc (sizeof (l->l_reloc_result[0]), relcount);

	if (l->l_reloc_result == NULL)
	  {
	    errstring = N_("\
%s: out of memory to store relocation results for %s\n");
	    _dl_fatal_printf (errstring, RTLD_PROGNAME, l->l_name);
	  }
      }
#endif
  }

  /* Mark the object so we know this work has been done.  */
  l->l_relocated = 1;

  /* Undo the segment protection changes.  */
  while (__builtin_expect (textrels != NULL, 0))
    {
      if (__mprotect (textrels->start, textrels->len, textrels->prot) < 0)
	{
	  errstring = N_("cannot restore segment prot after reloc");
	  goto call_error;
	}

#ifdef CLEAR_CACHE
      CLEAR_CACHE (textrels->start, textrels->start + textrels->len);
#endif

      textrels = textrels->next;
    }

  /* In case we can protect the data now that the relocations are
     done, do it.  */
  if (l->l_relro_size != 0)
    _dl_protect_relro (l);
}


void
_dl_protect_relro (struct link_map *l)
{
  ElfW(Addr) start = ALIGN_DOWN((l->l_addr
				 + l->l_relro_addr),
				GLRO(dl_pagesize));
  ElfW(Addr) end = ALIGN_DOWN((l->l_addr
			       + l->l_relro_addr
			       + l->l_relro_size),
			      GLRO(dl_pagesize));
  if (start != end
      && __mprotect ((void *) start, end - start, PROT_READ) < 0)
    {
      static const char errstring[] = N_("\
cannot apply additional memory protection after relocation");
      _dl_signal_error (errno, l->l_name, NULL, errstring);
    }
}

void
__attribute_noinline__
_dl_reloc_bad_type (struct link_map *map, unsigned int type, int plt)
{
#define DIGIT(b)	_itoa_lower_digits[(b) & 0xf];

  /* XXX We cannot translate these messages.  */
  static const char msg[2][32
#if __ELF_NATIVE_CLASS == 64
			   + 6
#endif
  ] = { "unexpected reloc type 0x",
	"unexpected PLT reloc type 0x" };
  char msgbuf[sizeof (msg[0])];
  char *cp;

  cp = __stpcpy (msgbuf, msg[plt]);
#if __ELF_NATIVE_CLASS == 64
  if (__builtin_expect(type > 0xff, 0))
    {
      *cp++ = DIGIT (type >> 28);
      *cp++ = DIGIT (type >> 24);
      *cp++ = DIGIT (type >> 20);
      *cp++ = DIGIT (type >> 16);
      *cp++ = DIGIT (type >> 12);
      *cp++ = DIGIT (type >> 8);
    }
#endif
  *cp++ = DIGIT (type >> 4);
  *cp++ = DIGIT (type);
  *cp = '\0';

  _dl_signal_error (0, map->l_name, NULL, msgbuf);
}
