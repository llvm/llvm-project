/* Look up a symbol in the loaded objects.
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

#include <alloca.h>
#include <libintl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <ldsodefs.h>
#include <dl-hash.h>
#include <dl-machine.h>
#include <sysdep-cancel.h>
#include <libc-lock.h>
#include <tls.h>
#include <atomic.h>
#include <elf_machine_sym_no_match.h>

#include <assert.h>

#define VERSTAG(tag)	(DT_NUM + DT_THISPROCNUM + DT_VERSIONTAGIDX (tag))

struct sym_val
  {
    const ElfW(Sym) *s;
    struct link_map *m;
  };


/* Statistics function.  */
#ifdef SHARED
# define bump_num_relocations() ++GL(dl_num_relocations)
#else
# define bump_num_relocations() ((void) 0)
#endif

/* Utility function for do_lookup_x. The caller is called with undef_name,
   ref, version, flags and type_class, and those are passed as the first
   five arguments. The caller then computes sym, symidx, strtab, and map
   and passes them as the next four arguments. Lastly the caller passes in
   versioned_sym and num_versions which are modified by check_match during
   the checking process.  */
static const ElfW(Sym) *
check_match (const char *const undef_name,
	     const ElfW(Sym) *const ref,
	     const struct r_found_version *const version,
	     const int flags,
	     const int type_class,
	     const ElfW(Sym) *const sym,
	     const Elf_Symndx symidx,
	     const char *const strtab,
	     const struct link_map *const map,
	     const ElfW(Sym) **const versioned_sym,
	     int *const num_versions)
{
  unsigned int stt = ELFW(ST_TYPE) (sym->st_info);
  assert (ELF_RTYPE_CLASS_PLT == 1);
  if (__glibc_unlikely ((sym->st_value == 0 /* No value.  */
			 && sym->st_shndx != SHN_ABS
			 && stt != STT_TLS)
			|| elf_machine_sym_no_match (sym)
			|| (type_class & (sym->st_shndx == SHN_UNDEF))))
    return NULL;

  /* Ignore all but STT_NOTYPE, STT_OBJECT, STT_FUNC,
     STT_COMMON, STT_TLS, and STT_GNU_IFUNC since these are no
     code/data definitions.  */
#define ALLOWED_STT \
  ((1 << STT_NOTYPE) | (1 << STT_OBJECT) | (1 << STT_FUNC) \
   | (1 << STT_COMMON) | (1 << STT_TLS) | (1 << STT_GNU_IFUNC))
  if (__glibc_unlikely (((1 << stt) & ALLOWED_STT) == 0))
    return NULL;

  if (sym != ref && strcmp (strtab + sym->st_name, undef_name))
    /* Not the symbol we are looking for.  */
    return NULL;

  const ElfW(Half) *verstab = map->l_versyms;
  if (version != NULL)
    {
      if (__glibc_unlikely (verstab == NULL))
	{
	  /* We need a versioned symbol but haven't found any.  If
	     this is the object which is referenced in the verneed
	     entry it is a bug in the library since a symbol must
	     not simply disappear.

	     It would also be a bug in the object since it means that
	     the list of required versions is incomplete and so the
	     tests in dl-version.c haven't found a problem.*/
	  assert (version->filename == NULL
		  || ! _dl_name_match_p (version->filename, map));

	  /* Otherwise we accept the symbol.  */
	}
      else
	{
	  /* We can match the version information or use the
	     default one if it is not hidden.  */
	  ElfW(Half) ndx = verstab[symidx] & 0x7fff;
	  if ((map->l_versions[ndx].hash != version->hash
	       || strcmp (map->l_versions[ndx].name, version->name))
	      && (version->hidden || map->l_versions[ndx].hash
		  || (verstab[symidx] & 0x8000)))
	    /* It's not the version we want.  */
	    return NULL;
	}
    }
  else
    {
      /* No specific version is selected.  There are two ways we
	 can got here:

	 - a binary which does not include versioning information
	 is loaded

	 - dlsym() instead of dlvsym() is used to get a symbol which
	 might exist in more than one form

	 If the library does not provide symbol version information
	 there is no problem at all: we simply use the symbol if it
	 is defined.

	 These two lookups need to be handled differently if the
	 library defines versions.  In the case of the old
	 unversioned application the oldest (default) version
	 should be used.  In case of a dlsym() call the latest and
	 public interface should be returned.  */
      if (verstab != NULL)
	{
	  if ((verstab[symidx] & 0x7fff)
	      >= ((flags & DL_LOOKUP_RETURN_NEWEST) ? 2 : 3))
	    {
	      /* Don't accept hidden symbols.  */
	      if ((verstab[symidx] & 0x8000) == 0
		  && (*num_versions)++ == 0)
		/* No version so far.  */
		*versioned_sym = sym;

	      return NULL;
	    }
	}
    }

  /* There cannot be another entry for this symbol so stop here.  */
  return sym;
}

/* Utility function for do_lookup_unique.  Add a symbol to TABLE.  */
static void
enter_unique_sym (struct unique_sym *table, size_t size,
                  unsigned int hash, const char *name,
                  const ElfW(Sym) *sym, const struct link_map *map)
{
  size_t idx = hash % size;
  size_t hash2 = 1 + hash % (size - 2);
  while (table[idx].name != NULL)
    {
      idx += hash2;
      if (idx >= size)
        idx -= size;
    }

  table[idx].hashval = hash;
  table[idx].name = name;
  table[idx].sym = sym;
  table[idx].map = map;
}

/* Mark MAP as NODELETE according to the lookup mode in FLAGS.  During
   initial relocation, NODELETE state is pending only.  */
static void
mark_nodelete (struct link_map *map, int flags)
{
  if (flags & DL_LOOKUP_FOR_RELOCATE)
    map->l_nodelete_pending = true;
  else
    map->l_nodelete_active = true;
}

/* Return true if MAP is marked as NODELETE according to the lookup
   mode in FLAGS> */
static bool
is_nodelete (struct link_map *map, int flags)
{
  /* Non-pending NODELETE always counts.  Pending NODELETE only counts
     during initial relocation processing.  */
  return map->l_nodelete_active
    || ((flags & DL_LOOKUP_FOR_RELOCATE) && map->l_nodelete_pending);
}

/* Utility function for do_lookup_x. Lookup an STB_GNU_UNIQUE symbol
   in the unique symbol table, creating a new entry if necessary.
   Return the matching symbol in RESULT.  */
static void
do_lookup_unique (const char *undef_name, uint_fast32_t new_hash,
		  struct link_map *map, struct sym_val *result,
		  int type_class, const ElfW(Sym) *sym, const char *strtab,
		  const ElfW(Sym) *ref, const struct link_map *undef_map,
		  int flags)
{
  /* We have to determine whether we already found a symbol with this
     name before.  If not then we have to add it to the search table.
     If we already found a definition we have to use it.  */

  struct unique_sym_table *tab
    = &GL(dl_ns)[map->l_ns]._ns_unique_sym_table;

  __rtld_lock_lock_recursive (tab->lock);

  struct unique_sym *entries = tab->entries;
  size_t size = tab->size;
  if (entries != NULL)
    {
      size_t idx = new_hash % size;
      size_t hash2 = 1 + new_hash % (size - 2);
      while (1)
	{
	  if (entries[idx].hashval == new_hash
	      && strcmp (entries[idx].name, undef_name) == 0)
	    {
	      if ((type_class & ELF_RTYPE_CLASS_COPY) != 0)
		{
		  /* We possibly have to initialize the central
		     copy from the copy addressed through the
		     relocation.  */
		  result->s = sym;
		  result->m = map;
		}
	      else
		{
		  result->s = entries[idx].sym;
		  result->m = (struct link_map *) entries[idx].map;
		}
	      __rtld_lock_unlock_recursive (tab->lock);
	      return;
	    }

	  if (entries[idx].name == NULL)
	    break;

	  idx += hash2;
	  if (idx >= size)
	    idx -= size;
	}

      if (size * 3 <= tab->n_elements * 4)
	{
	  /* Expand the table.  */
#ifdef RTLD_CHECK_FOREIGN_CALL
	  /* This must not happen during runtime relocations.  */
	  assert (!RTLD_CHECK_FOREIGN_CALL);
#endif
	  size_t newsize = _dl_higher_prime_number (size + 1);
	  struct unique_sym *newentries
	    = calloc (sizeof (struct unique_sym), newsize);
	  if (newentries == NULL)
	    {
	    nomem:
	      __rtld_lock_unlock_recursive (tab->lock);
	      _dl_fatal_printf ("out of memory\n");
	    }

	  for (idx = 0; idx < size; ++idx)
	    if (entries[idx].name != NULL)
	      enter_unique_sym (newentries, newsize, entries[idx].hashval,
                                entries[idx].name, entries[idx].sym,
                                entries[idx].map);

	  tab->free (entries);
	  tab->size = newsize;
	  size = newsize;
	  entries = tab->entries = newentries;
	  tab->free = __rtld_free;
	}
    }
  else
    {
#ifdef RTLD_CHECK_FOREIGN_CALL
      /* This must not happen during runtime relocations.  */
      assert (!RTLD_CHECK_FOREIGN_CALL);
#endif

#ifdef SHARED
      /* If tab->entries is NULL, but tab->size is not, it means
	 this is the second, conflict finding, lookup for
	 LD_TRACE_PRELINKING in _dl_debug_bindings.  Don't
	 allocate anything and don't enter anything into the
	 hash table.  */
      if (__glibc_unlikely (tab->size))
	{
	  assert (GLRO(dl_debug_mask) & DL_DEBUG_PRELINK);
	  goto success;
	}
#endif

#define INITIAL_NUNIQUE_SYM_TABLE 31
      size = INITIAL_NUNIQUE_SYM_TABLE;
      entries = calloc (sizeof (struct unique_sym), size);
      if (entries == NULL)
	goto nomem;

      tab->entries = entries;
      tab->size = size;
      tab->free = __rtld_free;
    }

  if ((type_class & ELF_RTYPE_CLASS_COPY) != 0)
    enter_unique_sym (entries, size, new_hash, strtab + sym->st_name, ref,
	   undef_map);
  else
    {
      enter_unique_sym (entries, size,
                        new_hash, strtab + sym->st_name, sym, map);

      if (map->l_type == lt_loaded && !is_nodelete (map, flags))
	{
	  /* Make sure we don't unload this object by
	     setting the appropriate flag.  */
	  if (__glibc_unlikely (GLRO (dl_debug_mask) & DL_DEBUG_BINDINGS))
	    _dl_debug_printf ("\
marking %s [%lu] as NODELETE due to unique symbol\n",
			      map->l_name, map->l_ns);
	  mark_nodelete (map, flags);
	}
    }
  ++tab->n_elements;

#ifdef SHARED
 success:
#endif
  __rtld_lock_unlock_recursive (tab->lock);

  result->s = sym;
  result->m = (struct link_map *) map;
}

/* Inner part of the lookup functions.  We return a value > 0 if we
   found the symbol, the value 0 if nothing is found and < 0 if
   something bad happened.  */
static int
__attribute_noinline__
do_lookup_x (const char *undef_name, uint_fast32_t new_hash,
	     unsigned long int *old_hash, const ElfW(Sym) *ref,
	     struct sym_val *result, struct r_scope_elem *scope, size_t i,
	     const struct r_found_version *const version, int flags,
	     struct link_map *skip, int type_class, struct link_map *undef_map)
{
  size_t n = scope->r_nlist;
  /* Make sure we read the value before proceeding.  Otherwise we
     might use r_list pointing to the initial scope and r_nlist being
     the value after a resize.  That is the only path in dl-open.c not
     protected by GSCOPE.  A read barrier here might be to expensive.  */
  __asm volatile ("" : "+r" (n), "+m" (scope->r_list));
  struct link_map **list = scope->r_list;

  do
    {
      const struct link_map *map = list[i]->l_real;

      /* Here come the extra test needed for `_dl_lookup_symbol_skip'.  */
      if (map == skip)
	continue;

      /* Don't search the executable when resolving a copy reloc.  */
      if ((type_class & ELF_RTYPE_CLASS_COPY) && map->l_type == lt_executable)
	continue;

      /* Do not look into objects which are going to be removed.  */
      if (map->l_removed)
	continue;

      /* Print some debugging info if wanted.  */
      if (__glibc_unlikely (GLRO(dl_debug_mask) & DL_DEBUG_SYMBOLS))
	_dl_debug_printf ("symbol=%s;  lookup in file=%s [%lu]\n",
			  undef_name, DSO_FILENAME (map->l_name),
			  map->l_ns);

      /* If the hash table is empty there is nothing to do here.  */
      if (map->l_nbuckets == 0)
	continue;

      Elf_Symndx symidx;
      int num_versions = 0;
      const ElfW(Sym) *versioned_sym = NULL;

      /* The tables for this map.  */
      const ElfW(Sym) *symtab = (const void *) D_PTR (map, l_info[DT_SYMTAB]);
      const char *strtab = (const void *) D_PTR (map, l_info[DT_STRTAB]);

      const ElfW(Sym) *sym;
      const ElfW(Addr) *bitmask = map->l_gnu_bitmask;
      if (__glibc_likely (bitmask != NULL))
	{
	  ElfW(Addr) bitmask_word
	    = bitmask[(new_hash / __ELF_NATIVE_CLASS)
		      & map->l_gnu_bitmask_idxbits];

	  unsigned int hashbit1 = new_hash & (__ELF_NATIVE_CLASS - 1);
	  unsigned int hashbit2 = ((new_hash >> map->l_gnu_shift)
				   & (__ELF_NATIVE_CLASS - 1));

	  if (__glibc_unlikely ((bitmask_word >> hashbit1)
				& (bitmask_word >> hashbit2) & 1))
	    {
	      Elf32_Word bucket = map->l_gnu_buckets[new_hash
						     % map->l_nbuckets];
	      if (bucket != 0)
		{
		  const Elf32_Word *hasharr = &map->l_gnu_chain_zero[bucket];

		  do
		    if (((*hasharr ^ new_hash) >> 1) == 0)
		      {
			symidx = ELF_MACHINE_HASH_SYMIDX (map, hasharr);
			sym = check_match (undef_name, ref, version, flags,
					   type_class, &symtab[symidx], symidx,
					   strtab, map, &versioned_sym,
					   &num_versions);
			if (sym != NULL)
			  goto found_it;
		      }
		  while ((*hasharr++ & 1u) == 0);
		}
	    }
	  /* No symbol found.  */
	  symidx = SHN_UNDEF;
	}
      else
	{
	  if (*old_hash == 0xffffffff)
	    *old_hash = _dl_elf_hash (undef_name);

	  /* Use the old SysV-style hash table.  Search the appropriate
	     hash bucket in this object's symbol table for a definition
	     for the same symbol name.  */
	  for (symidx = map->l_buckets[*old_hash % map->l_nbuckets];
	       symidx != STN_UNDEF;
	       symidx = map->l_chain[symidx])
	    {
	      sym = check_match (undef_name, ref, version, flags,
				 type_class, &symtab[symidx], symidx,
				 strtab, map, &versioned_sym,
				 &num_versions);
	      if (sym != NULL)
		goto found_it;
	    }
	}

      /* If we have seen exactly one versioned symbol while we are
	 looking for an unversioned symbol and the version is not the
	 default version we still accept this symbol since there are
	 no possible ambiguities.  */
      sym = num_versions == 1 ? versioned_sym : NULL;

      if (sym != NULL)
	{
	found_it:
	  /* When UNDEF_MAP is NULL, which indicates we are called from
	     do_lookup_x on relocation against protected data, we skip
	     the data definion in the executable from copy reloc.  */
	  if (ELF_RTYPE_CLASS_EXTERN_PROTECTED_DATA
	      && undef_map == NULL
	      && map->l_type == lt_executable
	      && type_class == ELF_RTYPE_CLASS_EXTERN_PROTECTED_DATA)
	    {
	      const ElfW(Sym) *s;
	      unsigned int i;

#if ! ELF_MACHINE_NO_RELA
	      if (map->l_info[DT_RELA] != NULL
		  && map->l_info[DT_RELASZ] != NULL
		  && map->l_info[DT_RELASZ]->d_un.d_val != 0)
		{
		  const ElfW(Rela) *rela
		    = (const ElfW(Rela) *) D_PTR (map, l_info[DT_RELA]);
		  unsigned int rela_count
		    = map->l_info[DT_RELASZ]->d_un.d_val / sizeof (*rela);

		  for (i = 0; i < rela_count; i++, rela++)
		    if (elf_machine_type_class (ELFW(R_TYPE) (rela->r_info))
			== ELF_RTYPE_CLASS_COPY)
		      {
			s = &symtab[ELFW(R_SYM) (rela->r_info)];
			if (!strcmp (strtab + s->st_name, undef_name))
			  goto skip;
		      }
		}
#endif
#if ! ELF_MACHINE_NO_REL
	      if (map->l_info[DT_REL] != NULL
		  && map->l_info[DT_RELSZ] != NULL
		  && map->l_info[DT_RELSZ]->d_un.d_val != 0)
		{
		  const ElfW(Rel) *rel
		    = (const ElfW(Rel) *) D_PTR (map, l_info[DT_REL]);
		  unsigned int rel_count
		    = map->l_info[DT_RELSZ]->d_un.d_val / sizeof (*rel);

		  for (i = 0; i < rel_count; i++, rel++)
		    if (elf_machine_type_class (ELFW(R_TYPE) (rel->r_info))
			== ELF_RTYPE_CLASS_COPY)
		      {
			s = &symtab[ELFW(R_SYM) (rel->r_info)];
			if (!strcmp (strtab + s->st_name, undef_name))
			  goto skip;
		      }
		}
#endif
	    }

	  /* Hidden and internal symbols are local, ignore them.  */
	  if (__glibc_unlikely (dl_symbol_visibility_binds_local_p (sym)))
	    goto skip;

	  switch (ELFW(ST_BIND) (sym->st_info))
	    {
	    case STB_WEAK:
	      /* Weak definition.  Use this value if we don't find another.  */
	      if (__glibc_unlikely (GLRO(dl_dynamic_weak)))
		{
		  if (! result->s)
		    {
		      result->s = sym;
		      result->m = (struct link_map *) map;
		    }
		  break;
		}
	      /* FALLTHROUGH */
	    case STB_GLOBAL:
	      /* Global definition.  Just what we need.  */
	      result->s = sym;
	      result->m = (struct link_map *) map;
	      return 1;

	    case STB_GNU_UNIQUE:;
	      do_lookup_unique (undef_name, new_hash, (struct link_map *) map,
				result, type_class, sym, strtab, ref,
				undef_map, flags);
	      return 1;

	    default:
	      /* Local symbols are ignored.  */
	      break;
	    }
	}

skip:
      ;
    }
  while (++i < n);

  /* We have not found anything until now.  */
  return 0;
}


static uint_fast32_t
dl_new_hash (const char *s)
{
  uint_fast32_t h = 5381;
  for (unsigned char c = *s; c != '\0'; c = *++s)
    h = h * 33 + c;
  return h & 0xffffffff;
}


/* Add extra dependency on MAP to UNDEF_MAP.  */
static int
add_dependency (struct link_map *undef_map, struct link_map *map, int flags)
{
  struct link_map *runp;
  unsigned int i;
  int result = 0;

  /* Avoid self-references and references to objects which cannot be
     unloaded anyway.  */
  if (undef_map == map)
    return 0;

  /* Avoid references to objects which cannot be unloaded anyway.  We
     do not need to record dependencies if this object goes away
     during dlopen failure, either.  IFUNC resolvers with relocation
     dependencies may pick an dependency which can be dlclose'd, but
     such IFUNC resolvers are undefined anyway.  */
  assert (map->l_type == lt_loaded);
  if (is_nodelete (map, flags))
    return 0;

  struct link_map_reldeps *l_reldeps
    = atomic_forced_read (undef_map->l_reldeps);

  /* Make sure l_reldeps is read before l_initfini.  */
  atomic_read_barrier ();

  /* Determine whether UNDEF_MAP already has a reference to MAP.  First
     look in the normal dependencies.  */
  struct link_map **l_initfini = atomic_forced_read (undef_map->l_initfini);
  if (l_initfini != NULL)
    {
      for (i = 0; l_initfini[i] != NULL; ++i)
	if (l_initfini[i] == map)
	  return 0;
    }

  /* No normal dependency.  See whether we already had to add it
     to the special list of dynamic dependencies.  */
  unsigned int l_reldepsact = 0;
  if (l_reldeps != NULL)
    {
      struct link_map **list = &l_reldeps->list[0];
      l_reldepsact = l_reldeps->act;
      for (i = 0; i < l_reldepsact; ++i)
	if (list[i] == map)
	  return 0;
    }

  /* Save serial number of the target MAP.  */
  unsigned long long serial = map->l_serial;

  /* Make sure nobody can unload the object while we are at it.  */
  if (__glibc_unlikely (flags & DL_LOOKUP_GSCOPE_LOCK))
    {
      /* We can't just call __rtld_lock_lock_recursive (GL(dl_load_lock))
	 here, that can result in ABBA deadlock.  */
      THREAD_GSCOPE_RESET_FLAG ();
      __rtld_lock_lock_recursive (GL(dl_load_lock));
      /* While MAP value won't change, after THREAD_GSCOPE_RESET_FLAG ()
	 it can e.g. point to unallocated memory.  So avoid the optimizer
	 treating the above read from MAP->l_serial as ensurance it
	 can safely dereference it.  */
      map = atomic_forced_read (map);

      /* From this point on it is unsafe to dereference MAP, until it
	 has been found in one of the lists.  */

      /* Redo the l_initfini check in case undef_map's l_initfini
	 changed in the mean time.  */
      if (undef_map->l_initfini != l_initfini
	  && undef_map->l_initfini != NULL)
	{
	  l_initfini = undef_map->l_initfini;
	  for (i = 0; l_initfini[i] != NULL; ++i)
	    if (l_initfini[i] == map)
	      goto out_check;
	}

      /* Redo the l_reldeps check if undef_map's l_reldeps changed in
	 the mean time.  */
      if (undef_map->l_reldeps != NULL)
	{
	  if (undef_map->l_reldeps != l_reldeps)
	    {
	      struct link_map **list = &undef_map->l_reldeps->list[0];
	      l_reldepsact = undef_map->l_reldeps->act;
	      for (i = 0; i < l_reldepsact; ++i)
		if (list[i] == map)
		  goto out_check;
	    }
	  else if (undef_map->l_reldeps->act > l_reldepsact)
	    {
	      struct link_map **list
		= &undef_map->l_reldeps->list[0];
	      i = l_reldepsact;
	      l_reldepsact = undef_map->l_reldeps->act;
	      for (; i < l_reldepsact; ++i)
		if (list[i] == map)
		  goto out_check;
	    }
	}
    }
  else
    __rtld_lock_lock_recursive (GL(dl_load_lock));

  /* The object is not yet in the dependency list.  Before we add
     it make sure just one more time the object we are about to
     reference is still available.  There is a brief period in
     which the object could have been removed since we found the
     definition.  */
  runp = GL(dl_ns)[undef_map->l_ns]._ns_loaded;
  while (runp != NULL && runp != map)
    runp = runp->l_next;

  if (runp != NULL)
    {
      /* The object is still available.  */

      /* MAP could have been dlclosed, freed and then some other dlopened
	 library could have the same link_map pointer.  */
      if (map->l_serial != serial)
	goto out_check;

      /* Redo the NODELETE check, as when dl_load_lock wasn't held
	 yet this could have changed.  */
      if (is_nodelete (map, flags))
	goto out;

      /* If the object with the undefined reference cannot be removed ever
	 just make sure the same is true for the object which contains the
	 definition.  */
      if (undef_map->l_type != lt_loaded || is_nodelete (map, flags))
	{
	  if (__glibc_unlikely (GLRO (dl_debug_mask) & DL_DEBUG_BINDINGS)
	      && !is_nodelete (map, flags))
	    {
	      if (undef_map->l_name[0] == '\0')
		_dl_debug_printf ("\
marking %s [%lu] as NODELETE due to reference to main program\n",
				  map->l_name, map->l_ns);
	      else
		_dl_debug_printf ("\
marking %s [%lu] as NODELETE due to reference to %s [%lu]\n",
				  map->l_name, map->l_ns,
				  undef_map->l_name, undef_map->l_ns);
	    }
	  mark_nodelete (map, flags);
	  goto out;
	}

      /* Add the reference now.  */
      if (__glibc_unlikely (l_reldepsact >= undef_map->l_reldepsmax))
	{
	  /* Allocate more memory for the dependency list.  Since this
	     can never happen during the startup phase we can use
	     `realloc'.  */
	  struct link_map_reldeps *newp;
	  unsigned int max
	    = undef_map->l_reldepsmax ? undef_map->l_reldepsmax * 2 : 10;

#ifdef RTLD_PREPARE_FOREIGN_CALL
	  RTLD_PREPARE_FOREIGN_CALL;
#endif

	  newp = malloc (sizeof (*newp) + max * sizeof (struct link_map *));
	  if (newp == NULL)
	    {
	      /* If we didn't manage to allocate memory for the list this is
		 no fatal problem.  We simply make sure the referenced object
		 cannot be unloaded.  This is semantically the correct
		 behavior.  */
	      if (__glibc_unlikely (GLRO (dl_debug_mask) & DL_DEBUG_BINDINGS)
		  && !is_nodelete (map, flags))
		_dl_debug_printf ("\
marking %s [%lu] as NODELETE due to memory allocation failure\n",
				  map->l_name, map->l_ns);
	      /* In case of non-lazy binding, we could actually report
		 the memory allocation error, but for now, we use the
		 conservative approximation as well.  */
	      mark_nodelete (map, flags);
	      goto out;
	    }
	  else
	    {
	      if (l_reldepsact)
		memcpy (&newp->list[0], &undef_map->l_reldeps->list[0],
			l_reldepsact * sizeof (struct link_map *));
	      newp->list[l_reldepsact] = map;
	      newp->act = l_reldepsact + 1;
	      atomic_write_barrier ();
	      void *old = undef_map->l_reldeps;
	      undef_map->l_reldeps = newp;
	      undef_map->l_reldepsmax = max;
	      if (old)
		_dl_scope_free (old);
	    }
	}
      else
	{
	  undef_map->l_reldeps->list[l_reldepsact] = map;
	  atomic_write_barrier ();
	  undef_map->l_reldeps->act = l_reldepsact + 1;
	}

      /* Display information if we are debugging.  */
      if (__glibc_unlikely (GLRO(dl_debug_mask) & DL_DEBUG_FILES))
	_dl_debug_printf ("\
\nfile=%s [%lu];  needed by %s [%lu] (relocation dependency)\n\n",
			  DSO_FILENAME (map->l_name),
			  map->l_ns,
			  DSO_FILENAME (undef_map->l_name),
			  undef_map->l_ns);
    }
  else
    /* Whoa, that was bad luck.  We have to search again.  */
    result = -1;

 out:
  /* Release the lock.  */
  __rtld_lock_unlock_recursive (GL(dl_load_lock));

  if (__glibc_unlikely (flags & DL_LOOKUP_GSCOPE_LOCK))
    THREAD_GSCOPE_SET_FLAG ();

  return result;

 out_check:
  if (map->l_serial != serial)
    result = -1;
  goto out;
}

static void
_dl_debug_bindings (const char *undef_name, struct link_map *undef_map,
		    const ElfW(Sym) **ref, struct sym_val *value,
		    const struct r_found_version *version, int type_class,
		    int protected);


/* Search loaded objects' symbol tables for a definition of the symbol
   UNDEF_NAME, perhaps with a requested version for the symbol.

   We must never have calls to the audit functions inside this function
   or in any function which gets called.  If this would happen the audit
   code might create a thread which can throw off all the scope locking.  */
lookup_t
_dl_lookup_symbol_x (const char *undef_name, struct link_map *undef_map,
		     const ElfW(Sym) **ref,
		     struct r_scope_elem *symbol_scope[],
		     const struct r_found_version *version,
		     int type_class, int flags, struct link_map *skip_map)
{
  const uint_fast32_t new_hash = dl_new_hash (undef_name);
  unsigned long int old_hash = 0xffffffff;
  struct sym_val current_value = { NULL, NULL };
  struct r_scope_elem **scope = symbol_scope;

  bump_num_relocations ();

  /* DL_LOOKUP_RETURN_NEWEST does not make sense for versioned
     lookups.  */
  assert (version == NULL || !(flags & DL_LOOKUP_RETURN_NEWEST));

  size_t i = 0;
  if (__glibc_unlikely (skip_map != NULL))
    /* Search the relevant loaded objects for a definition.  */
    while ((*scope)->r_list[i] != skip_map)
      ++i;

  /* Search the relevant loaded objects for a definition.  */
  for (size_t start = i; *scope != NULL; start = 0, ++scope)
    if (do_lookup_x (undef_name, new_hash, &old_hash, *ref,
		     &current_value, *scope, start, version, flags,
		     skip_map, type_class, undef_map) != 0)
      break;

  if (__glibc_unlikely (current_value.s == NULL))
    {
      if ((*ref == NULL || ELFW(ST_BIND) ((*ref)->st_info) != STB_WEAK)
	  && !(GLRO(dl_debug_mask) & DL_DEBUG_UNUSED))
	{
	  /* We could find no value for a strong reference.  */
	  const char *reference_name = undef_map ? undef_map->l_name : "";
	  const char *versionstr = version ? ", version " : "";
	  const char *versionname = (version && version->name
				     ? version->name : "");
	  struct dl_exception exception;
	  /* XXX We cannot translate the message.  */
	  _dl_exception_create_format
	    (&exception, DSO_FILENAME (reference_name),
	     "undefined symbol: %s%s%s",
	     undef_name, versionstr, versionname);
	  _dl_signal_cexception (0, &exception, N_("symbol lookup error"));
	  _dl_exception_free (&exception);
	}
      *ref = NULL;
      return 0;
    }

  int protected = (*ref
		   && ELFW(ST_VISIBILITY) ((*ref)->st_other) == STV_PROTECTED);
  if (__glibc_unlikely (protected != 0))
    {
      /* It is very tricky.  We need to figure out what value to
	 return for the protected symbol.  */
      if (type_class == ELF_RTYPE_CLASS_PLT)
	{
	  if (current_value.s != NULL && current_value.m != undef_map)
	    {
	      current_value.s = *ref;
	      current_value.m = undef_map;
	    }
	}
      else
	{
	  struct sym_val protected_value = { NULL, NULL };

	  for (scope = symbol_scope; *scope != NULL; i = 0, ++scope)
	    if (do_lookup_x (undef_name, new_hash, &old_hash, *ref,
			     &protected_value, *scope, i, version, flags,
			     skip_map,
			     (ELF_RTYPE_CLASS_EXTERN_PROTECTED_DATA
			      && ELFW(ST_TYPE) ((*ref)->st_info) == STT_OBJECT
			      && type_class == ELF_RTYPE_CLASS_EXTERN_PROTECTED_DATA)
			     ? ELF_RTYPE_CLASS_EXTERN_PROTECTED_DATA
			     : ELF_RTYPE_CLASS_PLT, NULL) != 0)
	      break;

	  if (protected_value.s != NULL && protected_value.m != undef_map)
	    {
	      current_value.s = *ref;
	      current_value.m = undef_map;
	    }
	}
    }

  /* We have to check whether this would bind UNDEF_MAP to an object
     in the global scope which was dynamically loaded.  In this case
     we have to prevent the latter from being unloaded unless the
     UNDEF_MAP object is also unloaded.  */
  if (__glibc_unlikely (current_value.m->l_type == lt_loaded)
      /* Don't do this for explicit lookups as opposed to implicit
	 runtime lookups.  */
      && (flags & DL_LOOKUP_ADD_DEPENDENCY) != 0
      /* Add UNDEF_MAP to the dependencies.  */
      && add_dependency (undef_map, current_value.m, flags) < 0)
      /* Something went wrong.  Perhaps the object we tried to reference
	 was just removed.  Try finding another definition.  */
      return _dl_lookup_symbol_x (undef_name, undef_map, ref,
				  (flags & DL_LOOKUP_GSCOPE_LOCK)
				  ? undef_map->l_scope : symbol_scope,
				  version, type_class, flags, skip_map);

  /* The object is used.  */
  if (__glibc_unlikely (current_value.m->l_used == 0))
    current_value.m->l_used = 1;

  if (__glibc_unlikely (GLRO(dl_debug_mask)
			& (DL_DEBUG_BINDINGS|DL_DEBUG_PRELINK)))
    _dl_debug_bindings (undef_name, undef_map, ref,
			&current_value, version, type_class, protected);

  *ref = current_value.s;
  return LOOKUP_VALUE (current_value.m);
}


/* Cache the location of MAP's hash table.  */

void
_dl_setup_hash (struct link_map *map)
{
  Elf_Symndx *hash;

  if (__glibc_likely (map->l_info[ELF_MACHINE_GNU_HASH_ADDRIDX] != NULL))
    {
      Elf32_Word *hash32
	= (void *) D_PTR (map, l_info[ELF_MACHINE_GNU_HASH_ADDRIDX]);
      map->l_nbuckets = *hash32++;
      Elf32_Word symbias = *hash32++;
      Elf32_Word bitmask_nwords = *hash32++;
      /* Must be a power of two.  */
      assert ((bitmask_nwords & (bitmask_nwords - 1)) == 0);
      map->l_gnu_bitmask_idxbits = bitmask_nwords - 1;
      map->l_gnu_shift = *hash32++;

      map->l_gnu_bitmask = (ElfW(Addr) *) hash32;
      hash32 += __ELF_NATIVE_CLASS / 32 * bitmask_nwords;

      map->l_gnu_buckets = hash32;
      hash32 += map->l_nbuckets;
      map->l_gnu_chain_zero = hash32 - symbias;

      /* Initialize MIPS xhash translation table.  */
      ELF_MACHINE_XHASH_SETUP (hash32, symbias, map);

      return;
    }

  if (!map->l_info[DT_HASH])
    return;
  hash = (void *) D_PTR (map, l_info[DT_HASH]);

  map->l_nbuckets = *hash++;
  /* Skip nchain.  */
  hash++;
  map->l_buckets = hash;
  hash += map->l_nbuckets;
  map->l_chain = hash;
}


static void
_dl_debug_bindings (const char *undef_name, struct link_map *undef_map,
		    const ElfW(Sym) **ref, struct sym_val *value,
		    const struct r_found_version *version, int type_class,
		    int protected)
{
  const char *reference_name = undef_map->l_name;

  if (GLRO(dl_debug_mask) & DL_DEBUG_BINDINGS)
    {
      _dl_debug_printf ("binding file %s [%lu] to %s [%lu]: %s symbol `%s'",
			DSO_FILENAME (reference_name),
			undef_map->l_ns,
			DSO_FILENAME (value->m->l_name),
			value->m->l_ns,
			protected ? "protected" : "normal", undef_name);
      if (version)
	_dl_debug_printf_c (" [%s]\n", version->name);
      else
	_dl_debug_printf_c ("\n");
    }
#ifdef SHARED
  if (GLRO(dl_debug_mask) & DL_DEBUG_PRELINK)
    {
/* ELF_RTYPE_CLASS_XXX must match RTYPE_CLASS_XXX used by prelink with
   LD_TRACE_PRELINKING.  */
#define RTYPE_CLASS_VALID	8
#define RTYPE_CLASS_PLT		(8|1)
#define RTYPE_CLASS_COPY	(8|2)
#define RTYPE_CLASS_TLS		(8|4)
#if ELF_RTYPE_CLASS_PLT != 0 && ELF_RTYPE_CLASS_PLT != 1
# error ELF_RTYPE_CLASS_PLT must be 0 or 1!
#endif
#if ELF_RTYPE_CLASS_COPY != 0 && ELF_RTYPE_CLASS_COPY != 2
# error ELF_RTYPE_CLASS_COPY must be 0 or 2!
#endif
      int conflict = 0;
      struct sym_val val = { NULL, NULL };

      if ((GLRO(dl_trace_prelink_map) == NULL
	   || GLRO(dl_trace_prelink_map) == GL(dl_ns)[LM_ID_BASE]._ns_loaded)
	  && undef_map != GL(dl_ns)[LM_ID_BASE]._ns_loaded)
	{
	  const uint_fast32_t new_hash = dl_new_hash (undef_name);
	  unsigned long int old_hash = 0xffffffff;
	  struct unique_sym *saved_entries
	    = GL(dl_ns)[LM_ID_BASE]._ns_unique_sym_table.entries;

	  GL(dl_ns)[LM_ID_BASE]._ns_unique_sym_table.entries = NULL;
	  do_lookup_x (undef_name, new_hash, &old_hash, *ref, &val,
		       undef_map->l_local_scope[0], 0, version, 0, NULL,
		       type_class, undef_map);
	  if (val.s != value->s || val.m != value->m)
	    conflict = 1;
	  else if (__glibc_unlikely (undef_map->l_symbolic_in_local_scope)
		   && val.s
		   && __glibc_unlikely (ELFW(ST_BIND) (val.s->st_info)
					== STB_GNU_UNIQUE))
	    {
	      /* If it is STB_GNU_UNIQUE and undef_map's l_local_scope
		 contains any DT_SYMBOLIC libraries, unfortunately there
		 can be conflicts even if the above is equal.  As symbol
		 resolution goes from the last library to the first and
		 if a STB_GNU_UNIQUE symbol is found in some late DT_SYMBOLIC
		 library, it would be the one that is looked up.  */
	      struct sym_val val2 = { NULL, NULL };
	      size_t n;
	      struct r_scope_elem *scope = undef_map->l_local_scope[0];

	      for (n = 0; n < scope->r_nlist; n++)
		if (scope->r_list[n] == val.m)
		  break;

	      for (n++; n < scope->r_nlist; n++)
		if (scope->r_list[n]->l_info[DT_SYMBOLIC] != NULL
		    && do_lookup_x (undef_name, new_hash, &old_hash, *ref,
				    &val2,
				    &scope->r_list[n]->l_symbolic_searchlist,
				    0, version, 0, NULL, type_class,
				    undef_map) > 0)
		  {
		    conflict = 1;
		    val = val2;
		    break;
		  }
	    }
	  GL(dl_ns)[LM_ID_BASE]._ns_unique_sym_table.entries = saved_entries;
	}

      if (value->s)
	{
	  /* Keep only ELF_RTYPE_CLASS_PLT and ELF_RTYPE_CLASS_COPY
	     bits since since prelink only uses them.  */
	  type_class &= ELF_RTYPE_CLASS_PLT | ELF_RTYPE_CLASS_COPY;
	  if (__glibc_unlikely (ELFW(ST_TYPE) (value->s->st_info)
				== STT_TLS))
	    /* Clear the RTYPE_CLASS_VALID bit in RTYPE_CLASS_TLS.  */
	    type_class = RTYPE_CLASS_TLS & ~RTYPE_CLASS_VALID;
	  else if (__glibc_unlikely (ELFW(ST_TYPE) (value->s->st_info)
				     == STT_GNU_IFUNC))
	    /* Set the RTYPE_CLASS_VALID bit.  */
	    type_class |= RTYPE_CLASS_VALID;
	}

      if (conflict
	  || GLRO(dl_trace_prelink_map) == undef_map
	  || GLRO(dl_trace_prelink_map) == NULL
	  || type_class >= 4)
	{
	  _dl_printf ("%s 0x%0*zx 0x%0*zx -> 0x%0*zx 0x%0*zx ",
		      conflict ? "conflict" : "lookup",
		      (int) sizeof (ElfW(Addr)) * 2,
		      (size_t) undef_map->l_map_start,
		      (int) sizeof (ElfW(Addr)) * 2,
		      (size_t) (((ElfW(Addr)) *ref) - undef_map->l_map_start),
		      (int) sizeof (ElfW(Addr)) * 2,
		      (size_t) (value->s ? value->m->l_map_start : 0),
		      (int) sizeof (ElfW(Addr)) * 2,
		      (size_t) (value->s ? value->s->st_value : 0));

	  if (conflict)
	    _dl_printf ("x 0x%0*zx 0x%0*zx ",
			(int) sizeof (ElfW(Addr)) * 2,
			(size_t) (val.s ? val.m->l_map_start : 0),
			(int) sizeof (ElfW(Addr)) * 2,
			(size_t) (val.s ? val.s->st_value : 0));

	  _dl_printf ("/%x %s\n", type_class, undef_name);
	}
    }
#endif
}
