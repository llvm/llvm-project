/* Look up a symbol in a single specified object.
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

#include <ldsodefs.h>
#include <string.h>
#include <elf_machine_sym_no_match.h>
#include <dl-hash.h>

/* This function corresponds to do_lookup_x in elf/dl-lookup.c.  The
   variant here is simplified because it requires symbol
   versioning.  */
static const ElfW(Sym) *
check_match (const struct link_map *const map, const char *const undef_name,
             const char *version, uint32_t version_hash,
             const Elf_Symndx symidx)
{
  const ElfW(Sym) *symtab = (const void *) D_PTR (map, l_info[DT_SYMTAB]);
  const ElfW(Sym) *sym = &symtab[symidx];

  unsigned int stt = ELFW(ST_TYPE) (sym->st_info);
  if (__glibc_unlikely ((sym->st_value == 0 /* No value.  */
                         && sym->st_shndx != SHN_ABS
                         && stt != STT_TLS)
                        || elf_machine_sym_no_match (sym)))
    return NULL;

  /* Ignore all but STT_NOTYPE, STT_OBJECT, STT_FUNC,
     STT_COMMON, STT_TLS, and STT_GNU_IFUNC since these are no
     code/data definitions.  */
#define ALLOWED_STT \
  ((1 << STT_NOTYPE) | (1 << STT_OBJECT) | (1 << STT_FUNC) \
   | (1 << STT_COMMON) | (1 << STT_TLS) | (1 << STT_GNU_IFUNC))
  if (__glibc_unlikely (((1 << stt) & ALLOWED_STT) == 0))
    return NULL;

  const char *strtab = (const void *) D_PTR (map, l_info[DT_STRTAB]);

  if (strcmp (strtab + sym->st_name, undef_name) != 0)
    /* Not the symbol we are looking for.  */
    return NULL;

  ElfW(Half) ndx = map->l_versyms[symidx] & 0x7fff;
  if (map->l_versions[ndx].hash != version_hash
      || strcmp (map->l_versions[ndx].name, version) != 0)
    /* It's not the version we want.  */
    return NULL;

  return sym;
}


/* This function corresponds to do_lookup_x in elf/dl-lookup.c.  The
   variant here is simplified because it does not search object
   dependencies.  It is optimized for a successful lookup.  */
const ElfW(Sym) *
_dl_lookup_direct (struct link_map *map,
                   const char *undef_name, uint32_t new_hash,
                   const char *version, uint32_t version_hash)
{
  const ElfW(Addr) *bitmask = map->l_gnu_bitmask;
  if (__glibc_likely (bitmask != NULL))
    {
      Elf32_Word bucket = map->l_gnu_buckets[new_hash % map->l_nbuckets];
      if (bucket != 0)
        {
          const Elf32_Word *hasharr = &map->l_gnu_chain_zero[bucket];

          do
            if (((*hasharr ^ new_hash) >> 1) == 0)
              {
                Elf_Symndx symidx = ELF_MACHINE_HASH_SYMIDX (map, hasharr);
                const ElfW(Sym) *sym = check_match (map, undef_name,
                                                    version, version_hash,
                                                    symidx);
                if (sym != NULL)
                  return sym;
              }
          while ((*hasharr++ & 1u) == 0);
        }
    }
  else
    {
      /* Fallback code for lack of GNU_HASH support.  */
      uint32_t old_hash = _dl_elf_hash (undef_name);

      /* Use the old SysV-style hash table.  Search the appropriate
         hash bucket in this object's symbol table for a definition
         for the same symbol name.  */
      for (Elf_Symndx symidx = map->l_buckets[old_hash % map->l_nbuckets];
           symidx != STN_UNDEF;
           symidx = map->l_chain[symidx])
        {
          const ElfW(Sym) *sym = check_match (map, undef_name,
                                              version, version_hash, symidx);
          if (sym != NULL)
            return sym;
        }
    }

  return NULL;
}
