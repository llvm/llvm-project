/* Configuration of lookup functions.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#define ELF_FUNCTION_PTR_IS_SPECIAL
#define DL_UNMAP_IS_SPECIAL

#include <dl-fptr.h>

/* We do not support copy relocations for IA-64.  */
#define DL_NO_COPY_RELOCS

/* Forward declaration.  */
struct link_map;

extern void *_dl_symbol_address (struct link_map *map, const Elf64_Sym *ref);
rtld_hidden_proto (_dl_symbol_address)

#define DL_SYMBOL_ADDRESS(map, ref) _dl_symbol_address(map, ref)

extern Elf64_Addr _dl_lookup_address (const void *address);

#define DL_LOOKUP_ADDRESS(addr) _dl_lookup_address (addr)

extern void attribute_hidden _dl_unmap (struct link_map *map);

#define DL_UNMAP(map) _dl_unmap (map)

#define DL_DT_FUNCTION_ADDRESS(map, start, attr, addr)			\
  attr volatile unsigned long int fptr[2];					\
  fptr[0] = (unsigned long int) (start);					\
  fptr[1] = (map)->l_info[DT_PLTGOT]->d_un.d_ptr;			\
  addr = (ElfW(Addr)) fptr;						\

#define DL_CALL_DT_INIT(map, start, argc, argv, env)	\
{							\
  ElfW(Addr) addr;					\
  DL_DT_FUNCTION_ADDRESS(map, start, , addr)		\
  dl_init_t init = (dl_init_t) addr; 			\
  init (argc, argv, env);				\
}

#define DL_CALL_DT_FINI(map, start)		\
{						\
  ElfW(Addr) addr;				\
  DL_DT_FUNCTION_ADDRESS(map, start, , addr)	\
  fini_t fini = (fini_t) addr;			\
  fini ();					\
}

/* The type of the return value of fixup/profile_fixup.  */
#define DL_FIXUP_VALUE_TYPE struct fdesc
/* Construct a value of type DL_FIXUP_VALUE_TYPE from a code address
   and a link map.  */
#define DL_FIXUP_MAKE_VALUE(map, addr) \
  ((struct fdesc) { (addr), (map)->l_info[DT_PLTGOT]->d_un.d_ptr })
/* Extract the code address from a value of type DL_FIXUP_MAKE_VALUE.
 */
#define DL_FIXUP_VALUE_CODE_ADDR(value) (value).ip

#define DL_FIXUP_VALUE_ADDR(value) ((uintptr_t) &(value))
#define DL_FIXUP_ADDR_VALUE(addr) (*(struct fdesc *) (addr))
