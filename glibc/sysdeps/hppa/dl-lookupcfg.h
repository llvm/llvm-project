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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#define ELF_FUNCTION_PTR_IS_SPECIAL
#define DL_UNMAP_IS_SPECIAL

#include <dl-fptr.h>

/* Forward declaration.  */
struct link_map;

void *_dl_symbol_address (struct link_map *map, const ElfW(Sym) *ref);
rtld_hidden_proto (_dl_symbol_address)

#define DL_SYMBOL_ADDRESS(map, ref) _dl_symbol_address(map, ref)

Elf32_Addr _dl_lookup_address (const void *address);

#define DL_LOOKUP_ADDRESS(addr) _dl_lookup_address ((const void *) addr)

void attribute_hidden _dl_unmap (struct link_map *map);

#define DL_UNMAP(map) _dl_unmap (map)

#define DL_DT_FUNCTION_ADDRESS(map, start, attr, addr)			\
  attr volatile unsigned int fptr[2];					\
 /* The test for "start & 2" below is to accommodate old binaries which	\
    violated the ELF ABI by pointing DT_INIT and DT_FINI at a function	\
    descriptor.  */							\
  if ((ElfW(Addr)) (start) & 2)						\
    addr = (ElfW(Addr)) start;						\
  else									\
    {									\
      fptr[0] = (unsigned int) (start);					\
      fptr[1] = (map)->l_info[DT_PLTGOT]->d_un.d_ptr;			\
      /* Set bit 30 to indicate to $$dyncall that this is a PLABEL. */	\
      addr = (ElfW(Addr))((unsigned int)fptr | 2);			\
    }									\

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

/* The type of the return value of fixup/profile_fixup */
#define DL_FIXUP_VALUE_TYPE struct fdesc

/* Construct a fixup value from the address and linkmap */
#define DL_FIXUP_MAKE_VALUE(map, addr) \
  (map) ? ((struct fdesc) { (addr), (map)->l_info[DT_PLTGOT]->d_un.d_ptr }) \
	: ((struct fdesc) { 0, 0 })

/* Extract the code address from a fixup value */
#define DL_FIXUP_VALUE_CODE_ADDR(value) ((value).ip)
#define DL_FIXUP_VALUE_ADDR(value) ((uintptr_t) &(value))
#define DL_FIXUP_ADDR_VALUE(addr) (*(struct fdesc *) (addr))
