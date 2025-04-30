/* Support for relocating static PIE.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#if ENABLE_STATIC_PIE
/* Mark symbols hidden in static PIE for early self relocation to work.  */
# pragma GCC visibility push(hidden)
#include <unistd.h>
#include <ldsodefs.h>
#include "dynamic-link.h"

/* Relocate static executable with PIE.  */

void
_dl_relocate_static_pie (void)
{
  struct link_map *main_map = _dl_get_dl_main_map ();

# define STATIC_PIE_BOOTSTRAP
# define BOOTSTRAP_MAP (main_map)
# define RESOLVE_MAP(sym, version, flags) BOOTSTRAP_MAP
# include "dynamic-link.h"

  /* Figure out the run-time load address of static PIE.  */
  main_map->l_addr = elf_machine_load_address ();

  /* Read our own dynamic section and fill in the info array.  */
  main_map->l_ld = ((void *) main_map->l_addr + elf_machine_dynamic ());
  elf_get_dynamic_info (main_map, NULL);

# ifdef ELF_MACHINE_BEFORE_RTLD_RELOC
  ELF_MACHINE_BEFORE_RTLD_RELOC (main_map->l_info);
# endif

  /* Relocate ourselves so we can do normal function calls and
     data access using the global offset table.  */
  ELF_DYNAMIC_RELOCATE (main_map, 0, 0, 0);
  main_map->l_relocated = 1;

  /* Initialize _r_debug.  */
  struct r_debug *r = _dl_debug_initialize (0, LM_ID_BASE);
  r->r_state = RT_CONSISTENT;

  /* Set up debugging before the debugger is notified for the first
     time.  */
# ifdef ELF_MACHINE_DEBUG_SETUP
  /* Some machines (e.g. MIPS) don't use DT_DEBUG in this way.  */
  ELF_MACHINE_DEBUG_SETUP (main_map, r);
# else
  if (main_map->l_info[DT_DEBUG] != NULL)
    /* There is a DT_DEBUG entry in the dynamic section.  Fill it in
       with the run-time address of the r_debug structure  */
    main_map->l_info[DT_DEBUG]->d_un.d_ptr = (ElfW(Addr)) r;
# endif
}
#endif
