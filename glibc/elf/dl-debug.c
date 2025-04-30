/* Communicate dynamic linker state to the debugger at runtime.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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


/* These are the members in the public `struct link_map' type.
   Sanity check that the internal type and the public type match.  */
#define VERIFY_MEMBER(name) \
  (offsetof (struct link_map_public, name) == offsetof (struct link_map, name))
extern const int verify_link_map_members[(VERIFY_MEMBER (l_addr)
					  && VERIFY_MEMBER (l_name)
					  && VERIFY_MEMBER (l_ld)
					  && VERIFY_MEMBER (l_next)
					  && VERIFY_MEMBER (l_prev))
					 ? 1 : -1];

/* This structure communicates dl state to the debugger.  The debugger
   normally finds it via the DT_DEBUG entry in the dynamic section, but in
   a statically-linked program there is no dynamic section for the debugger
   to examine and it looks for this particular symbol name.  */
struct r_debug _r_debug;


/* Initialize _r_debug if it has not already been done.  The argument is
   the run-time load address of the dynamic linker, to be put in
   _r_debug.r_ldbase.  Returns the address of _r_debug.  */

struct r_debug *
_dl_debug_initialize (ElfW(Addr) ldbase, Lmid_t ns)
{
  struct r_debug *r;

  if (ns == LM_ID_BASE)
    r = &_r_debug;
  else
    r = &GL(dl_ns)[ns]._ns_debug;

  if (r->r_map == NULL || ldbase != 0)
    {
      /* Tell the debugger where to find the map of loaded objects.  */
      r->r_version = 1	/* R_DEBUG_VERSION XXX */;
      r->r_ldbase = ldbase ?: _r_debug.r_ldbase;
      r->r_map = (void *) GL(dl_ns)[ns]._ns_loaded;
      r->r_brk = (ElfW(Addr)) &_dl_debug_state;
    }

  return r;
}


/* This function exists solely to have a breakpoint set on it by the
   debugger.  The debugger is supposed to find this function's address by
   examining the r_brk member of struct r_debug, but GDB 4.15 in fact looks
   for this particular symbol name in the PT_INTERP file.  */
void
_dl_debug_state (void)
{
}
rtld_hidden_def (_dl_debug_state)
