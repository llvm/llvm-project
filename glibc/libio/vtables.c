/* libio vtable validation.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <dlfcn.h>
#include <libioP.h>
#include <stdio.h>
#include <ldsodefs.h>

#ifdef SHARED

void (*IO_accept_foreign_vtables) (void) attribute_hidden;

/* Used to detected multiple libcs.  */
extern struct dl_open_hook *_dl_open_hook;
libc_hidden_proto (_dl_open_hook);

#else  /* !SHARED */

/* Used to check whether static dlopen support is needed.  */
# pragma weak __dlopen

#endif

void attribute_hidden
_IO_vtable_check (void)
{
#ifdef SHARED
  /* Honor the compatibility flag.  */
  void (*flag) (void) = atomic_load_relaxed (&IO_accept_foreign_vtables);
#ifdef PTR_DEMANGLE
  PTR_DEMANGLE (flag);
#endif
  if (flag == &_IO_vtable_check)
    return;

  /* In case this libc copy is in a non-default namespace, we always
     need to accept foreign vtables because there is always a
     possibility that FILE * objects are passed across the linking
     boundary.  */
  {
    Dl_info di;
    struct link_map *l;
    if (!rtld_active ()
        || (_dl_addr (_IO_vtable_check, &di, &l, NULL) != 0
            && l->l_ns != LM_ID_BASE))
      return;
  }

#else /* !SHARED */
  /* We cannot perform vtable validation in the static dlopen case
     because FILE * handles might be passed back and forth across the
     boundary.  Therefore, we disable checking in this case.  */
  if (__dlopen != NULL)
    return;
#endif

  __libc_fatal ("Fatal error: glibc detected an invalid stdio handle\n");
}

/* Some variants of libstdc++ interpose _IO_2_1_stdin_ etc. and
   install their own vtables directly, without calling _IO_init or
   other functions.  Detect this by looking at the vtables values
   during startup, and disable vtable validation in this case.  */
#ifdef SHARED
__attribute__ ((constructor))
static void
check_stdfiles_vtables (void)
{
  if (_IO_2_1_stdin_.vtable != &_IO_file_jumps
      || _IO_2_1_stdout_.vtable != &_IO_file_jumps
      || _IO_2_1_stderr_.vtable != &_IO_file_jumps)
    IO_set_accept_foreign_vtables (&_IO_vtable_check);
}
#endif
