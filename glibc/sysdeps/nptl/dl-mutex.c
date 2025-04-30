/* Recursive locking implementation for the dynamic loader.  NPTL version.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

/* Use the mutex implementation in libc (assuming PTHREAD_IN_LIBC).  */

#include <assert.h>
#include <first-versions.h>
#include <ldsodefs.h>

__typeof (pthread_mutex_lock) *___rtld_mutex_lock attribute_relro;
__typeof (pthread_mutex_unlock) *___rtld_mutex_unlock attribute_relro;

void
__rtld_mutex_init (void)
{
  /* There is an implicit assumption here that the lock counters are
     zero and this function is called while nothing is locked.  For
     early initialization of the mutex functions this is true because
     it happens directly in dl_main in elf/rtld.c, and not some ELF
     constructor while holding loader locks.  */

  struct link_map *libc_map = GL (dl_ns)[LM_ID_BASE].libc_map;

  const ElfW(Sym) *sym
    = _dl_lookup_direct (libc_map, "pthread_mutex_lock",
                         0x4f152227, /* dl_new_hash output.  */
                         FIRST_VERSION_libc_pthread_mutex_lock_STRING,
                         FIRST_VERSION_libc_pthread_mutex_lock_HASH);
  assert (sym != NULL);
  ___rtld_mutex_lock = DL_SYMBOL_ADDRESS (libc_map, sym);

  sym = _dl_lookup_direct (libc_map, "pthread_mutex_unlock",
                           0x7dd7aaaa, /* dl_new_hash output.  */
                           FIRST_VERSION_libc_pthread_mutex_unlock_STRING,
                           FIRST_VERSION_libc_pthread_mutex_unlock_HASH);
  assert (sym != NULL);
  ___rtld_mutex_unlock = DL_SYMBOL_ADDRESS (libc_map, sym);
}
