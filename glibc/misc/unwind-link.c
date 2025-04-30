/* Dynamic loading of the libgcc unwinder.
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

#ifdef SHARED

#include <assert.h>
#include <dlfcn.h>
#include <gnu/lib-names.h>
#include <unwind-link.h>
#include <libc-lock.h>

/* Statically allocate the object, so that we do not have to deal with
   malloc failure.  __libc_unwind_link_get must not fail if libgcc_s
   has already been loaded by other means.  */
static struct unwind_link global;

/* dlopen handle.  Also used for the double-checked locking idiom.  */
static void *global_libgcc_handle;

/* We cannot use __libc_once because the pthread_once implementation
   may depend on unwinding.  */
__libc_lock_define (static, lock);

struct unwind_link *
__libc_unwind_link_get (void)
{
  /* Double-checked locking idiom.  Synchronizes with the release MO
     store at the end of this function.  */
  if (atomic_load_acquire (&global_libgcc_handle) != NULL)
   return &global;

  /* Initialize a copy of the data, so that we do not need about
     unlocking in case the dynamic loader somehow triggers
     unwinding.  */
  void *local_libgcc_handle = __libc_dlopen (LIBGCC_S_SO);
  if (local_libgcc_handle == NULL)
    {
      __libc_lock_unlock (lock);
      return NULL;
    }

  struct unwind_link local;
  local.ptr__Unwind_Backtrace
    = __libc_dlsym (local_libgcc_handle, "_Unwind_Backtrace");
  local.ptr__Unwind_ForcedUnwind
    = __libc_dlsym (local_libgcc_handle, "_Unwind_ForcedUnwind");
  local.ptr__Unwind_GetCFA
    = __libc_dlsym (local_libgcc_handle, "_Unwind_GetCFA");
#if UNWIND_LINK_GETIP
  local.ptr__Unwind_GetIP
    = __libc_dlsym (local_libgcc_handle, "_Unwind_GetIP");
#endif
  local.ptr__Unwind_Resume
    = __libc_dlsym (local_libgcc_handle, "_Unwind_Resume");
#if UNWIND_LINK_FRAME_STATE_FOR
  local.ptr___frame_state_for
    = __libc_dlsym (local_libgcc_handle, "__frame_state_for");
#endif
  local.ptr_personality
    = __libc_dlsym (local_libgcc_handle, "__gcc_personality_v0");
  UNWIND_LINK_EXTRA_INIT

  /* If a symbol is missing, libgcc_s has somehow been corrupted.  */
  assert (local.ptr__Unwind_Backtrace != NULL);
  assert (local.ptr__Unwind_ForcedUnwind != NULL);
  assert (local.ptr__Unwind_GetCFA != NULL);
#if UNWIND_LINK_GETIP
  assert (local.ptr__Unwind_GetIP != NULL);
#endif
  assert (local.ptr__Unwind_Resume != NULL);
  assert (local.ptr_personality != NULL);

#ifdef PTR_MANGLE
  PTR_MANGLE (local.ptr__Unwind_Backtrace);
  PTR_MANGLE (local.ptr__Unwind_ForcedUnwind);
  PTR_MANGLE (local.ptr__Unwind_GetCFA);
# if UNWIND_LINK_GETIP
  PTR_MANGLE (local.ptr__Unwind_GetIP);
# endif
  PTR_MANGLE (local.ptr__Unwind_Resume);
# if UNWIND_LINK_FRAME_STATE_FOR
  PTR_MANGLE (local.ptr___frame_state_for);
# endif
  PTR_MANGLE (local.ptr_personality);
#endif

  __libc_lock_lock (lock);
  if (atomic_load_relaxed (&global_libgcc_handle) != NULL)
    /* This thread lost the race.  Clean up.  */
    __libc_dlclose (local_libgcc_handle);
  else
    {
      global = local;

      /* Completes the double-checked locking idiom.  */
      atomic_store_release (&global_libgcc_handle, local_libgcc_handle);
    }

  __libc_lock_unlock (lock);
  return &global;
}
libc_hidden_def (__libc_unwind_link_get)

void
__libc_unwind_link_after_fork (void)
{
  if (__libc_lock_trylock (lock) == 0)
    /* The lock was not acquired during the fork.  This covers both
       the initialized and uninitialized case.  */
    __libc_lock_unlock (lock);
  else
    {
      /* Initialization was in progress in another thread.
         Reinitialize the lock.  */
      __libc_lock_init (lock);
      global_libgcc_handle = NULL;
    }
}

void __libc_freeres_fn_section
__libc_unwind_link_freeres (void)
{
  if (global_libgcc_handle != NULL)
    {
      __libc_dlclose (global_libgcc_handle );
      global_libgcc_handle = NULL;
    }
}

#endif /* SHARED */
