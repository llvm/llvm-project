/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
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

#include <stdlib.h>
#include <unistd.h>
#include <ldsodefs.h>
#include <sysdep.h>

#ifndef SHARED
#include <hwcapinfo.h>
#endif

/* The main work is done in the generic function.  */
#define LIBC_START_MAIN generic_start_main
#define LIBC_START_DISABLE_INLINE
#define LIBC_START_MAIN_AUXVEC_ARG
#define MAIN_AUXVEC_ARG
#define INIT_MAIN_ARGS
#include <csu/libc-start.c>

struct startup_info
  {
    void *sda_base;
    int (*main) (int, char **, char **, void *);
    int (*init) (int, char **, char **, void *);
    void (*fini) (void);
  };

int
__libc_start_main_impl (int argc, char **argv,
			char **ev,
			ElfW (auxv_t) * auxvec,
			void (*rtld_fini) (void),
			struct startup_info *stinfo,
			char **stack_on_entry)
{
  /* the PPC SVR4 ABI says that the top thing on the stack will
     be a NULL pointer, so if not we assume that we're being called
     as a statically-linked program by Linux...  */
  if (*stack_on_entry != NULL)
    {
      char **temp;
      /* ...in which case, we have argc as the top thing on the
         stack, followed by argv (NULL-terminated), envp (likewise),
         and the auxiliary vector.  */
      /* 32/64-bit agnostic load from stack */
      argc = *(long int *) stack_on_entry;
      argv = stack_on_entry + 1;
      ev = argv + argc + 1;
#ifdef HAVE_AUX_VECTOR
      temp = ev;
      while (*temp != NULL)
	++temp;
      auxvec = (ElfW (auxv_t) *)++ temp;
#endif
      rtld_fini = NULL;
    }

  for (ElfW (auxv_t) * av = auxvec; av->a_type != AT_NULL; ++av)
    switch (av->a_type)
      {
      /* For the static case, we also need _dl_hwcap, _dl_hwcap2 and
         _dl_platform, so we can call
         __tcb_parse_hwcap_and_convert_at_platform ().  */
#ifndef SHARED
      case AT_HWCAP:
	_dl_hwcap = (unsigned long int) av->a_un.a_val;
	break;
      case AT_HWCAP2:
	_dl_hwcap2 = (unsigned long int) av->a_un.a_val;
	break;
      case AT_PLATFORM:
	_dl_platform = (void *) av->a_un.a_val;
	break;
#endif
      }

  /* Initialize hwcap/hwcap2 and platform data so it can be copied to
     the TCB later in __libc_setup_tls (). (static case only).  */
#ifndef SHARED
  __tcb_parse_hwcap_and_convert_at_platform ();
#endif

  return generic_start_main (stinfo->main, argc, argv, auxvec,
			     stinfo->init, stinfo->fini, rtld_fini,
			     stack_on_entry);
}
DEFINE_LIBC_START_MAIN_VERSION
