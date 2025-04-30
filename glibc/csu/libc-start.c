/* Perform initialization and invoke main.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
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

/* Note: This code is only part of the startup code proper for
   statically linked binaries.  For dynamically linked binaries, it
   resides in libc.so.  */

/* Mark symbols hidden in static PIE for early self relocation to work.  */
#include <errno.h>
#include <stdint.h>
#include <sys/mman.h>
#if BUILD_PIE_DEFAULT
# pragma GCC visibility push(hidden)
#endif

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <ldsodefs.h>
#include <libc-diag.h>
#include <libc-internal.h>
#include <elf/libc-early-init.h>
#include <stdbool.h>
#include <elf-initfini.h>
#include <shlib-compat.h>

#include <elf/dl-tunables.h>
#include <_ns_unmigratable.h>

extern void __libc_init_first (int argc, char **argv, char **envp);

#include <tls.h>
#ifndef SHARED
# include <dl-osinfo.h>
# ifndef THREAD_SET_STACK_GUARD
/* Only exported for architectures that don't store the stack guard canary
   in thread local area.  */
uintptr_t __stack_chk_guard attribute_relro;
# endif
# ifndef  THREAD_SET_POINTER_GUARD
/* Only exported for architectures that don't store the pointer guard
   value in thread local area.  */
uintptr_t __pointer_chk_guard_local attribute_relro attribute_hidden;
# endif
#endif

#ifndef SHARED
# include <link.h>
# include <dl-irel.h>

# ifdef ELF_MACHINE_IRELA
#  define IREL_T	ElfW(Rela)
#  define IPLT_START	__rela_iplt_start
#  define IPLT_END	__rela_iplt_end
#  define IREL		elf_irela
# elif defined ELF_MACHINE_IREL
#  define IREL_T	ElfW(Rel)
#  define IPLT_START	__rel_iplt_start
#  define IPLT_END	__rel_iplt_end
#  define IREL		elf_irel
# endif

static void
apply_irel (void)
{
# ifdef IREL
  /* We use weak references for these so that we'll still work with a linker
     that doesn't define them.  Such a linker doesn't support IFUNC at all
     and so uses won't work, but a statically-linked program that doesn't
     use any IFUNC symbols won't have a problem.  */
  extern const IREL_T IPLT_START[] __attribute__ ((weak));
  extern const IREL_T IPLT_END[] __attribute__ ((weak));
  for (const IREL_T *ipltent = IPLT_START; ipltent < IPLT_END; ++ipltent)
    IREL (ipltent);
# endif
}
#endif


#ifdef LIBC_START_MAIN
# ifdef LIBC_START_DISABLE_INLINE
#  define STATIC static
# else
#  define STATIC static inline __attribute__ ((always_inline))
# endif
# define DO_DEFINE_LIBC_START_MAIN_VERSION 0
#else
# define STATIC
# define LIBC_START_MAIN __libc_start_main_impl
# define DO_DEFINE_LIBC_START_MAIN_VERSION 1
#endif

#ifdef MAIN_AUXVEC_ARG
/* main gets passed a pointer to the auxiliary.  */
# define MAIN_AUXVEC_DECL	, void *
# define MAIN_AUXVEC_PARAM	, auxvec
#else
# define MAIN_AUXVEC_DECL
# define MAIN_AUXVEC_PARAM
#endif

#ifndef ARCH_INIT_CPU_FEATURES
# define ARCH_INIT_CPU_FEATURES()
#endif

/* Obtain the definition of __libc_start_call_main.  */
#include <libc_start_call_main.h>

#ifdef SHARED
/* Initialization for dynamic executables.  Find the main executable
   link map and run its init functions.  */
static void
call_init (int argc, char **argv, char **env)
{
  /* Obtain the main map of the executable.  */
  struct link_map *l = GL(dl_ns)[LM_ID_BASE]._ns_loaded;

  /* DT_PREINIT_ARRAY is not processed here.  It is already handled in
     _dl_init in elf/dl-init.c.  Also see the call_init function in
     the same file.  */

  if (ELF_INITFINI && l->l_info[DT_INIT] != NULL)
    DL_CALL_DT_INIT(l, l->l_addr + l->l_info[DT_INIT]->d_un.d_ptr,
		    argc, argv, env);

  ElfW(Dyn) *init_array = l->l_info[DT_INIT_ARRAY];
  if (init_array != NULL)
    {
      unsigned int jm
	= l->l_info[DT_INIT_ARRAYSZ]->d_un.d_val / sizeof (ElfW(Addr));
      ElfW(Addr) *addrs = (void *) (init_array->d_un.d_ptr + l->l_addr);
      for (unsigned int j = 0; j < jm; ++j)
	((dl_init_t) addrs[j]) (argc, argv, env);
    }
}

#else /* !SHARED */

/* These magic symbols are provided by the linker.  */
extern void (*__preinit_array_start []) (int, char **, char **)
  attribute_hidden;
extern void (*__preinit_array_end []) (int, char **, char **)
  attribute_hidden;
extern void (*__init_array_start []) (int, char **, char **)
  attribute_hidden;
extern void (*__init_array_end []) (int, char **, char **)
  attribute_hidden;
extern void (*__fini_array_start []) (void) attribute_hidden;
extern void (*__fini_array_end []) (void) attribute_hidden;

# if ELF_INITFINI
/* These function symbols are provided for the .init/.fini section entry
   points automagically by the linker.  */
extern void _init (void);
extern void _fini (void);
# endif

/* Initialization for static executables.  There is no dynamic
   segment, so we access the symbols directly.  */
static void
call_init (int argc, char **argv, char **envp)
{
  /* For static executables, preinit happens right before init.  */
  {
    const size_t size = __preinit_array_end - __preinit_array_start;
    size_t i;
    for (i = 0; i < size; i++)
      (*__preinit_array_start [i]) (argc, argv, envp);
  }

# if ELF_INITFINI
  _init ();
# endif

  const size_t size = __init_array_end - __init_array_start;
  for (size_t i = 0; i < size; i++)
      (*__init_array_start [i]) (argc, argv, envp);
}

/* Likewise for the destructor.  */
static void
call_fini (void *unused)
{
  size_t i = __fini_array_end - __fini_array_start;
  while (i-- > 0)
    (*__fini_array_start [i]) ();

# if ELF_INITFINI
  _fini ();
# endif
}

#endif /* !SHARED */

#include <libc-start.h>

STATIC int LIBC_START_MAIN (int (*main) (int, char **, char **
					 MAIN_AUXVEC_DECL),
			    int argc,
			    char **argv,
#ifdef LIBC_START_MAIN_AUXVEC_ARG
			    ElfW(auxv_t) *auxvec,
#endif
			    __typeof (main) init,
			    void (*fini) (void),
			    void (*rtld_fini) (void),
			    void *stack_end)
     __attribute__ ((noreturn));


/* Note: The init and fini parameters are no longer used.  fini is
   completely unused, init is still called if not NULL, but the
   current startup code always passes NULL.  (In the future, it would
   be possible to use fini to pass a version code if init is NULL, to
   indicate the link-time glibc without introducing a hard
   incompatibility for new programs with older glibc versions.)

   For dynamically linked executables, the dynamic segment is used to
   locate constructors and destructors.  For statically linked
   executables, the relevant symbols are access directly.  */
STATIC int
LIBC_START_MAIN (int (*main) (int, char **, char ** MAIN_AUXVEC_DECL),
		 int argc, char **argv,
#ifdef LIBC_START_MAIN_AUXVEC_ARG
		 ElfW(auxv_t) *auxvec,
#endif
		 __typeof (main) init,
		 void (*fini) (void),
		 void (*rtld_fini) (void), void *stack_end)
{
#ifndef SHARED
  char **ev = &argv[argc + 1];

  __environ = ev;

  /* Store the lowest stack address.  This is done in ld.so if this is
     the code for the DSO.  */
  __libc_stack_end = stack_end;

# ifdef HAVE_AUX_VECTOR
  /* First process the auxiliary vector since we need to find the
     program header to locate an eventually present PT_TLS entry.  */
#  ifndef LIBC_START_MAIN_AUXVEC_ARG
  ElfW(auxv_t) *auxvec;
  {
    char **evp = ev;
    while (*evp++ != NULL)
      ;
    auxvec = (ElfW(auxv_t) *) evp;
  }
#  endif
  _dl_aux_init (auxvec);
  if (GL(dl_phdr) == NULL)
# endif
    {
      /* Starting from binutils-2.23, the linker will define the
         magic symbol __ehdr_start to point to our own ELF header
         if it is visible in a segment that also includes the phdrs.
         So we can set up _dl_phdr and _dl_phnum even without any
         information from auxv.  */

      extern const ElfW(Ehdr) __ehdr_start
# if BUILD_PIE_DEFAULT
	__attribute__ ((visibility ("hidden")));
# else
	__attribute__ ((weak, visibility ("hidden")));
      if (&__ehdr_start != NULL)
# endif
        {
          assert (__ehdr_start.e_phentsize == sizeof *GL(dl_phdr));
          GL(dl_phdr) = (const void *) &__ehdr_start + __ehdr_start.e_phoff;
          GL(dl_phnum) = __ehdr_start.e_phnum;
        }
    }

  /* Initialize very early so that tunables can use it.  */
  __libc_init_secure ();

  __tunables_init (__environ);

  ARCH_INIT_CPU_FEATURES ();

  /* Do static pie self relocation after tunables and cpu features
     are setup for ifunc resolvers. Before this point relocations
     must be avoided.  */
  _dl_relocate_static_pie ();

  /* Perform IREL{,A} relocations.  */
  ARCH_SETUP_IREL ();

  /* The stack guard goes into the TCB, so initialize it early.  */
  ARCH_SETUP_TLS ();

  /* In some architectures, IREL{,A} relocations happen after TLS setup in
     order to let IFUNC resolvers benefit from TCB information, e.g. powerpc's
     hwcap and platform fields available in the TCB.  */
  ARCH_APPLY_IREL ();

  /* Set up the stack checker's canary.  */
  uintptr_t stack_chk_guard = _dl_setup_stack_chk_guard (_dl_random);
# ifdef THREAD_SET_STACK_GUARD
  THREAD_SET_STACK_GUARD (stack_chk_guard);
# else
  __stack_chk_guard = stack_chk_guard;
# endif

# ifdef DL_SYSDEP_OSCHECK
  {
    /* This needs to run to initiliaze _dl_osversion before TLS
       setup might check it.  */
    DL_SYSDEP_OSCHECK (__libc_fatal);
  }
# endif

  /* Initialize libpthread if linked in.  */
  if (__pthread_initialize_minimal != NULL)
    __pthread_initialize_minimal ();

  /* Set up the pointer guard value.  */
  uintptr_t pointer_chk_guard = _dl_setup_pointer_guard (_dl_random,
							 stack_chk_guard);
# ifdef THREAD_SET_POINTER_GUARD
  THREAD_SET_POINTER_GUARD (pointer_chk_guard);
# else
  __pointer_chk_guard_local = pointer_chk_guard;
# endif

#endif /* !SHARED  */

  /* Register the destructor of the dynamic linker if there is any.  */
  if (__glibc_likely (rtld_fini != NULL))
    __cxa_atexit ((void (*) (void *)) rtld_fini, NULL, NULL);

#ifndef SHARED
  /* Perform early initialization.  In the shared case, this function
     is called from the dynamic loader as early as possible.  */
  __libc_early_init (true);

  /* Call the initializer of the libc.  This is only needed here if we
     are compiling for the static library in which case we haven't
     run the constructors in `_dl_start_user'.  */
  __libc_init_first (argc, argv, __environ);

  /* Register the destructor of the statically-linked program.  */
  __cxa_atexit (call_fini, NULL, NULL);

  /* Some security at this point.  Prevent starting a SUID binary where
     the standard file descriptors are not opened.  We have to do this
     only for statically linked applications since otherwise the dynamic
     loader did the work already.  */
  if (__builtin_expect (__libc_enable_secure, 0))
    __libc_check_standard_fds ();
#endif /* !SHARED */

  /* Call the initializer of the program, if any.  */
#ifdef SHARED
  if (__builtin_expect (GLRO(dl_debug_mask) & DL_DEBUG_IMPCALLS, 0))
    GLRO(dl_debug_printf) ("\ninitialize program: %s\n\n", argv[0]);

  if (init != NULL)
    /* This is a legacy program which supplied its own init
       routine.  */
    (*init) (argc, argv, __environ MAIN_AUXVEC_PARAM);
  else
    /* This is a current program.  Use the dynamic segment to find
       constructors.  */
    call_init (argc, argv, __environ);
#else /* !SHARED */
  call_init (argc, argv, __environ);
#endif /* SHARED */

#ifdef SHARED
  /* Auditing checkpoint: we have a new object.  */
  if (__glibc_unlikely (GLRO(dl_naudit) > 0))
    {
      struct audit_ifaces *afct = GLRO(dl_audit);
      struct link_map *head = GL(dl_ns)[LM_ID_BASE]._ns_loaded;
      for (unsigned int cnt = 0; cnt < GLRO(dl_naudit); ++cnt)
	{
	  if (afct->preinit != NULL)
	    afct->preinit (&link_map_audit_state (head, cnt)->cookie);

	  afct = afct->next;
	}
    }
#endif

#ifdef SHARED
  if (__glibc_unlikely (GLRO(dl_debug_mask) & DL_DEBUG_IMPCALLS))
    GLRO(dl_debug_printf) ("\ntransferring control: %s\n\n", argv[0]);
#endif

#ifndef SHARED
  _dl_debug_initialize (0, LM_ID_BASE);
#endif

  __ns_main_app_started = 1;

  if (getenv("TRAP_MAIN_STACK")) {
    //we do not support dynamic grow of the stack so instead just allocate the default limit of 8MB
    const uint64_t main_stack_size = 0x800000;
    const uint64_t page_size = sysconf(_SC_PAGESIZE);
    const uint64_t main_stack_size_with_guard = main_stack_size + page_size;

    void *addr = mmap(NULL, main_stack_size_with_guard, PROT_READ | PROT_WRITE,
                      MAP_ANONYMOUS | MAP_PRIVATE | MAP_STACK, -1, 0);

    // modify last page (lowest address) as prot_none to catch stack exhaustion
    if (addr != MAP_FAILED && mprotect(addr, page_size, PROT_NONE) == 0) {
      //move addr to top address is it grows down and change rsp
      asm("addq %1, %0\n\t"
          "movq %0, %%rsp\n\t"
          : "=m"(addr)
          : "r"(main_stack_size_with_guard)
          : "rsp");
    }
  }
  __libc_start_call_main(main, argc, argv MAIN_AUXVEC_PARAM);
  //__libc_start_call_main calls exit so we do not reach here
  __ns_main_app_started = 0;
}

/* Starting with glibc 2.34, the init parameter is always NULL.  Older
   libcs are not prepared to handle that.  The macro
   DEFINE_LIBC_START_MAIN_VERSION creates GLIBC_2.34 alias, so that
   newly linked binaries reflect that dependency.  The macros below
   expect that the exported function is called
   __libc_start_main_impl.  */
#ifdef SHARED
# define DEFINE_LIBC_START_MAIN_VERSION \
  DEFINE_LIBC_START_MAIN_VERSION_1 \
  strong_alias (__libc_start_main_impl, __libc_start_main_alias_2)	\
  versioned_symbol (libc, __libc_start_main_alias_2, __libc_start_main, \
		    GLIBC_2_34);

# if SHLIB_COMPAT(libc, GLIBC_2_0, GLIBC_2_34)
#  define DEFINE_LIBC_START_MAIN_VERSION_1 \
  strong_alias (__libc_start_main_impl, __libc_start_main_alias_1)	\
  compat_symbol (libc, __libc_start_main_alias_1, __libc_start_main, GLIBC_2_0);
#  else
#  define DEFINE_LIBC_START_MAIN_VERSION_1
# endif
#else  /* !SHARED */
/* Enable calling the function under its exported name.  */
# define DEFINE_LIBC_START_MAIN_VERSION \
  strong_alias (__libc_start_main_impl, __libc_start_main)
#endif

/* Only define the version information if LIBC_START_MAIN was not set.
   If there is a wrapper file, it must expand
   DEFINE_LIBC_START_MAIN_VERSION on its own.  */
#if DO_DEFINE_LIBC_START_MAIN_VERSION
DEFINE_LIBC_START_MAIN_VERSION
#endif
