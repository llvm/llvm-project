/* Run-time dynamic linker data structures for loaded ELF shared objects.
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

#ifndef	_LDSODEFS_H
#define	_LDSODEFS_H	1

#include <features.h>

#include <stdbool.h>
#define __need_size_t
#define __need_NULL
#include <stddef.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>

#include <elf.h>
#include <dlfcn.h>
#include <fpu_control.h>
#include <sys/mman.h>
#include <link.h>
#include <dl-lookupcfg.h>
#include <dl-sysdep.h>
#include <libc-lock.h>
#include <hp-timing.h>
#include <tls.h>
#include <list_t.h>

__BEGIN_DECLS

#define VERSYMIDX(sym)	(DT_NUM + DT_THISPROCNUM + DT_VERSIONTAGIDX (sym))
#define VALIDX(tag)	(DT_NUM + DT_THISPROCNUM + DT_VERSIONTAGNUM \
			 + DT_EXTRANUM + DT_VALTAGIDX (tag))
#define ADDRIDX(tag)	(DT_NUM + DT_THISPROCNUM + DT_VERSIONTAGNUM \
			 + DT_EXTRANUM + DT_VALNUM + DT_ADDRTAGIDX (tag))

/* Type of GNU hash which the machine uses.  */
#ifndef ELF_MACHINE_GNU_HASH_ADDRIDX
# define ELF_MACHINE_GNU_HASH_ADDRIDX ADDRIDX (DT_GNU_HASH)
#endif

/* Calculate the index of a symbol in GNU hash.  */
#ifndef ELF_MACHINE_HASH_SYMIDX
# define ELF_MACHINE_HASH_SYMIDX(map, hasharr) \
  ((hasharr) - (map)->l_gnu_chain_zero)
#endif

/* Setup MIPS xhash.  Defined only for MIPS.  */
#ifndef ELF_MACHINE_XHASH_SETUP
# define ELF_MACHINE_XHASH_SETUP(hash32, symbias, map) \
  ((void) (hash32), (void) (symbias), (void) (map))
#endif

/* We use this macro to refer to ELF types independent of the native wordsize.
   `ElfW(TYPE)' is used in place of `Elf32_TYPE' or `Elf64_TYPE'.  */
#define ELFW(type)	_ElfW (ELF, __ELF_NATIVE_CLASS, type)

/* All references to the value of l_info[DT_PLTGOT],
  l_info[DT_STRTAB], l_info[DT_SYMTAB], l_info[DT_RELA],
  l_info[DT_REL], l_info[DT_JMPREL], and l_info[VERSYMIDX (DT_VERSYM)]
  have to be accessed via the D_PTR macro.  The macro is needed since for
  most architectures the entry is already relocated - but for some not
  and we need to relocate at access time.  */
#ifdef DL_RO_DYN_SECTION
# define D_PTR(map, i) ((map)->i->d_un.d_ptr + (map)->l_addr)
#else
# define D_PTR(map, i) (map)->i->d_un.d_ptr
#endif

/* Result of the lookup functions and how to retrieve the base address.  */
typedef struct link_map *lookup_t;
#define LOOKUP_VALUE(map) map
#define LOOKUP_VALUE_ADDRESS(map, set) ((set) || (map) ? (map)->l_addr : 0)

/* Calculate the address of symbol REF using the base address from map MAP,
   if non-NULL.  Don't check for NULL map if MAP_SET is TRUE.  */
#define SYMBOL_ADDRESS(map, ref, map_set)				\
  ((ref) == NULL ? 0							\
   : (__glibc_unlikely ((ref)->st_shndx == SHN_ABS) ? 0			\
      : LOOKUP_VALUE_ADDRESS (map, map_set)) + (ref)->st_value)

/* Type of a constructor function, in DT_INIT, DT_INIT_ARRAY,
   DT_PREINIT_ARRAY.  */
typedef void (*dl_init_t) (int, char **, char **);

/* On some architectures a pointer to a function is not just a pointer
   to the actual code of the function but rather an architecture
   specific descriptor. */
#ifndef ELF_FUNCTION_PTR_IS_SPECIAL
# define DL_SYMBOL_ADDRESS(map, ref) \
 (void *) SYMBOL_ADDRESS (map, ref, false)
# define DL_LOOKUP_ADDRESS(addr) ((ElfW(Addr)) (addr))
# define DL_CALL_DT_INIT(map, start, argc, argv, env) \
 ((dl_init_t) (start)) (argc, argv, env)
# define DL_CALL_DT_FINI(map, start) ((fini_t) (start)) ()
#endif

/* On some architectures dladdr can't use st_size of all symbols this way.  */
#define DL_ADDR_SYM_MATCH(L, SYM, MATCHSYM, ADDR) \
  ((ADDR) >= (L)->l_addr + (SYM)->st_value				\
   && ((((SYM)->st_shndx == SHN_UNDEF || (SYM)->st_size == 0)		\
	&& (ADDR) == (L)->l_addr + (SYM)->st_value)			\
       || (ADDR) < (L)->l_addr + (SYM)->st_value + (SYM)->st_size)	\
   && ((MATCHSYM) == NULL || (MATCHSYM)->st_value < (SYM)->st_value))

/* According to the ELF gABI no STV_HIDDEN or STV_INTERNAL symbols are
   expected to be present in dynamic symbol tables as they should have
   been either removed or converted to STB_LOCAL binding by the static
   linker.  However some GNU binutils versions produce such symbols in
   some cases.  To prevent such symbols present in a buggy binary from
   preempting global symbols we filter them out with this predicate.  */
static __always_inline bool
dl_symbol_visibility_binds_local_p (const ElfW(Sym) *sym)
{
  return (ELFW(ST_VISIBILITY) (sym->st_other) == STV_HIDDEN
	  || ELFW(ST_VISIBILITY) (sym->st_other) == STV_INTERNAL);
}

/* Unmap a loaded object, called by _dl_close (). */
#ifndef DL_UNMAP_IS_SPECIAL
# define DL_UNMAP(map)	_dl_unmap_segments (map)
#endif

/* Reloc type classes as returned by elf_machine_type_class().
   ELF_RTYPE_CLASS_PLT means this reloc should not be satisfied by
   some PLT symbol, ELF_RTYPE_CLASS_COPY means this reloc should not be
   satisfied by any symbol in the executable.  Some architectures do
   not support copy relocations.  In this case we define the macro to
   zero so that the code for handling them gets automatically optimized
   out.  ELF_RTYPE_CLASS_EXTERN_PROTECTED_DATA means address of protected
   data defined in the shared library may be external, i.e., due to copy
   relocation.  */
#define ELF_RTYPE_CLASS_PLT 1
#ifndef DL_NO_COPY_RELOCS
# define ELF_RTYPE_CLASS_COPY 2
#else
# define ELF_RTYPE_CLASS_COPY 0
#endif
/* If DL_EXTERN_PROTECTED_DATA is defined, address of protected data
   defined in the shared library may be external, i.e., due to copy
   relocation.   */
#ifdef DL_EXTERN_PROTECTED_DATA
# define ELF_RTYPE_CLASS_EXTERN_PROTECTED_DATA 4
#else
# define ELF_RTYPE_CLASS_EXTERN_PROTECTED_DATA 0
#endif

/* ELF uses the PF_x macros to specify the segment permissions, mmap
   uses PROT_xxx.  In most cases the three macros have the values 1, 2,
   and 3 but not in a matching order.  The following macros allows
   converting from the PF_x values to PROT_xxx values.  */
#define PF_TO_PROT \
  ((PROT_READ << (PF_R * 4))						      \
   | (PROT_WRITE << (PF_W * 4))						      \
   | (PROT_EXEC << (PF_X * 4))						      \
   | ((PROT_READ | PROT_WRITE) << ((PF_R | PF_W) * 4))			      \
   | ((PROT_READ | PROT_EXEC) << ((PF_R | PF_X) * 4))			      \
   | ((PROT_WRITE | PROT_EXEC) << (PF_W | PF_X) * 4)			      \
   | ((PROT_READ | PROT_WRITE | PROT_EXEC) << ((PF_R | PF_W | PF_X) * 4)))

/* The filename itself, or the main program name, if available.  */
#define DSO_FILENAME(name) ((name)[0] ? (name)				      \
			    : (rtld_progname ?: "<main program>"))

#define RTLD_PROGNAME (rtld_progname ?: "<program name unknown>")

/* For the version handling we need an array with only names and their
   hash values.  */
struct r_found_version
  {
    const char *name;
    ElfW(Word) hash;

    int hidden;
    const char *filename;
  };

/* We want to cache information about the searches for shared objects.  */

enum r_dir_status { unknown, nonexisting, existing };

struct r_search_path_elem
  {
    /* This link is only used in the `all_dirs' member of `r_search_path'.  */
    struct r_search_path_elem *next;

    /* Strings saying where the definition came from.  */
    const char *what;
    const char *where;

    /* Basename for this search path element.  The string must end with
       a slash character.  */
    const char *dirname;
    size_t dirnamelen;

    enum r_dir_status status[0];
  };

struct r_strlenpair
  {
    const char *str;
    size_t len;
  };


/* A data structure for a simple single linked list of strings.  */
struct libname_list
  {
    const char *name;		/* Name requested (before search).  */
    struct libname_list *next;	/* Link to next name for this object.  */
    int dont_free;		/* Flag whether this element should be freed
				   if the object is not entirely unloaded.  */
  };


/* Bit masks for the objects which valid callers can come from to
   functions with restricted interface.  */
enum allowmask
  {
    allow_libc = 1,
    allow_libdl = 2,
    allow_libpthread = 4,
    allow_ldso = 8
  };


struct audit_ifaces
{
  void (*activity) (uintptr_t *, unsigned int);
  char *(*objsearch) (const char *, uintptr_t *, unsigned int);
  unsigned int (*objopen) (struct link_map *, Lmid_t, uintptr_t *);
  void (*preinit) (uintptr_t *);
  union
  {
    uintptr_t (*symbind32) (Elf32_Sym *, unsigned int, uintptr_t *,
			    uintptr_t *, unsigned int *, const char *);
    uintptr_t (*symbind64) (Elf64_Sym *, unsigned int, uintptr_t *,
			    uintptr_t *, unsigned int *, const char *);
  };
  union
  {
#ifdef ARCH_PLTENTER_MEMBERS
    ARCH_PLTENTER_MEMBERS;
#endif
  };
  union
  {
#ifdef ARCH_PLTEXIT_MEMBERS
    ARCH_PLTEXIT_MEMBERS;
#endif
  };
  unsigned int (*objclose) (uintptr_t *);

  struct audit_ifaces *next;
};


/* Test whether given NAME matches any of the names of the given object.  */
extern int _dl_name_match_p (const char *__name, const struct link_map *__map)
     attribute_hidden;

/* Compute next higher prime number.  */
extern unsigned long int _dl_higher_prime_number (unsigned long int n)
     attribute_hidden;

/* A stripped down strtoul-like implementation.  */
uint64_t _dl_strtoul (const char *, char **) attribute_hidden;

/* Function used as argument for `_dl_receive_error' function.  The
   arguments are the error code, error string, and the objname the
   error occurred in.  */
typedef void (*receiver_fct) (int, const char *, const char *);

/* Internal functions of the run-time dynamic linker.
   These can be accessed if you link again the dynamic linker
   as a shared library, as in `-lld' or `/lib/ld.so' explicitly;
   but are not normally of interest to user programs.

   The `-ldl' library functions in <dlfcn.h> provide a simple
   user interface to run-time dynamic linking.  */


#ifndef SHARED
# define EXTERN extern
# define GL(name) _##name
#else
# define EXTERN
# if IS_IN (rtld)
#  define GL(name) _rtld_local._##name
# else
#  define GL(name) _rtld_global._##name
# endif
struct rtld_global
{
#endif
  /* Don't change the order of the following elements.  'dl_loaded'
     must remain the first element.  Forever.  */

/* Non-shared code has no support for multiple namespaces.  */
#ifdef SHARED
# define DL_NNS 16
#else
# define DL_NNS 1
#endif
  EXTERN struct link_namespaces
  {
    /* A pointer to the map for the main map.  */
    struct link_map *_ns_loaded;
    /* Number of object in the _dl_loaded list.  */
    unsigned int _ns_nloaded;
    /* Direct pointer to the searchlist of the main object.  */
    struct r_scope_elem *_ns_main_searchlist;
    /* This is zero at program start to signal that the global scope map is
       allocated by rtld.  Later it keeps the size of the map.  It might be
       reset if in _dl_close if the last global object is removed.  */
    unsigned int _ns_global_scope_alloc;

    /* During dlopen, this is the number of objects that still need to
       be added to the global scope map.  It has to be taken into
       account when resizing the map, for future map additions after
       recursive dlopen calls from ELF constructors.  */
    unsigned int _ns_global_scope_pending_adds;

    /* Once libc.so has been loaded into the namespace, this points to
       its link map.  */
    struct link_map *libc_map;

    /* Search table for unique objects.  */
    struct unique_sym_table
    {
      __rtld_lock_define_recursive (, lock)
      struct unique_sym
      {
	uint32_t hashval;
	const char *name;
	const ElfW(Sym) *sym;
	const struct link_map *map;
      } *entries;
      size_t size;
      size_t n_elements;
      void (*free) (void *);
    } _ns_unique_sym_table;
    /* Keep track of changes to each namespace' list.  */
    struct r_debug _ns_debug;
  } _dl_ns[DL_NNS];
  /* One higher than index of last used namespace.  */
  EXTERN size_t _dl_nns;

  /* During the program run we must not modify the global data of
     loaded shared object simultanously in two threads.  Therefore we
     protect `_dl_open' and `_dl_close' in dl-close.c.

     This must be a recursive lock since the initializer function of
     the loaded object might as well require a call to this function.
     At this time it is not anymore a problem to modify the tables.  */
  __rtld_lock_define_recursive (EXTERN, _dl_load_lock)
  /* This lock is used to keep __dl_iterate_phdr from inspecting the
     list of loaded objects while an object is added to or removed
     from that list.  */
  __rtld_lock_define_recursive (EXTERN, _dl_load_write_lock)

  /* Incremented whenever something may have been added to dl_loaded.  */
  EXTERN unsigned long long _dl_load_adds;

  /* The object to be initialized first.  */
  EXTERN struct link_map *_dl_initfirst;

  /* Map of shared object to be profiled.  */
  EXTERN struct link_map *_dl_profile_map;

  /* Counters for the number of relocations performed.  */
  EXTERN unsigned long int _dl_num_relocations;
  EXTERN unsigned long int _dl_num_cache_relocations;

  /* List of search directories.  */
  EXTERN struct r_search_path_elem *_dl_all_dirs;

  /* Structure describing the dynamic linker itself.  */
  EXTERN struct link_map _dl_rtld_map;
#ifdef SHARED
  /* Used to store the audit information for the link map of the
     dynamic loader.  */
  struct auditstate _dl_rtld_auditstate[DL_NNS];
#endif

#if !PTHREAD_IN_LIBC && defined SHARED \
    && defined __rtld_lock_default_lock_recursive
  EXTERN void (*_dl_rtld_lock_recursive) (void *);
  EXTERN void (*_dl_rtld_unlock_recursive) (void *);
#endif

  /* Get architecture specific definitions.  */
#define PROCINFO_DECL
#ifndef PROCINFO_CLASS
# define PROCINFO_CLASS EXTERN
#endif
#include <dl-procruntime.c>

#if !PTHREAD_IN_LIBC
  /* If loading a shared object requires that we make the stack executable
     when it was not, we do it by calling this function.
     It returns an errno code or zero on success.  */
  EXTERN int (*_dl_make_stack_executable_hook) (void **);
#endif

  /* Prevailing state of the stack, PF_X indicating it's executable.  */
  EXTERN ElfW(Word) _dl_stack_flags;

  /* Flag signalling whether there are gaps in the module ID allocation.  */
  EXTERN bool _dl_tls_dtv_gaps;
  /* Highest dtv index currently needed.  */
  EXTERN size_t _dl_tls_max_dtv_idx;
  /* Information about the dtv slots.  */
  EXTERN struct dtv_slotinfo_list
  {
    size_t len;
    struct dtv_slotinfo_list *next;
    struct dtv_slotinfo
    {
      size_t gen;
      struct link_map *map;
    } slotinfo[];
  } *_dl_tls_dtv_slotinfo_list;
  /* Number of modules in the static TLS block.  */
  EXTERN size_t _dl_tls_static_nelem;
  /* Size actually allocated in the static TLS block.  */
  EXTERN size_t _dl_tls_static_used;
  /* Remaining amount of static TLS that may be used for optimizing
     dynamic TLS access (e.g. with TLSDESC).  */
  EXTERN size_t _dl_tls_static_optional;

/* Number of additional entries in the slotinfo array of each slotinfo
   list element.  A large number makes it almost certain take we never
   have to iterate beyond the first element in the slotinfo list.  */
#define TLS_SLOTINFO_SURPLUS (62)

/* Number of additional slots in the dtv allocated.  */
#define DTV_SURPLUS	(14)

  /* Initial dtv of the main thread, not allocated with normal malloc.  */
  EXTERN void *_dl_initial_dtv;
  /* Generation counter for the dtv.  */
  EXTERN size_t _dl_tls_generation;

#if !THREAD_GSCOPE_IN_TCB
  EXTERN void (*_dl_init_static_tls) (struct link_map *);
#endif

  /* Scopes to free after next THREAD_GSCOPE_WAIT ().  */
  EXTERN struct dl_scope_free_list
  {
    size_t count;
    void *list[50];
  } *_dl_scope_free_list;
#if THREAD_GSCOPE_IN_TCB
  /* List of active thread stacks, with memory managed by glibc.  */
  EXTERN list_t _dl_stack_used;

  /* List of thread stacks that were allocated by the application.  */
  EXTERN list_t _dl_stack_user;

  /* List of queued thread stacks.  */
  EXTERN list_t _dl_stack_cache;

  /* Total size of all stacks in the cache (sum over stackblock_size).  */
  EXTERN size_t _dl_stack_cache_actsize;

  /* We need to record what list operations we are going to do so
     that, in case of an asynchronous interruption due to a fork()
     call, we can correct for the work.  */
  EXTERN uintptr_t _dl_in_flight_stack;

  /* Mutex protecting the stack lists.  */
  EXTERN int _dl_stack_cache_lock;
#else
  EXTERN int _dl_thread_gscope_count;
#endif
#ifdef SHARED
};
# define __rtld_global_attribute__
# if IS_IN (rtld)
#  ifdef HAVE_SDATA_SECTION
#   define __rtld_local_attribute__ \
	    __attribute__ ((visibility ("hidden"), section (".sdata")))
#   undef __rtld_global_attribute__
#   define __rtld_global_attribute__ __attribute__ ((section (".sdata")))
#  else
#   define __rtld_local_attribute__ __attribute__ ((visibility ("hidden")))
#  endif
extern struct rtld_global _rtld_local __rtld_local_attribute__;
#  undef __rtld_local_attribute__
# endif
extern struct rtld_global _rtld_global __rtld_global_attribute__;
# undef __rtld_global_attribute__
#endif

#ifndef SHARED
# define GLRO(name) _##name
#else
# if IS_IN (rtld)
#  define GLRO(name) _rtld_local_ro._##name
# else
#  define GLRO(name) _rtld_global_ro._##name
# endif
struct rtld_global_ro
{
#endif

  /* If nonzero the appropriate debug information is printed.  */
  EXTERN int _dl_debug_mask;
#define DL_DEBUG_LIBS	    (1 << 0)
#define DL_DEBUG_IMPCALLS   (1 << 1)
#define DL_DEBUG_BINDINGS   (1 << 2)
#define DL_DEBUG_SYMBOLS    (1 << 3)
#define DL_DEBUG_VERSIONS   (1 << 4)
#define DL_DEBUG_RELOC      (1 << 5)
#define DL_DEBUG_FILES      (1 << 6)
#define DL_DEBUG_STATISTICS (1 << 7)
#define DL_DEBUG_UNUSED	    (1 << 8)
#define DL_DEBUG_SCOPES	    (1 << 9)
/* These two are used only internally.  */
#define DL_DEBUG_HELP       (1 << 10)
#define DL_DEBUG_PRELINK    (1 << 11)

  /* OS version.  */
  EXTERN unsigned int _dl_osversion;
  /* Platform name.  */
  EXTERN const char *_dl_platform;
  EXTERN size_t _dl_platformlen;

  /* Cached value of `getpagesize ()'.  */
  EXTERN size_t _dl_pagesize;

  /* Cached value of `sysconf (_SC_MINSIGSTKSZ)'.  */
  EXTERN size_t _dl_minsigstacksize;

  /* Do we read from ld.so.cache?  */
  EXTERN int _dl_inhibit_cache;

  /* Copy of the content of `_dl_main_searchlist' at startup time.  */
  EXTERN struct r_scope_elem _dl_initial_searchlist;

  /* CLK_TCK as reported by the kernel.  */
  EXTERN int _dl_clktck;

  /* If nonzero print warnings messages.  */
  EXTERN int _dl_verbose;

  /* File descriptor to write debug messages to.  */
  EXTERN int _dl_debug_fd;

  /* Do we do lazy relocations?  */
  EXTERN int _dl_lazy;

  /* Nonzero if runtime lookups should not update the .got/.plt.  */
  EXTERN int _dl_bind_not;

  /* Nonzero if references should be treated as weak during runtime
     linking.  */
  EXTERN int _dl_dynamic_weak;

  /* Default floating-point control word.  */
  EXTERN fpu_control_t _dl_fpu_control;

  /* Expected cache ID.  */
  EXTERN int _dl_correct_cache_id;

  /* Mask for hardware capabilities that are available.  */
  EXTERN uint64_t _dl_hwcap;

#if !HAVE_TUNABLES
  /* Mask for important hardware capabilities we honour. */
  EXTERN uint64_t _dl_hwcap_mask;
#endif

#ifdef HAVE_AUX_VECTOR
  /* Pointer to the auxv list supplied to the program at startup.  */
  EXTERN ElfW(auxv_t) *_dl_auxv;
#endif

  /* Get architecture specific definitions.  */
#include <dl-procinfo.c>

  /* Names of shared object for which the RPATH should be ignored.  */
  EXTERN const char *_dl_inhibit_rpath;

  /* Location of the binary.  */
  EXTERN const char *_dl_origin_path;

  /* -1 if the dynamic linker should honor library load bias,
     0 if not, -2 use the default (honor biases for normal
     binaries, don't honor for PIEs).  */
  EXTERN ElfW(Addr) _dl_use_load_bias;

  /* Size of the static TLS block.  */
  EXTERN size_t _dl_tls_static_size;

  /* Alignment requirement of the static TLS block.  */
  EXTERN size_t _dl_tls_static_align;

  /* Size of surplus space in the static TLS area for dynamically
     loaded modules with IE-model TLS or for TLSDESC optimization.
     See comments in elf/dl-tls.c where it is initialized.  */
  EXTERN size_t _dl_tls_static_surplus;

  /* Name of the shared object to be profiled (if any).  */
  EXTERN const char *_dl_profile;
  /* Filename of the output file.  */
  EXTERN const char *_dl_profile_output;
  /* Name of the object we want to trace the prelinking.  */
  EXTERN const char *_dl_trace_prelink;
  /* Map of shared object to be prelink traced.  */
  EXTERN struct link_map *_dl_trace_prelink_map;

  /* All search directories defined at startup.  This is assigned a
     non-NULL pointer by the ld.so startup code (after initialization
     to NULL), so this can also serve as an indicator whether a copy
     of ld.so is initialized and active.  See the rtld_active function
     below.  */
  EXTERN struct r_search_path_elem *_dl_init_all_dirs;

#ifdef NEED_DL_SYSINFO
  /* Syscall handling improvements.  This is very specific to x86.  */
  EXTERN uintptr_t _dl_sysinfo;
#endif

#ifdef NEED_DL_SYSINFO_DSO
  /* The vsyscall page is a virtual DSO pre-mapped by the kernel.
     This points to its ELF header.  */
  EXTERN const ElfW(Ehdr) *_dl_sysinfo_dso;

  /* At startup time we set up the normal DSO data structure for it,
     and this points to it.  */
  EXTERN struct link_map *_dl_sysinfo_map;

# define PROCINFO_DECL
# ifndef PROCINFO_CLASS
#  define PROCINFO_CLASS EXTERN
# endif
# include <dl-vdso-setup.c>
#endif

  /* Mask for more hardware capabilities that are available on some
     platforms.  */
  EXTERN uint64_t _dl_hwcap2;

#ifdef SHARED
  /* We add a function table to _rtld_global which is then used to
     call the function instead of going through the PLT.  The result
     is that we can avoid exporting the functions and we do not jump
     PLT relocations in libc.so.  */
  void (*_dl_debug_printf) (const char *, ...)
       __attribute__ ((__format__ (__printf__, 1, 2)));
  void (*_dl_mcount) (ElfW(Addr) frompc, ElfW(Addr) selfpc);
  lookup_t (*_dl_lookup_symbol_x) (const char *, struct link_map *,
				   const ElfW(Sym) **, struct r_scope_elem *[],
				   const struct r_found_version *, int, int,
				   struct link_map *);
  void *(*_dl_open) (const char *file, int mode, const void *caller_dlopen,
		     Lmid_t nsid, int argc, char *argv[], char *env[]);
  void (*_dl_close) (void *map);
  /* libdl in a secondary namespace (after dlopen) must use
     _dl_catch_error from the main namespace, so it has to be
     exported in some way.  */
  int (*_dl_catch_error) (const char **objname, const char **errstring,
			  bool *mallocedp, void (*operate) (void *),
			  void *args);
  /* libdl in a secondary namespace must use free from the base
     namespace.  */
  void (*_dl_error_free) (void *);
  void *(*_dl_tls_get_addr_soft) (struct link_map *, struct pthread *);
#ifdef HAVE_DL_DISCOVER_OSVERSION
  int (*_dl_discover_osversion) (void);
#endif

  /* Dynamic linker operations used after static dlopen.  */
  const struct dlfcn_hook *_dl_dlfcn_hook;

  /* List of auditing interfaces.  */
  struct audit_ifaces *_dl_audit;
  unsigned int _dl_naudit;
};
# define __rtld_global_attribute__
# if IS_IN (rtld)
#  define __rtld_local_attribute__ __attribute__ ((visibility ("hidden")))
extern struct rtld_global_ro _rtld_local_ro
    attribute_relro __rtld_local_attribute__;
extern struct rtld_global_ro _rtld_global_ro
    attribute_relro __rtld_global_attribute__;
#  undef __rtld_local_attribute__
# else
/* We cheat a bit here.  We declare the variable as as const even
   though it is at startup.  */
extern const struct rtld_global_ro _rtld_global_ro
    attribute_relro __rtld_global_attribute__;
# endif
# undef __rtld_global_attribute__
#endif
#undef EXTERN

#ifndef SHARED
/* dl-support.c defines these and initializes them early on.  */
extern const ElfW(Phdr) *_dl_phdr;
extern size_t _dl_phnum;
#endif

#if PTHREAD_IN_LIBC
/* This function changes the permissions of all stacks (not just those
   of the main stack).  */
int _dl_make_stacks_executable (void **stack_endp) attribute_hidden;
#else
/* This is the initial value of GL(dl_make_stack_executable_hook).
   A threads library can change it.  The ld.so implementation changes
   the permissions of the main stack only.  */
extern int _dl_make_stack_executable (void **stack_endp);
rtld_hidden_proto (_dl_make_stack_executable)
#endif

/* Variable pointing to the end of the stack (or close to it).  This value
   must be constant over the runtime of the application.  Some programs
   might use the variable which results in copy relocations on some
   platforms.  But this does not matter, ld.so can always use the local
   copy.  */
extern void *__libc_stack_end
#ifndef LIBC_STACK_END_NOT_RELRO
     attribute_relro
#endif
     ;
rtld_hidden_proto (__libc_stack_end)

/* Parameters passed to the dynamic linker.  */
extern int _dl_argc attribute_hidden attribute_relro;
extern char **_dl_argv
#ifndef DL_ARGV_NOT_RELRO
     attribute_relro
#endif
     ;
rtld_hidden_proto (_dl_argv)
#if IS_IN (rtld)
extern unsigned int _dl_skip_args attribute_hidden
# ifndef DL_ARGV_NOT_RELRO
     attribute_relro
# endif
     ;
extern unsigned int _dl_skip_args_internal attribute_hidden
# ifndef DL_ARGV_NOT_RELRO
     attribute_relro
# endif
     ;
#endif
#define rtld_progname _dl_argv[0]

/* Flag set at startup and cleared when the last initializer has run.  */
extern int _dl_starting_up;
weak_extern (_dl_starting_up)
rtld_hidden_proto (_dl_starting_up)

/* Random data provided by the kernel.  */
extern void *_dl_random attribute_hidden attribute_relro;

/* Write message on the debug file descriptor.  The parameters are
   interpreted as for a `printf' call.  All the lines start with a
   tag showing the PID.  */
extern void _dl_debug_printf (const char *fmt, ...)
     __attribute__ ((__format__ (__printf__, 1, 2))) attribute_hidden;

/* Write message on the debug file descriptor.  The parameters are
   interpreted as for a `printf' call.  All the lines buf the first
   start with a tag showing the PID.  */
extern void _dl_debug_printf_c (const char *fmt, ...)
     __attribute__ ((__format__ (__printf__, 1, 2))) attribute_hidden;


/* Write a message on the specified descriptor FD.  The parameters are
   interpreted as for a `printf' call.  */
#if IS_IN (rtld) || !defined (SHARED)
extern void _dl_dprintf (int fd, const char *fmt, ...)
     __attribute__ ((__format__ (__printf__, 2, 3)))
     attribute_hidden;
#else
__attribute__ ((always_inline, __format__ (__printf__, 2, 3)))
static inline void
_dl_dprintf (int fd, const char *fmt, ...)
{
  /* Use local declaration to avoid includign <stdio.h>.  */
  extern int __dprintf(int fd, const char *format, ...) attribute_hidden;
#if defined(__clang__)
  va_list va;
  va_start(va, fmt);
  __dprintf (fd, fmt, va);
  va_end(va);
#else
  __dprintf (fd, fmt, __builtin_va_arg_pack ());
#endif
}
#endif

/* Write LENGTH bytes at BUFFER to FD, like write.  Returns the number
   of bytes written on success, or a negative error constant on
   failure.  */
ssize_t _dl_write (int fd, const void *buffer, size_t length)
  attribute_hidden;

/* Write a message on the specified descriptor standard output.  The
   parameters are interpreted as for a `printf' call.  */
void _dl_printf (const char *fmt, ...)
  attribute_hidden __attribute__ ((__format__ (__printf__, 1, 2)));

/* Write a message on the specified descriptor standard error.  The
   parameters are interpreted as for a `printf' call.  */
void _dl_error_printf (const char *fmt, ...)
  attribute_hidden __attribute__ ((__format__ (__printf__, 1, 2)));

/* Write a message on the specified descriptor standard error and exit
   the program.  The parameters are interpreted as for a `printf' call.  */
void _dl_fatal_printf (const char *fmt, ...)
  __attribute__ ((__format__ (__printf__, 1, 2), __noreturn__));
rtld_hidden_proto (_dl_fatal_printf)

/* An exception raised by the _dl_signal_error function family and
   caught by _dl_catch_error function family.  Exceptions themselves
   are copied as part of the raise operation, but the strings are
   not.  */
struct dl_exception
{
  const char *objname;
  const char *errstring;

  /* This buffer typically stores both objname and errstring
     above.  */
  char *message_buffer;
};

/* Creates a new exception.  This calls malloc; if allocation fails,
   dummy values are inserted.  OBJECT is the name of the problematical
   shared object, or null if its a general problem.  ERRSTRING is a
   string describing the specific problem.  */
void _dl_exception_create (struct dl_exception *, const char *object,
			   const char *errstring)
  __attribute__ ((nonnull (1, 3)));
rtld_hidden_proto (_dl_exception_create)

/* Used internally to implement dlerror message freeing.  See
   include/dlfcn.h and dlfcn/dlerror.c.  */
void _dl_error_free (void *ptr) attribute_hidden;

/* Like _dl_exception_create, but create errstring from a format
   string FMT.  Currently, only "%s" and "%%" are supported as format
   directives.  */
void _dl_exception_create_format (struct dl_exception *, const char *objname,
				  const char *fmt, ...)
  __attribute__ ((nonnull (1, 3), format (printf, 3, 4)));
rtld_hidden_proto (_dl_exception_create_format)

/* Deallocate the exception, freeing allocated buffers (if
   possible).  */
void _dl_exception_free (struct dl_exception *)
  __attribute__ ((nonnull (1)));
rtld_hidden_proto (_dl_exception_free)

/* This function is called by all the internal dynamic linker
   functions when they encounter an error.  ERRCODE is either an
   `errno' code or zero; it specifies the return value of
   _dl_catch_error.  OCCASION is included in the error message if the
   process is terminated immediately.  */
void _dl_signal_exception (int errcode, struct dl_exception *,
			   const char *occasion)
  __attribute__ ((__noreturn__));
libc_hidden_proto (_dl_signal_exception)

/* Like _dl_signal_exception, but creates the exception first.  */
extern void _dl_signal_error (int errcode, const char *object,
			      const char *occasion, const char *errstring)
     __attribute__ ((__noreturn__));
libc_hidden_proto (_dl_signal_error)

/* Like _dl_signal_exception, but may return when called in the
   context of _dl_receive_error.  This is only used during ld.so
   bootstrap.  In static and profiled builds, this is equivalent to
   _dl_signal_exception.  */
#if IS_IN (rtld)
extern void _dl_signal_cexception (int errcode, struct dl_exception *,
				   const char *occasion) attribute_hidden;
#else
__attribute__ ((always_inline))
static inline void
_dl_signal_cexception (int errcode, struct dl_exception *exception,
		       const char *occasion)
{
  _dl_signal_exception (errcode, exception, occasion);
}
#endif

/* See _dl_signal_cexception above.  */
#if IS_IN (rtld)
extern void _dl_signal_cerror (int errcode, const char *object,
			       const char *occasion, const char *errstring)
     attribute_hidden;
#else
__attribute__ ((always_inline))
static inline void
_dl_signal_cerror (int errcode, const char *object,
			       const char *occasion, const char *errstring)
{
  _dl_signal_error (errcode, object, occasion, errstring);
}
#endif

/* Call OPERATE, receiving errors from `dl_signal_cerror'.  Unlike
   `_dl_catch_error' the operation is resumed after the OPERATE
   function returns.
   ARGS is passed as argument to OPERATE.  */
extern void _dl_receive_error (receiver_fct fct, void (*operate) (void *),
			       void *args) attribute_hidden;

/* Call OPERATE, catching errors from `_dl_signal_error' and related
   functions.  If there is no error, *ERRSTRING is set to null.  If
   there is an error, *ERRSTRING is set to a string constructed from
   the strings passed to _dl_signal_error, and the error code passed
   is the return value and *OBJNAME is set to the object name which
   experienced the problems.  ERRSTRING if nonzero points to a
   malloc'ed string which the caller has to free after use.  ARGS is
   passed as argument to OPERATE.  MALLOCEDP is set to true only if
   the returned string is allocated using the libc's malloc.  */
extern int _dl_catch_error (const char **objname, const char **errstring,
			    bool *mallocedp, void (*operate) (void *),
			    void *args);
libc_hidden_proto (_dl_catch_error)

/* Used for initializing GLRO (_dl_catch_error).  */
extern __typeof__ (_dl_catch_error) _rtld_catch_error attribute_hidden;

/* Call OPERATE (ARGS).  If no error occurs, set *EXCEPTION to zero.
   Otherwise, store a copy of the raised exception in *EXCEPTION,
   which has to be freed by _dl_exception_free.  As a special case, if
   EXCEPTION is null, call OPERATE (ARGS) with exception handling
   disabled (so that exceptions are fatal).  */
int _dl_catch_exception (struct dl_exception *exception,
			 void (*operate) (void *), void *args);
libc_hidden_proto (_dl_catch_exception)

/* Open the shared object NAME and map in its segments.
   LOADER's DT_RPATH is used in searching for NAME.
   If the object is already opened, returns its existing map.  */
extern struct link_map *_dl_map_object (struct link_map *loader,
					const char *name,
					int type, int trace_mode, int mode,
					Lmid_t nsid) attribute_hidden;

/* Call _dl_map_object on the dependencies of MAP, and set up
   MAP->l_searchlist.  PRELOADS points to a vector of NPRELOADS previously
   loaded objects that will be inserted into MAP->l_searchlist after MAP
   but before its dependencies.  */
extern void _dl_map_object_deps (struct link_map *map,
				 struct link_map **preloads,
				 unsigned int npreloads, int trace_mode,
				 int open_mode)
     attribute_hidden;

/* Cache the locations of MAP's hash table.  */
extern void _dl_setup_hash (struct link_map *map) attribute_hidden;


/* Collect the directories in the search path for LOADER's dependencies.
   The data structure is defined in <dlfcn.h>.  If COUNTING is true,
   SI->dls_cnt and SI->dls_size are set; if false, those must be as set
   by a previous call with COUNTING set, and SI must point to SI->dls_size
   bytes to be used in filling in the result.  */
extern void _dl_rtld_di_serinfo (struct link_map *loader,
				 Dl_serinfo *si, bool counting);

/* Process PT_GNU_PROPERTY program header PH in module L after
   PT_LOAD segments are mapped from file FD.  */
void _dl_process_pt_gnu_property (struct link_map *l, int fd,
				  const ElfW(Phdr) *ph);


/* Search loaded objects' symbol tables for a definition of the symbol
   referred to by UNDEF.  *SYM is the symbol table entry containing the
   reference; it is replaced with the defining symbol, and the base load
   address of the defining object is returned.  SYMBOL_SCOPE is a
   null-terminated list of object scopes to search; each object's
   l_searchlist (i.e. the segment of the dependency tree starting at that
   object) is searched in turn.  REFERENCE_NAME should name the object
   containing the reference; it is used in error messages.
   TYPE_CLASS describes the type of symbol we are looking for.  */
enum
  {
    /* If necessary add dependency between user and provider object.  */
    DL_LOOKUP_ADD_DEPENDENCY = 1,
    /* Return most recent version instead of default version for
       unversioned lookup.  */
    DL_LOOKUP_RETURN_NEWEST = 2,
    /* Set if dl_lookup* called with GSCOPE lock held.  */
    DL_LOOKUP_GSCOPE_LOCK = 4,
    /* Set if dl_lookup is called for non-lazy relocation processing
       from _dl_relocate_object in elf/dl-reloc.c.  */
    DL_LOOKUP_FOR_RELOCATE = 8,
  };

/* Lookup versioned symbol.  */
extern lookup_t _dl_lookup_symbol_x (const char *undef,
				     struct link_map *undef_map,
				     const ElfW(Sym) **sym,
				     struct r_scope_elem *symbol_scope[],
				     const struct r_found_version *version,
				     int type_class, int flags,
				     struct link_map *skip_map)
     attribute_hidden;


/* Restricted version of _dl_lookup_symbol_x.  Searches MAP (and only
   MAP) for the symbol UNDEF_NAME, with GNU hash NEW_HASH (computed
   with dl_new_hash), symbol version VERSION, and symbol version hash
   VERSION_HASH (computed with _dl_elf_hash).  Returns a pointer to
   the symbol table entry in MAP on success, or NULL on failure.  MAP
   must have symbol versioning information, or otherwise the result is
   undefined.  */
const ElfW(Sym) *_dl_lookup_direct (struct link_map *map,
				    const char *undef_name,
				    uint32_t new_hash,
				    const char *version,
				    uint32_t version_hash) attribute_hidden;

/* Add the new link_map NEW to the end of the namespace list.  */
extern void _dl_add_to_namespace_list (struct link_map *new, Lmid_t nsid)
     attribute_hidden;

/* Allocate a `struct link_map' for a new object being loaded.  */
extern struct link_map *_dl_new_object (char *realname, const char *libname,
					int type, struct link_map *loader,
					int mode, Lmid_t nsid)
     attribute_hidden;

/* Relocate the given object (if it hasn't already been).
   SCOPE is passed to _dl_lookup_symbol in symbol lookups.
   If RTLD_LAZY is set in RELOC-MODE, don't relocate its PLT.  */
extern void _dl_relocate_object (struct link_map *map,
				 struct r_scope_elem *scope[],
				 int reloc_mode, int consider_profiling)
     attribute_hidden;

/* Protect PT_GNU_RELRO area.  */
extern void _dl_protect_relro (struct link_map *map) attribute_hidden;

/* Call _dl_signal_error with a message about an unhandled reloc type.
   TYPE is the result of ELFW(R_TYPE) (r_info), i.e. an R_<CPU>_* value.
   PLT is nonzero if this was a PLT reloc; it just affects the message.  */
extern void _dl_reloc_bad_type (struct link_map *map,
				unsigned int type, int plt)
     attribute_hidden __attribute__ ((__noreturn__));

/* Resolve conflicts if prelinking.  */
extern void _dl_resolve_conflicts (struct link_map *l,
				   ElfW(Rela) *conflict,
				   ElfW(Rela) *conflictend)
     attribute_hidden;

/* Check the version dependencies of all objects available through
   MAP.  If VERBOSE print some more diagnostics.  */
extern int _dl_check_all_versions (struct link_map *map, int verbose,
				   int trace_mode) attribute_hidden;

/* Check the version dependencies for MAP.  If VERBOSE print some more
   diagnostics.  */
extern int _dl_check_map_versions (struct link_map *map, int verbose,
				   int trace_mode) attribute_hidden;

/* Initialize the object in SCOPE by calling the constructors with
   ARGC, ARGV, and ENV as the parameters.  */
extern void _dl_init (struct link_map *main_map, int argc, char **argv,
		      char **env) attribute_hidden;

/* Call the finalizer functions of all shared objects whose
   initializer functions have completed.  */
extern void _dl_fini (void) attribute_hidden;

/* Sort array MAPS according to dependencies of the contained objects.  */
extern void _dl_sort_maps (struct link_map **maps, unsigned int nmaps,
			   char *used, bool for_fini) attribute_hidden;

/* The dynamic linker calls this function before and having changing
   any shared object mappings.  The `r_state' member of `struct r_debug'
   says what change is taking place.  This function's address is
   the value of the `r_brk' member.  */
extern void _dl_debug_state (void);
rtld_hidden_proto (_dl_debug_state)

/* Initialize `struct r_debug' if it has not already been done.  The
   argument is the run-time load address of the dynamic linker, to be put
   in the `r_ldbase' member.  Returns the address of the structure.  */
extern struct r_debug *_dl_debug_initialize (ElfW(Addr) ldbase, Lmid_t ns)
     attribute_hidden;

/* Initialize the basic data structure for the search paths.  SOURCE
   is either "LD_LIBRARY_PATH" or "--library-path".
   GLIBC_HWCAPS_PREPEND adds additional glibc-hwcaps subdirectories to
   search.  GLIBC_HWCAPS_MASK is used to filter the built-in
   subdirectories if not NULL.  */
extern void _dl_init_paths (const char *library_path, const char *source,
			    const char *glibc_hwcaps_prepend,
			    const char *glibc_hwcaps_mask)
  attribute_hidden;

/* Gather the information needed to install the profiling tables and start
   the timers.  */
extern void _dl_start_profile (void) attribute_hidden;

/* The actual functions used to keep book on the calls.  */
extern void _dl_mcount (ElfW(Addr) frompc, ElfW(Addr) selfpc);
rtld_hidden_proto (_dl_mcount)

/* This function is simply a wrapper around the _dl_mcount function
   which does not require a FROMPC parameter since this is the
   calling function.  */
extern void _dl_mcount_wrapper (void *selfpc);

/* Show the members of the auxiliary array passed up from the kernel.  */
extern void _dl_show_auxv (void) attribute_hidden;

/* Return all environment variables starting with `LD_', one after the
   other.  */
extern char *_dl_next_ld_env_entry (char ***position) attribute_hidden;

/* Return an array with the names of the important hardware
   capabilities.  PREPEND is a colon-separated list of glibc-hwcaps
   directories to search first.  MASK is a colon-separated list used
   to filter the built-in glibc-hwcaps subdirectories.  The length of
   the array is written to *SZ, and the maximum of all strings length
   is written to *MAX_CAPSTRLEN.  */
const struct r_strlenpair *_dl_important_hwcaps (const char *prepend,
						 const char *mask,
						 size_t *sz,
						 size_t *max_capstrlen)
  attribute_hidden;

/* Look up NAME in ld.so.cache and return the file name stored there,
   or null if none is found.  Caller must free returned string.  */
extern char *_dl_load_cache_lookup (const char *name) attribute_hidden;

/* If the system does not support MAP_COPY we cannot leave the file open
   all the time since this would create problems when the file is replaced.
   Therefore we provide this function to close the file and open it again
   once needed.  */
extern void _dl_unload_cache (void) attribute_hidden;

/* System-dependent function to read a file's whole contents in the
   most convenient manner available.  *SIZEP gets the size of the
   file.  On error MAP_FAILED is returned.  */
extern void *_dl_sysdep_read_whole_file (const char *file, size_t *sizep,
					 int prot) attribute_hidden;

/* System-specific function to do initial startup for the dynamic linker.
   After this, file access calls and getenv must work.  This is responsible
   for setting __libc_enable_secure if we need to be secure (e.g. setuid),
   and for setting _dl_argc and _dl_argv, and then calling _dl_main.  */
extern ElfW(Addr) _dl_sysdep_start (void **start_argptr,
				    void (*dl_main) (const ElfW(Phdr) *phdr,
						     ElfW(Word) phnum,
						     ElfW(Addr) *user_entry,
						     ElfW(auxv_t) *auxv))
     attribute_hidden;

extern void _dl_sysdep_start_cleanup (void) attribute_hidden;


/* Determine next available module ID and set the L l_tls_modid.  */
extern void _dl_assign_tls_modid (struct link_map *l) attribute_hidden;

/* Count the modules with TLS segments.  */
extern size_t _dl_count_modids (void) attribute_hidden;

/* Calculate offset of the TLS blocks in the static TLS block.  */
extern void _dl_determine_tlsoffset (void) attribute_hidden;

/* Calculate the size of the static TLS surplus, when the given
   number of audit modules are loaded.  */
void _dl_tls_static_surplus_init (size_t naudit) attribute_hidden;

/* This function is called very early from dl_main to set up TLS and
   other thread-related data structures.  */
void __tls_pre_init_tp (void) attribute_hidden;

/* This function is called after processor-specific initialization of
   the TCB and thread pointer via TLS_INIT_TP, to complete very early
   initialization of the thread library.  */
void __tls_init_tp (void) attribute_hidden;

#ifndef SHARED
/* Set up the TCB for statically linked applications.  This is called
   early during startup because we always use TLS (for errno and the
   stack protector, among other things).  */
void __libc_setup_tls (void);

# if ENABLE_STATIC_PIE
/* Relocate static executable with PIE.  */
extern void _dl_relocate_static_pie (void) attribute_hidden;

/* Get a pointer to _dl_main_map.  */
extern struct link_map * _dl_get_dl_main_map (void)
  __attribute__ ((visibility ("hidden")));
# else
#  define _dl_relocate_static_pie()
# endif
#endif

/* Initialization of libpthread for statically linked applications.
   If libpthread is not linked in, this is an empty function.  */
void __pthread_initialize_minimal (void) weak_function;

/* Allocate memory for static TLS block (unless MEM is nonzero) and dtv.  */
extern void *_dl_allocate_tls (void *mem);
rtld_hidden_proto (_dl_allocate_tls)

/* Get size and alignment requirements of the static TLS block.  */
extern void _dl_get_tls_static_info (size_t *sizep, size_t *alignp);

extern void _dl_allocate_static_tls (struct link_map *map) attribute_hidden;

/* These are internal entry points to the two halves of _dl_allocate_tls,
   only used within rtld.c itself at startup time.  */
extern void *_dl_allocate_tls_storage (void) attribute_hidden;
extern void *_dl_allocate_tls_init (void *);
rtld_hidden_proto (_dl_allocate_tls_init)

/* Deallocate memory allocated with _dl_allocate_tls.  */
extern void _dl_deallocate_tls (void *tcb, bool dealloc_tcb);
rtld_hidden_proto (_dl_deallocate_tls)

extern void _dl_nothread_init_static_tls (struct link_map *) attribute_hidden;

/* Find origin of the executable.  */
extern const char *_dl_get_origin (void) attribute_hidden;

/* Count DSTs.  */
extern size_t _dl_dst_count (const char *name) attribute_hidden;

/* Substitute DST values.  */
extern char *_dl_dst_substitute (struct link_map *l, const char *name,
				 char *result) attribute_hidden;

/* Open the shared object NAME, relocate it, and run its initializer if it
   hasn't already been run.  MODE is as for `dlopen' (see <dlfcn.h>).  If
   the object is already opened, returns its existing map.  */
extern void *_dl_open (const char *name, int mode, const void *caller,
		       Lmid_t nsid, int argc, char *argv[], char *env[])
     attribute_hidden;

/* Free or queue for freeing scope OLD.  If other threads might be
   in the middle of _dl_fixup, _dl_profile_fixup or dl*sym using the
   old scope, OLD can't be freed until no thread is using it.  */
extern int _dl_scope_free (void *) attribute_hidden;


/* Add module to slot information data.  If DO_ADD is false, only the
   required memory is allocated.  Must be called with GL
   (dl_load_lock) acquired.  If the function has already been called
   for the link map L with !do_add, then this function will not raise
   an exception, otherwise it is possible that it encounters a memory
   allocation failure.  */
extern void _dl_add_to_slotinfo (struct link_map *l, bool do_add)
  attribute_hidden;

/* Update slot information data for at least the generation of the
   module with the given index.  */
extern struct link_map *_dl_update_slotinfo (unsigned long int req_modid)
     attribute_hidden;

/* Look up the module's TLS block as for __tls_get_addr,
   but never touch anything.  Return null if it's not allocated yet.  */
extern void *_dl_tls_get_addr_soft (struct link_map *l, struct pthread *)
     attribute_hidden;

extern int _dl_addr_inside_object (struct link_map *l, const ElfW(Addr) addr)
     attribute_hidden;

/* Show show of an object.  */
extern void _dl_show_scope (struct link_map *new, int from)
     attribute_hidden;

extern struct link_map *_dl_find_dso_for_object (const ElfW(Addr) addr);
rtld_hidden_proto (_dl_find_dso_for_object)

/* Initialization which is normally done by the dynamic linker.  */
extern void _dl_non_dynamic_init (void)
     attribute_hidden;

/* Used by static binaries to check the auxiliary vector.  */
extern void _dl_aux_init (ElfW(auxv_t) *av)
     attribute_hidden;

/* Initialize the static TLS space for the link map in all existing
   threads. */
#if THREAD_GSCOPE_IN_TCB
void _dl_init_static_tls (struct link_map *map) attribute_hidden;
#endif
static inline void
dl_init_static_tls (struct link_map *map)
{
#if THREAD_GSCOPE_IN_TCB
  /* The stack list is available to ld.so, so the initialization can
     be handled within ld.so directly.  */
  _dl_init_static_tls (map);
#else
  GL (dl_init_static_tls) (map);
#endif
}

#ifndef SHARED
/* Called before relocating ld.so during static dlopen.  This can be
   used to partly initialize the dormant ld.so copy in the static
   dlopen namespace.  */
void __rtld_static_init (struct link_map *map) attribute_hidden;
#endif

/* Return true if the ld.so copy in this namespace is actually active
   and working.  If false, the dl_open/dlfcn hooks have to be used to
   call into the outer dynamic linker (which happens after static
   dlopen).  */
#ifdef SHARED
static inline bool
rtld_active (void)
{
  /* The default-initialized variable does not have a non-zero
     dl_init_all_dirs member, so this allows us to recognize an
     initialized and active ld.so copy.  */
  return GLRO(dl_init_all_dirs) != NULL;
}

static inline struct auditstate *
link_map_audit_state (struct link_map *l, size_t index)
{
  if (l == &GL (dl_rtld_map))
    /* The auditstate array is stored separately.  */
    return &GL (dl_rtld_auditstate) [index];
  else
    {
      /* The auditstate array follows the link map in memory.  */
      struct auditstate *base = (struct auditstate *) (l + 1);
      return &base[index];
    }
}
#endif /* SHARED */

#if PTHREAD_IN_LIBC && defined SHARED
/* Recursive locking implementation for use within the dynamic loader.
   Used to define the __rtld_lock_lock_recursive and
   __rtld_lock_unlock_recursive via <libc-lock.h>.  Initialized to a
   no-op dummy implementation early.  Similar
   to GL (dl_rtld_lock_recursive) and GL (dl_rtld_unlock_recursive)
   in !PTHREAD_IN_LIBC builds.  */
extern int (*___rtld_mutex_lock) (pthread_mutex_t *) attribute_hidden;
extern int (*___rtld_mutex_unlock) (pthread_mutex_t *lock) attribute_hidden;

/* Called after libc has been loaded, but before RELRO is activated.
   Used to initialize the function pointers to the actual
   implementations.  */
void __rtld_mutex_init (void) attribute_hidden;
#else /* !PTHREAD_IN_LIBC */
static inline void
__rtld_mutex_init (void)
{
  /* The initialization happens later (!PTHREAD_IN_LIBC) or is not
     needed at all (!SHARED).  */
}
#endif /* !PTHREAD_IN_LIBC */

#if THREAD_GSCOPE_IN_TCB
void __thread_gscope_wait (void) attribute_hidden;
# define THREAD_GSCOPE_WAIT() __thread_gscope_wait ()
#endif

__END_DECLS

#endif /* ldsodefs.h */
