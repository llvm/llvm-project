#ifndef _DLFCN_H
#include <dlfcn/dlfcn.h>
#ifndef _ISOMAC
#include <link.h>		/* For ElfW.  */
#include <stdbool.h>

/* Internally used flag.  */
#define __RTLD_DLOPEN	0x80000000
#define __RTLD_SPROF	0x40000000
#define __RTLD_OPENEXEC	0x20000000
#define __RTLD_CALLMAP	0x10000000
#define __RTLD_AUDIT	0x08000000
#define __RTLD_SECURE	0x04000000 /* Apply additional security checks.  */
#define __RTLD_NOIFUNC	0x02000000 /* Suppress calling ifunc functions.  */

#define __LM_ID_CALLER	-2

/* These variables are defined and initialized in the startup code.  */
extern int __libc_argc attribute_hidden;
extern char **__libc_argv attribute_hidden;

/* Now define the internal interfaces.  */

/* Use RTLD_NOW here because:
   1. In pthread_cancel_init we want to use RTLD_NOW to reduce the stack usage
      of future cancellation operations, particularly when the target thread
      is running with a small stack.  Likewise for consistency we do the same
      thing in __libgcc_s_init.  RTLD_NOW will rarely make a difference for
      __libgcc_s_init because unwinding is already in progress, so libgcc_s.so
      has already been loaded if its unwinder is used (Bug 22636).
   2. It allows us to provide robust fallback code at dlopen time for
      incorrectly configured systems that mix old libnss_* modules
      with newly installed libraries e.g. old libnss_dns.so.2 with new
      libresolv.so.2.  Using RTLD_LAZY here causes a failure at the
      time the symbol is called and at that point it is much harder to
      safely return an error (Bug 22766).

   The use of RTLD_NOW also impacts gconv module loading, backtracing
   (where the unwinder form libgcc_s.so is used), and IDNA functions
   (which load libidn2), all of which load their respective DSOs on
   demand, and so should not impact program startup.  That is to say
   that the DSOs are loaded as part of an API call and therefore we
   will be calling that family of API functions shortly so RTLD_NOW or
   RTLD_LAZY is not a big difference in performance, but RTLD_NOW has
   better error handling semantics for the library.  */
#define __libc_dlopen(name) \
  __libc_dlopen_mode (name, RTLD_NOW | __RTLD_DLOPEN)
extern void *__libc_dlopen_mode  (const char *__name, int __mode)
  attribute_hidden;
extern void *__libc_dlsym   (void *__map, const char *__name)
  attribute_hidden;
extern void *__libc_dlvsym (void *map, const char *name, const char *version)
  attribute_hidden;
extern int   __libc_dlclose (void *__map)
  attribute_hidden;

/* Locate shared object containing the given address.  */
#ifdef ElfW
extern int _dl_addr (const void *address, Dl_info *info,
		     struct link_map **mapp, const ElfW(Sym) **symbolp)
  attribute_hidden;
#endif

struct link_map;

/* Close an object previously opened by _dl_open.  */
extern void _dl_close (void *map) attribute_hidden;
/* Same as above, but without locking and safety checks for user
   provided map arguments.  */
extern void _dl_close_worker (struct link_map *map, bool force)
    attribute_hidden;

/* Look up NAME in shared object HANDLE (which may be RTLD_DEFAULT or
   RTLD_NEXT).  WHO is the calling function, for RTLD_NEXT.  Returns
   the symbol value, which may be NULL.  */
extern void *_dl_sym (void *handle, const char *name, void *who)
  attribute_hidden;

/* Look up version VERSION of symbol NAME in shared object HANDLE
   (which may be RTLD_DEFAULT or RTLD_NEXT).  WHO is the calling
   function, for RTLD_NEXT.  Returns the symbol value, which may be
   NULL.  */
extern void *_dl_vsym (void *handle, const char *name, const char *version,
		       void *who) attribute_hidden;

/* Helper function for <dlfcn.h> functions.  Runs the OPERATE function via
   _dl_catch_error.  Returns zero for success, nonzero for failure; and
   arranges for `dlerror' to return the error details.
   ARGS is passed as argument to OPERATE.  */
extern int _dlerror_run (void (*operate) (void *), void *args) attribute_hidden;

/* This structure is used to make the outer (statically linked)
   implementation of dlopen and related functions to the inner libc
   after static dlopen, via the GLRO (dl_dlfcn_hook) pointer.  */
struct dlfcn_hook
{
  /* Public interfaces.  */
  void *(*dlopen) (const char *file, int mode, void *dl_caller);
  int (*dlclose) (void *handle);
  void *(*dlsym) (void *handle, const char *name, void *dl_caller);
  void *(*dlvsym) (void *handle, const char *name, const char *version,
		   void *dl_caller);
  char *(*dlerror) (void);
  int (*dladdr) (const void *address, Dl_info *info);
  int (*dladdr1) (const void *address, Dl_info *info,
		  void **extra_info, int flags);
  int (*dlinfo) (void *handle, int request, void *arg);
  void *(*dlmopen) (Lmid_t nsid, const char *file, int mode, void *dl_caller);

  /* Internal interfaces.  */
  void* (*libc_dlopen_mode)  (const char *__name, int __mode);
  void* (*libc_dlsym)  (void *map, const char *name);
  void* (*libc_dlvsym)  (void *map, const char *name, const char *version);
  int   (*libc_dlclose) (void *map);
};

/* Note: These prototypes are for initializing _dlfcn_hook in static
   builds; see __rtld_static_init.  Internal calls in glibc should use
   the __libc_dl* functions defined in elf/dl-libc.c instead.  */

extern void *__dlopen (const char *file, int mode, void *caller);
extern void *__dlmopen (Lmid_t nsid, const char *file, int mode,
			void *dl_caller);
extern int __dlclose (void *handle);
extern void *__dlsym (void *handle, const char *name, void *dl_caller);
extern void *__dlvsym (void *handle, const char *name, const char *version,
		       void *dl_caller);
extern int __dladdr (const void *address, Dl_info *info);
extern int __dladdr1 (const void *address, Dl_info *info,
		      void **extra_info, int flags);
extern int __dlinfo (void *handle, int request, void *arg);
extern char *__dlerror (void);

#endif
#endif
