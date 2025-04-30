/* Common extra functions.
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

/* This header file should only contain definitions compatible with
   C90.  (Using __attribute__ is fine because <features.h> provides a
   fallback.)  */

#ifndef SUPPORT_H
#define SUPPORT_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <sys/cdefs.h>
/* For mode_t.  */
#include <sys/stat.h>
/* For ssize_t and off64_t.  */
#include <sys/types.h>
/* For locale_t.  */
#include <locale.h>

__BEGIN_DECLS

/* Write a message to standard output.  Can be used in signal
   handlers.  */
void write_message (const char *message) __attribute__ ((nonnull (1)));

/* Avoid all the buffer overflow messages on stderr.  */
void ignore_stderr (void);

/* Set fortification error handler.  Used when tests want to verify that bad
   code is caught by the library.  */
void set_fortify_handler (void (*handler) (int sig));

/* Report an out-of-memory error for the allocation of SIZE bytes in
   FUNCTION, terminating the process.  */
void oom_error (const char *function, size_t size)
  __attribute__ ((nonnull (1)));

/* Return a pointer to a memory region of SIZE bytes.  The memory is
   initialized to zero and will be shared with subprocesses (across
   fork).  The returned pointer must be freed using
   support_shared_free; it is not compatible with the malloc
   functions.  */
void *support_shared_allocate (size_t size);

/* Deallocate a pointer returned by support_shared_allocate.  */
void support_shared_free (void *);

/* Write CONTENTS to the file PATH.  Create or truncate the file as
   needed.  The file mode is 0666 masked by the umask.  Terminate the
   process on error.  */
void support_write_file_string (const char *path, const char *contents);

/* Quote the contents of the byte array starting at BLOB, of LENGTH
   bytes, in such a way that the result string can be included in a C
   literal (in single/double quotes, without putting the quotes into
   the result).  */
char *support_quote_blob (const void *blob, size_t length);

/* Quote the contents of the string, in such a way that the result
   string can be included in a C literal (in single/double quotes,
   without putting the quotes into the result).  */
char *support_quote_string (const char *);

/* Returns non-zero if the file descriptor is a regular file on a file
   system which supports holes (that is, seeking and writing does not
   allocate storage for the range of zeros).  FD must refer to a
   regular file open for writing, and initially empty.  */
int support_descriptor_supports_holes (int fd);

/* Error-checking wrapper functions which terminate the process on
   error.  */

extern void *xmalloc (size_t n)
  __attribute_malloc__ __attribute_alloc_size__ ((1)) __attr_dealloc_free
  __returns_nonnull;
extern void *xcalloc (size_t n, size_t s)
  __attribute_malloc__ __attribute_alloc_size__ ((1, 2)) __attr_dealloc_free
  __returns_nonnull;
extern void *xrealloc (void *o, size_t n)
  __attribute_malloc__ __attribute_alloc_size__ ((2)) __attr_dealloc_free;
extern char *xstrdup (const char *) __attribute_malloc__ __attr_dealloc_free
  __returns_nonnull;
void *xposix_memalign (size_t alignment, size_t n)
  __attribute_malloc__ __attribute_alloc_size__ ((2)) __attr_dealloc_free
  __returns_nonnull;
char *xasprintf (const char *format, ...)
  __attribute__ ((format (printf, 1, 2), malloc)) __attr_dealloc_free
  __returns_nonnull;
char *xstrdup (const char *) __attr_dealloc_free __returns_nonnull;
char *xstrndup (const char *, size_t) __attr_dealloc_free __returns_nonnull;
char *xsetlocale (int category, const char *locale);
locale_t xnewlocale (int category_mask, const char *locale, locale_t base);
char *xuselocale (locale_t newloc);

/* These point to the TOP of the source/build tree, not your (or
   support's) subdirectory.  */
extern const char support_srcdir_root[];
extern const char support_objdir_root[];

/* Corresponds to the path to the runtime linker used by the testsuite,
   e.g. OBJDIR_PATH/elf/ld-linux-x86-64.so.2  */
extern const char support_objdir_elf_ldso[];

/* Corresponds to the --prefix= passed to configure.  */
extern const char support_install_prefix[];
/* Corresponds to the install's lib/ or lib64/ directory.  */
extern const char support_libdir_prefix[];
/* Corresponds to the install's bin/ directory.  */
extern const char support_bindir_prefix[];
/* Corresponds to the install's sbin/ directory.  */
extern const char support_sbindir_prefix[];
/* Corresponds to the install's system /lib or /lib64 directory.  */
extern const char support_slibdir_prefix[];
/* Corresponds to the install's sbin/ directory (without prefix).  */
extern const char support_install_rootsbindir[];
/* Corresponds to the install's compiled locale directory.  */
extern const char support_complocaledir_prefix[];

/* Copies the file at the path FROM to TO.  If TO does not exist, it
   is created.  If TO is a regular file, it is truncated before
   copying.  The file mode is copied, but the permissions are not.  */
extern void support_copy_file (const char *from, const char *to);

extern ssize_t support_copy_file_range (int, off64_t *, int, off64_t *,
					size_t, unsigned int);

/* Return true if PATH supports 64-bit time_t interfaces for file
   operations (such as fstatat or utimensat).  */
extern bool support_path_support_time64_value (const char *path, int64_t at,
					       int64_t mt);
static __inline bool support_path_support_time64 (const char *path)
{
  /* 1s and 2s after y2038 limit.  */
  return support_path_support_time64_value (path, 0x80000001ULL,
					    0x80000002ULL);
}

/* Return true if stat supports nanoseconds resolution.  PATH is used
   for tests and its ctime may change.  */
extern bool support_stat_nanoseconds (const char *path);

/* Return true if select modify the timeout to reflect the amount of time
   no slept.  */
extern bool support_select_modifies_timeout (void);

/* Return true if select normalize the timeout input by taking in account
   tv_usec larger than 1000000.  */
extern bool support_select_normalizes_timeout (void);

/* Create a timer that trigger after SEC seconds and NSEC nanoseconds.  If
   REPEAT is true the timer will repeat indefinitely.  If CALLBACK is not
   NULL, the function will be called when the timer expires; otherwise a
   dummy empty function is used instead.
   This is implemented with POSIX per-process timer with SIGEV_SIGNAL.  */
timer_t support_create_timer (uint64_t sec, long int nsec, bool repeat,
			      void (*callback)(int));
/* Disable the timer TIMER.  */
void support_delete_timer (timer_t timer);

struct support_stack
{
  void *stack;
  size_t size;
  size_t guardsize;
};

/* Allocate stack suitable to used with xclone or sigaltstack call. The stack
   will have a minimum size of SIZE + MINSIGSTKSZ bytes, rounded up to a whole
   number of pages.  There will be a large (at least 1 MiB) inaccessible guard
   bands on either side of it.
   The returned value on ALLOC_BASE and ALLOC_SIZE will be the usable stack
   region, excluding the GUARD_SIZE allocated area.
   It also terminates the process on error.  */
struct support_stack support_stack_alloc (size_t size);

/* Deallocate the STACK.  */
void support_stack_free (struct support_stack *stack);

__END_DECLS

#endif /* SUPPORT_H */
