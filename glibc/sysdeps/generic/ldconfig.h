/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Jaeger <aj@suse.de>, 1999.

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

#ifndef _LDCONFIG_H
#define _LDCONFIG_H

#include <stddef.h>
#include <stdint.h>
#include <sys/stat.h>

#define FLAG_ANY			-1
#define FLAG_TYPE_MASK			0x00ff
#define FLAG_LIBC4			0x0000
#define FLAG_ELF			0x0001
#define FLAG_ELF_LIBC5			0x0002
#define FLAG_ELF_LIBC6			0x0003
#define FLAG_REQUIRED_MASK		0xff00
#define FLAG_SPARC_LIB64		0x0100
#define FLAG_IA64_LIB64			0x0200
#define FLAG_X8664_LIB64		0x0300
#define FLAG_S390_LIB64			0x0400
#define FLAG_POWERPC_LIB64		0x0500
#define FLAG_MIPS64_LIBN32		0x0600
#define FLAG_MIPS64_LIBN64		0x0700
#define FLAG_X8664_LIBX32		0x0800
#define FLAG_ARM_LIBHF			0x0900
#define FLAG_AARCH64_LIB64		0x0a00
#define FLAG_ARM_LIBSF			0x0b00
#define FLAG_MIPS_LIB32_NAN2008		0x0c00
#define FLAG_MIPS64_LIBN32_NAN2008	0x0d00
#define FLAG_MIPS64_LIBN64_NAN2008	0x0e00
#define FLAG_RISCV_FLOAT_ABI_SOFT	0x0f00
#define FLAG_RISCV_FLOAT_ABI_DOUBLE	0x1000

/* Name of auxiliary cache.  */
#define _PATH_LDCONFIG_AUX_CACHE "/var/cache/ldconfig/aux-cache"

/* Declared in cache.c.  */
extern void print_cache (const char *cache_name);

extern void init_cache (void);

extern void save_cache (const char *cache_name);

struct glibc_hwcaps_subdirectory;

/* Return a struct describing the subdirectory for NAME.  Reuse an
   existing struct if it exists.  */
struct glibc_hwcaps_subdirectory *new_glibc_hwcaps_subdirectory
  (const char *name);

/* Returns the name that was specified when
   add_glibc_hwcaps_subdirectory was called.  */
const char *glibc_hwcaps_subdirectory_name
  (const struct glibc_hwcaps_subdirectory *);

extern void add_to_cache (const char *path, const char *filename,
			  const char *soname, int flags,
			  unsigned int osversion, unsigned int isa_level,
			  uint64_t hwcap,
			  struct glibc_hwcaps_subdirectory *);

extern void init_aux_cache (void);

extern void load_aux_cache (const char *aux_cache_name);

extern int search_aux_cache (struct stat64 *stat_buf, int *flags,
			     unsigned int *osversion,
			     unsigned int *isa_level, char **soname);

extern void add_to_aux_cache (struct stat64 *stat_buf, int flags,
			      unsigned int osversion,
			      unsigned int isa_level, const char *soname);

extern void save_aux_cache (const char *aux_cache_name);

/* Declared in readlib.c.  */
extern int process_file (const char *real_file_name, const char *file_name,
			 const char *lib, int *flag,
			 unsigned int *osversion, unsigned int *isa_level,
			 char **soname, int is_link,
			 struct stat64 *stat_buf);

extern char *implicit_soname (const char *lib, int flag);

/* Declared in readelflib.c.  */
extern int process_elf_file (const char *file_name, const char *lib,
			     int *flag, unsigned int *osversion,
			     unsigned int *isa_level, char **soname,
			     void *file_contents, size_t file_length);

/* Declared in chroot_canon.c.  */
extern char *chroot_canon (const char *chroot, const char *name);

/* Declared in ldconfig.c.  */
extern int opt_verbose;

enum opt_format
  {
    opt_format_old = 0,	/* Use struct cache_file.  */
    opt_format_compat = 1, /* Use both, old format followed by new.  */
    opt_format_new = 2,	/* Use struct cache_file_new.  */
  };

extern enum opt_format opt_format;

/* Prototypes for a few program-wide used functions.  */
#include <programs/xmalloc.h>

#endif /* ! _LDCONFIG_H  */
