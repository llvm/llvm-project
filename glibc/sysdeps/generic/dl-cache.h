/* Support for reading /etc/ld.so.cache files written by Linux ldconfig.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
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

#ifndef _DL_CACHE_H
#define _DL_CACHE_H

#include <endian.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifndef _DL_CACHE_DEFAULT_ID
# define _DL_CACHE_DEFAULT_ID	3
#endif

#ifndef _dl_cache_check_flags
# define _dl_cache_check_flags(flags)			\
  ((flags) == 1 || (flags) == _DL_CACHE_DEFAULT_ID)
#endif

#ifndef LD_SO_CACHE
# define LD_SO_CACHE SYSCONFDIR "/ld.so.cache"
#endif

#ifndef add_system_dir
# define add_system_dir(dir) add_dir (dir)
#endif

#define CACHEMAGIC "ld.so-1.7.0"

/* libc5 and glibc 2.0/2.1 use the same format.  For glibc 2.2 another
   format has been added in a compatible way:
   The beginning of the string table is used for the new table:
	old_magic
	nlibs
	libs[0]
	...
	libs[nlibs-1]
	pad, new magic needs to be aligned
	     - this is string[0] for the old format
	new magic - this is string[0] for the new format
	newnlibs
	...
	newlibs[0]
	...
	newlibs[newnlibs-1]
	string 1
	string 2
	...
*/
struct file_entry
{
  int32_t flags;		/* This is 1 for an ELF library.  */
  uint32_t key, value;		/* String table indices.  */
};

struct cache_file
{
  char magic[sizeof CACHEMAGIC - 1];
  unsigned int nlibs;
  struct file_entry libs[0];
};

#define CACHEMAGIC_NEW "glibc-ld.so.cache"
#define CACHE_VERSION "1.1"
#define CACHEMAGIC_VERSION_NEW CACHEMAGIC_NEW CACHE_VERSION


struct file_entry_new
{
  union
  {
    /* Fields shared with struct file_entry.  */
    struct file_entry entry;
    /* Also expose these fields directly.  */
    struct
    {
      int32_t flags;		/* This is 1 for an ELF library.  */
      uint32_t key, value;	/* String table indices.  */
    };
  };
  uint32_t osversion;		/* Required OS version.	 */
  uint64_t hwcap;		/* Hwcap entry.	 */
};

/* This bit in the hwcap field of struct file_entry_new indicates that
   the lower 32 bits contain an index into the
   cache_extension_tag_glibc_hwcaps section.  Older glibc versions do
   not know about this HWCAP bit, so they will ignore these
   entries.  */
#define DL_CACHE_HWCAP_EXTENSION (1ULL << 62)

/* The number of the ISA level bits in the upper 32 bits of the hwcap
   field.  */
#define DL_CACHE_HWCAP_ISA_LEVEL_COUNT 10

/* The mask of the ISA level bits in the hwcap field.  */
#define DL_CACHE_HWCAP_ISA_LEVEL_MASK \
  ((1 << DL_CACHE_HWCAP_ISA_LEVEL_COUNT) -1)

/* Return true if the ENTRY->hwcap value indicates that
   DL_CACHE_HWCAP_EXTENSION is used.  */
static inline bool
dl_cache_hwcap_extension (struct file_entry_new *entry)
{
  /* This is an hwcap extension if only the DL_CACHE_HWCAP_EXTENSION bit
     is set, ignoring the lower 32 bits as well as the ISA level bits in
     the upper 32 bits.  */
  return (((entry->hwcap >> 32) & ~DL_CACHE_HWCAP_ISA_LEVEL_MASK)
	  == (DL_CACHE_HWCAP_EXTENSION >> 32));
}

/* See flags member of struct cache_file_new below.  */
enum
  {
    /* No endianness information available.  An old ldconfig version
       without endianness support wrote the file.  */
    cache_file_new_flags_endian_unset = 0,

    /* Cache is invalid and should be ignored.  */
    cache_file_new_flags_endian_invalid = 1,

    /* Cache format is little endian.  */
    cache_file_new_flags_endian_little = 2,

    /* Cache format is big endian.  */
    cache_file_new_flags_endian_big = 3,

    /* Bit mask to extract the cache_file_new_flags_endian_*
       values.  */
    cache_file_new_flags_endian_mask = 3,

    /* Expected value of the endian bits in the flags member for the
       current architecture.  */
    cache_file_new_flags_endian_current
      = (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
	 ? cache_file_new_flags_endian_little
	 : cache_file_new_flags_endian_big),
  };

struct cache_file_new
{
  char magic[sizeof CACHEMAGIC_NEW - 1];
  char version[sizeof CACHE_VERSION - 1];
  uint32_t nlibs;		/* Number of entries.  */
  uint32_t len_strings;		/* Size of string table. */

  /* flags & cache_file_new_flags_endian_mask is one of the values
     cache_file_new_flags_endian_unset, cache_file_new_flags_endian_invalid,
     cache_file_new_flags_endian_little, cache_file_new_flags_endian_big.

     The remaining bits are unused and should be generated as zero and
     ignored by readers.  */
  uint8_t flags;

  uint8_t padding_unsed[3];	/* Not used, for future extensions.  */

  /* File offset of the extension directory.  See struct
     cache_extension below.  Must be a multiple of four.  */
  uint32_t extension_offset;

  uint32_t unused[3];		/* Leave space for future extensions
				   and align to 8 byte boundary.  */
  struct file_entry_new libs[0]; /* Entries describing libraries.  */
  /* After this the string table of size len_strings is found.	*/
};
_Static_assert (sizeof (struct cache_file_new) == 48,
		"size of struct cache_file_new");

/* Returns false if *CACHE has the wrong endianness for this
   architecture, and true if the endianness matches (or is
   unknown).  */
static inline bool
cache_file_new_matches_endian (const struct cache_file_new *cache)
{
  /* A zero value for cache->flags means that no endianness
     information is available.  */
  return cache->flags == 0
    || ((cache->flags & cache_file_new_flags_endian_big)
	== cache_file_new_flags_endian_current);
}


/* Randomly chosen magic value, which allows for additional
   consistency verification.  */
enum { cache_extension_magic = (uint32_t) -358342284 };

/* Tag values for different kinds of extension sections.  Similar to
   SHT_* constants.  */
enum cache_extension_tag
  {
   /* Array of bytes containing the glibc version that generated this
      cache file.  */
   cache_extension_tag_generator,

   /* glibc-hwcaps subdirectory information.  An array of uint32_t
      values, which are indices into the string table.  The strings
      are sorted lexicographically (according to strcmp).  The extra
      level of indirection (instead of using string table indices
      directly) allows the dynamic loader to compute the preference
      order of the hwcaps names more efficiently.

      For this section, 4-byte alignment is required, and the section
      size must be a multiple of 4.  */
   cache_extension_tag_glibc_hwcaps,

   /* Total number of known cache extension tags.  */
   cache_extension_count
  };

/* Element in the array following struct cache_extension.  Similar to
   an ELF section header.  */
struct cache_extension_section
{
  /* Type of the extension section.  A enum cache_extension_tag value.  */
  uint32_t tag;

  /* Extension-specific flags.  Currently generated as zero.  */
  uint32_t flags;

  /* Offset from the start of the file for the data in this extension
     section.  Specific extensions can have alignment constraints.  */
  uint32_t offset;

  /* Length in bytes of the extension data.  Specific extensions may
     have size requirements.  */
  uint32_t size;
};

/* The extension directory in the cache.  An array of struct
   cache_extension_section entries.  */
struct cache_extension
{
  uint32_t magic;		/* Always cache_extension_magic.  */
  uint32_t count;		/* Number of following entries.  */

  /* count section descriptors of type struct cache_extension_section
     follow.  */
  struct cache_extension_section sections[];
};

/* A relocated version of struct cache_extension_section.  */
struct cache_extension_loaded
{
  /* Address and size of this extension section.  base is NULL if the
     section is missing from the file.  */
  const void *base;
  size_t size;

  /* Flags from struct cache_extension_section.  */
  uint32_t flags;
};

/* All supported extension sections, relocated.  Filled in by
   cache_extension_load below.  */
struct cache_extension_all_loaded
{
  struct cache_extension_loaded sections[cache_extension_count];
};

/* Performs basic data validation based on section tag, and removes
   the sections which are invalid.  */
static void
cache_extension_verify (struct cache_extension_all_loaded *loaded)
{
  {
    /* Section must not be empty, it must be aligned at 4 bytes, and
       the size must be a multiple of 4.  */
    struct cache_extension_loaded *hwcaps
      = &loaded->sections[cache_extension_tag_glibc_hwcaps];
    if (hwcaps->size == 0
	|| ((uintptr_t) hwcaps->base % 4) != 0
	|| (hwcaps->size % 4) != 0)
      {
	hwcaps->base = NULL;
	hwcaps->size = 0;
	hwcaps->flags = 0;
      }
  }
}

static bool __attribute__ ((unused))
cache_extension_load (const struct cache_file_new *cache,
		      const void *file_base, size_t file_size,
		      struct cache_extension_all_loaded *loaded)
{
  memset (loaded, 0, sizeof (*loaded));
  if (cache->extension_offset == 0)
    /* No extensions present.  This is not a format error.  */
    return true;
  if ((cache->extension_offset % 4) != 0)
    /* Extension offset is misaligned.  */
    return false;
  size_t size_tmp;
  if (__builtin_add_overflow (cache->extension_offset,
			      sizeof (struct cache_extension), &size_tmp)
      || size_tmp > file_size)
    /* Extension extends beyond the end of the file.  */
    return false;
  const struct cache_extension *ext = file_base + cache->extension_offset;
  if (ext->magic != cache_extension_magic)
    return false;
  if (__builtin_mul_overflow (ext->count,
			      sizeof (struct cache_extension_section),
			      &size_tmp)
      || __builtin_add_overflow (cache->extension_offset
				 + sizeof (struct cache_extension), size_tmp,
				 &size_tmp)
      || size_tmp > file_size)
    /* Extension array extends beyond the end of the file.  */
    return false;
  for (uint32_t i = 0; i < ext->count; ++i)
    {
      if (__builtin_add_overflow (ext->sections[i].offset,
				  ext->sections[i].size, &size_tmp)
	  || size_tmp > file_size)
	/* Extension data extends beyond the end of the file.  */
	return false;

      uint32_t tag = ext->sections[i].tag;
      if (tag >= cache_extension_count)
	/* Tag is out of range and unrecognized.  */
	continue;
      loaded->sections[tag].base = file_base + ext->sections[i].offset;
      loaded->sections[tag].size = ext->sections[i].size;
      loaded->sections[tag].flags = ext->sections[i].flags;
    }
  cache_extension_verify (loaded);
  return true;
}

/* Used to align cache_file_new.  */
#define ALIGN_CACHE(addr)				\
(((addr) + __alignof__ (struct cache_file_new) -1)	\
 & (~(__alignof__ (struct cache_file_new) - 1)))

extern int _dl_cache_libcmp (const char *p1, const char *p2) attribute_hidden;

#endif /* _DL_CACHE_H */
