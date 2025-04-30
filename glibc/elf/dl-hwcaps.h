/* Hardware capability support for run-time dynamic loader.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#ifndef _DL_HWCAPS_H
#define _DL_HWCAPS_H

#include <stdint.h>
#include <stddef.h>

#include <elf/dl-tunables.h>

#if HAVE_TUNABLES
# define GET_HWCAP_MASK() TUNABLE_GET (glibc, cpu, hwcap_mask, uint64_t, NULL)
#else
# ifdef SHARED
#   define GET_HWCAP_MASK() GLRO(dl_hwcap_mask)
# else
/* HWCAP_MASK is ignored in static binaries when built without tunables.  */
#  define GET_HWCAP_MASK() (0)
# endif
#endif

#define GLIBC_HWCAPS_SUBDIRECTORY "glibc-hwcaps"
#define GLIBC_HWCAPS_PREFIX GLIBC_HWCAPS_SUBDIRECTORY "/"

/* Used by _dl_hwcaps_split below, to split strings at ':'
   separators.  */
struct dl_hwcaps_split
{
  const char *segment;          /* Start of the current segment.  */
  size_t length;                /* Number of bytes until ':' or NUL.  */
};

/* Prepare *S to parse SUBJECT, for future _dl_hwcaps_split calls.  If
   SUBJECT is NULL, it is treated as the empty string.  */
static inline void
_dl_hwcaps_split_init (struct dl_hwcaps_split *s, const char *subject)
{
  s->segment = subject;
  /* The initial call to _dl_hwcaps_split will not skip anything.  */
  s->length = 0;
}

/* Extract the next non-empty string segment, up to ':' or the null
   terminator.  Return true if one more segment was found, or false if
   the end of the string was reached.  On success, S->segment is the
   start of the segment found, and S->length is its length.
   (Typically, S->segment[S->length] is not null.)  */
_Bool _dl_hwcaps_split (struct dl_hwcaps_split *s) attribute_hidden;

/* Similar to dl_hwcaps_split, but with bit-based and name-based
   masking.  */
struct dl_hwcaps_split_masked
{
  struct dl_hwcaps_split split;

  /* For used by the iterator implementation.  */
  const char *mask;
  uint32_t bitmask;
};

/* Prepare *S for iteration with _dl_hwcaps_split_masked.  Only HWCAP
   names in SUBJECT whose bit is set in BITMASK and whose name is in
   MASK will be returned.  SUBJECT must not contain empty HWCAP names.
   If MASK is NULL, no name-based masking is applied.  Likewise for
   BITMASK if BITMASK is -1 (infinite number of bits).  */
static inline void
_dl_hwcaps_split_masked_init (struct dl_hwcaps_split_masked *s,
                              const char *subject,
                              uint32_t bitmask, const char *mask)
{
  _dl_hwcaps_split_init (&s->split, subject);
  s->bitmask = bitmask;
  s->mask = mask;
}

/* Like _dl_hwcaps_split, but apply masking.  */
_Bool _dl_hwcaps_split_masked (struct dl_hwcaps_split_masked *s)
  attribute_hidden;

/* Returns true if the colon-separated HWCAP list HWCAPS contains the
   capability NAME (with length NAME_LENGTH).  If HWCAPS is NULL, the
   function returns true.  */
_Bool _dl_hwcaps_contains (const char *hwcaps, const char *name,
                           size_t name_length) attribute_hidden;

/* Colon-separated string of glibc-hwcaps subdirectories, without the
   "glibc-hwcaps/" prefix.  The most preferred subdirectory needs to
   be listed first.  Up to 32 subdirectories are supported, limited by
   the width of the uint32_t mask.  */
extern const char _dl_hwcaps_subdirs[] attribute_hidden;

/* Returns a bitmap of active subdirectories in _dl_hwcaps_subdirs.
   Bit 0 (the LSB) corresponds to the first substring in
   _dl_hwcaps_subdirs, bit 1 to the second substring, and so on.
   There is no direct correspondence between HWCAP bitmasks and this
   bitmask.  */
uint32_t _dl_hwcaps_subdirs_active (void) attribute_hidden;

/* Returns a bitmask that marks the last ACTIVE subdirectories in a
   _dl_hwcaps_subdirs_active string (containing SUBDIRS directories in
   total) as active.  Intended for use in _dl_hwcaps_subdirs_active
   implementations (if a contiguous tail of the list in
   _dl_hwcaps_subdirs is selected).  */
static inline uint32_t
_dl_hwcaps_subdirs_build_bitmask (int subdirs, int active)
{
  /* Leading subdirectories that are not active.  */
  int inactive = subdirs - active;
  if (inactive == 32)
    return 0;

  uint32_t mask;
  if (subdirs != 32)
    mask = (1U << subdirs) - 1;
  else
    mask = -1;
  return mask ^ ((1U << inactive) - 1);
}

/* Pre-computed glibc-hwcaps subdirectory priorities.  Used in
   dl-cache.c to quickly find the proprieties for the stored HWCAP
   names.  */
struct dl_hwcaps_priority
{
  /* The name consists of name_length bytes at name (not necessarily
     null-terminated).  */
  const char *name;
  uint32_t name_length;

  /* Priority of this name.  A positive number.  */
  uint32_t priority;
};

/* Pre-computed hwcaps priorities.  Set up by
   _dl_important_hwcaps.  */
extern struct dl_hwcaps_priority *_dl_hwcaps_priorities attribute_hidden;
extern uint32_t _dl_hwcaps_priorities_length attribute_hidden;

#endif /* _DL_HWCAPS_H */
