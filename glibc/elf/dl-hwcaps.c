/* Hardware capability support for run-time dynamic loader.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

#include <assert.h>
#include <elf.h>
#include <errno.h>
#include <libintl.h>
#include <unistd.h>
#include <ldsodefs.h>

#include <dl-procinfo.h>
#include <dl-hwcaps.h>

/* This is the result of counting the substrings in a colon-separated
   hwcaps string.  */
struct hwcaps_counts
{
  /* Number of substrings.  */
  size_t count;

  /* Sum of the individual substring lengths (without separators or
     null terminators).  */
  size_t total_length;

  /* Maximum length of an individual substring.  */
  size_t maximum_length;
};

/* Update *COUNTS according to the contents of HWCAPS.  Skip over
   entries whose bit is not set in MASK.  */
static void
update_hwcaps_counts (struct hwcaps_counts *counts, const char *hwcaps,
		      uint32_t bitmask, const char *mask)
{
  struct dl_hwcaps_split_masked sp;
  _dl_hwcaps_split_masked_init (&sp, hwcaps, bitmask, mask);
  while (_dl_hwcaps_split_masked (&sp))
    {
      ++counts->count;
      counts->total_length += sp.split.length;
      if (sp.split.length > counts->maximum_length)
	counts->maximum_length = sp.split.length;
    }
}

/* State for copy_hwcaps.  Must be initialized to point to
   the storage areas for the array and the strings themselves.  */
struct copy_hwcaps
{
  struct r_strlenpair *next_pair;
  char *next_string;
};

/* Copy HWCAPS into the string pairs and strings, advancing *TARGET.
   Skip over entries whose bit is not set in MASK.  */
static void
copy_hwcaps (struct copy_hwcaps *target, const char *hwcaps,
	     uint32_t bitmask, const char *mask)
{
  struct dl_hwcaps_split_masked sp;
  _dl_hwcaps_split_masked_init (&sp, hwcaps, bitmask, mask);
  while (_dl_hwcaps_split_masked (&sp))
    {
      target->next_pair->str = target->next_string;
      char *slash = __mempcpy (__mempcpy (target->next_string,
					  GLIBC_HWCAPS_PREFIX,
					  strlen (GLIBC_HWCAPS_PREFIX)),
			       sp.split.segment, sp.split.length);
      *slash = '/';
      target->next_pair->len
	= strlen (GLIBC_HWCAPS_PREFIX) + sp.split.length + 1;
      ++target->next_pair;
      target->next_string = slash + 1;
    }
}

struct dl_hwcaps_priority *_dl_hwcaps_priorities;
uint32_t _dl_hwcaps_priorities_length;

/* Allocate _dl_hwcaps_priorities and fill it with data.  */
static void
compute_priorities (size_t total_count, const char *prepend,
		    uint32_t bitmask, const char *mask)
{
  _dl_hwcaps_priorities = malloc (total_count
				  * sizeof (*_dl_hwcaps_priorities));
  if (_dl_hwcaps_priorities == NULL)
    _dl_signal_error (ENOMEM, NULL, NULL,
		      N_("cannot create HWCAP priorities"));
  _dl_hwcaps_priorities_length = total_count;

  /* First the prepended subdirectories.  */
  size_t i = 0;
  {
    struct dl_hwcaps_split sp;
    _dl_hwcaps_split_init (&sp, prepend);
    while (_dl_hwcaps_split (&sp))
      {
	_dl_hwcaps_priorities[i].name = sp.segment;
	_dl_hwcaps_priorities[i].name_length = sp.length;
	_dl_hwcaps_priorities[i].priority = i + 1;
	++i;
      }
  }

  /* Then the built-in subdirectories that are actually active.  */
  {
    struct dl_hwcaps_split_masked sp;
    _dl_hwcaps_split_masked_init (&sp, _dl_hwcaps_subdirs, bitmask, mask);
    while (_dl_hwcaps_split_masked (&sp))
      {
	_dl_hwcaps_priorities[i].name = sp.split.segment;
	_dl_hwcaps_priorities[i].name_length = sp.split.length;
	_dl_hwcaps_priorities[i].priority = i + 1;
	++i;
      }
  }
  assert (i == total_count);
}

/* Sort the _dl_hwcaps_priorities array by name.  */
static void
sort_priorities_by_name (void)
{
  /* Insertion sort.  There is no need to link qsort into the dynamic
     loader for such a short array.  */
  for (size_t i = 1; i < _dl_hwcaps_priorities_length; ++i)
    for (size_t j = i; j > 0; --j)
      {
	struct dl_hwcaps_priority *previous = _dl_hwcaps_priorities + j - 1;
	struct dl_hwcaps_priority *current = _dl_hwcaps_priorities + j;

	/* Bail out if current is greater or equal to the previous
	   value.  */
	uint32_t to_compare;
	if (current->name_length < previous->name_length)
	  to_compare = current->name_length;
	else
	  to_compare = previous->name_length;
	int cmp = memcmp (current->name, previous->name, to_compare);
	if (cmp > 0
	    || (cmp == 0 && current->name_length >= previous->name_length))
	  break;

	/* Swap *previous and *current.  */
	struct dl_hwcaps_priority tmp = *previous;
	*previous = *current;
	*current = tmp;
      }
}

/* Return an array of useful/necessary hardware capability names.  */
const struct r_strlenpair *
_dl_important_hwcaps (const char *glibc_hwcaps_prepend,
		      const char *glibc_hwcaps_mask,
		      size_t *sz, size_t *max_capstrlen)
{
  uint64_t hwcap_mask = GET_HWCAP_MASK();
  /* Determine how many important bits are set.  */
  uint64_t masked = GLRO(dl_hwcap) & hwcap_mask;
  size_t cnt = GLRO (dl_platform) != NULL;
  size_t n, m;
  struct r_strlenpair *result;
  struct r_strlenpair *rp;
  char *cp;

  /* glibc-hwcaps subdirectories.  These are exempted from the power
     set construction below.  */
  uint32_t hwcaps_subdirs_active = _dl_hwcaps_subdirs_active ();
  struct hwcaps_counts hwcaps_counts =  { 0, };
  update_hwcaps_counts (&hwcaps_counts, glibc_hwcaps_prepend, -1, NULL);
  update_hwcaps_counts (&hwcaps_counts, _dl_hwcaps_subdirs,
			hwcaps_subdirs_active, glibc_hwcaps_mask);
  compute_priorities (hwcaps_counts.count, glibc_hwcaps_prepend,
		      hwcaps_subdirs_active, glibc_hwcaps_mask);
  sort_priorities_by_name ();

  /* Each hwcaps subdirectory has a GLIBC_HWCAPS_PREFIX string prefix
     and a "/" suffix once stored in the result.  */
  hwcaps_counts.maximum_length += strlen (GLIBC_HWCAPS_PREFIX) + 1;
  size_t total = (hwcaps_counts.count * (strlen (GLIBC_HWCAPS_PREFIX) + 1)
		  + hwcaps_counts.total_length);

  /* Count the number of bits set in the masked value.  */
  for (n = 0; (~((1ULL << n) - 1) & masked) != 0; ++n)
    if ((masked & (1ULL << n)) != 0)
      ++cnt;

  /* For TLS enabled builds always add 'tls'.  */
  ++cnt;

  /* Create temporary data structure to generate result table.  */
  struct r_strlenpair temp[cnt];
  m = 0;
  for (n = 0; masked != 0; ++n)
    if ((masked & (1ULL << n)) != 0)
      {
	temp[m].str = _dl_hwcap_string (n);
	temp[m].len = strlen (temp[m].str);
	masked ^= 1ULL << n;
	++m;
      }
  if (GLRO (dl_platform) != NULL)
    {
      temp[m].str = GLRO (dl_platform);
      temp[m].len = GLRO (dl_platformlen);
      ++m;
    }

  temp[m].str = "tls";
  temp[m].len = 3;
  ++m;

  assert (m == cnt);

  /* Determine the total size of all strings together.  */
  if (cnt == 1)
    total += temp[0].len + 1;
  else
    {
      total += temp[0].len + temp[cnt - 1].len + 2;
      if (cnt > 2)
	{
	  total <<= 1;
	  for (n = 1; n + 1 < cnt; ++n)
	    total += temp[n].len + 1;
	  if (cnt > 3
	      && (cnt >= sizeof (size_t) * 8
		  || total + (sizeof (*result) << 3)
		     >= (1UL << (sizeof (size_t) * 8 - cnt + 3))))
	    _dl_signal_error (ENOMEM, NULL, NULL,
			      N_("cannot create capability list"));

	  total <<= cnt - 3;
	}
    }

  *sz = hwcaps_counts.count + (1 << cnt);

  /* This is the overall result, including both glibc-hwcaps
     subdirectories and the legacy hwcaps subdirectories using the
     power set construction.  */
  struct r_strlenpair *overall_result
    = malloc (*sz * sizeof (*result) + total);
  if (overall_result == NULL)
    _dl_signal_error (ENOMEM, NULL, NULL,
		      N_("cannot create capability list"));

  /* Fill in the glibc-hwcaps subdirectories.  */
  {
    struct copy_hwcaps target;
    target.next_pair = overall_result;
    target.next_string = (char *) (overall_result + *sz);
    copy_hwcaps (&target, glibc_hwcaps_prepend, -1, NULL);
    copy_hwcaps (&target, _dl_hwcaps_subdirs,
		 hwcaps_subdirs_active, glibc_hwcaps_mask);
    /* Set up the write target for the power set construction.  */
    result = target.next_pair;
    cp = target.next_string;
  }


  /* Power set construction begins here.  We use a very compressed way
     to store the various combinations of capability names.  */

  if (cnt == 1)
    {
      result[0].str = cp;
      result[0].len = temp[0].len + 1;
      result[1].str = cp;
      result[1].len = 0;
      cp = __mempcpy (cp, temp[0].str, temp[0].len);
      *cp = '/';
      if (result[0].len > hwcaps_counts.maximum_length)
	*max_capstrlen = result[0].len;
      else
	*max_capstrlen = hwcaps_counts.maximum_length;

      return overall_result;
    }

  /* Fill in the information.  This follows the following scheme
     (indices from TEMP for four strings):
	entry #0: 0, 1, 2, 3	binary: 1111
	      #1: 0, 1, 3		1101
	      #2: 0, 2, 3		1011
	      #3: 0, 3			1001
     This allows the representation of all possible combinations of
     capability names in the string.  First generate the strings.  */
  result[1].str = result[0].str = cp;
#define add(idx) \
      cp = __mempcpy (__mempcpy (cp, temp[idx].str, temp[idx].len), "/", 1);
  if (cnt == 2)
    {
      add (1);
      add (0);
    }
  else
    {
      n = 1 << (cnt - 1);
      do
	{
	  n -= 2;

	  /* We always add the last string.  */
	  add (cnt - 1);

	  /* Add the strings which have the bit set in N.  */
	  for (m = cnt - 2; m > 0; --m)
	    if ((n & (1 << m)) != 0)
	      add (m);

	  /* Always add the first string.  */
	  add (0);
	}
      while (n != 0);
    }
#undef add

  /* Now we are ready to install the string pointers and length.  */
  for (n = 0; n < (1UL << cnt); ++n)
    result[n].len = 0;
  n = cnt;
  do
    {
      size_t mask = 1 << --n;

      rp = result;
      for (m = 1 << cnt; m > 0; ++rp)
	if ((--m & mask) != 0)
	  rp->len += temp[n].len + 1;
    }
  while (n != 0);

  /* The first half of the strings all include the first string.  */
  n = (1 << cnt) - 2;
  rp = &result[2];
  while (n != (1UL << (cnt - 1)))
    {
      if ((--n & 1) != 0)
	rp[0].str = rp[-2].str + rp[-2].len;
      else
	rp[0].str = rp[-1].str;
      ++rp;
    }

  /* The second half starts right after the first part of the string of
     the corresponding entry in the first half.  */
  do
    {
      rp[0].str = rp[-(1 << (cnt - 1))].str + temp[cnt - 1].len + 1;
      ++rp;
    }
  while (--n != 0);

  /* The maximum string length.  */
  if (result[0].len > hwcaps_counts.maximum_length)
    *max_capstrlen = result[0].len;
  else
    *max_capstrlen = hwcaps_counts.maximum_length;

  return overall_result;
}
