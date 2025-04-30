/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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
#include <limits.h>
#include <stdio.h>
#include <stdio_ext.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <stdint.h>
#include <alloc_buffer.h>

#include <timezone/tzfile.h>

int __use_tzfile;
static dev_t tzfile_dev;
static ino64_t tzfile_ino;
static time_t tzfile_mtime;

struct ttinfo
  {
    int offset;			/* Seconds east of GMT.  */
    unsigned char isdst;	/* Used to set tm_isdst.  */
    unsigned char idx;		/* Index into `zone_names'.  */
    unsigned char isstd;	/* Transition times are in standard time.  */
    unsigned char isgmt;	/* Transition times are in GMT.  */
  };

struct leap
  {
    __time64_t transition;	/* Time the transition takes effect.  */
    long int change;		/* Seconds of correction to apply.  */
  };

static size_t num_transitions;
libc_freeres_ptr (static __time64_t *transitions);
static unsigned char *type_idxs;
static size_t num_types;
static struct ttinfo *types;
static char *zone_names;
static long int rule_stdoff;
static long int rule_dstoff;
static size_t num_leaps;
static struct leap *leaps;
static char *tzspec;

#include <endian.h>
#include <byteswap.h>

/* Decode the four bytes at PTR as a signed integer in network byte order.  */
static inline int
__attribute ((always_inline))
decode (const void *ptr)
{
  if (BYTE_ORDER == BIG_ENDIAN && sizeof (int) == 4)
    return *(const int *) ptr;
  if (sizeof (int) == 4)
    return bswap_32 (*(const int *) ptr);

  const unsigned char *p = ptr;
  int result = *p & (1 << (CHAR_BIT - 1)) ? ~0 : 0;

  result = (result << 8) | *p++;
  result = (result << 8) | *p++;
  result = (result << 8) | *p++;
  result = (result << 8) | *p++;

  return result;
}


static inline int64_t
__attribute ((always_inline))
decode64 (const void *ptr)
{
  if ((BYTE_ORDER == BIG_ENDIAN))
    return *(const int64_t *) ptr;

  return bswap_64 (*(const int64_t *) ptr);
}


void
__tzfile_read (const char *file, size_t extra, char **extrap)
{
  static const char default_tzdir[] = TZDIR;
  size_t num_isstd, num_isgmt;
  FILE *f;
  struct tzhead tzhead;
  size_t chars;
  size_t i;
  int was_using_tzfile = __use_tzfile;
  int trans_width = 4;
  char *new = NULL;

  _Static_assert (sizeof (__time64_t) == 8,
		  "__time64_t must be eight bytes");

  __use_tzfile = 0;

  if (file == NULL)
    /* No user specification; use the site-wide default.  */
    file = TZDEFAULT;
  else if (*file == '\0')
    /* User specified the empty string; use UTC with no leap seconds.  */
    goto ret_free_transitions;
  else
    {
      /* We must not allow to read an arbitrary file in a setuid
	 program.  So we fail for any file which is not in the
	 directory hierachy starting at TZDIR
	 and which is not the system wide default TZDEFAULT.  */
      if (__libc_enable_secure
	  && ((*file == '/'
	       && memcmp (file, TZDEFAULT, sizeof TZDEFAULT)
	       && memcmp (file, default_tzdir, sizeof (default_tzdir) - 1))
	      || strstr (file, "../") != NULL))
	/* This test is certainly a bit too restrictive but it should
	   catch all critical cases.  */
	goto ret_free_transitions;
    }

  if (*file != '/')
    {
      const char *tzdir;

      tzdir = getenv ("TZDIR");
      if (tzdir == NULL || *tzdir == '\0')
	tzdir = default_tzdir;
      if (__asprintf (&new, "%s/%s", tzdir, file) == -1)
	goto ret_free_transitions;
      file = new;
    }

  /* If we were already using tzfile, check whether the file changed.  */
  struct __stat64_t64 st;
  if (was_using_tzfile
      && __stat64_time64 (file, &st) == 0
      && tzfile_ino == st.st_ino && tzfile_dev == st.st_dev
      && tzfile_mtime == st.st_mtime)
    goto done;  /* Nothing to do.  */

  /* Note the file is opened with cancellation in the I/O functions
     disabled and if available FD_CLOEXEC set.  */
  f = fopen (file, "rce");
  if (f == NULL)
    goto ret_free_transitions;

  /* Get information about the file we are actually using.  */
  if (__fstat64_time64 (__fileno (f), &st) != 0)
    goto lose;

  free ((void *) transitions);
  transitions = NULL;

  /* Remember the inode and device number and modification time.  */
  tzfile_dev = st.st_dev;
  tzfile_ino = st.st_ino;
  tzfile_mtime = st.st_mtime;

  /* No threads reading this stream.  */
  __fsetlocking (f, FSETLOCKING_BYCALLER);

 read_again:
  if (__builtin_expect (__fread_unlocked ((void *) &tzhead, sizeof (tzhead),
					  1, f) != 1, 0)
      || memcmp (tzhead.tzh_magic, TZ_MAGIC, sizeof (tzhead.tzh_magic)) != 0)
    goto lose;

  num_transitions = (size_t) decode (tzhead.tzh_timecnt);
  num_types = (size_t) decode (tzhead.tzh_typecnt);
  chars = (size_t) decode (tzhead.tzh_charcnt);
  num_leaps = (size_t) decode (tzhead.tzh_leapcnt);
  num_isstd = (size_t) decode (tzhead.tzh_ttisstdcnt);
  num_isgmt = (size_t) decode (tzhead.tzh_ttisutcnt);

  if (__glibc_unlikely (num_isstd > num_types || num_isgmt > num_types))
    goto lose;

  if (trans_width == 4 && tzhead.tzh_version[0] != '\0')
    {
      /* We use the 8-byte format.  */
      trans_width = 8;

      /* Position the stream before the second header.  */
      size_t to_skip = (num_transitions * (4 + 1)
			+ num_types * 6
			+ chars
			+ num_leaps * 8
			+ num_isstd
			+ num_isgmt);
      if (fseek (f, to_skip, SEEK_CUR) != 0)
	goto lose;

      goto read_again;
    }

  /* Compute the size of the POSIX time zone specification in the
     file.  */
  size_t tzspec_len;
  if (trans_width == 8)
    {
      off_t rem = st.st_size - __ftello (f);
      if (__builtin_expect (rem < 0
			    || (size_t) rem < (num_transitions * (8 + 1)
					       + num_types * 6
					       + chars), 0))
	goto lose;
      tzspec_len = (size_t) rem - (num_transitions * (8 + 1)
				   + num_types * 6
				   + chars);
      if (__builtin_expect (num_leaps > SIZE_MAX / 12
			    || tzspec_len < num_leaps * 12, 0))
	goto lose;
      tzspec_len -= num_leaps * 12;
      if (__glibc_unlikely (tzspec_len < num_isstd))
	goto lose;
      tzspec_len -= num_isstd;
      if (__glibc_unlikely (tzspec_len == 0 || tzspec_len - 1 < num_isgmt))
	goto lose;
      tzspec_len -= num_isgmt + 1;
      if (tzspec_len == 0)
	goto lose;
    }
  else
    tzspec_len = 0;

  /* The file is parsed into a single heap allocation, comprising of
     the following arrays:

     __time64_t transitions[num_transitions];
     struct leap leaps[num_leaps];
     struct ttinfo types[num_types];
     unsigned char type_idxs[num_types];
     char zone_names[chars];
     char tzspec[tzspec_len];
     char extra_array[extra]; // Stored into *pextras if requested.

     The piece-wise allocations from buf below verify that no
     overflow/wraparound occurred in these computations.

     The order of the suballocations is important for alignment
     purposes.  __time64_t outside a struct may require more alignment
     then inside a struct on some architectures, so it must come
     first. */
  _Static_assert (__alignof (__time64_t) >= __alignof (struct leap),
		  "alignment of __time64_t");
  _Static_assert (__alignof (struct leap) >= __alignof (struct ttinfo),
		  "alignment of struct leap");
  struct alloc_buffer buf;
  {
    size_t total_size = (num_transitions * sizeof (__time64_t)
			 + num_leaps * sizeof (struct leap)
			 + num_types * sizeof (struct ttinfo)
			 + num_transitions /* type_idxs */
			 + chars /* zone_names */
			 + tzspec_len + extra);
    transitions = malloc (total_size);
    if (transitions == NULL)
      goto lose;
    buf = alloc_buffer_create (transitions, total_size);
  }

  /* The address of the first allocation is already stored in the
     pointer transitions.  */
  (void) alloc_buffer_alloc_array (&buf, __time64_t, num_transitions);
  leaps = alloc_buffer_alloc_array (&buf, struct leap, num_leaps);
  types = alloc_buffer_alloc_array (&buf, struct ttinfo, num_types);
  type_idxs = alloc_buffer_alloc_array (&buf, unsigned char, num_transitions);
  zone_names = alloc_buffer_alloc_array (&buf, char, chars);
  if (trans_width == 8)
    tzspec = alloc_buffer_alloc_array (&buf, char, tzspec_len);
  else
    tzspec = NULL;
  if (extra > 0)
    *extrap = alloc_buffer_alloc_array (&buf, char, extra);
  if (alloc_buffer_has_failed (&buf))
    goto lose;

  if (__glibc_unlikely (__fread_unlocked (transitions, trans_width,
					  num_transitions, f)
			!= num_transitions)
      || __glibc_unlikely (__fread_unlocked (type_idxs, 1, num_transitions, f)
			   != num_transitions))
	goto lose;

  /* Check for bogus indices in the data file, so we can hereafter
     safely use type_idxs[T] as indices into `types' and never crash.  */
  for (i = 0; i < num_transitions; ++i)
    if (__glibc_unlikely (type_idxs[i] >= num_types))
      goto lose;

  if (trans_width == 4)
    {
      /* Decode the transition times, stored as 4-byte integers in
	 network (big-endian) byte order.  We work from the end of the
	 array so as not to clobber the next element to be
	 processed.  */
      i = num_transitions;
      while (i-- > 0)
	transitions[i] = decode ((char *) transitions + i * 4);
    }
  else if (BYTE_ORDER != BIG_ENDIAN)
    {
      /* Decode the transition times, stored as 8-byte integers in
	 network (big-endian) byte order.  */
      for (i = 0; i < num_transitions; ++i)
	transitions[i] = decode64 ((char *) transitions + i * 8);
    }

  for (i = 0; i < num_types; ++i)
    {
      unsigned char x[4];
      int c;
      if (__builtin_expect (__fread_unlocked (x, 1,
					      sizeof (x), f) != sizeof (x),
			    0))
	goto lose;
      c = __getc_unlocked (f);
      if (__glibc_unlikely ((unsigned int) c > 1u))
	goto lose;
      types[i].isdst = c;
      c = __getc_unlocked (f);
      if (__glibc_unlikely ((size_t) c > chars))
	/* Bogus index in data file.  */
	goto lose;
      types[i].idx = c;
      types[i].offset = decode (x);
    }

  if (__glibc_unlikely (__fread_unlocked (zone_names, 1, chars, f) != chars))
    goto lose;

  for (i = 0; i < num_leaps; ++i)
    {
      unsigned char x[8];
      if (__builtin_expect (__fread_unlocked (x, 1, trans_width, f)
			    != trans_width, 0))
	goto lose;
      if (trans_width == 4)
	leaps[i].transition = decode (x);
      else
	leaps[i].transition = decode64 (x);

      if (__glibc_unlikely (__fread_unlocked (x, 1, 4, f) != 4))
	goto lose;
      leaps[i].change = (long int) decode (x);
    }

  for (i = 0; i < num_isstd; ++i)
    {
      int c = __getc_unlocked (f);
      if (__glibc_unlikely (c == EOF))
	goto lose;
      types[i].isstd = c != 0;
    }
  while (i < num_types)
    types[i++].isstd = 0;

  for (i = 0; i < num_isgmt; ++i)
    {
      int c = __getc_unlocked (f);
      if (__glibc_unlikely (c == EOF))
	goto lose;
      types[i].isgmt = c != 0;
    }
  while (i < num_types)
    types[i++].isgmt = 0;

  /* Read the POSIX TZ-style information if possible.  */
  if (tzspec != NULL)
    {
      assert (tzspec_len > 0);
      /* Skip over the newline first.  */
      if (__getc_unlocked (f) != '\n'
	  || (__fread_unlocked (tzspec, 1, tzspec_len - 1, f)
	      != tzspec_len - 1))
	tzspec = NULL;
      else
	tzspec[tzspec_len - 1] = '\0';
    }

  /* Don't use an empty TZ string.  */
  if (tzspec != NULL && tzspec[0] == '\0')
    tzspec = NULL;

  fclose (f);

  /* First "register" all timezone names.  */
  for (i = 0; i < num_types; ++i)
    if (__tzstring (&zone_names[types[i].idx]) == NULL)
      goto ret_free_transitions;

  /* Find the standard and daylight time offsets used by the rule file.
     We choose the offsets in the types of each flavor that are
     transitioned to earliest in time.  */
  __tzname[0] = NULL;
  __tzname[1] = NULL;
  for (i = num_transitions; i > 0; )
    {
      int type = type_idxs[--i];
      int dst = types[type].isdst;

      if (__tzname[dst] == NULL)
	{
	  int idx = types[type].idx;

	  __tzname[dst] = __tzstring (&zone_names[idx]);

	  if (__tzname[1 - dst] != NULL)
	    break;
	}
    }
  if (__tzname[0] == NULL)
    {
      /* This should only happen if there are no transition rules.
	 In this case there should be only one single type.  */
      assert (num_types == 1);
      __tzname[0] = __tzstring (zone_names);
    }
  if (__tzname[1] == NULL)
    __tzname[1] = __tzname[0];

  if (num_transitions == 0)
    /* Use the first rule (which should also be the only one).  */
    rule_stdoff = rule_dstoff = types[0].offset;
  else
    {
      int stdoff_set = 0, dstoff_set = 0;
      rule_stdoff = rule_dstoff = 0;
      i = num_transitions - 1;
      do
	{
	  if (!stdoff_set && !types[type_idxs[i]].isdst)
	    {
	      stdoff_set = 1;
	      rule_stdoff = types[type_idxs[i]].offset;
	    }
	  else if (!dstoff_set && types[type_idxs[i]].isdst)
	    {
	      dstoff_set = 1;
	      rule_dstoff = types[type_idxs[i]].offset;
	    }
	  if (stdoff_set && dstoff_set)
	    break;
	}
      while (i-- > 0);

      if (!dstoff_set)
	rule_dstoff = rule_stdoff;
    }

  __daylight = rule_stdoff != rule_dstoff;
  __timezone = -rule_stdoff;

 done:
  __use_tzfile = 1;
  free (new);
  return;

 lose:
  fclose (f);
 ret_free_transitions:
  free (new);
  free ((void *) transitions);
  transitions = NULL;
}

/* The user specified a hand-made timezone, but not its DST rules.
   We will use the names and offsets from the user, and the rules
   from the TZDEFRULES file.  */

void
__tzfile_default (const char *std, const char *dst,
		  int stdoff, int dstoff)
{
  size_t stdlen = strlen (std) + 1;
  size_t dstlen = strlen (dst) + 1;
  size_t i;
  int isdst;
  char *cp;

  __tzfile_read (TZDEFRULES, stdlen + dstlen, &cp);
  if (!__use_tzfile)
    return;

  if (num_types < 2)
    {
      __use_tzfile = 0;
      return;
    }

  /* Ignore the zone names read from the file and use the given ones
     instead.  */
  __mempcpy (__mempcpy (cp, std, stdlen), dst, dstlen);
  zone_names = cp;

  /* Now there are only two zones, regardless of what the file contained.  */
  num_types = 2;

  /* Now correct the transition times for the user-specified standard and
     daylight offsets from GMT.  */
  isdst = 0;
  for (i = 0; i < num_transitions; ++i)
    {
      struct ttinfo *trans_type = &types[type_idxs[i]];

      /* We will use only types 0 (standard) and 1 (daylight).
	 Fix up this transition to point to whichever matches
	 the flavor of its original type.  */
      type_idxs[i] = trans_type->isdst;

      if (trans_type->isgmt)
	/* The transition time is in GMT.  No correction to apply.  */ ;
      else if (isdst && !trans_type->isstd)
	/* The type says this transition is in "local wall clock time", and
	   wall clock time as of the previous transition was DST.  Correct
	   for the difference between the rule's DST offset and the user's
	   DST offset.  */
	transitions[i] += dstoff - rule_dstoff;
      else
	/* This transition is in "local wall clock time", and wall clock
	   time as of this iteration is non-DST.  Correct for the
	   difference between the rule's standard offset and the user's
	   standard offset.  */
	transitions[i] += stdoff - rule_stdoff;

      /* The DST state of "local wall clock time" for the next iteration is
	 as specified by this transition.  */
      isdst = trans_type->isdst;
    }

  /* Now that we adjusted the transitions to the requested offsets,
     reset the rule_stdoff and rule_dstoff values appropriately.  They
     are used elsewhere.  */
  rule_stdoff = stdoff;
  rule_dstoff = dstoff;

  /* Reset types 0 and 1 to describe the user's settings.  */
  types[0].idx = 0;
  types[0].offset = stdoff;
  types[0].isdst = 0;
  types[1].idx = stdlen;
  types[1].offset = dstoff;
  types[1].isdst = 1;

  /* Reset the zone names to point to the user's names.  */
  __tzname[0] = (char *) std;
  __tzname[1] = (char *) dst;

  /* Set the timezone.  */
  __timezone = -types[0].offset;

  /* Invalidate the tzfile attribute cache to force rereading
     TZDEFRULES the next time it is used.  */
  tzfile_dev = 0;
  tzfile_ino = 0;
  tzfile_mtime = 0;
}

void
__tzfile_compute (__time64_t timer, int use_localtime,
		  long int *leap_correct, int *leap_hit,
		  struct tm *tp)
{
  size_t i;

  if (use_localtime)
    {
      __tzname[0] = NULL;
      __tzname[1] = NULL;

      if (__glibc_unlikely (num_transitions == 0 || timer < transitions[0]))
	{
	  /* TIMER is before any transition (or there are no transitions).
	     Choose the first non-DST type
	     (or the first if they're all DST types).  */
	  i = 0;
	  while (i < num_types && types[i].isdst)
	    {
	      if (__tzname[1] == NULL)
		__tzname[1] = __tzstring (&zone_names[types[i].idx]);

	      ++i;
	    }

	  if (i == num_types)
	    i = 0;
	  __tzname[0] = __tzstring (&zone_names[types[i].idx]);
	  if (__tzname[1] == NULL)
	    {
	      size_t j = i;
	      while (j < num_types)
		if (types[j].isdst)
		  {
		    __tzname[1] = __tzstring (&zone_names[types[j].idx]);
		    break;
		  }
		else
		  ++j;
	    }
	}
      else if (__glibc_unlikely (timer >= transitions[num_transitions - 1]))
	{
	  if (__glibc_unlikely (tzspec == NULL))
	    {
	    use_last:
	      i = num_transitions;
	      goto found;
	    }

	  /* Parse the POSIX TZ-style string.  */
	  __tzset_parse_tz (tzspec);

	  /* Convert to broken down structure.  If this fails do not
	     use the string.  */
	  if (__glibc_unlikely (! __offtime (timer, 0, tp)))
	    goto use_last;

	  /* Use the rules from the TZ string to compute the change.  */
	  __tz_compute (timer, tp, 1);

	  /* If tzspec comes from posixrules loaded by __tzfile_default,
	     override the STD and DST zone names with the ones user
	     requested in TZ envvar.  */
	  if (__glibc_unlikely (zone_names == (char *) &leaps[num_leaps]))
	    {
	      assert (num_types == 2);
	      __tzname[0] = __tzstring (zone_names);
	      __tzname[1] = __tzstring (&zone_names[strlen (zone_names) + 1]);
	    }

	  goto leap;
	}
      else
	{
	  /* Find the first transition after TIMER, and
	     then pick the type of the transition before it.  */
	  size_t lo = 0;
	  size_t hi = num_transitions - 1;
	  /* Assume that DST is changing twice a year and guess
	     initial search spot from it.  Half of a gregorian year
	     has on average 365.2425 * 86400 / 2 = 15778476 seconds.
	     The value i can be truncated if size_t is smaller than
	     __time64_t, but this is harmless because it is just
	     a guess.  */
	  i = (transitions[num_transitions - 1] - timer) / 15778476;
	  if (i < num_transitions)
	    {
	      i = num_transitions - 1 - i;
	      if (timer < transitions[i])
		{
		  if (i < 10 || timer >= transitions[i - 10])
		    {
		      /* Linear search.  */
		      while (timer < transitions[i - 1])
			--i;
		      goto found;
		    }
		  hi = i - 10;
		}
	      else
		{
		  if (i + 10 >= num_transitions || timer < transitions[i + 10])
		    {
		      /* Linear search.  */
		      while (timer >= transitions[i])
			++i;
		      goto found;
		    }
		  lo = i + 10;
		}
	    }

	  /* Binary search.  */
	  /* assert (timer >= transitions[lo] && timer < transitions[hi]); */
	  while (lo + 1 < hi)
	    {
	      i = (lo + hi) / 2;
	      if (timer < transitions[i])
		hi = i;
	      else
		lo = i;
	    }
	  i = hi;

	found:
	  /* assert (timer >= transitions[i - 1]
	     && (i == num_transitions || timer < transitions[i])); */
	  __tzname[types[type_idxs[i - 1]].isdst]
	    = __tzstring (&zone_names[types[type_idxs[i - 1]].idx]);
	  size_t j = i;
	  while (j < num_transitions)
	    {
	      int type = type_idxs[j];
	      int dst = types[type].isdst;
	      int idx = types[type].idx;

	      if (__tzname[dst] == NULL)
		{
		  __tzname[dst] = __tzstring (&zone_names[idx]);

		  if (__tzname[1 - dst] != NULL)
		    break;
		}

	      ++j;
	    }

	  if (__glibc_unlikely (__tzname[0] == NULL))
	    __tzname[0] = __tzname[1];

	  i = type_idxs[i - 1];
	}

      struct ttinfo *info = &types[i];
      __daylight = rule_stdoff != rule_dstoff;
      __timezone = -rule_stdoff;

      if (__tzname[0] == NULL)
	{
	  /* This should only happen if there are no transition rules.
	     In this case there should be only one single type.  */
	  assert (num_types == 1);
	  __tzname[0] = __tzstring (zone_names);
	}
      if (__tzname[1] == NULL)
	/* There is no daylight saving time.  */
	__tzname[1] = __tzname[0];
      tp->tm_isdst = info->isdst;
      assert (strcmp (&zone_names[info->idx], __tzname[tp->tm_isdst]) == 0);
      tp->tm_zone = __tzname[tp->tm_isdst];
      tp->tm_gmtoff = info->offset;
    }

 leap:
  *leap_correct = 0L;
  *leap_hit = 0;

  /* Find the last leap second correction transition time before TIMER.  */
  i = num_leaps;
  do
    if (i-- == 0)
      return;
  while (timer < leaps[i].transition);

  /* Apply its correction.  */
  *leap_correct = leaps[i].change;

  if (timer == leaps[i].transition /* Exactly at the transition time.  */
      && ((i == 0 && leaps[i].change > 0)
	  || leaps[i].change > leaps[i - 1].change))
    {
      *leap_hit = 1;
      while (i > 0
	     && leaps[i].transition == leaps[i - 1].transition + 1
	     && leaps[i].change == leaps[i - 1].change + 1)
	{
	  ++*leap_hit;
	  --i;
	}
    }
}
