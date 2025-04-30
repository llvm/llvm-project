/* Code to load locale data from the locale archive file.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

#include <locale.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include <errno.h>
#include <assert.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/param.h>

#include "localeinfo.h"
#include "locarchive.h"
#include <not-cancel.h>

/* Define the hash function.  We define the function as static inline.  */
#define compute_hashval static inline compute_hashval
#define hashval_t uint32_t
#include "hashval.h"
#undef compute_hashval


/* Name of the locale archive file.  */
static const char archfname[] = COMPLOCALEDIR "/locale-archive";

/* Size of initial mapping window, optimal if large enough to
   cover the header plus the initial locale.  */
#define ARCHIVE_MAPPING_WINDOW	(2 * 1024 * 1024)

#ifndef MAP_COPY
/* This is not quite as good as MAP_COPY since unexamined pages
   can change out from under us and give us inconsistent data.
   But we rely on the user not to diddle the system's live archive.
   Even though we only ever use PROT_READ, using MAP_SHARED would
   not give the system sufficient freedom to e.g. let the on disk
   file go away because it doesn't know we won't call mprotect later.  */
# define MAP_COPY MAP_PRIVATE
#endif
#ifndef MAP_FILE
 /* Some systems do not have this flag; it is superfluous.  */
# define MAP_FILE 0
#endif

/* Record of contiguous pages already mapped from the locale archive.  */
struct archmapped
{
  void *ptr;
  uint32_t from;
  uint32_t len;
  struct archmapped *next;
};
static struct archmapped *archmapped;

/* This describes the mapping at the beginning of the file that contains
   the header data.  There could be data in the following partial page,
   so this is searched like any other.  Once the archive has been used,
   ARCHMAPPED points to this; if mapping the archive header failed,
   then headmap.ptr is null.  */
static struct archmapped headmap;
static struct __stat64_t64 archive_stat; /* stat of archive when header mapped.  */

/* Record of locales that we have already loaded from the archive.  */
struct locale_in_archive
{
  struct locale_in_archive *next;
  char *name;
  struct __locale_data *data[__LC_LAST];
};
static struct locale_in_archive *archloaded;


/* Local structure and subroutine of _nl_load_archive, see below.  */
struct range
{
  uint32_t from;
  uint32_t len;
  int category;
  void *result;
};

static int
rangecmp (const void *p1, const void *p2)
{
  return ((struct range *) p1)->from - ((struct range *) p2)->from;
}


/* Calculate the amount of space needed for all the tables described
   by the given header.  Note we do not include the empty table space
   that has been preallocated in the file, so our mapping may not be
   large enough if localedef adds data to the file in place.  However,
   doing that would permute the header fields while we are accessing
   them and thus not be safe anyway, so we don't allow for that.  */
static inline off_t
calculate_head_size (const struct locarhead *h)
{
  off_t namehash_end = (h->namehash_offset
			+ h->namehash_size * sizeof (struct namehashent));
  off_t string_end =  h->string_offset + h->string_used;
  off_t locrectab_end = (h->locrectab_offset
			 + h->locrectab_used * sizeof (struct locrecent));
  return MAX (namehash_end, MAX (string_end, locrectab_end));
}


/* Find the locale *NAMEP in the locale archive, and return the
   internalized data structure for its CATEGORY data.  If this locale has
   already been loaded from the archive, just returns the existing data
   structure.  If successful, sets *NAMEP to point directly into the mapped
   archive string table; that way, the next call can short-circuit strcmp.  */
struct __locale_data *
_nl_load_locale_from_archive (int category, const char **namep)
{
  const char *name = *namep;
  struct
  {
    void *addr;
    size_t len;
  } results[__LC_LAST];
  struct locale_in_archive *lia;
  struct locarhead *head;
  struct namehashent *namehashtab;
  struct locrecent *locrec;
  struct archmapped *mapped;
  struct archmapped *last;
  unsigned long int hval;
  size_t idx;
  size_t incr;
  struct range ranges[__LC_LAST - 1];
  int nranges;
  int cnt;
  size_t ps = __sysconf (_SC_PAGE_SIZE);
  int fd = -1;

  /* Check if we have already loaded this locale from the archive.
     If we previously loaded the locale but found bogons in the data,
     then we will have stored a null pointer to return here.  */
  for (lia = archloaded; lia != NULL; lia = lia->next)
    if (name == lia->name || !strcmp (name, lia->name))
      {
	*namep = lia->name;
	return lia->data[category];
      }

  {
    /* If the name contains a codeset, then we normalize the name before
       doing the lookup.  */
    const char *p = strchr (name, '.');
    if (p != NULL && p[1] != '@' && p[1] != '\0')
      {
	const char *rest = __strchrnul (++p, '@');
	const char *normalized_codeset = _nl_normalize_codeset (p, rest - p);
	if (normalized_codeset == NULL)	/* malloc failure */
	  return NULL;
	if (strncmp (normalized_codeset, p, rest - p) != 0
	    || normalized_codeset[rest - p] != '\0')
	  {
	    /* There is a normalized codeset name that is different from
	       what was specified; reconstruct a new locale name using it.  */
	    size_t normlen = strlen (normalized_codeset);
	    size_t restlen = strlen (rest) + 1;
	    char *newname = alloca (p - name + normlen + restlen);
	    memcpy (__mempcpy (__mempcpy (newname, name, p - name),
			       normalized_codeset, normlen),
		    rest, restlen);
	    name = newname;
	  }
	free ((char *) normalized_codeset);
      }
  }

  /* Make sure the archive is loaded.  */
  if (archmapped == NULL)
    {
      void *result;
      size_t headsize, mapsize;

      /* We do this early as a sign that we have tried to open the archive.
	 If headmap.ptr remains null, that's an indication that we tried
	 and failed, so we won't try again.  */
      archmapped = &headmap;

      /* The archive has never been opened.  */
      fd = __open_nocancel (archfname, O_RDONLY|O_LARGEFILE|O_CLOEXEC);
      if (fd < 0)
	/* Cannot open the archive, for whatever reason.  */
	return NULL;

      if (__fstat64_time64 (fd, &archive_stat) == -1)
	{
	  /* stat failed, very strange.  */
	close_and_out:
	  if (fd >= 0)
	    __close_nocancel_nostatus (fd);
	  return NULL;
	}


      /* Map an initial window probably large enough to cover the header
	 and the first locale's data.  With a large address space, we can
	 just map the whole file and be sure everything is covered.  */

      mapsize = (sizeof (void *) > 4 ? archive_stat.st_size
		 : MIN (archive_stat.st_size, ARCHIVE_MAPPING_WINDOW));

      result = __mmap64 (NULL, mapsize, PROT_READ, MAP_FILE|MAP_COPY, fd, 0);
      if (result == MAP_FAILED)
	goto close_and_out;

      /* Check whether the file is large enough for the sizes given in
	 the header.  Theoretically an archive could be so large that
	 just the header fails to fit in our initial mapping window.  */
      headsize = calculate_head_size ((const struct locarhead *) result);
      if (headsize > mapsize)
	{
	  (void) __munmap (result, mapsize);
	  if (sizeof (void *) > 4 || headsize > archive_stat.st_size)
	    /* The file is not big enough for the header.  Bogus.  */
	    goto close_and_out;

	  /* Freakishly long header.  */
	  /* XXX could use mremap when available */
	  mapsize = (headsize + ps - 1) & ~(ps - 1);
	  result = __mmap64 (NULL, mapsize, PROT_READ, MAP_FILE|MAP_COPY,
			     fd, 0);
	  if (result == MAP_FAILED)
	    goto close_and_out;
	}

      if (sizeof (void *) > 4 || mapsize >= archive_stat.st_size)
	{
	  /* We've mapped the whole file already, so we can be
	     sure we won't need this file descriptor later.  */
	  __close_nocancel_nostatus (fd);
	  fd = -1;
	}

      headmap.ptr = result;
      /* headmap.from already initialized to zero.  */
      headmap.len = mapsize;
    }

  /* If there is no archive or it cannot be loaded for some reason fail.  */
  if (__glibc_unlikely (headmap.ptr == NULL))
    goto close_and_out;

  /* We have the archive available.  To find the name we first have to
     determine its hash value.  */
  hval = compute_hashval (name, strlen (name));

  head = headmap.ptr;
  namehashtab = (struct namehashent *) ((char *) head
					+ head->namehash_offset);

  /* Avoid division by 0 if the file is corrupted.  */
  if (__glibc_unlikely (head->namehash_size <= 2))
    goto close_and_out;

  idx = hval % head->namehash_size;
  incr = 1 + hval % (head->namehash_size - 2);

  /* If the name_offset field is zero this means this is a
     deleted entry and therefore no entry can be found.  */
  while (1)
    {
      if (namehashtab[idx].name_offset == 0)
	/* Not found.  */
	goto close_and_out;

      if (namehashtab[idx].hashval == hval
	  && strcmp (name, headmap.ptr + namehashtab[idx].name_offset) == 0)
	/* Found the entry.  */
	break;

      idx += incr;
      if (idx >= head->namehash_size)
	idx -= head->namehash_size;
    }

  /* We found an entry.  It might be a placeholder for a removed one.  */
  if (namehashtab[idx].locrec_offset == 0)
    goto close_and_out;

  locrec = (struct locrecent *) (headmap.ptr + namehashtab[idx].locrec_offset);

  if (sizeof (void *) > 4 /* || headmap.len == archive_stat.st_size */)
    {
      /* We already have the whole locale archive mapped in.  */
      assert (headmap.len == archive_stat.st_size);
      for (cnt = 0; cnt < __LC_LAST; ++cnt)
	if (cnt != LC_ALL)
	  {
	    if (locrec->record[cnt].offset + locrec->record[cnt].len
		> headmap.len)
	      /* The archive locrectab contains bogus offsets.  */
	      goto close_and_out;
	    results[cnt].addr = headmap.ptr + locrec->record[cnt].offset;
	    results[cnt].len = locrec->record[cnt].len;
	  }
    }
  else
    {
      /* Get the offsets of the data files and sort them.  */
      for (cnt = nranges = 0; cnt < __LC_LAST; ++cnt)
	if (cnt != LC_ALL)
	  {
	    ranges[nranges].from = locrec->record[cnt].offset;
	    ranges[nranges].len = locrec->record[cnt].len;
	    ranges[nranges].category = cnt;
	    ranges[nranges].result = NULL;

	    ++nranges;
	  }

      qsort (ranges, nranges, sizeof (ranges[0]), rangecmp);

      /* The information about mmap'd blocks is kept in a list.
	 Skip over the blocks which are before the data we need.  */
      last = mapped = archmapped;
      for (cnt = 0; cnt < nranges; ++cnt)
	{
	  int upper;
	  size_t from;
	  size_t to;
	  void *addr;
	  struct archmapped *newp;

	  /* Determine whether the appropriate page is already mapped.  */
	  while (mapped != NULL
		 && (mapped->from + mapped->len
		     <= ranges[cnt].from + ranges[cnt].len))
	    {
	      last = mapped;
	      mapped = mapped->next;
	    }

	  /* Do we have a match?  */
	  if (mapped != NULL
	      && mapped->from <= ranges[cnt].from
	      && (ranges[cnt].from + ranges[cnt].len
		  <= mapped->from + mapped->len))
	    {
	      /* Yep, already loaded.  */
	      results[ranges[cnt].category].addr = ((char *) mapped->ptr
						    + ranges[cnt].from
						    - mapped->from);
	      results[ranges[cnt].category].len = ranges[cnt].len;
	      continue;
	    }

	  /* Map the range with the locale data from the file.  We will
	     try to cover as much of the locale as possible.  I.e., if the
	     next category (next as in "next offset") is on the current or
	     immediately following page we use it as well.  */
	  assert (powerof2 (ps));
	  from = ranges[cnt].from & ~(ps - 1);
	  upper = cnt;
	  do
	    {
	      to = ranges[upper].from + ranges[upper].len;
	      if (to > (size_t) archive_stat.st_size)
		/* The archive locrectab contains bogus offsets.  */
		goto close_and_out;
	      to = (to + ps - 1) & ~(ps - 1);

	      /* If a range is already mmaped in, stop.	 */
	      if (mapped != NULL && ranges[upper].from >= mapped->from)
		break;

	      ++upper;
	    }
	  /* Loop while still in contiguous pages. */
	  while (upper < nranges && ranges[upper].from < to + ps);

	  /* Open the file if it hasn't happened yet.  */
	  if (fd == -1)
	    {
	      struct __stat64_t64 st;
	      fd = __open_nocancel (archfname,
				    O_RDONLY|O_LARGEFILE|O_CLOEXEC);
	      if (fd == -1)
		/* Cannot open the archive, for whatever reason.  */
		return NULL;
	      /* Now verify we think this is really the same archive file
		 we opened before.  If it has been changed we cannot trust
		 the header we read previously.  */
	      if (__fstat64_time64 (fd, &st) < 0
		  || st.st_size != archive_stat.st_size
		  || st.st_mtime != archive_stat.st_mtime
		  || st.st_dev != archive_stat.st_dev
		  || st.st_ino != archive_stat.st_ino)
		goto close_and_out;
	    }

	  /* Map the range from the archive.  */
	  addr = __mmap64 (NULL, to - from, PROT_READ, MAP_FILE|MAP_COPY,
			   fd, from);
	  if (addr == MAP_FAILED)
	    goto close_and_out;

	  /* Allocate a record for this mapping.  */
	  newp = (struct archmapped *) malloc (sizeof (struct archmapped));
	  if (newp == NULL)
	    {
	      (void) __munmap (addr, to - from);
	      goto close_and_out;
	    }

	  /* And queue it.  */
	  newp->ptr = addr;
	  newp->from = from;
	  newp->len = to - from;
	  assert (last->next == mapped);
	  newp->next = mapped;
	  last->next = newp;
	  last = newp;

	  /* Determine the load addresses for the category data.  */
	  do
	    {
	      assert (ranges[cnt].from >= from);
	      results[ranges[cnt].category].addr = ((char *) addr
						    + ranges[cnt].from - from);
	      results[ranges[cnt].category].len = ranges[cnt].len;
	    }
	  while (++cnt < upper);
	  --cnt;		/* The 'for' will increase 'cnt' again.  */
	}
    }

  /* We don't need the file descriptor any longer.  */
  if (fd >= 0)
    __close_nocancel_nostatus (fd);
  fd = -1;

  /* We succeeded in mapping all the necessary regions of the archive.
     Now we need the expected data structures to point into the data.  */

  lia = malloc (sizeof *lia);
  if (__glibc_unlikely (lia == NULL))
    return NULL;

  lia->name = __strdup (*namep);
  if (__glibc_unlikely (lia->name == NULL))
    {
      free (lia);
      return NULL;
    }

  lia->next = archloaded;
  archloaded = lia;

  for (cnt = 0; cnt < __LC_LAST; ++cnt)
    if (cnt != LC_ALL)
      {
	lia->data[cnt] = _nl_intern_locale_data (cnt,
						 results[cnt].addr,
						 results[cnt].len);
	if (__glibc_likely (lia->data[cnt] != NULL))
	  {
	    /* _nl_intern_locale_data leaves us these fields to initialize.  */
	    lia->data[cnt]->alloc = ld_archive;
	    lia->data[cnt]->name = lia->name;

	    /* We do this instead of bumping the count each time we return
	       this data because the mappings stay around forever anyway
	       and we might as well hold on to a little more memory and not
	       have to rebuild it on the next lookup of the same thing.
	       If we were to maintain the usage_count normally and let the
	       structures be freed, we would have to remove the elements
	       from archloaded too.  */
	    lia->data[cnt]->usage_count = UNDELETABLE;
	  }
      }

  *namep = lia->name;
  return lia->data[category];
}

void __libc_freeres_fn_section
_nl_archive_subfreeres (void)
{
  struct locale_in_archive *lia;
  struct archmapped *am;

  /* Toss out our cached locales.  */
  lia = archloaded;
  while (lia != NULL)
    {
      int category;
      struct locale_in_archive *dead = lia;
      lia = lia->next;

      free (dead->name);
      for (category = 0; category < __LC_LAST; ++category)
	if (category != LC_ALL && dead->data[category] != NULL)
	  {
	    /* _nl_unload_locale just does this free for the archive case.  */
	    if (dead->data[category]->private.cleanup)
	      (*dead->data[category]->private.cleanup) (dead->data[category]);

	    free (dead->data[category]);
	  }
      free (dead);
    }
  archloaded = NULL;

  if (archmapped != NULL)
    {
      /* Now toss all the mapping windows, which we know nothing is using any
	 more because we just tossed all the locales that point into them.  */

      assert (archmapped == &headmap);
      archmapped = NULL;
      (void) __munmap (headmap.ptr, headmap.len);
      am = headmap.next;
      while (am != NULL)
	{
	  struct archmapped *dead = am;
	  am = am->next;
	  (void) __munmap (dead->ptr, dead->len);
	  free (dead);
	}
    }
}
