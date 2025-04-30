/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2002.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published
   by the Free Software Foundation; version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <https://www.gnu.org/licenses/>.  */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <assert.h>
#include <dirent.h>
#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <inttypes.h>
#include <libintl.h>
#include <locale.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdio_ext.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <sys/shm.h>
#include <sys/stat.h>

#include <libc-mmap.h>
#include <libc-pointer-arith.h>
#include "../../crypt/md5.h"
#include "../localeinfo.h"
#include "../locarchive.h"
#include "localedef.h"
#include "locfile.h"

/* Define the hash function.  We define the function as static inline.
   We must change the name so as not to conflict with simple-hash.h.  */
#define compute_hashval static archive_hashval
#define hashval_t uint32_t
#include "hashval.h"
#undef compute_hashval

extern const char *output_prefix;

#define ARCHIVE_NAME COMPLOCALEDIR "/locale-archive"

static const char *locnames[] =
  {
#define DEFINE_CATEGORY(category, category_name, items, a) \
  [category] = category_name,
#include "categories.def"
#undef  DEFINE_CATEGORY
  };


/* Size of the initial archive header.  */
#define INITIAL_NUM_NAMES	900
#define INITIAL_SIZE_STRINGS	7500
#define INITIAL_NUM_LOCREC	420
#define INITIAL_NUM_SUMS	2000


/* Get and set values (possibly endian-swapped) in structures mapped
   from or written directly to locale archives.  */
#define GET(FIELD)	maybe_swap_uint32 (FIELD)
#define SET(FIELD, VALUE)	((FIELD) = maybe_swap_uint32 (VALUE))
#define INC(FIELD, INCREMENT)	SET (FIELD, GET (FIELD) + (INCREMENT))


/* Size of the reserved address space area.  */
#define RESERVE_MMAP_SIZE	512 * 1024 * 1024

/* To prepare for enlargements of the mmaped area reserve some address
   space.  On some machines, being a file mapping rather than an anonymous
   mapping affects the address selection.  So do this mapping from the
   actual file, even though it's only a dummy to reserve address space.  */
static void *
prepare_address_space (int fd, size_t total, size_t *reserved, int *xflags,
		       void **mmap_base, size_t *mmap_len)
{
  if (total < RESERVE_MMAP_SIZE)
    {
      void *p = mmap64 (NULL, RESERVE_MMAP_SIZE, PROT_NONE, MAP_SHARED, fd, 0);
      if (p != MAP_FAILED)
	{
	  void *aligned_p = PTR_ALIGN_UP (p, MAP_FIXED_ALIGNMENT);
	  size_t align_adjust = aligned_p - p;
	  *mmap_base = p;
	  *mmap_len = RESERVE_MMAP_SIZE;
	  assert (align_adjust < RESERVE_MMAP_SIZE);
	  *reserved = RESERVE_MMAP_SIZE - align_adjust;
	  *xflags = MAP_FIXED;
	  return aligned_p;
	}
    }

  *reserved = total;
  *xflags = 0;
  *mmap_base = NULL;
  *mmap_len = 0;
  return NULL;
}


static void
create_archive (const char *archivefname, struct locarhandle *ah)
{
  int fd;
  char fname[strlen (archivefname) + sizeof (".XXXXXX")];
  struct locarhead head;
  size_t total;

  strcpy (stpcpy (fname, archivefname), ".XXXXXX");

  /* Create a temporary file in the correct directory.  */
  fd = mkstemp (fname);
  if (fd == -1)
    error (EXIT_FAILURE, errno, _("cannot create temporary file: %s"), fname);

  /* Create the initial content of the archive.  */
  SET (head.magic, AR_MAGIC);
  SET (head.serial, 0);
  SET (head.namehash_offset, sizeof (struct locarhead));
  SET (head.namehash_used, 0);
  SET (head.namehash_size, next_prime (INITIAL_NUM_NAMES));

  SET (head.string_offset,
       (GET (head.namehash_offset)
	+ GET (head.namehash_size) * sizeof (struct namehashent)));
  SET (head.string_used, 0);
  SET (head.string_size, INITIAL_SIZE_STRINGS);

  SET (head.locrectab_offset,
       GET (head.string_offset) + GET (head.string_size));
  SET (head.locrectab_used, 0);
  SET (head.locrectab_size, INITIAL_NUM_LOCREC);

  SET (head.sumhash_offset,
       (GET (head.locrectab_offset)
	+ GET (head.locrectab_size) * sizeof (struct locrecent)));
  SET (head.sumhash_used, 0);
  SET (head.sumhash_size, next_prime (INITIAL_NUM_SUMS));

  total = (GET (head.sumhash_offset)
	   + GET (head.sumhash_size) * sizeof (struct sumhashent));

  /* Write out the header and create room for the other data structures.  */
  if (TEMP_FAILURE_RETRY (write (fd, &head, sizeof (head))) != sizeof (head))
    {
      int errval = errno;
      unlink (fname);
      error (EXIT_FAILURE, errval, _("cannot initialize archive file"));
    }

  if (ftruncate64 (fd, total) != 0)
    {
      int errval = errno;
      unlink (fname);
      error (EXIT_FAILURE, errval, _("cannot resize archive file"));
    }

  size_t reserved, mmap_len;
  int xflags;
  void *mmap_base;
  void *p = prepare_address_space (fd, total, &reserved, &xflags, &mmap_base,
				   &mmap_len);

  /* Map the header and all the administration data structures.  */
  p = mmap64 (p, total, PROT_READ | PROT_WRITE, MAP_SHARED | xflags, fd, 0);
  if (p == MAP_FAILED)
    {
      int errval = errno;
      unlink (fname);
      error (EXIT_FAILURE, errval, _("cannot map archive header"));
    }

  /* Now try to rename it.  We don't use the rename function since
     this would overwrite a file which has been created in
     parallel.  */
  if (link (fname, archivefname) == -1)
    {
      int errval = errno;

      /* We cannot use the just created file.  */
      close (fd);
      unlink (fname);

      if (errval == EEXIST)
	{
	  /* There is already an archive.  Must have been a localedef run
	     which happened in parallel.  Simply open this file then.  */
	  open_archive (ah, false);
	  return;
	}

      error (EXIT_FAILURE, errval, _("failed to create new locale archive"));
    }

  /* Remove the temporary name.  */
  unlink (fname);

  /* Make the file globally readable.  */
  if (fchmod (fd, S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH) == -1)
    {
      int errval = errno;
      unlink (archivefname);
      error (EXIT_FAILURE, errval,
	     _("cannot change mode of new locale archive"));
    }

  ah->fname = NULL;
  ah->fd = fd;
  ah->mmap_base = mmap_base;
  ah->mmap_len = mmap_len;
  ah->addr = p;
  ah->mmaped = total;
  ah->reserved = reserved;
}


/* This structure and qsort comparator function are used below to sort an
   old archive's locrec table in order of data position in the file.  */
struct oldlocrecent
{
  unsigned int cnt;
  struct locrecent *locrec;
};

static int
oldlocrecentcmp (const void *a, const void *b)
{
  struct locrecent *la = ((const struct oldlocrecent *) a)->locrec;
  struct locrecent *lb = ((const struct oldlocrecent *) b)->locrec;
  uint32_t start_a = -1, end_a = 0;
  uint32_t start_b = -1, end_b = 0;
  int cnt;

  for (cnt = 0; cnt < __LC_LAST; ++cnt)
    if (cnt != LC_ALL)
      {
	if (GET (la->record[cnt].offset) < start_a)
	  start_a = GET (la->record[cnt].offset);
	if (GET (la->record[cnt].offset) + GET (la->record[cnt].len) > end_a)
	  end_a = GET (la->record[cnt].offset) + GET (la->record[cnt].len);
      }
  assert (start_a != (uint32_t)-1);
  assert (end_a != 0);

  for (cnt = 0; cnt < __LC_LAST; ++cnt)
    if (cnt != LC_ALL)
      {
	if (GET (lb->record[cnt].offset) < start_b)
	  start_b = GET (lb->record[cnt].offset);
	if (GET (lb->record[cnt].offset) + GET (lb->record[cnt].len) > end_b)
	  end_b = GET (lb->record[cnt].offset) + GET (lb->record[cnt].len);
      }
  assert (start_b != (uint32_t)-1);
  assert (end_b != 0);

  if (start_a != start_b)
    return (int)start_a - (int)start_b;
  return (int)end_a - (int)end_b;
}


/* forward decls for below */
static uint32_t add_locale (struct locarhandle *ah, const char *name,
			    locale_data_t data, bool replace);
static void add_alias (struct locarhandle *ah, const char *alias,
		       bool replace, const char *oldname,
		       uint32_t *locrec_offset_p);


static bool
file_data_available_p (struct locarhandle *ah, uint32_t offset, uint32_t size)
{
  if (offset < ah->mmaped && offset + size <= ah->mmaped)
    return true;

  struct stat64 st;
  if (fstat64 (ah->fd, &st) != 0)
    return false;

  if (st.st_size > ah->reserved)
    return false;

  size_t start = ALIGN_DOWN (ah->mmaped, MAP_FIXED_ALIGNMENT);
  void *p = mmap64 (ah->addr + start, st.st_size - start,
		    PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED,
		    ah->fd, start);
  if (p == MAP_FAILED)
    {
      ah->mmaped = start;
      return false;
    }

  ah->mmaped = st.st_size;
  return true;
}


static int
compare_from_file (struct locarhandle *ah, void *p1, uint32_t offset2,
		   uint32_t size)
{
  void *p2 = xmalloc (size);
  if (pread (ah->fd, p2, size, offset2) != size)
    record_error (4, errno,
		  _("cannot read data from locale archive"));

  int res = memcmp (p1, p2, size);
  free (p2);
  return res;
}


static void
enlarge_archive (struct locarhandle *ah, const struct locarhead *head)
{
  struct stat64 st;
  int fd;
  struct locarhead newhead;
  size_t total;
  unsigned int cnt, loccnt;
  struct namehashent *oldnamehashtab;
  struct locarhandle new_ah;
  size_t prefix_len = output_prefix ? strlen (output_prefix) : 0;
  char archivefname[prefix_len + sizeof (ARCHIVE_NAME)];
  char fname[prefix_len + sizeof (ARCHIVE_NAME) + sizeof (".XXXXXX") - 1];

  if (output_prefix)
    memcpy (archivefname, output_prefix, prefix_len);
  strcpy (archivefname + prefix_len, ARCHIVE_NAME);
  strcpy (stpcpy (fname, archivefname), ".XXXXXX");

  /* Not all of the old file has to be mapped.  Change this now this
     we will have to access the whole content.  */
  if (fstat64 (ah->fd, &st) != 0)
  enomap:
    error (EXIT_FAILURE, errno, _("cannot map locale archive file"));

  if (st.st_size < ah->reserved)
    ah->addr = mmap64 (ah->addr, st.st_size, PROT_READ | PROT_WRITE,
		       MAP_SHARED | MAP_FIXED, ah->fd, 0);
  else
    {
      if (ah->mmap_base)
	munmap (ah->mmap_base, ah->mmap_len);
      else
	munmap (ah->addr, ah->reserved);
      ah->addr = mmap64 (NULL, st.st_size, PROT_READ | PROT_WRITE,
			 MAP_SHARED, ah->fd, 0);
      ah->reserved = st.st_size;
      ah->mmap_base = NULL;
      ah->mmap_len = 0;
      head = ah->addr;
    }
  if (ah->addr == MAP_FAILED)
    goto enomap;
  ah->mmaped = st.st_size;

  /* Create a temporary file in the correct directory.  */
  fd = mkstemp (fname);
  if (fd == -1)
    error (EXIT_FAILURE, errno, _("cannot create temporary file: %s"), fname);

  /* Copy the existing head information.  */
  newhead = *head;

  /* Create the new archive header.  The sizes of the various tables
     should be double from what is currently used.  */
  SET (newhead.namehash_size,
       MAX (next_prime (2 * GET (newhead.namehash_used)),
	    GET (newhead.namehash_size)));
  if (verbose)
    printf ("name: size: %u, used: %d, new: size: %u\n",
	    GET (head->namehash_size),
	    GET (head->namehash_used), GET (newhead.namehash_size));

  SET (newhead.string_offset, (GET (newhead.namehash_offset)
			       + (GET (newhead.namehash_size)
				  * sizeof (struct namehashent))));
  /* Keep the string table size aligned to 4 bytes, so that
     all the struct { uint32_t } types following are happy.  */
  SET (newhead.string_size, MAX ((2 * GET (newhead.string_used) + 3) & -4,
				 GET (newhead.string_size)));

  SET (newhead.locrectab_offset,
       GET (newhead.string_offset) + GET (newhead.string_size));
  SET (newhead.locrectab_size, MAX (2 * GET (newhead.locrectab_used),
				    GET (newhead.locrectab_size)));

  SET (newhead.sumhash_offset, (GET (newhead.locrectab_offset)
				+ (GET (newhead.locrectab_size)
				   * sizeof (struct locrecent))));
  SET (newhead.sumhash_size,
       MAX (next_prime (2 * GET (newhead.sumhash_used)),
	    GET (newhead.sumhash_size)));

  total = (GET (newhead.sumhash_offset)
	   + GET (newhead.sumhash_size) * sizeof (struct sumhashent));

  /* The new file is empty now.  */
  SET (newhead.namehash_used, 0);
  SET (newhead.string_used, 0);
  SET (newhead.locrectab_used, 0);
  SET (newhead.sumhash_used, 0);

  /* Write out the header and create room for the other data structures.  */
  if (TEMP_FAILURE_RETRY (write (fd, &newhead, sizeof (newhead)))
      != sizeof (newhead))
    {
      int errval = errno;
      unlink (fname);
      error (EXIT_FAILURE, errval, _("cannot initialize archive file"));
    }

  if (ftruncate64 (fd, total) != 0)
    {
      int errval = errno;
      unlink (fname);
      error (EXIT_FAILURE, errval, _("cannot resize archive file"));
    }

  size_t reserved, mmap_len;
  int xflags;
  void *mmap_base;
  void *p = prepare_address_space (fd, total, &reserved, &xflags, &mmap_base,
				   &mmap_len);

  /* Map the header and all the administration data structures.  */
  p = mmap64 (p, total, PROT_READ | PROT_WRITE, MAP_SHARED | xflags, fd, 0);
  if (p == MAP_FAILED)
    {
      int errval = errno;
      unlink (fname);
      error (EXIT_FAILURE, errval, _("cannot map archive header"));
    }

  /* Lock the new file.  */
  if (lockf64 (fd, F_LOCK, total) != 0)
    {
      int errval = errno;
      unlink (fname);
      error (EXIT_FAILURE, errval, _("cannot lock new archive"));
    }

  new_ah.mmaped = total;
  new_ah.mmap_base = mmap_base;
  new_ah.mmap_len = mmap_len;
  new_ah.addr = p;
  new_ah.fd = fd;
  new_ah.reserved = reserved;

  /* Walk through the hash name hash table to find out what data is
     still referenced and transfer it into the new file.  */
  oldnamehashtab = (struct namehashent *) ((char *) ah->addr
					   + GET (head->namehash_offset));

  /* Sort the old locrec table in order of data position.  */
  struct oldlocrecent oldlocrecarray[GET (head->namehash_size)];
  for (cnt = 0, loccnt = 0; cnt < GET (head->namehash_size); ++cnt)
    if (GET (oldnamehashtab[cnt].locrec_offset) != 0)
      {
	oldlocrecarray[loccnt].cnt = cnt;
	oldlocrecarray[loccnt++].locrec
	  = (struct locrecent *) ((char *) ah->addr
				  + GET (oldnamehashtab[cnt].locrec_offset));
      }
  qsort (oldlocrecarray, loccnt, sizeof (struct oldlocrecent),
	 oldlocrecentcmp);

  uint32_t last_locrec_offset = 0;
  for (cnt = 0; cnt < loccnt; ++cnt)
    {
      /* Insert this entry in the new hash table.  */
      locale_data_t old_data;
      unsigned int idx;
      struct locrecent *oldlocrec = oldlocrecarray[cnt].locrec;

      for (idx = 0; idx < __LC_LAST; ++idx)
	if (idx != LC_ALL)
	  {
	    old_data[idx].size = GET (oldlocrec->record[idx].len);
	    old_data[idx].addr
	      = ((char *) ah->addr + GET (oldlocrec->record[idx].offset));

	    __md5_buffer (old_data[idx].addr, old_data[idx].size,
			  old_data[idx].sum);
	  }

      if (cnt > 0 && oldlocrecarray[cnt - 1].locrec == oldlocrec)
	{
	  const char *oldname
	    = ((char *) ah->addr
	       + GET (oldnamehashtab[oldlocrecarray[cnt
						    - 1].cnt].name_offset));

	  add_alias
	    (&new_ah,
	     ((char *) ah->addr
	      + GET (oldnamehashtab[oldlocrecarray[cnt].cnt].name_offset)),
	     0, oldname, &last_locrec_offset);
	  continue;
	}

      last_locrec_offset =
	add_locale
	(&new_ah,
	 ((char *) ah->addr
	  + GET (oldnamehashtab[oldlocrecarray[cnt].cnt].name_offset)),
	 old_data, 0);
      if (last_locrec_offset == 0)
	error (EXIT_FAILURE, 0, _("cannot extend locale archive file"));
    }

  /* Make the file globally readable.  */
  if (fchmod (fd, S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH) == -1)
    {
      int errval = errno;
      unlink (fname);
      error (EXIT_FAILURE, errval,
	     _("cannot change mode of resized locale archive"));
    }

  /* Rename the new file.  */
  if (rename (fname, archivefname) != 0)
    {
      int errval = errno;
      unlink (fname);
      error (EXIT_FAILURE, errval, _("cannot rename new archive"));
    }

  /* Close the old file.  */
  close_archive (ah);

  /* Add the information for the new one.  */
  *ah = new_ah;
}


void
open_archive (struct locarhandle *ah, bool readonly)
{
  struct stat64 st;
  struct stat64 st2;
  int fd;
  struct locarhead head;
  int retry = 0;
  size_t prefix_len = output_prefix ? strlen (output_prefix) : 0;
  char default_fname[prefix_len + sizeof (ARCHIVE_NAME)];
  const char *archivefname = ah->fname;

  /* If ah has a non-NULL fname open that otherwise open the default.  */
  if (archivefname == NULL)
    {
      archivefname = default_fname;
      if (output_prefix)
        memcpy (default_fname, output_prefix, prefix_len);
      strcpy (default_fname + prefix_len, ARCHIVE_NAME);
    }

  while (1)
    {
      /* Open the archive.  We must have exclusive write access.  */
      fd = open64 (archivefname, readonly ? O_RDONLY : O_RDWR);
      if (fd == -1)
	{
	  /* Maybe the file does not yet exist? If we are opening
	     the default locale archive we ignore the failure and
	     list an empty archive, otherwise we print an error
	     and exit.  */
	  if (errno == ENOENT && archivefname == default_fname)
	    {
	      if (readonly)
		{
		  static const struct locarhead nullhead =
		    {
		      .namehash_used = 0,
		      .namehash_offset = 0,
		      .namehash_size = 0
		    };

		  ah->addr = (void *) &nullhead;
		  ah->fd = -1;
		}
	      else
		create_archive (archivefname, ah);

	      return;
	    }
	  else
	    error (EXIT_FAILURE, errno, _("cannot open locale archive \"%s\""),
		   archivefname);
	}

      if (fstat64 (fd, &st) < 0)
	error (EXIT_FAILURE, errno, _("cannot stat locale archive \"%s\""),
	       archivefname);

      if (!readonly && lockf64 (fd, F_LOCK, sizeof (struct locarhead)) == -1)
	{
	  close (fd);

	  if (retry++ < max_locarchive_open_retry)
	    {
	      struct timespec req;

	      /* Wait for a bit.  */
	      req.tv_sec = 0;
	      req.tv_nsec = 1000000 * (random () % 500 + 1);
	      (void) nanosleep (&req, NULL);

	      continue;
	    }

	  error (EXIT_FAILURE, errno, _("cannot lock locale archive \"%s\""),
		 archivefname);
	}

      /* One more check.  Maybe another process replaced the archive file
	 with a new, larger one since we opened the file.  */
      if (stat64 (archivefname, &st2) == -1
	  || st.st_dev != st2.st_dev
	  || st.st_ino != st2.st_ino)
	{
	  (void) lockf64 (fd, F_ULOCK, sizeof (struct locarhead));
	  close (fd);
	  continue;
	}

      /* Leave the loop.  */
      break;
    }

  /* Read the header.  */
  if (TEMP_FAILURE_RETRY (read (fd, &head, sizeof (head))) != sizeof (head))
    {
      (void) lockf64 (fd, F_ULOCK, sizeof (struct locarhead));
      error (EXIT_FAILURE, errno, _("cannot read archive header"));
    }

  ah->fd = fd;
  ah->mmaped = st.st_size;

  size_t reserved, mmap_len;
  int xflags;
  void *mmap_base;
  void *p = prepare_address_space (fd, st.st_size, &reserved, &xflags,
				   &mmap_base, &mmap_len);

  /* Map the entire file.  We might need to compare the category data
     in the file with the newly added data.  */
  ah->addr = mmap64 (p, st.st_size, PROT_READ | (readonly ? 0 : PROT_WRITE),
		     MAP_SHARED | xflags, fd, 0);
  if (ah->addr == MAP_FAILED)
    {
      (void) lockf64 (fd, F_ULOCK, sizeof (struct locarhead));
      error (EXIT_FAILURE, errno, _("cannot map archive header"));
    }
  ah->reserved = reserved;
  ah->mmap_base = mmap_base;
  ah->mmap_len = mmap_len;
}


void
close_archive (struct locarhandle *ah)
{
  if (ah->fd != -1)
    {
      if (ah->mmap_base)
	munmap (ah->mmap_base, ah->mmap_len);
      else
	munmap (ah->addr, ah->reserved);
      close (ah->fd);
    }
}

#include "../../intl/explodename.c"
#include "../../intl/l10nflist.c"

static struct namehashent *
insert_name (struct locarhandle *ah,
	     const char *name, size_t name_len, bool replace)
{
  const struct locarhead *const head = ah->addr;
  struct namehashent *namehashtab
    = (struct namehashent *) ((char *) ah->addr
			      + GET (head->namehash_offset));
  unsigned int insert_idx, idx, incr;

  /* Hash value of the locale name.  */
  uint32_t hval = archive_hashval (name, name_len);

  insert_idx = -1;
  idx = hval % GET (head->namehash_size);
  incr = 1 + hval % (GET (head->namehash_size) - 2);

  /* If the name_offset field is zero this means this is a
     deleted entry and therefore no entry can be found.  */
  while (GET (namehashtab[idx].name_offset) != 0)
    {
      if (GET (namehashtab[idx].hashval) == hval
	  && (strcmp (name,
		      (char *) ah->addr + GET (namehashtab[idx].name_offset))
	      == 0))
	{
	  /* Found the entry.  */
	  if (GET (namehashtab[idx].locrec_offset) != 0 && ! replace)
	    {
	      if (! be_quiet)
		error (0, 0, _("locale '%s' already exists"), name);
	      return NULL;
	    }

	  break;
	}

      if (GET (namehashtab[idx].hashval) == hval && ! be_quiet)
	{
	  error (0, 0, "hash collision (%u) %s, %s",
		 hval, name,
		 (char *) ah->addr + GET (namehashtab[idx].name_offset));
	}

      /* Remember the first place we can insert the new entry.  */
      if (GET (namehashtab[idx].locrec_offset) == 0 && insert_idx == -1)
	insert_idx = idx;

      idx += incr;
      if (idx >= GET (head->namehash_size))
	idx -= GET (head->namehash_size);
    }

  /* Add as early as possible.  */
  if (insert_idx != -1)
    idx = insert_idx;

  SET (namehashtab[idx].hashval, hval); /* no-op if replacing an old entry.  */
  return &namehashtab[idx];
}

static void
add_alias (struct locarhandle *ah, const char *alias, bool replace,
	   const char *oldname, uint32_t *locrec_offset_p)
{
  uint32_t locrec_offset = *locrec_offset_p;
  struct locarhead *head = ah->addr;
  const size_t name_len = strlen (alias);
  struct namehashent *namehashent = insert_name (ah, alias, strlen (alias),
						 replace);
  if (namehashent == NULL && ! replace)
    return;

  if (GET (namehashent->name_offset) == 0)
    {
      /* We are adding a new hash entry for this alias.
	 Determine whether we have to resize the file.  */
      if (GET (head->string_used) + name_len + 1 > GET (head->string_size)
	  || (100 * GET (head->namehash_used)
	      > 75 * GET (head->namehash_size)))
	{
	  /* The current archive is not large enough.  */
	  enlarge_archive (ah, head);

	  /* The locrecent might have moved, so we have to look up
	     the old name afresh.  */
	  namehashent = insert_name (ah, oldname, strlen (oldname), true);
	  assert (GET (namehashent->name_offset) != 0);
	  assert (GET (namehashent->locrec_offset) != 0);
	  *locrec_offset_p = GET (namehashent->locrec_offset);

	  /* Tail call to try the whole thing again.  */
	  add_alias (ah, alias, replace, oldname, locrec_offset_p);
	  return;
	}

      /* Add the name string.  */
      memcpy (ah->addr + GET (head->string_offset) + GET (head->string_used),
	      alias, name_len + 1);
      SET (namehashent->name_offset,
	   GET (head->string_offset) + GET (head->string_used));
      INC (head->string_used, name_len + 1);

      INC (head->namehash_used, 1);
    }

  if (GET (namehashent->locrec_offset) != 0)
    {
      /* Replacing an existing entry.
	 Mark that we are no longer using the old locrecent.  */
      struct locrecent *locrecent
	= (struct locrecent *) ((char *) ah->addr
				+ GET (namehashent->locrec_offset));
      INC (locrecent->refs, -1);
    }

  /* Point this entry at the locrecent installed for the main name.  */
  SET (namehashent->locrec_offset, locrec_offset);
}

static int			/* qsort comparator used below */
cmpcategorysize (const void *a, const void *b)
{
  if (*(const void **) a == NULL)
    return 1;
  if (*(const void **) b == NULL)
    return -1;
  return ((*(const struct locale_category_data **) a)->size
	  - (*(const struct locale_category_data **) b)->size);
}

/* Check the content of the archive for duplicates.  Add the content
   of the files if necessary.  Returns the locrec_offset.  */
static uint32_t
add_locale (struct locarhandle *ah,
	    const char *name, locale_data_t data, bool replace)
{
  /* First look for the name.  If it already exists and we are not
     supposed to replace it don't do anything.  If it does not exist
     we have to allocate a new locale record.  */
  size_t name_len = strlen (name);
  uint32_t file_offsets[__LC_LAST];
  unsigned int num_new_offsets = 0;
  struct sumhashent *sumhashtab;
  uint32_t hval;
  unsigned int cnt, idx;
  struct locarhead *head;
  struct namehashent *namehashent;
  unsigned int incr;
  struct locrecent *locrecent;
  off64_t lastoffset;
  char *ptr;
  struct locale_category_data *size_order[__LC_LAST];
  /* Page size alignment is a minor optimization for locality; use a
     common value here rather than making the localedef output depend
     on the page size of the system on which localedef is run.  See
     <https://sourceware.org/glibc/wiki/Development_Todo/Master#Locale_archive_alignment>
     for more discussion.  */
  const size_t pagesz = 4096;
  int small_mask;

  head = ah->addr;
  sumhashtab = (struct sumhashent *) ((char *) ah->addr
				      + GET (head->sumhash_offset));

  memset (file_offsets, 0, sizeof (file_offsets));

  size_order[LC_ALL] = NULL;
  for (cnt = 0; cnt < __LC_LAST; ++cnt)
    if (cnt != LC_ALL)
      size_order[cnt] = &data[cnt];

  /* Sort the array in ascending order of data size.  */
  qsort (size_order, __LC_LAST, sizeof size_order[0], cmpcategorysize);

  small_mask = 0;
  data[LC_ALL].size = 0;
  for (cnt = 0; cnt < __LC_LAST; ++cnt)
    if (size_order[cnt] != NULL)
      {
	const size_t rounded_size = (size_order[cnt]->size + 15) & -16;
	if (data[LC_ALL].size + rounded_size > 2 * pagesz)
	  {
	    /* This category makes the small-categories block
	       stop being small, so this is the end of the road.  */
	    do
	      size_order[cnt++] = NULL;
	    while (cnt < __LC_LAST);
	    break;
	  }
	data[LC_ALL].size += rounded_size;
	small_mask |= 1 << (size_order[cnt] - data);
      }

  /* Copy the data for all the small categories into the LC_ALL
     pseudo-category.  */

  data[LC_ALL].addr = alloca (data[LC_ALL].size);
  memset (data[LC_ALL].addr, 0, data[LC_ALL].size);

  ptr = data[LC_ALL].addr;
  for (cnt = 0; cnt < __LC_LAST; ++cnt)
    if (small_mask & (1 << cnt))
      {
	memcpy (ptr, data[cnt].addr, data[cnt].size);
	ptr += (data[cnt].size + 15) & -16;
      }
  __md5_buffer (data[LC_ALL].addr, data[LC_ALL].size, data[LC_ALL].sum);

  /* For each locale category data set determine whether the same data
     is already somewhere in the archive.  */
  for (cnt = 0; cnt < __LC_LAST; ++cnt)
    if (small_mask == 0 ? cnt != LC_ALL : !(small_mask & (1 << cnt)))
      {
	++num_new_offsets;

	/* Compute the hash value of the checksum to determine a
	   starting point for the search in the MD5 hash value
	   table.  */
	hval = archive_hashval (data[cnt].sum, 16);

	idx = hval % GET (head->sumhash_size);
	incr = 1 + hval % (GET (head->sumhash_size) - 2);

	while (GET (sumhashtab[idx].file_offset) != 0)
	  {
	    if (memcmp (data[cnt].sum, sumhashtab[idx].sum, 16) == 0)
	      {
		/* Check the content, there could be a collision of
		   the hash sum.

		   Unfortunately the sumhashent record does not include
		   the size of the stored data.  So we have to search for
		   it.  */
		locrecent
		  = (struct locrecent *) ((char *) ah->addr
					  + GET (head->locrectab_offset));
		size_t iloc;
		for (iloc = 0; iloc < GET (head->locrectab_used); ++iloc)
		  if (GET (locrecent[iloc].refs) != 0
		      && (GET (locrecent[iloc].record[cnt].offset)
			  == GET (sumhashtab[idx].file_offset)))
		    break;

		if (iloc != GET (head->locrectab_used)
		    && data[cnt].size == GET (locrecent[iloc].record[cnt].len)
		    /* We have to compare the content.  Either we can
		       have the data mmaped or we have to read from
		       the file.  */
		    && (file_data_available_p
			(ah, GET (sumhashtab[idx].file_offset),
			 data[cnt].size)
			? memcmp (data[cnt].addr,
				  (char *) ah->addr
				  + GET (sumhashtab[idx].file_offset),
				  data[cnt].size) == 0
			: compare_from_file (ah, data[cnt].addr,
					     GET (sumhashtab[idx].file_offset),
					     data[cnt].size) == 0))
		  {
		    /* Found it.  */
		    file_offsets[cnt] = GET (sumhashtab[idx].file_offset);
		    --num_new_offsets;
		    break;
		  }
	      }

	    idx += incr;
	    if (idx >= GET (head->sumhash_size))
	      idx -= GET (head->sumhash_size);
	  }
      }

  /* Find a slot for the locale name in the hash table.  */
  namehashent = insert_name (ah, name, name_len, replace);
  if (namehashent == NULL)	/* Already exists and !REPLACE.  */
    return 0;

  /* Determine whether we have to resize the file.  */
  if ((100 * (GET (head->sumhash_used) + num_new_offsets)
       > 75 * GET (head->sumhash_size))
      || (GET (namehashent->locrec_offset) == 0
	  && (GET (head->locrectab_used) == GET (head->locrectab_size)
	      || (GET (head->string_used) + name_len + 1
		  > GET (head->string_size))
	      || (100 * GET (head->namehash_used)
		  > 75 * GET (head->namehash_size)))))
    {
      /* The current archive is not large enough.  */
      enlarge_archive (ah, head);
      return add_locale (ah, name, data, replace);
    }

  /* Add the locale data which is not yet in the archive.  */
  for (cnt = 0, lastoffset = 0; cnt < __LC_LAST; ++cnt)
    if ((small_mask == 0 ? cnt != LC_ALL : !(small_mask & (1 << cnt)))
	&& file_offsets[cnt] == 0)
      {
	/* The data for this section is not yet available in the
	   archive.  Append it.  */
	off64_t lastpos;
	uint32_t md5hval;

	lastpos = lseek64 (ah->fd, 0, SEEK_END);
	if (lastpos == (off64_t) -1)
	  error (EXIT_FAILURE, errno, _("cannot add to locale archive"));

	/* If block of small categories would cross page boundary,
	   align it unless it immediately follows a large category.  */
	if (cnt == LC_ALL && lastoffset != lastpos
	    && ((((lastpos & (pagesz - 1)) + data[cnt].size + pagesz - 1)
		 & -pagesz)
		> ((data[cnt].size + pagesz - 1) & -pagesz)))
	  {
	    size_t sz = pagesz - (lastpos & (pagesz - 1));
	    char *zeros = alloca (sz);

	    memset (zeros, 0, sz);
	    if (TEMP_FAILURE_RETRY (write (ah->fd, zeros, sz) != sz))
	      error (EXIT_FAILURE, errno,
		     _("cannot add to locale archive"));

	    lastpos += sz;
	  }

	/* Align all data to a 16 byte boundary.  */
	if ((lastpos & 15) != 0)
	  {
	    static const char zeros[15] = { 0, };

	    if (TEMP_FAILURE_RETRY (write (ah->fd, zeros, 16 - (lastpos & 15)))
		!= 16 - (lastpos & 15))
	      error (EXIT_FAILURE, errno, _("cannot add to locale archive"));

	    lastpos += 16 - (lastpos & 15);
	  }

	/* Remember the position.  */
	file_offsets[cnt] = lastpos;
	lastoffset = lastpos + data[cnt].size;

	/* Write the data.  */
	if (TEMP_FAILURE_RETRY (write (ah->fd, data[cnt].addr, data[cnt].size))
	    != data[cnt].size)
	  error (EXIT_FAILURE, errno, _("cannot add to locale archive"));

	/* Add the hash value to the hash table.  */
	md5hval = archive_hashval (data[cnt].sum, 16);

	idx = md5hval % GET (head->sumhash_size);
	incr = 1 + md5hval % (GET (head->sumhash_size) - 2);

	while (GET (sumhashtab[idx].file_offset) != 0)
	  {
	    idx += incr;
	    if (idx >= GET (head->sumhash_size))
	      idx -= GET (head->sumhash_size);
	  }

	memcpy (sumhashtab[idx].sum, data[cnt].sum, 16);
	SET (sumhashtab[idx].file_offset, file_offsets[cnt]);

	INC (head->sumhash_used, 1);
      }

  lastoffset = file_offsets[LC_ALL];
  for (cnt = 0; cnt < __LC_LAST; ++cnt)
    if (small_mask & (1 << cnt))
      {
	file_offsets[cnt] = lastoffset;
	lastoffset += (data[cnt].size + 15) & -16;
      }

  if (GET (namehashent->name_offset) == 0)
    {
      /* Add the name string.  */
      memcpy ((char *) ah->addr + GET (head->string_offset)
	      + GET (head->string_used),
	      name, name_len + 1);
      SET (namehashent->name_offset,
	   GET (head->string_offset) + GET (head->string_used));
      INC (head->string_used, name_len + 1);
      INC (head->namehash_used, 1);
    }

  if (GET (namehashent->locrec_offset == 0))
    {
      /* Allocate a name location record.  */
      SET (namehashent->locrec_offset, (GET (head->locrectab_offset)
					+ (GET (head->locrectab_used)
					   * sizeof (struct locrecent))));
      INC (head->locrectab_used, 1);
      locrecent = (struct locrecent *) ((char *) ah->addr
					+ GET (namehashent->locrec_offset));
      SET (locrecent->refs, 1);
    }
  else
    {
      /* If there are other aliases pointing to this locrecent,
	 we still need a new one.  If not, reuse the old one.  */

      locrecent = (struct locrecent *) ((char *) ah->addr
					+ GET (namehashent->locrec_offset));
      if (GET (locrecent->refs) > 1)
	{
	  INC (locrecent->refs, -1);
	  SET (namehashent->locrec_offset, (GET (head->locrectab_offset)
					    + (GET (head->locrectab_used)
					       * sizeof (struct locrecent))));
	  INC (head->locrectab_used, 1);
	  locrecent
	    = (struct locrecent *) ((char *) ah->addr
				    + GET (namehashent->locrec_offset));
	  SET (locrecent->refs, 1);
	}
    }

  /* Fill in the table with the locations of the locale data.  */
  for (cnt = 0; cnt < __LC_LAST; ++cnt)
    {
      SET (locrecent->record[cnt].offset, file_offsets[cnt]);
      SET (locrecent->record[cnt].len, data[cnt].size);
    }

  return GET (namehashent->locrec_offset);
}


/* Check the content of the archive for duplicates.  Add the content
   of the files if necessary.  Add all the names, possibly overwriting
   old files.  */
int
add_locale_to_archive (struct locarhandle *ah, const char *name,
		       locale_data_t data, bool replace)
{
  char *normalized_name = NULL;
  uint32_t locrec_offset;

  /* First analyze the name to decide how to archive it.  */
  const char *language;
  const char *modifier;
  const char *territory;
  const char *codeset;
  const char *normalized_codeset;
  int mask = _nl_explode_name (strdupa (name),
			       &language, &modifier, &territory,
			       &codeset, &normalized_codeset);
  if (mask == -1)
    return -1;

  if (mask & XPG_NORM_CODESET)
    /* This name contains a codeset in unnormalized form.
       We will store it in the archive with a normalized name.  */
    asprintf (&normalized_name, "%s%s%s.%s%s%s",
	      language, territory == NULL ? "" : "_", territory ?: "",
	      (mask & XPG_NORM_CODESET) ? normalized_codeset : codeset,
	      modifier == NULL ? "" : "@", modifier ?: "");

  /* This call does the main work.  */
  locrec_offset = add_locale (ah, normalized_name ?: name, data, replace);
  if (locrec_offset == 0)
    {
      free (normalized_name);
      if (mask & XPG_NORM_CODESET)
	free ((char *) normalized_codeset);
      return -1;
    }

  if ((mask & XPG_CODESET) == 0)
    {
      /* This name lacks a codeset, so determine the locale's codeset and
	 add an alias for its name with normalized codeset appended.  */

      const struct
      {
	unsigned int magic;
	unsigned int nstrings;
	unsigned int strindex[0];
      } *filedata = data[LC_CTYPE].addr;
      codeset = (char *) filedata
	+ maybe_swap_uint32 (filedata->strindex[_NL_ITEM_INDEX
						(_NL_CTYPE_CODESET_NAME)]);
      char *normalized_codeset_name = NULL;

      normalized_codeset = _nl_normalize_codeset (codeset, strlen (codeset));
      mask |= XPG_NORM_CODESET;

      asprintf (&normalized_codeset_name, "%s%s%s.%s%s%s",
		language, territory == NULL ? "" : "_", territory ?: "",
		normalized_codeset,
		modifier == NULL ? "" : "@", modifier ?: "");

      add_alias (ah, normalized_codeset_name, replace,
		 normalized_name ?: name, &locrec_offset);
      free (normalized_codeset_name);
    }

  /* Now read the locale.alias files looking for lines whose
     right hand side matches our name after normalization.  */
  int result = 0;
  if (alias_file != NULL)
    {
      FILE *fp;
      fp = fopen (alias_file, "rm");
      if (fp == NULL)
	error (1, errno, _("locale alias file `%s' not found"),
	       alias_file);

      /* No threads present.  */
      __fsetlocking (fp, FSETLOCKING_BYCALLER);

      while (! feof_unlocked (fp))
	{
	  /* It is a reasonable approach to use a fix buffer here
	     because
	     a) we are only interested in the first two fields
	     b) these fields must be usable as file names and so must
	     not be that long  */
	  char buf[BUFSIZ];
	  char *alias;
	  char *value;
	  char *cp;

	  if (fgets_unlocked (buf, BUFSIZ, fp) == NULL)
	    /* EOF reached.  */
	    break;

	  cp = buf;
	  /* Ignore leading white space.  */
	  while (isspace (cp[0]) && cp[0] != '\n')
	    ++cp;

	  /* A leading '#' signals a comment line.  */
	  if (cp[0] != '\0' && cp[0] != '#' && cp[0] != '\n')
	    {
	      alias = cp++;
	      while (cp[0] != '\0' && !isspace (cp[0]))
		++cp;
	      /* Terminate alias name.  */
	      if (cp[0] != '\0')
		*cp++ = '\0';

	      /* Now look for the beginning of the value.  */
	      while (isspace (cp[0]))
		++cp;

	      if (cp[0] != '\0')
		{
		  value = cp++;
		  while (cp[0] != '\0' && !isspace (cp[0]))
		    ++cp;
		  /* Terminate value.  */
		  if (cp[0] == '\n')
		    {
		      /* This has to be done to make the following
			 test for the end of line possible.  We are
			 looking for the terminating '\n' which do not
			 overwrite here.  */
		      *cp++ = '\0';
		      *cp = '\n';
		    }
		  else if (cp[0] != '\0')
		    *cp++ = '\0';

		  /* Does this alias refer to our locale?  We will
		     normalize the right hand side and compare the
		     elements of the normalized form.  */
		  {
		    const char *rhs_language;
		    const char *rhs_modifier;
		    const char *rhs_territory;
		    const char *rhs_codeset;
		    const char *rhs_normalized_codeset;
		    int rhs_mask = _nl_explode_name (value,
						     &rhs_language,
						     &rhs_modifier,
						     &rhs_territory,
						     &rhs_codeset,
						     &rhs_normalized_codeset);
		    if (rhs_mask == -1)
		      {
			result = -1;
			goto out;
		      }
		    if (!strcmp (language, rhs_language)
			&& ((rhs_mask & XPG_CODESET)
			    /* He has a codeset, it must match normalized.  */
			    ? !strcmp ((mask & XPG_NORM_CODESET)
				       ? normalized_codeset : codeset,
				       (rhs_mask & XPG_NORM_CODESET)
				       ? rhs_normalized_codeset : rhs_codeset)
			    /* He has no codeset, we must also have none.  */
			    : (mask & XPG_CODESET) == 0)
			/* Codeset (or lack thereof) matches.  */
			&& !strcmp (territory ?: "", rhs_territory ?: "")
			&& !strcmp (modifier ?: "", rhs_modifier ?: ""))
		      /* We have a winner.  */
		      add_alias (ah, alias, replace,
				 normalized_name ?: name, &locrec_offset);
		    if (rhs_mask & XPG_NORM_CODESET)
		      free ((char *) rhs_normalized_codeset);
		  }
		}
	    }

	  /* Possibly not the whole line fits into the buffer.
	     Ignore the rest of the line.  */
	  while (strchr (cp, '\n') == NULL)
	    {
	      cp = buf;
	      if (fgets_unlocked (buf, BUFSIZ, fp) == NULL)
		/* Make sure the inner loop will be left.  The outer
		   loop will exit at the `feof' test.  */
		*cp = '\n';
	    }
	}

    out:
      fclose (fp);
    }

  free (normalized_name);

  if (mask & XPG_NORM_CODESET)
    free ((char *) normalized_codeset);

  return result;
}


int
add_locales_to_archive (size_t nlist, char *list[], bool replace)
{
  struct locarhandle ah;
  int result = 0;

  /* Open the archive.  This call never returns if we cannot
     successfully open the archive.  */
  ah.fname = NULL;
  open_archive (&ah, false);

  while (nlist-- > 0)
    {
      const char *fname = *list++;
      size_t fnamelen = strlen (fname);
      struct stat64 st;
      DIR *dirp;
      struct dirent64 *d;
      int seen;
      locale_data_t data;
      int cnt;

      if (! be_quiet)
	printf (_("Adding %s\n"), fname);

      /* First see whether this really is a directory and whether it
	 contains all the require locale category files.  */
      if (stat64 (fname, &st) < 0)
	{
	  error (0, 0, _("stat of \"%s\" failed: %s: ignored"), fname,
		 strerror (errno));
	  continue;
	}
      if (!S_ISDIR (st.st_mode))
	{
	  error (0, 0, _("\"%s\" is no directory; ignored"), fname);
	  continue;
	}

      dirp = opendir (fname);
      if (dirp == NULL)
	{
	  error (0, 0, _("cannot open directory \"%s\": %s: ignored"),
		 fname, strerror (errno));
	  continue;
	}

      seen = 0;
      while ((d = readdir64 (dirp)) != NULL)
	{
	  for (cnt = 0; cnt < __LC_LAST; ++cnt)
	    if (cnt != LC_ALL)
	      if (strcmp (d->d_name, locnames[cnt]) == 0)
		{
		  unsigned char d_type;

		  /* We have an object of the required name.  If it's
		     a directory we have to look at a file with the
		     prefix "SYS_".  Otherwise we have found what we
		     are looking for.  */
		  d_type = d->d_type;

		  if (d_type != DT_REG)
		    {
		      char fullname[fnamelen + 2 * strlen (d->d_name) + 7];

		      if (d_type == DT_UNKNOWN)
			{
			  strcpy (stpcpy (stpcpy (fullname, fname), "/"),
				  d->d_name);

			  if (stat64 (fullname, &st) == -1)
			    /* We cannot stat the file, ignore it.  */
			    break;

			  d_type = IFTODT (st.st_mode);
			}

		      if (d_type == DT_DIR)
			{
			  /* We have to do more tests.  The file is a
			     directory and it therefore must contain a
			     regular file with the same name except a
			     "SYS_" prefix.  */
			  char *t = stpcpy (stpcpy (fullname, fname), "/");
			  strcpy (stpcpy (stpcpy (t, d->d_name), "/SYS_"),
				  d->d_name);

			  if (stat64 (fullname, &st) == -1)
			    /* There is no SYS_* file or we cannot
			       access it.  */
			    break;

			  d_type = IFTODT (st.st_mode);
			}
		    }

		  /* If we found a regular file (eventually after
		     following a symlink) we are successful.  */
		  if (d_type == DT_REG)
		    ++seen;
		  break;
		}
	}

      closedir (dirp);

      if (seen != __LC_LAST - 1)
	{
	  /* We don't have all locale category files.  Ignore the name.  */
	  error (0, 0, _("incomplete set of locale files in \"%s\""),
		 fname);
	  continue;
	}

      /* Add the files to the archive.  To do this we first compute
	 sizes and the MD5 sums of all the files.  */
      for (cnt = 0; cnt < __LC_LAST; ++cnt)
	if (cnt != LC_ALL)
	  {
	    char fullname[fnamelen + 2 * strlen (locnames[cnt]) + 7];
	    int fd;

	    strcpy (stpcpy (stpcpy (fullname, fname), "/"), locnames[cnt]);
	    fd = open64 (fullname, O_RDONLY);
	    if (fd == -1 || fstat64 (fd, &st) == -1)
	      {
		/* Cannot read the file.  */
		if (fd != -1)
		  close (fd);
		break;
	      }

	    if (S_ISDIR (st.st_mode))
	      {
		char *t;
		close (fd);
		t = stpcpy (stpcpy (fullname, fname), "/");
		strcpy (stpcpy (stpcpy (t, locnames[cnt]), "/SYS_"),
			locnames[cnt]);

		fd = open64 (fullname, O_RDONLY);
		if (fd == -1 || fstat64 (fd, &st) == -1
		    || !S_ISREG (st.st_mode))
		  {
		    if (fd != -1)
		      close (fd);
		    break;
		  }
	      }

	    /* Map the file.  */
	    data[cnt].addr = mmap64 (NULL, st.st_size, PROT_READ, MAP_SHARED,
				     fd, 0);
	    if (data[cnt].addr == MAP_FAILED)
	      {
		/* Cannot map it.  */
		close (fd);
		break;
	      }

	    data[cnt].size = st.st_size;
	    __md5_buffer (data[cnt].addr, st.st_size, data[cnt].sum);

	    /* We don't need the file descriptor anymore.  */
	    close (fd);
	  }

      if (cnt != __LC_LAST)
	{
	  while (cnt-- > 0)
	    if (cnt != LC_ALL)
	      munmap (data[cnt].addr, data[cnt].size);

	  error (0, 0, _("cannot read all files in \"%s\": ignored"), fname);

	  continue;
	}

      result |= add_locale_to_archive (&ah, basename (fname), data, replace);

      for (cnt = 0; cnt < __LC_LAST; ++cnt)
	if (cnt != LC_ALL)
	  munmap (data[cnt].addr, data[cnt].size);
    }

  /* We are done.  */
  close_archive (&ah);

  return result;
}


int
delete_locales_from_archive (size_t nlist, char *list[])
{
  struct locarhandle ah;
  struct locarhead *head;
  struct namehashent *namehashtab;

  /* Open the archive.  This call never returns if we cannot
     successfully open the archive.  */
  ah.fname = NULL;
  open_archive (&ah, false);

  head = ah.addr;
  namehashtab = (struct namehashent *) ((char *) ah.addr
					+ GET (head->namehash_offset));

  while (nlist-- > 0)
    {
      const char *locname = *list++;
      uint32_t hval;
      unsigned int idx;
      unsigned int incr;

      /* Search for this locale in the archive.  */
      hval = archive_hashval (locname, strlen (locname));

      idx = hval % GET (head->namehash_size);
      incr = 1 + hval % (GET (head->namehash_size) - 2);

      /* If the name_offset field is zero this means this is no
	 deleted entry and therefore no entry can be found.  */
      while (GET (namehashtab[idx].name_offset) != 0)
	{
	  if (GET (namehashtab[idx].hashval) == hval
	      && (strcmp (locname,
			  ((char *) ah.addr
			   + GET (namehashtab[idx].name_offset)))
		  == 0))
	    {
	      /* Found the entry.  Now mark it as removed by zero-ing
		 the reference to the locale record.  */
	      SET (namehashtab[idx].locrec_offset, 0);
	      break;
	    }

	  idx += incr;
	  if (idx >= GET (head->namehash_size))
	    idx -= GET (head->namehash_size);
	}

      if (GET (namehashtab[idx].name_offset) == 0 && ! be_quiet)
	error (0, 0, _("locale \"%s\" not in archive"), locname);
    }

  close_archive (&ah);

  return 0;
}


struct nameent
{
  char *name;
  uint32_t locrec_offset;
};


struct dataent
{
  const unsigned char *sum;
  uint32_t file_offset;
  uint32_t nlink;
};


static int
nameentcmp (const void *a, const void *b)
{
  return strcmp (((const struct nameent *) a)->name,
		 ((const struct nameent *) b)->name);
}


static int
dataentcmp (const void *a, const void *b)
{
  if (((const struct dataent *) a)->file_offset
      < ((const struct dataent *) b)->file_offset)
    return -1;

  if (((const struct dataent *) a)->file_offset
      > ((const struct dataent *) b)->file_offset)
    return 1;

  return 0;
}


void
show_archive_content (const char *fname, int verbose)
{
  struct locarhandle ah;
  struct locarhead *head;
  struct namehashent *namehashtab;
  struct nameent *names;
  size_t cnt, used;

  /* Open the archive.  This call never returns if we cannot
     successfully open the archive.  */
  ah.fname = fname;
  open_archive (&ah, true);

  head = ah.addr;

  names = (struct nameent *) xmalloc (GET (head->namehash_used)
				      * sizeof (struct nameent));

  namehashtab = (struct namehashent *) ((char *) ah.addr
					+ GET (head->namehash_offset));
  for (cnt = used = 0; cnt < GET (head->namehash_size); ++cnt)
    if (GET (namehashtab[cnt].locrec_offset) != 0)
      {
	assert (used < GET (head->namehash_used));
	names[used].name = ah.addr + GET (namehashtab[cnt].name_offset);
	names[used++].locrec_offset = GET (namehashtab[cnt].locrec_offset);
      }

  /* Sort the names.  */
  qsort (names, used, sizeof (struct nameent), nameentcmp);

  if (verbose)
    {
      struct dataent *files;
      struct sumhashent *sumhashtab;
      int sumused;

      files = (struct dataent *) xmalloc (GET (head->sumhash_used)
					  * sizeof (struct dataent));

      sumhashtab = (struct sumhashent *) ((char *) ah.addr
					  + GET (head->sumhash_offset));
      for (cnt = sumused = 0; cnt < GET (head->sumhash_size); ++cnt)
	if (GET (sumhashtab[cnt].file_offset) != 0)
	  {
	    assert (sumused < GET (head->sumhash_used));
	    files[sumused].sum = (const unsigned char *) sumhashtab[cnt].sum;
	    files[sumused].file_offset = GET (sumhashtab[cnt].file_offset);
	    files[sumused++].nlink = 0;
	  }

      /* Sort by file locations.  */
      qsort (files, sumused, sizeof (struct dataent), dataentcmp);

      /* Compute nlink fields.  */
      for (cnt = 0; cnt < used; ++cnt)
	{
	  struct locrecent *locrec;
	  int idx;

	  locrec = (struct locrecent *) ((char *) ah.addr
					 + names[cnt].locrec_offset);
	  for (idx = 0; idx < __LC_LAST; ++idx)
	    if (GET (locrec->record[LC_ALL].offset) != 0
		? (idx == LC_ALL
		   || (GET (locrec->record[idx].offset)
		       < GET (locrec->record[LC_ALL].offset))
		   || ((GET (locrec->record[idx].offset)
			+ GET (locrec->record[idx].len))
		       > (GET (locrec->record[LC_ALL].offset)
			  + GET (locrec->record[LC_ALL].len))))
		: idx != LC_ALL)
	      {
		struct dataent *data, dataent;

		dataent.file_offset = GET (locrec->record[idx].offset);
		data = (struct dataent *) bsearch (&dataent, files, sumused,
						   sizeof (struct dataent),
						   dataentcmp);
		assert (data != NULL);
		++data->nlink;
	      }
	}

      /* Print it.  */
      for (cnt = 0; cnt < used; ++cnt)
	{
	  struct locrecent *locrec;
	  int idx, i;

	  locrec = (struct locrecent *) ((char *) ah.addr
					 + names[cnt].locrec_offset);
	  for (idx = 0; idx < __LC_LAST; ++idx)
	    if (idx != LC_ALL)
	      {
		struct dataent *data, dataent;

		dataent.file_offset = GET (locrec->record[idx].offset);
		if (GET (locrec->record[LC_ALL].offset) != 0
		    && (dataent.file_offset
			>= GET (locrec->record[LC_ALL].offset))
		    && (dataent.file_offset + GET (locrec->record[idx].len)
			<= (GET (locrec->record[LC_ALL].offset)
			    + GET (locrec->record[LC_ALL].len))))
		  dataent.file_offset = GET (locrec->record[LC_ALL].offset);

		data = (struct dataent *) bsearch (&dataent, files, sumused,
						   sizeof (struct dataent),
						   dataentcmp);
		printf ("%6d %7x %3d%c ",
			GET (locrec->record[idx].len),
			GET (locrec->record[idx].offset),
			data->nlink,
			(dataent.file_offset
			 == GET (locrec->record[LC_ALL].offset))
			? '+' : ' ');
		for (i = 0; i < 16; i += 4)
		    printf ("%02x%02x%02x%02x",
			    data->sum[i], data->sum[i + 1],
			    data->sum[i + 2], data->sum[i + 3]);
		printf (" %s/%s\n", names[cnt].name,
			idx == LC_MESSAGES ? "LC_MESSAGES/SYS_LC_MESSAGES"
			: locnames[idx]);
	      }
	}
      free (files);
    }
  else
    for (cnt = 0; cnt < used; ++cnt)
      puts (names[cnt].name);

  close_archive (&ah);

  exit (EXIT_SUCCESS);
}
