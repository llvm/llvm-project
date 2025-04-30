/* Functions to read locale data files.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1996.

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
#include <errno.h>
#include <fcntl.h>
#include <locale.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef _POSIX_MAPPED_FILES
# include <sys/mman.h>
#endif
#include <sys/stat.h>

#include <not-cancel.h>
#include "localeinfo.h"


static const size_t _nl_category_num_items[] =
{
#define DEFINE_CATEGORY(category, category_name, items, a) \
  [category] = _NL_ITEM_INDEX (_NL_NUM_##category),
#include "categories.def"
#undef	DEFINE_CATEGORY
};


#define NO_PAREN(arg, rest...) arg, ##rest

/* The size of the array must be specified explicitly because some of
   the 'items' may be subarrays, which will cause the compiler to deduce
   an incorrect size from the initializer.  */
#define DEFINE_CATEGORY(category, category_name, items, a) \
static const enum value_type _nl_value_type_##category     \
  [_NL_ITEM_INDEX (_NL_NUM_##category)] = { NO_PAREN items };
#define DEFINE_ELEMENT(element, element_name, optstd, type, rest...) \
  [_NL_ITEM_INDEX (element)] = type,
#include "categories.def"
#undef DEFINE_CATEGORY

static const enum value_type *const _nl_value_types[] =
{
#define DEFINE_CATEGORY(category, category_name, items, a) \
  [category] = _nl_value_type_##category,
#include "categories.def"
#undef DEFINE_CATEGORY
};


struct __locale_data *
_nl_intern_locale_data (int category, const void *data, size_t datasize)
{
  const struct
    {
      unsigned int magic;
      unsigned int nstrings;
      unsigned int strindex[0];
    } *const filedata = data;
  struct __locale_data *newdata;
  size_t cnt;

  if (__builtin_expect (datasize < sizeof *filedata, 0)
      || __builtin_expect (filedata->magic != LIMAGIC (category), 0))
    {
      /* Bad data file.  */
      __set_errno (EINVAL);
      return NULL;
    }

  if (__builtin_expect (filedata->nstrings < _nl_category_num_items[category],
			0)
      || (__builtin_expect (sizeof *filedata
			    + filedata->nstrings * sizeof (unsigned int)
			    >= datasize, 0)))
    {
      /* Insufficient data.  */
      __set_errno (EINVAL);
      return NULL;
    }

  newdata = malloc (sizeof *newdata
		    + filedata->nstrings * sizeof (union locale_data_value));
  if (newdata == NULL)
    return NULL;

  newdata->filedata = (void *) filedata;
  newdata->filesize = datasize;
  newdata->private.data = NULL;
  newdata->private.cleanup = NULL;
  newdata->usage_count = 0;
  newdata->use_translit = 0;
  newdata->nstrings = filedata->nstrings;
  for (cnt = 0; cnt < newdata->nstrings; ++cnt)
    {
      size_t idx = filedata->strindex[cnt];
      if (__glibc_unlikely (idx > (size_t) newdata->filesize))
	{
	puntdata:
	  free (newdata);
	  __set_errno (EINVAL);
	  return NULL;
	}

      /* Determine the type.  There is one special case: the LC_CTYPE
	 category can have more elements than there are in the
	 _nl_value_type_LC_XYZ array.  There are all pointers.  */
      switch (category)
	{
#define CATTEST(cat) \
	case LC_##cat:						\
	  if (cnt >= (sizeof (_nl_value_type_LC_##cat)		\
		      / sizeof (_nl_value_type_LC_##cat[0])))	\
	    goto puntdata;					\
	  break
	  CATTEST (NUMERIC);
	  CATTEST (TIME);
	  CATTEST (COLLATE);
	  CATTEST (MONETARY);
	  CATTEST (MESSAGES);
	  CATTEST (PAPER);
	  CATTEST (NAME);
	  CATTEST (ADDRESS);
	  CATTEST (TELEPHONE);
	  CATTEST (MEASUREMENT);
	  CATTEST (IDENTIFICATION);
	default:
	  assert (category == LC_CTYPE);
	  break;
	}

      if ((category == LC_CTYPE
	   && cnt >= (sizeof (_nl_value_type_LC_CTYPE)
		      / sizeof (_nl_value_type_LC_CTYPE[0])))
	  || __builtin_expect (_nl_value_types[category][cnt] != word, 1))
	newdata->values[cnt].string = newdata->filedata + idx;
      else
	{
	  if (!LOCFILE_ALIGNED_P (idx))
	    goto puntdata;
	  newdata->values[cnt].word =
	    *((const uint32_t *) (newdata->filedata + idx));
	}
    }

  return newdata;
}

void
_nl_load_locale (struct loaded_l10nfile *file, int category)
{
  int fd;
  void *filedata;
  struct __stat64_t64 st;
  struct __locale_data *newdata;
  int save_err;
  int alloc = ld_mapped;

  file->decided = 1;
  file->data = NULL;

  fd = __open_nocancel (file->filename, O_RDONLY | O_CLOEXEC);
  if (__builtin_expect (fd, 0) < 0)
    /* Cannot open the file.  */
    return;

  if (__glibc_unlikely (__fstat64_time64 (fd, &st) < 0))
    {
    puntfd:
      __close_nocancel_nostatus (fd);
      return;
    }
  if (__glibc_unlikely (S_ISDIR (st.st_mode)))
    {
      /* LOCALE/LC_foo is a directory; open LOCALE/LC_foo/SYS_LC_foo
	   instead.  */
      char *newp;
      size_t filenamelen;

      __close_nocancel_nostatus (fd);

      filenamelen = strlen (file->filename);
      newp = (char *) alloca (filenamelen
			      + 5 + _nl_category_name_sizes[category] + 1);
      __mempcpy (__mempcpy (__mempcpy (newp, file->filename, filenamelen),
			    "/SYS_", 5), _nl_category_names_get (category),
		 _nl_category_name_sizes[category] + 1);

      fd = __open_nocancel (newp, O_RDONLY | O_CLOEXEC);
      if (__builtin_expect (fd, 0) < 0)
	return;

      if (__glibc_unlikely (__fstat64_time64 (fd, &st) < 0))
	goto puntfd;
    }

  /* Map in the file's data.  */
  save_err = errno;
#ifdef _POSIX_MAPPED_FILES
# ifndef MAP_COPY
  /* Linux seems to lack read-only copy-on-write.  */
#  define MAP_COPY MAP_PRIVATE
# endif
# ifndef MAP_FILE
  /* Some systems do not have this flag; it is superfluous.  */
#  define MAP_FILE 0
# endif
  filedata = __mmap ((caddr_t) 0, st.st_size,
		     PROT_READ, MAP_FILE|MAP_COPY, fd, 0);
  if (__glibc_unlikely (filedata == MAP_FAILED))
    {
      filedata = NULL;
      if (__builtin_expect (errno, ENOSYS) == ENOSYS)
	{
#endif	/* _POSIX_MAPPED_FILES */
	  /* No mmap; allocate a buffer and read from the file.  */
	  alloc = ld_malloced;
	  filedata = malloc (st.st_size);
	  if (filedata != NULL)
	    {
	      off_t to_read = st.st_size;
	      ssize_t nread;
	      char *p = (char *) filedata;
	      while (to_read > 0)
		{
		  nread = __read_nocancel (fd, p, to_read);
		  if (__builtin_expect (nread, 1) <= 0)
		    {
		      free (filedata);
		      if (nread == 0)
			__set_errno (EINVAL); /* Bizarreness going on.  */
		      goto puntfd;
		    }
		  p += nread;
		  to_read -= nread;
		}
	      __set_errno (save_err);
	    }
#ifdef _POSIX_MAPPED_FILES
	}
    }
#endif	/* _POSIX_MAPPED_FILES */

  /* We have mapped the data, so we no longer need the descriptor.  */
  __close_nocancel_nostatus (fd);

  if (__glibc_unlikely (filedata == NULL))
    /* We failed to map or read the data.  */
    return;

  newdata = _nl_intern_locale_data (category, filedata, st.st_size);
  if (__glibc_unlikely (newdata == NULL))
    /* Bad data.  */
    {
#ifdef _POSIX_MAPPED_FILES
      if (alloc == ld_mapped)
	__munmap ((caddr_t) filedata, st.st_size);
#endif
      return;
    }

  /* _nl_intern_locale_data leaves us these fields to initialize.  */
  newdata->name = NULL;	/* This will be filled if necessary in findlocale.c. */
  newdata->alloc = alloc;

  file->data = newdata;
}

void
_nl_unload_locale (struct __locale_data *locale)
{
  if (locale->private.cleanup)
    (*locale->private.cleanup) (locale);

  switch (__builtin_expect (locale->alloc, ld_mapped))
    {
    case ld_malloced:
      free ((void *) locale->filedata);
      break;
    case ld_mapped:
#ifdef _POSIX_MAPPED_FILES
      __munmap ((caddr_t) locale->filedata, locale->filesize);
      break;
#endif
    case ld_archive:		/* Nothing to do.  */
      break;
    }

  if (__builtin_expect (locale->alloc, ld_mapped) != ld_archive)
    free ((char *) locale->name);

  free (locale);
}
