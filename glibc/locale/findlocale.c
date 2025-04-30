/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
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
#include <locale.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef _POSIX_MAPPED_FILES
# include <sys/mman.h>
#endif

#include "localeinfo.h"
#include "../iconv/gconv_charset.h"
#include "../iconv/gconv_int.h"


#ifdef NL_CURRENT_INDIRECT
# define DEFINE_CATEGORY(category, category_name, items, a) \
extern struct __locale_data _nl_C_##category; \
weak_extern (_nl_C_##category)
# include "categories.def"
# undef	DEFINE_CATEGORY

/* Array indexed by category of pointers to _nl_C_CATEGORY slots.
   Elements are zero for categories whose data is never used.  */
struct __locale_data *const _nl_C[] attribute_hidden =
  {
# define DEFINE_CATEGORY(category, category_name, items, a) \
    [category] = &_nl_C_##category,
# include "categories.def"
# undef	DEFINE_CATEGORY
  };
#else
# define _nl_C		(_nl_C_locobj.__locales)
#endif


/* For each category we keep a list of records for the locale files
   which are somehow addressed.  */
struct loaded_l10nfile *_nl_locale_file_list[__LC_LAST];

const char _nl_default_locale_path[] attribute_hidden = COMPLOCALEDIR;

/* Checks if the name is actually present, that is, not NULL and not
   empty.  */
static inline int
name_present (const char *name)
{
  return name != NULL && name[0] != '\0';
}

/* Checks that the locale name neither extremely long, nor contains a
   ".." path component (to prevent directory traversal).  */
static inline int
valid_locale_name (const char *name)
{
  /* Not set.  */
  size_t namelen = strlen (name);
  /* Name too long.  The limit is arbitrary and prevents stack overflow
     issues later.  */
  if (__glibc_unlikely (namelen > 255))
    return 0;
  /* Directory traversal attempt.  */
  static const char slashdot[4] = {'/', '.', '.', '/'};
  if (__glibc_unlikely (__memmem (name, namelen,
				  slashdot, sizeof (slashdot)) != NULL))
    return 0;
  if (namelen == 2 && __glibc_unlikely (name[0] == '.' && name [1] == '.'))
    return 0;
  if (namelen >= 3
      && __glibc_unlikely (((name[0] == '.'
			     && name[1] == '.'
			     && name[2] == '/')
			    || (name[namelen - 3] == '/'
				&& name[namelen - 2] == '.'
				&& name[namelen - 1] == '.'))))
    return 0;
  /* If there is a slash in the name, it must start with one.  */
  if (__glibc_unlikely (memchr (name, '/', namelen) != NULL) && name[0] != '/')
    return 0;
  return 1;
}

struct __locale_data *
_nl_find_locale (const char *locale_path, size_t locale_path_len,
		 int category, const char **name)
{
  int mask;
  /* Name of the locale for this category.  */
  const char *cloc_name = *name;
  const char *language;
  const char *modifier;
  const char *territory;
  const char *codeset;
  const char *normalized_codeset;
  struct loaded_l10nfile *locale_file;

  if (cloc_name[0] == '\0')
    {
      /* The user decides which locale to use by setting environment
	 variables.  */
      cloc_name = getenv ("LC_ALL");
      if (!name_present (cloc_name))
	cloc_name = getenv (_nl_category_names_get (category));
      if (!name_present (cloc_name))
	cloc_name = getenv ("LANG");
      if (!name_present (cloc_name))
	cloc_name = _nl_C_name;
    }

  /* We used to fall back to the C locale if the name contains a slash
     character '/', but we now check for directory traversal in
     valid_locale_name, so this is no longer necessary.  */

  if (__builtin_expect (strcmp (cloc_name, _nl_C_name), 1) == 0
      || __builtin_expect (strcmp (cloc_name, _nl_POSIX_name), 1) == 0)
    {
      /* We need not load anything.  The needed data is contained in
	 the library itself.  */
      *name = _nl_C_name;
      return _nl_C[category];
    }
  else if (!valid_locale_name (cloc_name))
    {
      __set_errno (EINVAL);
      return NULL;
    }

  *name = cloc_name;

  /* We really have to load some data.  First we try the archive,
     but only if there was no LOCPATH environment variable specified.  */
  if (__glibc_likely (locale_path == NULL))
    {
      struct __locale_data *data
	= _nl_load_locale_from_archive (category, name);
      if (__glibc_likely (data != NULL))
	return data;

      /* Nothing in the archive with the given name.  Expanding it as
	 an alias and retry.  */
      cloc_name = _nl_expand_alias (*name);
      if (cloc_name != NULL)
	{
	  data = _nl_load_locale_from_archive (category, &cloc_name);
	  if (__builtin_expect (data != NULL, 1))
	    return data;
	}

      /* Nothing in the archive.  Set the default path to search below.  */
      locale_path = _nl_default_locale_path;
      locale_path_len = sizeof _nl_default_locale_path;
    }
  else
    /* We really have to load some data.  First see whether the name is
       an alias.  Please note that this makes it impossible to have "C"
       or "POSIX" as aliases.  */
    cloc_name = _nl_expand_alias (*name);

  if (cloc_name == NULL)
    /* It is no alias.  */
    cloc_name = *name;

  /* Make a writable copy of the locale name.  */
  char *loc_name = strdupa (cloc_name);

  /* LOCALE can consist of up to four recognized parts for the XPG syntax:

		language[_territory[.codeset]][@modifier]

     Beside the first all of them are allowed to be missing.  If the
     full specified locale is not found, the less specific one are
     looked for.  The various part will be stripped off according to
     the following order:
		(1) codeset
		(2) normalized codeset
		(3) territory
		(4) modifier
   */
  mask = _nl_explode_name (loc_name, &language, &modifier, &territory,
			   &codeset, &normalized_codeset);
  if (mask == -1)
    /* Memory allocate problem.  */
    return NULL;

  /* If exactly this locale was already asked for we have an entry with
     the complete name.  */
  locale_file = _nl_make_l10nflist (&_nl_locale_file_list[category],
				    locale_path, locale_path_len, mask,
				    language, territory, codeset,
				    normalized_codeset, modifier,
				    _nl_category_names_get (category), 0);

  if (locale_file == NULL)
    {
      /* Find status record for addressed locale file.  We have to search
	 through all directories in the locale path.  */
      locale_file = _nl_make_l10nflist (&_nl_locale_file_list[category],
					locale_path, locale_path_len, mask,
					language, territory, codeset,
					normalized_codeset, modifier,
					_nl_category_names_get (category), 1);
      if (locale_file == NULL)
	/* This means we are out of core.  */
	return NULL;
    }

  /* The space for normalized_codeset is dynamically allocated.  Free it.  */
  if (mask & XPG_NORM_CODESET)
    free ((void *) normalized_codeset);

  if (locale_file->decided == 0)
    _nl_load_locale (locale_file, category);

  if (locale_file->data == NULL)
    {
      int cnt;
      for (cnt = 0; locale_file->successor[cnt] != NULL; ++cnt)
	{
	  if (locale_file->successor[cnt]->decided == 0)
	    _nl_load_locale (locale_file->successor[cnt], category);
	  if (locale_file->successor[cnt]->data != NULL)
	    break;
	}
      /* Move the entry we found (or NULL) to the first place of
	 successors.  */
      locale_file->successor[0] = locale_file->successor[cnt];
      locale_file = locale_file->successor[cnt];

      if (locale_file == NULL)
	return NULL;
    }

  /* The LC_CTYPE category allows to check whether a locale is really
     usable.  If the locale name contains a charset name and the
     charset name used in the locale (present in the LC_CTYPE data) is
     not the same (after resolving aliases etc) we reject the locale
     since using it would irritate users expecting the charset named
     in the locale name.  */
  if (codeset != NULL)
    {
      /* Get the codeset information from the locale file.  */
      static const int codeset_idx[] =
	{
	  [__LC_CTYPE] = _NL_ITEM_INDEX (CODESET),
	  [__LC_NUMERIC] = _NL_ITEM_INDEX (_NL_NUMERIC_CODESET),
	  [__LC_TIME] = _NL_ITEM_INDEX (_NL_TIME_CODESET),
	  [__LC_COLLATE] = _NL_ITEM_INDEX (_NL_COLLATE_CODESET),
	  [__LC_MONETARY] = _NL_ITEM_INDEX (_NL_MONETARY_CODESET),
	  [__LC_MESSAGES] = _NL_ITEM_INDEX (_NL_MESSAGES_CODESET),
	  [__LC_PAPER] = _NL_ITEM_INDEX (_NL_PAPER_CODESET),
	  [__LC_NAME] = _NL_ITEM_INDEX (_NL_NAME_CODESET),
	  [__LC_ADDRESS] = _NL_ITEM_INDEX (_NL_ADDRESS_CODESET),
	  [__LC_TELEPHONE] = _NL_ITEM_INDEX (_NL_TELEPHONE_CODESET),
	  [__LC_MEASUREMENT] = _NL_ITEM_INDEX (_NL_MEASUREMENT_CODESET),
	  [__LC_IDENTIFICATION] = _NL_ITEM_INDEX (_NL_IDENTIFICATION_CODESET)
	};
      const struct __locale_data *data;
      const char *locale_codeset;
      char *clocale_codeset;
      char *ccodeset;

      data = (const struct __locale_data *) locale_file->data;
      locale_codeset =
	(const char *) data->values[codeset_idx[category]].string;
      assert (locale_codeset != NULL);
      /* Note the length of the allocated memory: +3 for up to two slashes
	 and the NUL byte.  */
      clocale_codeset = (char *) alloca (strlen (locale_codeset) + 3);
      strip (clocale_codeset, locale_codeset);

      ccodeset = (char *) alloca (strlen (codeset) + 3);
      strip (ccodeset, codeset);

      if (__gconv_compare_alias (upstr (ccodeset, ccodeset),
				 upstr (clocale_codeset,
					clocale_codeset)) != 0)
	/* The codesets are not identical, don't use the locale.  */
	return NULL;
    }

  /* Determine the locale name for which loading succeeded.  This
     information comes from the file name.  The form is
     <path>/<locale>/LC_foo.  We must extract the <locale> part.  */
  if (((const struct __locale_data *) locale_file->data)->name == NULL)
    {
      char *cp, *endp;

      endp = strrchr (locale_file->filename, '/');
      cp = endp - 1;
      while (cp[-1] != '/')
	--cp;
      ((struct __locale_data *) locale_file->data)->name
	= __strndup (cp, endp - cp);
    }

  /* Determine whether the user wants transliteration or not.  */
  if (modifier != NULL
      && __strcasecmp_l (modifier, "TRANSLIT", _nl_C_locobj_ptr) == 0)
    ((struct __locale_data *) locale_file->data)->use_translit = 1;

  /* Increment the usage count.  */
  if (((const struct __locale_data *) locale_file->data)->usage_count
      < MAX_USAGE_COUNT)
    ++((struct __locale_data *) locale_file->data)->usage_count;

  return (struct __locale_data *) locale_file->data;
}


/* Calling this function assumes the lock for handling global locale data
   is acquired.  */
void
_nl_remove_locale (int locale, struct __locale_data *data)
{
  if (--data->usage_count == 0)
    {
      if (data->alloc != ld_archive)
	{
	  /* First search the entry in the list of loaded files.  */
	  struct loaded_l10nfile *ptr = _nl_locale_file_list[locale];

	  /* Search for the entry.  It must be in the list.  Otherwise it
	     is a bug and we crash badly.  */
	  while ((struct __locale_data *) ptr->data != data)
	    ptr = ptr->next;

	  /* Mark the data as not available anymore.  So when the data has
	     to be used again it is reloaded.  */
	  ptr->decided = 0;
	  ptr->data = NULL;
	}

      /* This does the real work.  */
      _nl_unload_locale (data);
    }
}
