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

#include <alloca.h>
#include <argz.h>
#include <errno.h>
#include <libc-lock.h>
#include <locale.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "localeinfo.h"

#ifdef NL_CURRENT_INDIRECT

/* For each category declare a special external symbol
   _nl_current_CATEGORY_used with a weak reference.
   This symbol will is defined in lc-CATEGORY.c and will be linked in
   if anything uses _nl_current_CATEGORY (also defined in that module).
   Also use a weak reference for the _nl_current_CATEGORY thread variable.  */

# define DEFINE_CATEGORY(category, category_name, items, a) \
    extern char _nl_current_##category##_used; \
    weak_extern (_nl_current_##category##_used) \
    weak_extern (_nl_current_##category)
# include "categories.def"
# undef	DEFINE_CATEGORY

/* Now define a table of flags based on those special weak symbols' values.
   _nl_current_used[CATEGORY] will be zero if _nl_current_CATEGORY is not
   linked in.  */
static char *const _nl_current_used[] =
  {
# define DEFINE_CATEGORY(category, category_name, items, a) \
    [category] = &_nl_current_##category##_used,
# include "categories.def"
# undef	DEFINE_CATEGORY
  };

# define CATEGORY_USED(category)	(_nl_current_used[category] != 0)

#else

/* The shared library always loads all the categories,
   and the current global settings are kept in _nl_global_locale.  */

# define CATEGORY_USED(category)	(1)

#endif


/* Define an array of category names (also the environment variable names).  */
const struct catnamestr_t _nl_category_names attribute_hidden =
  {
#define DEFINE_CATEGORY(category, category_name, items, a) \
    category_name,
#include "categories.def"
#undef DEFINE_CATEGORY
  };

const uint8_t _nl_category_name_idxs[__LC_LAST] attribute_hidden =
  {
#define DEFINE_CATEGORY(category, category_name, items, a) \
    [category] = offsetof (struct catnamestr_t, CATNAMEMF (__LINE__)),
#include "categories.def"
#undef DEFINE_CATEGORY
  };

/* An array of their lengths, for convenience.  */
const uint8_t _nl_category_name_sizes[] attribute_hidden =
  {
#define DEFINE_CATEGORY(category, category_name, items, a) \
    [category] = sizeof (category_name) - 1,
#include "categories.def"
#undef	DEFINE_CATEGORY
    [LC_ALL] = sizeof ("LC_ALL") - 1
  };


#ifdef NL_CURRENT_INDIRECT
# define WEAK_POSTLOAD(postload) weak_extern (postload)
#else
# define WEAK_POSTLOAD(postload) /* Need strong refs in static linking.  */
#endif

/* Declare the postload functions used below.  */
#undef	NO_POSTLOAD
#define NO_POSTLOAD _nl_postload_ctype /* Harmless thing known to exist.  */
#define DEFINE_CATEGORY(category, category_name, items, postload) \
extern void postload (void); WEAK_POSTLOAD (postload)
#include "categories.def"
#undef	DEFINE_CATEGORY
#undef	NO_POSTLOAD

/* Define an array indexed by category of postload functions to call after
   loading and installing that category's data.  */
static void (*const _nl_category_postload[]) (void) =
  {
#define DEFINE_CATEGORY(category, category_name, items, postload) \
    [category] = postload,
#include "categories.def"
#undef	DEFINE_CATEGORY
  };


/* Lock for protecting global data.  */
__libc_rwlock_define_initialized (, __libc_setlocale_lock attribute_hidden)

/* Defined in loadmsgcat.c.  */
extern int _nl_msg_cat_cntr;


/* Use this when we come along an error.  */
#define ERROR_RETURN							      \
  do {									      \
    __set_errno (EINVAL);						      \
    return NULL;							      \
  } while (0)


/* Construct a new composite name.  */
static char *
new_composite_name (int category, const char **newnames)
{
  size_t last_len = 0;
  size_t cumlen = 0;
  int i;
  char *new, *p;
  int same = 1;

  for (i = 0; i < __LC_LAST; ++i)
    if (i != LC_ALL)
      {
	const char *name = (category == LC_ALL ? newnames[i]
			    : category == i ? newnames[0]
			    : _nl_global_locale.__names[i]);
	last_len = strlen (name);
	cumlen += _nl_category_name_sizes[i] + 1 + last_len + 1;
	if (same && name != newnames[0] && strcmp (name, newnames[0]) != 0)
	  same = 0;
      }

  if (same)
    {
      /* All the categories use the same name.  */
      if (strcmp (newnames[0], _nl_C_name) == 0
	  || strcmp (newnames[0], _nl_POSIX_name) == 0)
	return (char *) _nl_C_name;

      new = malloc (last_len + 1);

      return new == NULL ? NULL : memcpy (new, newnames[0], last_len + 1);
    }

  new = malloc (cumlen);
  if (new == NULL)
    return NULL;
  p = new;
  for (i = 0; i < __LC_LAST; ++i)
    if (i != LC_ALL)
      {
	/* Add "CATEGORY=NAME;" to the string.  */
	const char *name = (category == LC_ALL ? newnames[i]
			    : category == i ? newnames[0]
			    : _nl_global_locale.__names[i]);
	p = __stpcpy (p, _nl_category_names_get (i));
	*p++ = '=';
	p = __stpcpy (p, name);
	*p++ = ';';
      }
  p[-1] = '\0';		/* Clobber the last ';'.  */
  return new;
}


/* Put NAME in _nl_global_locale.__names.  */
static void
setname (int category, const char *name)
{
  if (_nl_global_locale.__names[category] == name)
    return;

  if (_nl_global_locale.__names[category] != _nl_C_name)
    free ((char *) _nl_global_locale.__names[category]);

  _nl_global_locale.__names[category] = name;
}

/* Put DATA in *_nl_current[CATEGORY].  */
static void
setdata (int category, struct __locale_data *data)
{
  if (CATEGORY_USED (category))
    {
      _nl_global_locale.__locales[category] = data;
      if (_nl_category_postload[category])
	(*_nl_category_postload[category]) ();
    }
}

char *
setlocale (int category, const char *locale)
{
  char *locale_path;
  size_t locale_path_len;
  const char *locpath_var;
  char *composite;

  /* Sanity check for CATEGORY argument.  */
  if (__builtin_expect (category, 0) < 0
      || __builtin_expect (category, 0) >= __LC_LAST)
    ERROR_RETURN;

  /* Does user want name of current locale?  */
  if (locale == NULL)
    return (char *) _nl_global_locale.__names[category];

  /* Protect global data.  */
  __libc_rwlock_wrlock (__libc_setlocale_lock);

  if (strcmp (locale, _nl_global_locale.__names[category]) == 0)
    {
      /* Changing to the same thing.  */
      __libc_rwlock_unlock (__libc_setlocale_lock);

      return (char *) _nl_global_locale.__names[category];
    }

  /* We perhaps really have to load some data.  So we determine the
     path in which to look for the data now.  The environment variable
     `LOCPATH' must only be used when the binary has no SUID or SGID
     bit set.  If using the default path, we tell _nl_find_locale
     by passing null and it can check the canonical locale archive.  */
  locale_path = NULL;
  locale_path_len = 0;

  locpath_var = getenv ("LOCPATH");
  if (locpath_var != NULL && locpath_var[0] != '\0')
    {
      if (__argz_create_sep (locpath_var, ':',
			     &locale_path, &locale_path_len) != 0
	  || __argz_add_sep (&locale_path, &locale_path_len,
			     _nl_default_locale_path, ':') != 0)
	{
	  __libc_rwlock_unlock (__libc_setlocale_lock);
	  return NULL;
	}
    }

  if (category == LC_ALL)
    {
      /* The user wants to set all categories.  The desired locales
	 for the individual categories can be selected by using a
	 composite locale name.  This is a semi-colon separated list
	 of entries of the form `CATEGORY=VALUE'.  */
      const char *newnames[__LC_LAST];
      struct __locale_data *newdata[__LC_LAST];
      /* Copy of the locale argument, for in-place splitting.  */
      char *locale_copy = NULL;

      /* Set all name pointers to the argument name.  */
      for (category = 0; category < __LC_LAST; ++category)
	if (category != LC_ALL)
	  newnames[category] = (char *) locale;

      if (__glibc_unlikely (strchr (locale, ';') != NULL))
	{
	  /* This is a composite name.  Make a copy and split it up.  */
	  locale_copy = __strdup (locale);
	  if (__glibc_unlikely (locale_copy == NULL))
	    {
	      __libc_rwlock_unlock (__libc_setlocale_lock);
	      return NULL;
	    }
	  char *np = locale_copy;
	  char *cp;
	  int cnt;

	  while ((cp = strchr (np, '=')) != NULL)
	    {
	      for (cnt = 0; cnt < __LC_LAST; ++cnt)
		if (cnt != LC_ALL
		    && (size_t) (cp - np) == _nl_category_name_sizes[cnt]
		    && (memcmp (np, (_nl_category_names_get (cnt)), cp - np)
			== 0))
		  break;

	      if (cnt == __LC_LAST)
		{
		error_return:
		  __libc_rwlock_unlock (__libc_setlocale_lock);
		  free (locale_copy);

		  /* Bogus category name.  */
		  ERROR_RETURN;
		}

	      /* Found the category this clause sets.  */
	      newnames[cnt] = ++cp;
	      cp = strchr (cp, ';');
	      if (cp != NULL)
		{
		  /* Examine the next clause.  */
		  *cp = '\0';
		  np = cp + 1;
		}
	      else
		/* This was the last clause.  We are done.  */
		break;
	    }

	  for (cnt = 0; cnt < __LC_LAST; ++cnt)
	    if (cnt != LC_ALL && newnames[cnt] == locale)
	      /* The composite name did not specify all categories.  */
	      goto error_return;
	}

      /* Load the new data for each category.  */
      while (category-- > 0)
	if (category != LC_ALL)
	  {
	    newdata[category] = _nl_find_locale (locale_path, locale_path_len,
						 category,
						 &newnames[category]);

	    if (newdata[category] == NULL)
	      {
#ifdef NL_CURRENT_INDIRECT
		if (newnames[category] == _nl_C_name)
		  /* Null because it's the weak value of _nl_C_LC_FOO.  */
		  continue;
#endif
		break;
	      }

	    /* We must not simply free a global locale since we have
	       no control over the usage.  So we mark it as
	       un-deletable.  And yes, the 'if' is needed, the data
	       might be in read-only memory.  */
	    if (newdata[category]->usage_count != UNDELETABLE)
	      newdata[category]->usage_count = UNDELETABLE;

	    /* Make a copy of locale name.  */
	    if (newnames[category] != _nl_C_name)
	      {
		if (strcmp (newnames[category],
			    _nl_global_locale.__names[category]) == 0)
		  newnames[category] = _nl_global_locale.__names[category];
		else
		  {
		    newnames[category] = __strdup (newnames[category]);
		    if (newnames[category] == NULL)
		      break;
		  }
	      }
	  }

      /* Create new composite name.  */
      composite = (category >= 0
		   ? NULL : new_composite_name (LC_ALL, newnames));
      if (composite != NULL)
	{
	  /* Now we have loaded all the new data.  Put it in place.  */
	  for (category = 0; category < __LC_LAST; ++category)
	    if (category != LC_ALL)
	      {
		setdata (category, newdata[category]);
		setname (category, newnames[category]);
	      }
	  setname (LC_ALL, composite);

	  /* We successfully loaded a new locale.  Let the message catalog
	     functions know about this.  */
	  ++_nl_msg_cat_cntr;
	}
      else
	for (++category; category < __LC_LAST; ++category)
	  if (category != LC_ALL && newnames[category] != _nl_C_name
	      && newnames[category] != _nl_global_locale.__names[category])
	    free ((char *) newnames[category]);

      /* Critical section left.  */
      __libc_rwlock_unlock (__libc_setlocale_lock);

      /* Free the resources.  */
      free (locale_path);
      free (locale_copy);

      return composite;
    }
  else
    {
      struct __locale_data *newdata = NULL;
      const char *newname[1] = { locale };

      if (CATEGORY_USED (category))
	{
	  /* Only actually load the data if anything will use it.  */
	  newdata = _nl_find_locale (locale_path, locale_path_len, category,
				     &newname[0]);
	  if (newdata == NULL)
	    goto abort_single;

	  /* We must not simply free a global locale since we have no
	     control over the usage.  So we mark it as un-deletable.

	     Note: do not remove the `if', it's necessary to cope with
	     the builtin locale data.  */
	  if (newdata->usage_count != UNDELETABLE)
	    newdata->usage_count = UNDELETABLE;
	}

      /* Make a copy of locale name.  */
      if (newname[0] != _nl_C_name)
	{
	  newname[0] = __strdup (newname[0]);
	  if (newname[0] == NULL)
	    goto abort_single;
	}

      /* Create new composite name.  */
      composite = new_composite_name (category, newname);
      if (composite == NULL)
	{
	  if (newname[0] != _nl_C_name)
	    free ((char *) newname[0]);

	  /* Say that we don't have any data loaded.  */
	abort_single:
	  newname[0] = NULL;
	}
      else
	{
	  if (CATEGORY_USED (category))
	    setdata (category, newdata);

	  setname (category, newname[0]);
	  setname (LC_ALL, composite);

	  /* We successfully loaded a new locale.  Let the message catalog
	     functions know about this.  */
	  ++_nl_msg_cat_cntr;
	}

      /* Critical section left.  */
      __libc_rwlock_unlock (__libc_setlocale_lock);

      /* Free the resources (the locale path variable.  */
      free (locale_path);

      return (char *) newname[0];
    }
}
libc_hidden_def (setlocale)

static void __libc_freeres_fn_section
free_category (int category,
	       struct __locale_data *here, struct __locale_data *c_data)
{
  struct loaded_l10nfile *runp = _nl_locale_file_list[category];

  /* If this category is already "C" don't do anything.  */
  if (here != c_data)
    {
      /* We have to be prepared that sometime later we still
	 might need the locale information.  */
      setdata (category, c_data);
      setname (category, _nl_C_name);
    }

  while (runp != NULL)
    {
      struct loaded_l10nfile *curr = runp;
      struct __locale_data *data = (struct __locale_data *) runp->data;

      if (data != NULL && data != c_data)
	_nl_unload_locale (data);
      runp = runp->next;
      free ((char *) curr->filename);
      free (curr);
    }
}

/* This is called from iconv/gconv_db.c's free_mem, as locales must
   be freed before freeing gconv steps arrays.  */
void __libc_freeres_fn_section
_nl_locale_subfreeres (void)
{
#ifdef NL_CURRENT_INDIRECT
  /* We don't use the loop because we want to have individual weak
     symbol references here.  */
# define DEFINE_CATEGORY(category, category_name, items, a)		      \
  if (CATEGORY_USED (category))						      \
    {									      \
      extern struct __locale_data _nl_C_##category;			      \
      weak_extern (_nl_C_##category)					      \
      free_category (category, *_nl_current_##category, &_nl_C_##category);   \
    }
# include "categories.def"
# undef	DEFINE_CATEGORY
#else
  int category;

  for (category = 0; category < __LC_LAST; ++category)
    if (category != LC_ALL)
      free_category (category, _NL_CURRENT_DATA (category),
		     _nl_C_locobj.__locales[category]);
#endif

  setname (LC_ALL, _nl_C_name);

  /* This frees the data structures associated with the locale archive.
     The locales from the archive are not in the file list, so we have
     not called _nl_unload_locale on them above.  */
  _nl_archive_subfreeres ();
}
