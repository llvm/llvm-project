/* Handle list of needed message catalogs
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
   Written by Ulrich Drepper <drepper@gnu.org>, 1995.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation; either version 2.1 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>
#include <sys/types.h>
#include <stdlib.h>
#include <string.h>

#if defined HAVE_UNISTD_H || defined _LIBC
# include <unistd.h>
#endif

#include "gettextP.h"
#ifdef _LIBC
# include <libintl.h>
#else
# include "libgnuintl.h"
#endif

/* Handle multi-threaded applications.  */
#ifdef _LIBC
# include <libc-lock.h>
# define gl_rwlock_define_initialized __libc_rwlock_define_initialized
# define gl_rwlock_rdlock __libc_rwlock_rdlock
# define gl_rwlock_wrlock __libc_rwlock_wrlock
# define gl_rwlock_unlock __libc_rwlock_unlock
#else
# include "lock.h"
#endif

/* @@ end of prolog @@ */
/* List of already loaded domains.  */
static struct loaded_l10nfile *_nl_loaded_domains;


/* Return a data structure describing the message catalog described by
   the DOMAINNAME and CATEGORY parameters with respect to the currently
   established bindings.  */
struct loaded_l10nfile *
_nl_find_domain (const char *dirname, char *locale,
		 const char *domainname, struct binding *domainbinding)
{
  struct loaded_l10nfile *retval;
  const char *language;
  const char *modifier;
  const char *territory;
  const char *codeset;
  const char *normalized_codeset;
  const char *alias_value;
  int mask;

  /* LOCALE can consist of up to four recognized parts for the XPG syntax:

		language[_territory][.codeset][@modifier]

     Beside the first part all of them are allowed to be missing.  If
     the full specified locale is not found, the less specific one are
     looked for.  The various parts will be stripped off according to
     the following order:
		(1) codeset
		(2) normalized codeset
		(3) territory
		(4) modifier
   */

  /* We need to protect modifying the _NL_LOADED_DOMAINS data.  */
  gl_rwlock_define_initialized (static, lock);
  gl_rwlock_rdlock (lock);

  /* If we have already tested for this locale entry there has to
     be one data set in the list of loaded domains.  */
  retval = _nl_make_l10nflist (&_nl_loaded_domains, dirname,
			       strlen (dirname) + 1, 0, locale, NULL, NULL,
			       NULL, NULL, domainname, 0);

  gl_rwlock_unlock (lock);

  if (retval != NULL)
    {
      /* We know something about this locale.  */
      int cnt;

      if (retval->decided <= 0)
	_nl_load_domain (retval, domainbinding);

      if (retval->data != NULL)
	return retval;

      for (cnt = 0; retval->successor[cnt] != NULL; ++cnt)
	{
	  if (retval->successor[cnt]->decided <= 0)
	    _nl_load_domain (retval->successor[cnt], domainbinding);

	  if (retval->successor[cnt]->data != NULL)
	    break;
	}

      return retval;
      /* NOTREACHED */
    }

  /* See whether the locale value is an alias.  If yes its value
     *overwrites* the alias name.  No test for the original value is
     done.  */
  alias_value = _nl_expand_alias (locale);
  if (alias_value != NULL)
    {
      size_t len = strlen (alias_value) + 1;
      locale = (char *) malloc (len);
      if (locale == NULL)
	return NULL;

      memcpy (locale, alias_value, len);
    }

  /* Now we determine the single parts of the locale name.  First
     look for the language.  Termination symbols are `_', '.', and `@'.  */
  mask = _nl_explode_name (locale, &language, &modifier, &territory,
			   &codeset, &normalized_codeset);
  if (mask == -1)
    /* This means we are out of core.  */
    return NULL;

  /* We need to protect modifying the _NL_LOADED_DOMAINS data.  */
  gl_rwlock_wrlock (lock);

  /* Create all possible locale entries which might be interested in
     generalization.  */
  retval = _nl_make_l10nflist (&_nl_loaded_domains, dirname,
			       strlen (dirname) + 1, mask, language, territory,
			       codeset, normalized_codeset, modifier,
			       domainname, 1);

  gl_rwlock_unlock (lock);

  if (retval == NULL)
    /* This means we are out of core.  */
    goto out;

  if (retval->decided <= 0)
    _nl_load_domain (retval, domainbinding);
  if (retval->data == NULL)
    {
      int cnt;
      for (cnt = 0; retval->successor[cnt] != NULL; ++cnt)
	{
	  if (retval->successor[cnt]->decided <= 0)
	    _nl_load_domain (retval->successor[cnt], domainbinding);
	  if (retval->successor[cnt]->data != NULL)
	    break;
	}
    }

  /* The room for an alias was dynamically allocated.  Free it now.  */
  if (alias_value != NULL)
    free (locale);

out:
  /* The space for normalized_codeset is dynamically allocated.  Free it.  */
  if (mask & XPG_NORM_CODESET)
    free ((void *) normalized_codeset);

  return retval;
}


#ifdef _LIBC
/* This is called from iconv/gconv_db.c's free_mem, as locales must
   be freed before freeing gconv steps arrays.  */
void __libc_freeres_fn_section
_nl_finddomain_subfreeres (void)
{
  struct loaded_l10nfile *runp = _nl_loaded_domains;

  while (runp != NULL)
    {
      struct loaded_l10nfile *here = runp;
      if (runp->data != NULL)
	_nl_unload_domain ((struct loaded_domain *) runp->data);
      runp = runp->next;
      free ((char *) here->filename);
      free (here);
    }
}
#endif
