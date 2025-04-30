/* Handle configuration data.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <locale.h>
#include <search.h>
#include <stddef.h>
#include <stdio.h>
#include <stdio_ext.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/param.h>

#include <libc-lock.h>
#include <gconv_int.h>
#include <gconv_parseconfdir.h>

/* This is the default path where we look for module lists.  */
static const char default_gconv_path[] = GCONV_PATH;

/* Type to represent search path.  */
struct path_elem
{
  const char *name;
  size_t len;
};

/* The path elements, as determined by the __gconv_get_path function.
   All path elements end in a slash.  */
struct path_elem *__gconv_path_elem;
/* Maximum length of a single path element in __gconv_path_elem.  */
size_t __gconv_max_path_elem_len;

/* We use the following struct if we couldn't allocate memory.  */
static const struct path_elem empty_path_elem = { NULL, 0 };

/* Filename extension for the modules.  */
#ifndef MODULE_EXT
# define MODULE_EXT ".so"
#endif
static const char gconv_module_ext[] = MODULE_EXT;

/* We have a few builtin transformations.  */
static struct gconv_module builtin_modules[] =
{
#define BUILTIN_TRANSFORMATION(From, To, Cost, Name, Fct, BtowcFct, \
			       MinF, MaxF, MinT, MaxT) \
  {									      \
    .from_string = From,						      \
    .to_string = To,							      \
    .cost_hi = Cost,							      \
    .cost_lo = INT_MAX,							      \
    .module_name = Name							      \
  },
#define BUILTIN_ALIAS(From, To)

#include "gconv_builtin.h"

#undef BUILTIN_TRANSFORMATION
#undef BUILTIN_ALIAS
};

static const char builtin_aliases[] =
{
#define BUILTIN_TRANSFORMATION(From, To, Cost, Name, Fct, BtowcFct, \
			       MinF, MaxF, MinT, MaxT)
#define BUILTIN_ALIAS(From, To) From "\0" To "\0"

#include "gconv_builtin.h"

#undef BUILTIN_TRANSFORMATION
#undef BUILTIN_ALIAS
};


/* Value of the GCONV_PATH environment variable.  */
const char *__gconv_path_envvar;


/* Test whether there is already a matching module known.  */
static int
detect_conflict (const char *alias)
{
  struct gconv_module *node = __gconv_modules_db;

  while (node != NULL)
    {
      int cmpres = strcmp (alias, node->from_string);

      if (cmpres == 0)
	/* We have a conflict.  */
	return 1;
      else if (cmpres < 0)
	node = node->left;
      else
	node = node->right;
    }

  return node != NULL;
}


/* The actual code to add aliases.  */
static void
add_alias2 (const char *from, const char *to, const char *wp)
{
  /* Test whether this alias conflicts with any available module.  */
  if (detect_conflict (from))
    /* It does conflict, don't add the alias.  */
    return;

  struct gconv_alias *new_alias = (struct gconv_alias *)
    malloc (sizeof (struct gconv_alias) + (wp - from));
  if (new_alias != NULL)
    {
      void **inserted;

      new_alias->fromname = memcpy ((char *) new_alias
				    + sizeof (struct gconv_alias),
				    from, wp - from);
      new_alias->toname = new_alias->fromname + (to - from);

      inserted = (void **) __tsearch (new_alias, &__gconv_alias_db,
				      __gconv_alias_compare);
      if (inserted == NULL || *inserted != new_alias)
	/* Something went wrong, free this entry.  */
	free (new_alias);
    }
}


/* Add new alias.  */
static void
add_alias (char *rp)
{
  /* We now expect two more string.  The strings are normalized
     (converted to UPPER case) and strored in the alias database.  */
  char *from, *to, *wp;

  while (__isspace_l (*rp, _nl_C_locobj_ptr))
    ++rp;
  from = wp = rp;
  while (*rp != '\0' && !__isspace_l (*rp, _nl_C_locobj_ptr))
    *wp++ = __toupper_l (*rp++, _nl_C_locobj_ptr);
  if (*rp == '\0')
    /* There is no `to' string on the line.  Ignore it.  */
    return;
  *wp++ = '\0';
  to = ++rp;
  while (__isspace_l (*rp, _nl_C_locobj_ptr))
    ++rp;
  while (*rp != '\0' && !__isspace_l (*rp, _nl_C_locobj_ptr))
    *wp++ = __toupper_l (*rp++, _nl_C_locobj_ptr);
  if (to == wp)
    /* No `to' string, ignore the line.  */
    return;
  *wp++ = '\0';

  add_alias2 (from, to, wp);
}


/* Insert a data structure for a new module in the search tree.  */
static void
insert_module (struct gconv_module *newp, int tobefreed)
{
  struct gconv_module **rootp = &__gconv_modules_db;

  while (*rootp != NULL)
    {
      struct gconv_module *root = *rootp;
      int cmpres;

      cmpres = strcmp (newp->from_string, root->from_string);
      if (cmpres == 0)
	{
	  /* Both strings are identical.  Insert the string at the
	     end of the `same' list if it is not already there.  */
	  while (strcmp (newp->from_string, root->from_string) != 0
		 || strcmp (newp->to_string, root->to_string) != 0)
	    {
	      rootp = &root->same;
	      root = *rootp;
	      if (root == NULL)
		break;
	    }

	  if (root != NULL)
	    {
	      /* This is a no new conversion.  But maybe the cost is
		 better.  */
	      if (newp->cost_hi < root->cost_hi
		  || (newp->cost_hi == root->cost_hi
		      && newp->cost_lo < root->cost_lo))
		{
		  newp->left = root->left;
		  newp->right = root->right;
		  newp->same = root->same;
		  *rootp = newp;

		  free (root);
		}
	      else if (tobefreed)
		free (newp);
	      return;
	    }

	  break;
	}
      else if (cmpres < 0)
	rootp = &root->left;
      else
	rootp = &root->right;
    }

  /* Plug in the new node here.  */
  *rootp = newp;
}


/* Add new module.  */
static void
add_module (char *rp, const char *directory, size_t dir_len, int modcounter)
{
  /* We expect now
     1. `from' name
     2. `to' name
     3. filename of the module
     4. an optional cost value
  */
  struct gconv_alias fake_alias;
  struct gconv_module *new_module;
  char *from, *to, *module, *wp;
  int need_ext;
  int cost_hi;

  while (__isspace_l (*rp, _nl_C_locobj_ptr))
    ++rp;
  from = rp;
  while (*rp != '\0' && !__isspace_l (*rp, _nl_C_locobj_ptr))
    {
      *rp = __toupper_l (*rp, _nl_C_locobj_ptr);
      ++rp;
    }
  if (*rp == '\0')
    return;
  *rp++ = '\0';
  to = wp = rp;
  while (__isspace_l (*rp, _nl_C_locobj_ptr))
    ++rp;
  while (*rp != '\0' && !__isspace_l (*rp, _nl_C_locobj_ptr))
    *wp++ = __toupper_l (*rp++, _nl_C_locobj_ptr);
  if (*rp == '\0')
    return;
  *wp++ = '\0';
  do
    ++rp;
  while (__isspace_l (*rp, _nl_C_locobj_ptr));
  module = wp;
  while (*rp != '\0' && !__isspace_l (*rp, _nl_C_locobj_ptr))
    *wp++ = *rp++;
  if (*rp == '\0')
    {
      /* There is no cost, use one by default.  */
      *wp++ = '\0';
      cost_hi = 1;
    }
  else
    {
      /* There might be a cost value.  */
      char *endp;

      *wp++ = '\0';
      cost_hi = strtol (rp, &endp, 10);
      if (rp == endp || cost_hi < 1)
	/* No useful information.  */
	cost_hi = 1;
    }

  if (module[0] == '\0')
    /* No module name given.  */
    return;
  if (module[0] == '/')
    dir_len = 0;

  /* See whether we must add the ending.  */
  need_ext = 0;
  if (wp - module < (ptrdiff_t) sizeof (gconv_module_ext)
      || memcmp (wp - sizeof (gconv_module_ext), gconv_module_ext,
		 sizeof (gconv_module_ext)) != 0)
    /* We must add the module extension.  */
    need_ext = sizeof (gconv_module_ext) - 1;

  /* See whether we have already an alias with this name defined.  */
  fake_alias.fromname = strndupa (from, to - from);

  if (__tfind (&fake_alias, &__gconv_alias_db, __gconv_alias_compare) != NULL)
    /* This module duplicates an alias.  */
    return;

  new_module = (struct gconv_module *) calloc (1,
					       sizeof (struct gconv_module)
					       + (wp - from)
					       + dir_len + need_ext);
  if (new_module != NULL)
    {
      char *tmp;

      new_module->from_string = tmp = (char *) (new_module + 1);
      tmp = __mempcpy (tmp, from, to - from);

      new_module->to_string = tmp;
      tmp = __mempcpy (tmp, to, module - to);

      new_module->cost_hi = cost_hi;
      new_module->cost_lo = modcounter;

      new_module->module_name = tmp;

      if (dir_len != 0)
	tmp = __mempcpy (tmp, directory, dir_len);

      tmp = __mempcpy (tmp, module, wp - module);

      if (need_ext)
	memcpy (tmp - 1, gconv_module_ext, sizeof (gconv_module_ext));

      /* Now insert the new module data structure in our search tree.  */
      insert_module (new_module, 1);
    }
}


/* Determine the directories we are looking for data in.  This function should
   only be called from __gconv_read_conf.  */
static void
__gconv_get_path (void)
{
  struct path_elem *result;

  /* This function is only ever called when __gconv_path_elem is NULL.  */
  result = __gconv_path_elem;
  assert (result == NULL);

  /* Determine the complete path first.  */
  char *gconv_path;
  size_t gconv_path_len;
  char *elem;
  char *oldp;
  char *cp;
  int nelems;
  char *cwd;
  size_t cwdlen;

  if (__gconv_path_envvar == NULL)
    {
      /* No user-defined path.  Make a modifiable copy of the
         default path.  */
      gconv_path = strdupa (default_gconv_path);
      gconv_path_len = sizeof (default_gconv_path);
      cwd = NULL;
      cwdlen = 0;
    }
  else
    {
      /* Append the default path to the user-defined path.  */
      size_t user_len = strlen (__gconv_path_envvar);

      gconv_path_len = user_len + 1 + sizeof (default_gconv_path);
      gconv_path = alloca (gconv_path_len);
      __mempcpy (__mempcpy (__mempcpy (gconv_path, __gconv_path_envvar,
                                       user_len),
                            ":", 1),
                 default_gconv_path, sizeof (default_gconv_path));
      cwd = __getcwd (NULL, 0);
      cwdlen = __glibc_unlikely (cwd == NULL) ? 0 : strlen (cwd);
    }
  assert (default_gconv_path[0] == '/');

  /* In a first pass we calculate the number of elements.  */
  oldp = NULL;
  cp = strchr (gconv_path, ':');
  nelems = 1;
  while (cp != NULL)
    {
      if (cp != oldp + 1)
        ++nelems;
      oldp = cp;
      cp = strchr (cp + 1, ':');
    }

  /* Allocate the memory for the result.  */
  result = malloc ((nelems + 1)
                              * sizeof (struct path_elem)
                              + gconv_path_len + nelems
                              + (nelems - 1) * (cwdlen + 1));
  if (result != NULL)
    {
      char *strspace = (char *) &result[nelems + 1];
      int n = 0;

      /* Separate the individual parts.  */
      __gconv_max_path_elem_len = 0;
      elem = __strtok_r (gconv_path, ":", &gconv_path);
      assert (elem != NULL);
      do
        {
          result[n].name = strspace;
          if (elem[0] != '/')
            {
              assert (cwd != NULL);
              strspace = __mempcpy (strspace, cwd, cwdlen);
              *strspace++ = '/';
            }
          strspace = __stpcpy (strspace, elem);
          if (strspace[-1] != '/')
            *strspace++ = '/';

          result[n].len = strspace - result[n].name;
          if (result[n].len > __gconv_max_path_elem_len)
            __gconv_max_path_elem_len = result[n].len;

          *strspace++ = '\0';
          ++n;
        }
      while ((elem = __strtok_r (NULL, ":", &gconv_path)) != NULL);

      result[n].name = NULL;
      result[n].len = 0;
    }

  __gconv_path_elem = result ?: (struct path_elem *) &empty_path_elem;

  free (cwd);
}


/* Read all configuration files found in the user-specified and the default
   path.  This function should only be called once during the program's
   lifetime.  It disregards locking and synchronization because its only
   caller, __gconv_load_conf, handles this.  */
static void
__gconv_read_conf (void)
{
  int save_errno = errno;
  size_t cnt;

  /* First see whether we should use the cache.  */
  if (__gconv_load_cache () == 0)
    {
      /* Yes, we are done.  */
      __set_errno (save_errno);
      return;
    }

#ifndef STATIC_GCONV
  /* Find out where we have to look.  */
  __gconv_get_path ();

  for (cnt = 0; __gconv_path_elem[cnt].name != NULL; ++cnt)
    gconv_parseconfdir (__gconv_path_elem[cnt].name,
			__gconv_path_elem[cnt].len);
#endif

  /* Add the internal modules.  */
  for (cnt = 0; cnt < sizeof (builtin_modules) / sizeof (builtin_modules[0]);
       ++cnt)
    {
      struct gconv_alias fake_alias;

      fake_alias.fromname = (char *) builtin_modules[cnt].from_string;

      if (__tfind (&fake_alias, &__gconv_alias_db, __gconv_alias_compare)
	  != NULL)
	/* It'll conflict so don't add it.  */
	continue;

      insert_module (&builtin_modules[cnt], 0);
    }

  /* Add aliases for builtin conversions.  */
  const char *cp = builtin_aliases;
  do
    {
      const char *from = cp;
      const char *to = __rawmemchr (from, '\0') + 1;
      cp = __rawmemchr (to, '\0') + 1;

      add_alias2 (from, to, cp);
    }
  while (*cp != '\0');

  /* Restore the error number.  */
  __set_errno (save_errno);
}


/* This "once" variable is used to do a one-time load of the configuration.  */
__libc_once_define (static, once);


/* Read all configuration files found in the user-specified and the default
   path, but do it only "once" using __gconv_read_conf to do the actual
   work.  This is the function that must be called when reading iconv
   configuration.  */
void
__gconv_load_conf (void)
{
  __libc_once (once, __gconv_read_conf);
}


/* Free all resources if necessary.  */
libc_freeres_fn (free_mem)
{
  if (__gconv_path_elem != NULL && __gconv_path_elem != &empty_path_elem)
    free ((void *) __gconv_path_elem);
}
