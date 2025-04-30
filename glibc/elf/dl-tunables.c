/* The tunable framework.  See the README.tunables to know how to use the
   tunable in a glibc module.

   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

/* Mark symbols hidden in static PIE for early self relocation to work.  */
#if BUILD_PIE_DEFAULT
# pragma GCC visibility push(hidden)
#endif
#include <startup.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <stdlib.h>
#include <sysdep.h>
#include <fcntl.h>
#include <ldsodefs.h>
#include <array_length.h>

#define TUNABLES_INTERNAL 1
#include "dl-tunables.h"

#include <not-errno.h>

#if TUNABLES_FRONTEND == TUNABLES_FRONTEND_valstring
# define GLIBC_TUNABLES "GLIBC_TUNABLES"
#endif

#if TUNABLES_FRONTEND == TUNABLES_FRONTEND_valstring
static char *
tunables_strdup (const char *in)
{
  size_t i = 0;

  while (in[i++] != '\0');
  char *out = __sbrk (i);

  /* For most of the tunables code, we ignore user errors.  However,
     this is a system error - and running out of memory at program
     startup should be reported, so we do.  */
  if (out == (void *)-1)
    _dl_fatal_printf ("sbrk() failure while processing tunables\n");

  i--;

  while (i-- > 0)
    out[i] = in[i];

  return out;
}
#endif

static char **
get_next_env (char **envp, char **name, size_t *namelen, char **val,
	      char ***prev_envp)
{
  while (envp != NULL && *envp != NULL)
    {
      char **prev = envp;
      char *envline = *envp++;
      int len = 0;

      while (envline[len] != '\0' && envline[len] != '=')
	len++;

      /* Just the name and no value, go to the next one.  */
      if (envline[len] == '\0')
	continue;

      *name = envline;
      *namelen = len;
      *val = &envline[len + 1];
      *prev_envp = prev;

      return envp;
    }

  return NULL;
}

static void
do_tunable_update_val (tunable_t *cur, const tunable_val_t *valp,
		       const tunable_num_t *minp,
		       const tunable_num_t *maxp)
{
  tunable_num_t val, min, max;

  if (cur->type.type_code == TUNABLE_TYPE_STRING)
    {
      cur->val.strval = valp->strval;
      cur->initialized = true;
      return;
    }

  bool unsigned_cmp = unsigned_tunable_type (cur->type.type_code);

  val = valp->numval;
  min = minp != NULL ? *minp : cur->type.min;
  max = maxp != NULL ? *maxp : cur->type.max;

  /* We allow only increasingly restrictive bounds.  */
  if (tunable_val_lt (min, cur->type.min, unsigned_cmp))
    min = cur->type.min;

  if (tunable_val_gt (max, cur->type.max, unsigned_cmp))
    max = cur->type.max;

  /* Skip both bounds if they're inconsistent.  */
  if (tunable_val_gt (min, max, unsigned_cmp))
    {
      min = cur->type.min;
      max = cur->type.max;
    }

  /* Bail out if the bounds are not valid.  */
  if (tunable_val_lt (val, min, unsigned_cmp)
      || tunable_val_lt (max, val, unsigned_cmp))
    return;

  cur->val.numval = val;
  cur->type.min = min;
  cur->type.max = max;
  cur->initialized = true;
}

/* Validate range of the input value and initialize the tunable CUR if it looks
   good.  */
static void
tunable_initialize (tunable_t *cur, const char *strval)
{
  tunable_val_t val;

  if (cur->type.type_code != TUNABLE_TYPE_STRING)
    val.numval = (tunable_num_t) _dl_strtoul (strval, NULL);
  else
    val.strval = strval;
  do_tunable_update_val (cur, &val, NULL, NULL);
}

void
__tunable_set_val (tunable_id_t id, tunable_val_t *valp, tunable_num_t *minp,
		   tunable_num_t *maxp)
{
  tunable_t *cur = &tunable_list[id];

  do_tunable_update_val (cur, valp, minp, maxp);
}

#if TUNABLES_FRONTEND == TUNABLES_FRONTEND_valstring
/* Parse the tunable string TUNESTR and adjust it to drop any tunables that may
   be unsafe for AT_SECURE processes so that it can be used as the new
   environment variable value for GLIBC_TUNABLES.  VALSTRING is the original
   environment variable string which we use to make NULL terminated values so
   that we don't have to allocate memory again for it.  */
static void
parse_tunables (char *tunestr, char *valstring)
{
  if (tunestr == NULL || *tunestr == '\0')
    return;

  char *p = tunestr;
  size_t off = 0;

  while (true)
    {
      char *name = p;
      size_t len = 0;

      /* First, find where the name ends.  */
      while (p[len] != '=' && p[len] != ':' && p[len] != '\0')
	len++;

      /* If we reach the end of the string before getting a valid name-value
	 pair, bail out.  */
      if (p[len] == '\0')
	{
	  if (__libc_enable_secure)
	    tunestr[off] = '\0';
	  return;
	}

      /* We did not find a valid name-value pair before encountering the
	 colon.  */
      if (p[len]== ':')
	{
	  p += len + 1;
	  continue;
	}

      p += len + 1;

      /* Take the value from the valstring since we need to NULL terminate it.  */
      char *value = &valstring[p - tunestr];
      len = 0;

      while (p[len] != ':' && p[len] != '\0')
	len++;

      /* Add the tunable if it exists.  */
      for (size_t i = 0; i < sizeof (tunable_list) / sizeof (tunable_t); i++)
	{
	  tunable_t *cur = &tunable_list[i];

	  if (tunable_is_name (cur->name, name))
	    {
	      /* If we are in a secure context (AT_SECURE) then ignore the
		 tunable unless it is explicitly marked as secure.  Tunable
		 values take precedence over their envvar aliases.  We write
		 the tunables that are not SXID_ERASE back to TUNESTR, thus
		 dropping all SXID_ERASE tunables and any invalid or
		 unrecognized tunables.  */
	      if (__libc_enable_secure)
		{
		  if (cur->security_level != TUNABLE_SECLEVEL_SXID_ERASE)
		    {
		      if (off > 0)
			tunestr[off++] = ':';

		      const char *n = cur->name;

		      while (*n != '\0')
			tunestr[off++] = *n++;

		      tunestr[off++] = '=';

		      for (size_t j = 0; j < len; j++)
			tunestr[off++] = value[j];
		    }

		  if (cur->security_level != TUNABLE_SECLEVEL_NONE)
		    break;
		}

	      value[len] = '\0';
	      tunable_initialize (cur, value);
	      break;
	    }
	}

      if (p[len] != '\0')
	p += len + 1;
    }
}
#endif

/* Enable the glibc.malloc.check tunable in SETUID/SETGID programs only when
   the system administrator has created the /etc/suid-debug file.  This is a
   special case where we want to conditionally enable/disable a tunable even
   for setuid binaries.  We use the special version of access() to avoid
   setting ERRNO, which is a TLS variable since TLS has not yet been set
   up.  */
static __always_inline void
maybe_enable_malloc_check (void)
{
  tunable_id_t id = TUNABLE_ENUM_NAME (glibc, malloc, check);
  if (__libc_enable_secure && __access_noerrno ("/etc/suid-debug", F_OK) == 0)
    tunable_list[id].security_level = TUNABLE_SECLEVEL_NONE;
}

/* Initialize the tunables list from the environment.  For now we only use the
   ENV_ALIAS to find values.  Later we will also use the tunable names to find
   values.  */
void
__tunables_init (char **envp)
{
  char *envname = NULL;
  char *envval = NULL;
  size_t len = 0;
  char **prev_envp = envp;

  maybe_enable_malloc_check ();

  while ((envp = get_next_env (envp, &envname, &len, &envval,
			       &prev_envp)) != NULL)
    {
#if TUNABLES_FRONTEND == TUNABLES_FRONTEND_valstring
      if (tunable_is_name (GLIBC_TUNABLES, envname))
	{
	  char *new_env = tunables_strdup (envname);
	  if (new_env != NULL)
	    parse_tunables (new_env + len + 1, envval);
	  /* Put in the updated envval.  */
	  *prev_envp = new_env;
	  continue;
	}
#endif

      for (int i = 0; i < sizeof (tunable_list) / sizeof (tunable_t); i++)
	{
	  tunable_t *cur = &tunable_list[i];

	  /* Skip over tunables that have either been set already or should be
	     skipped.  */
	  if (cur->initialized || cur->env_alias[0] == '\0')
	    continue;

	  const char *name = cur->env_alias;

	  /* We have a match.  Initialize and move on to the next line.  */
	  if (tunable_is_name (name, envname))
	    {
	      /* For AT_SECURE binaries, we need to check the security settings of
		 the tunable and decide whether we read the value and also whether
		 we erase the value so that child processes don't inherit them in
		 the environment.  */
	      if (__libc_enable_secure)
		{
		  if (cur->security_level == TUNABLE_SECLEVEL_SXID_ERASE)
		    {
		      /* Erase the environment variable.  */
		      char **ep = prev_envp;

		      while (*ep != NULL)
			{
			  if (tunable_is_name (name, *ep))
			    {
			      char **dp = ep;

			      do
				dp[0] = dp[1];
			      while (*dp++);
			    }
			  else
			    ++ep;
			}
		      /* Reset the iterator so that we read the environment again
			 from the point we erased.  */
		      envp = prev_envp;
		    }

		  if (cur->security_level != TUNABLE_SECLEVEL_NONE)
		    continue;
		}

	      tunable_initialize (cur, envval);
	      break;
	    }
	}
    }
}

void
__tunables_print (void)
{
  for (int i = 0; i < array_length (tunable_list); i++)
    {
      const tunable_t *cur = &tunable_list[i];
      if (cur->type.type_code == TUNABLE_TYPE_STRING
	  && cur->val.strval == NULL)
	_dl_printf ("%s:\n", cur->name);
      else
	{
	  _dl_printf ("%s: ", cur->name);
	  switch (cur->type.type_code)
	    {
	    case TUNABLE_TYPE_INT_32:
	      _dl_printf ("%d (min: %d, max: %d)\n",
			  (int) cur->val.numval,
			  (int) cur->type.min,
			  (int) cur->type.max);
	      break;
	    case TUNABLE_TYPE_UINT_64:
	      _dl_printf ("0x%lx (min: 0x%lx, max: 0x%lx)\n",
			  (long int) cur->val.numval,
			  (long int) cur->type.min,
			  (long int) cur->type.max);
	      break;
	    case TUNABLE_TYPE_SIZE_T:
	      _dl_printf ("0x%zx (min: 0x%zx, max: 0x%zx)\n",
			  (size_t) cur->val.numval,
			  (size_t) cur->type.min,
			  (size_t) cur->type.max);
	      break;
	    case TUNABLE_TYPE_STRING:
	      _dl_printf ("%s\n", cur->val.strval);
	      break;
	    default:
	      __builtin_unreachable ();
	    }
	}
    }
}

/* Set the tunable value.  This is called by the module that the tunable exists
   in. */
void
__tunable_get_val (tunable_id_t id, void *valp, tunable_callback_t callback)
{
  tunable_t *cur = &tunable_list[id];

  switch (cur->type.type_code)
    {
    case TUNABLE_TYPE_UINT_64:
	{
	  *((uint64_t *) valp) = (uint64_t) cur->val.numval;
	  break;
	}
    case TUNABLE_TYPE_INT_32:
	{
	  *((int32_t *) valp) = (int32_t) cur->val.numval;
	  break;
	}
    case TUNABLE_TYPE_SIZE_T:
	{
	  *((size_t *) valp) = (size_t) cur->val.numval;
	  break;
	}
    case TUNABLE_TYPE_STRING:
	{
	  *((const char **)valp) = cur->val.strval;
	  break;
	}
    default:
      __builtin_unreachable ();
    }

  if (cur->initialized && callback != NULL)
    callback (&cur->val);
}

rtld_hidden_def (__tunable_get_val)
